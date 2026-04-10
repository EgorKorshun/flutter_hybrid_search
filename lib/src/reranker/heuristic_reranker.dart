import 'dart:typed_data';

import '../models/score_breakdown.dart';
import '../models/search_entry.dart';
import '../models/search_result.dart';
import '../ranking/search_ranking.dart';
import 'reranker.dart';

/// Rule-based reranker that combines cosine similarity with keyword signals.
///
/// Scoring formula for each candidate:
///
/// ```
/// score = vectorScore
///       + ftsBoost      (if found by FTS5 MATCH)
///       + typoBoost     (if found by typo matching but NOT by FTS5)
///       + conciseBoost  (if question is short and on-topic)
/// ```
///
/// After scoring, candidates are sorted, oversampled (2×), deduplicated by
/// normalised question text, and finally passed through
/// [SearchRanking.singleIfPerfect].
///
/// Every [SearchResult] produced by this reranker includes a [ScoreBreakdown]
/// that records each signal's individual contribution.
///
/// This reranker requires **no model files** and works on every platform.
/// It is the default (and fallback) reranker in [HybridSearchEngine].
///
/// ## Boost values
///
/// | Signal | Default |
/// |---|---|
/// | FTS5 exact match | +0.5 ([SearchRanking.ftsBoost]) |
/// | Typo-only match | +0.7 ([SearchRanking.typoBoost]) |
/// | Concise question | +0.0 – +0.5 ([SearchRanking.conciseMatchBoost]) |
///
/// All boost constants are configurable via [SearchRanking] at compile time.
class HeuristicReranker implements RerankerInterface {
  /// Creates a [HeuristicReranker].
  const HeuristicReranker();

  // Oversampling factor: take `limit × _oversample` candidates before
  // deduplication to ensure [limit] unique results after duplicate removal.
  static const int _oversample = 2;
  static const String _method = 'heuristic';

  @override
  List<SearchResult> rerank(
    String query,
    RerankerCandidates candidates,
    Set<int> keywordMatchIds, {
    int limit = 3,
    Float32List? queryEmbedding,
    Set<int>? ftsIds,
    List<String>? contentWords,
  }) {
    if (candidates.isEmpty) return <SearchResult>[];

    final List<String> words =
        contentWords ?? SearchRanking.queryWordsForFts(query);
    // Typo-only IDs: matched by typo tolerance but NOT by FTS5.
    final Set<int> typoOnly =
        ftsIds != null ? keywordMatchIds.difference(ftsIds) : <int>{};

    // Score every candidate with the combined signal, capturing the breakdown.
    final Map<int, ScoreBreakdown> breakdownById = <int, ScoreBreakdown>{};
    final List<(int, double)> scored = <(int, double)>[
      for (final (
            :SearchEntry entry,
            :double vectorScore,
            embedding: Float32List? _,
          ) in candidates)
        () {
          final ({double total, double fts, double typo, double concise}) r =
              _combinedScore(
            entry: entry,
            base: vectorScore,
            ftsIds: ftsIds,
            typoOnly: typoOnly,
            queryWords: words,
          );
          breakdownById[entry.id] = ScoreBreakdown(
            vectorScore: vectorScore,
            ftsScore: r.fts,
            typoScore: r.typo,
            conciseScore: r.concise,
            totalScore: r.total,
          );
          return (entry.id, r.total);
        }(),
    ];

    // Sort descending by combined score.
    scored.sort(((int, double) a, (int, double) b) => b.$2.compareTo(a.$2));

    // Build lookup maps for deduplication.
    final Map<int, SearchEntry> byId = <int, SearchEntry>{
      for (final c in candidates) c.entry.id: c.entry,
    };
    final Map<int, double> scoreById = <int, double>{
      for (final (int id, double s) in scored) id: s,
    };

    final List<SearchResult> results = _deduplicate(
      scored
          .take(limit * _oversample)
          .map<int>(((int, double) r) => r.$1)
          .toList(),
      byId,
      scoreById,
      breakdownById,
      limit,
    );

    return SearchRanking.singleIfPerfect(results);
  }

  /// Computes the combined score and per-signal components for a single [entry].
  ({double total, double fts, double typo, double concise}) _combinedScore({
    required SearchEntry entry,
    required double base,
    required Set<int>? ftsIds,
    required Set<int> typoOnly,
    required List<String> queryWords,
  }) {
    final double fts = (ftsIds != null && ftsIds.contains(entry.id))
        ? SearchRanking.ftsBoost
        : 0.0;
    final double typo =
        typoOnly.contains(entry.id) ? SearchRanking.typoBoost : 0.0;
    final double concise =
        SearchRanking.conciseMatchBoostFor(queryWords, entry.question);

    return (total: base + fts + typo + concise, fts: fts, typo: typo, concise: concise);
  }

  /// Returns up to [limit] unique results, deduplicating by lowercased
  /// question text. Each result includes a [ScoreBreakdown].
  List<SearchResult> _deduplicate(
    List<int> ids,
    Map<int, SearchEntry> byId,
    Map<int, double> scoreById,
    Map<int, ScoreBreakdown> breakdownById,
    int limit,
  ) {
    final List<SearchResult> out = <SearchResult>[];
    final Set<String> seen = <String>{};

    for (final int id in ids) {
      if (out.length >= limit) break;
      final SearchEntry? entry = byId[id];
      if (entry == null) continue;
      final String key = entry.question.trim().toLowerCase();
      if (seen.contains(key)) continue;
      seen.add(key);
      out.add(SearchResult(
        entry: entry,
        score: scoreById[id] ?? 0.0,
        method: _method,
        breakdown: breakdownById[id],
      ));
    }

    return out;
  }
}
