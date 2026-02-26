import 'dart:typed_data';

import '../models/search_entry.dart';
import '../models/search_result.dart';

/// The list of candidates fed into a [RerankerInterface].
///
/// Each record contains:
/// - [entry] — the candidate [SearchEntry].
/// - [vectorScore] — cosine similarity with the query vector (0–1).
/// - [embedding] — precomputed entry embedding (nullable; required for
///   learned rerankers, optional for heuristic ones).
typedef RerankerCandidates
    = List<({SearchEntry entry, double vectorScore, Float32List? embedding})>;

/// Contract for reranking an initial list of search candidates.
///
/// A reranker refines the rough ordering produced by vector search by
/// combining additional signals — keyword matches, question conciseness,
/// entry embeddings, or a learned model.
///
/// ## Built-in implementations
///
/// - [HeuristicReranker] — rule-based signal combination (default fallback,
///   no model required).
///
/// ## Custom implementation
///
/// ```dart
/// class MyReranker implements RerankerInterface {
///   const MyReranker();
///
///   @override
///   List<SearchResult> rerank(
///     String query,
///     RerankerCandidates candidates,
///     Set<int> keywordMatchIds, {
///     int limit = 3,
///     Float32List? queryEmbedding,
///     Set<int>? ftsIds,
///     List<String>? contentWords,
///   }) {
///     // Sort by vectorScore only, take top [limit].
///     final sorted = candidates.toList()
///       ..sort((a, b) => b.vectorScore.compareTo(a.vectorScore));
///     return sorted.take(limit).map((c) => SearchResult(
///       entry: c.entry,
///       score: c.vectorScore,
///       method: 'custom',
///     )).toList();
///   }
/// }
/// ```
abstract interface class RerankerInterface {
  /// Reranks [candidates] and returns at most [limit] [SearchResult]s.
  ///
  /// Parameters:
  /// - [query] — the original user query string.
  /// - [candidates] — entries with their cosine scores and optional embeddings.
  /// - [keywordMatchIds] — IDs that matched keywords (FTS5 + typo-tolerant).
  /// - [limit] — maximum results to return.
  /// - [queryEmbedding] — query vector (for learned rerankers).
  /// - [ftsIds] — subset of [keywordMatchIds] from exact FTS5 matches.
  /// - [contentWords] — stopword-stripped query tokens for boost computation.
  ///
  /// Returns results sorted by descending combined score, deduplicated by
  /// question text.
  List<SearchResult> rerank(
    String query,
    RerankerCandidates candidates,
    Set<int> keywordMatchIds, {
    int limit = 3,
    Float32List? queryEmbedding,
    Set<int>? ftsIds,
    List<String>? contentWords,
  });
}
