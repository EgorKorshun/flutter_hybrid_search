import '../models/search_result.dart';

/// Pure-function ranking utilities for the hybrid search pipeline.
///
/// Provides boost constants, typo-tolerance matching, FTS query building,
/// and score combination helpers. All methods are static and stateless,
/// making them straightforward to unit-test and reuse in custom rerankers.
///
/// ## Boost signal overview
///
/// | Signal | Boost | Description |
/// |---|---|---|
/// | FTS exact match | +[ftsBoost] (0.5) | Entry found by FTS5 `MATCH` query |
/// | Typo match | +[typoBoost] (0.7) | Entry matched with 1-char edit distance |
/// | Concise question | +[conciseMatchBoost] (0.5) | Short on-topic question |
///
/// ## Usage
///
/// ```dart
/// // Build FTS5 MATCH query from content words.
/// final query = SearchRanking.buildFtsMatchQuery(['flutter', 'state']);
/// // → "question: flutter OR question: state"
///
/// // Detect typo-tolerant match.
/// final matches = SearchRanking.questionMatchesWithTypo(['datt'], 'What is Dart?');
/// // → true  (datt ≈ dart)
///
/// // Perfect-match shortcutting.
/// final final = SearchRanking.singleIfPerfect(results);
/// ```
abstract final class SearchRanking {
  SearchRanking._();

  // -------------------------------------------------------------------------
  // Boost constants
  // -------------------------------------------------------------------------

  /// Score boost added when an entry is found by the FTS5 `MATCH` query.
  ///
  /// FTS matches are exact or phrase matches on the [questionColumn], so
  /// they deserve a strong positive signal.
  static const double ftsBoost = 0.5;

  /// Score boost added when an entry matches the query with 1-character
  /// typo tolerance but was NOT found by FTS5.
  ///
  /// Higher than [ftsBoost] because typo matches catch user errors that FTS
  /// misses, and are less common (so they are more discriminative).
  static const double typoBoost = 0.7;

  /// Maximum additional boost for a concise, on-topic question match.
  ///
  /// Applied in full when the question contains exactly the same words as
  /// the query. Reduced for one or two extra words. See [conciseMatchBoostFor].
  static const double conciseMatchBoost = 0.5;

  /// Cosine similarity threshold above which a single result is returned
  /// immediately, bypassing the normal top-k selection.
  ///
  /// When exactly one candidate scores ≥ [perfectScoreThreshold], it is
  /// returned as the sole result — showing tangential matches alongside a
  /// perfect answer degrades UX.
  static const double perfectScoreThreshold = 0.999;

  // -------------------------------------------------------------------------
  // Query word extraction
  // -------------------------------------------------------------------------

  /// Extracts normalised words from [query] for FTS matching and overlap
  /// filtering.
  ///
  /// Pipeline: trim → lowercase → strip punctuation → split on whitespace.
  ///
  /// Example:
  /// ```dart
  /// SearchRanking.queryWordsForFts('What is Flutter?');
  /// // → ['what', 'is', 'flutter']
  /// ```
  static List<String> queryWordsForFts(String query) {
    final String normalised = query
        .trim()
        .toLowerCase()
        .replaceAll(_reNonWord, ' ')
        .replaceAll(_reWhitespace, ' ')
        .trim();

    return normalised.split(' ').where((String w) => w.isNotEmpty).toList();
  }

  // -------------------------------------------------------------------------
  // FTS query building
  // -------------------------------------------------------------------------

  /// Builds an FTS5 `MATCH` query string for [words] restricted to [column].
  ///
  /// Each word is OR-combined so that entries matching any of the query
  /// words are returned. The column restriction avoids false positives from
  /// matches in the answer text.
  ///
  /// Example:
  /// ```dart
  /// SearchRanking.buildFtsMatchQuery(['dart', 'flutter']);
  /// // → "question: dart OR question: flutter"
  /// ```
  static String buildFtsMatchQuery(
    List<String> words, {
    String column = 'question',
  }) {
    if (words.isEmpty) return '';
    return words.map((String w) => '$column: ${_escapeFts(w)}').join(' OR ');
  }

  /// Escapes a word for safe inclusion in an FTS5 MATCH expression.
  static String _escapeFts(String word) => word.replaceAll('"', '""');

  // -------------------------------------------------------------------------
  // Keyword overlap filtering
  // -------------------------------------------------------------------------

  /// Counts how many of [queryWords] appear (exactly or within one typo) in
  /// [question].
  ///
  /// Used to filter results that have zero keyword overlap with the query —
  /// a safety net against purely semantic hallucinations.
  ///
  /// Returns `0` if no words match.
  static int countQueryWordsMatchingQuestion(
    List<String> queryWords,
    String question,
  ) {
    final List<String> questionWords = _normaliseWords(question);
    int count = 0;
    for (final String qw in queryWords) {
      if (questionWords.any((String w) => _withinOneTypo(qw, w))) count++;
    }
    return count;
  }

  /// Returns `true` if at least one word in [queryWords] matches a word in
  /// [question] within one-character edit distance.
  static bool questionMatchesWithTypo(
    List<String> queryWords,
    String question,
  ) =>
      countQueryWordsMatchingQuestion(queryWords, question) > 0;

  // -------------------------------------------------------------------------
  // Concise match boost
  // -------------------------------------------------------------------------

  /// Returns a boost value rewarding short, on-topic questions.
  ///
  /// A concise question is one where:
  /// - All [queryWords] appear in [question] (exact or typo).
  /// - [question] has at most `queryWords.length + maxExtraWords` words.
  ///
  /// The boost decreases as the question grows:
  ///
  /// | Extra words beyond query | Boost factor |
  /// |---|---|
  /// | 0 | 1.0 × [conciseMatchBoost] |
  /// | 1 | 0.7 × [conciseMatchBoost] |
  /// | 2+ | 0.4 × [conciseMatchBoost] |
  ///
  /// Example: for query `"What is Dart?"` (3 content words after stopword
  /// removal = `["dart"]`), the question `"What is Dart?"` scores full
  /// boost, while `"Dart vs Kotlin vs Java"` scores none.
  static double conciseMatchBoostFor(
    List<String> queryWords,
    String question, {
    int maxExtraWords = 1,
  }) {
    if (queryWords.isEmpty) return 0.0;

    final List<String> questionWords = _normaliseWords(question);
    if (questionWords.length > queryWords.length + maxExtraWords) return 0.0;

    final int matched = countQueryWordsMatchingQuestion(queryWords, question);
    if (matched < queryWords.length) return 0.0;

    final int extra = questionWords.length - queryWords.length;
    if (extra <= 0) return conciseMatchBoost;
    if (extra == 1) return conciseMatchBoost * 0.7;
    return conciseMatchBoost * 0.4;
  }

  // -------------------------------------------------------------------------
  // Score combination helpers
  // -------------------------------------------------------------------------

  /// Returns top [limit] IDs by combined vector + FTS boost score.
  ///
  /// Each entry's score = vectorScore + [ftsBoost] if it is in [ftsIds].
  static List<int> topIdsByCombinedScore(
    Map<int, double> idToVectorScore,
    Set<int> ftsIds, {
    required int limit,
    double boost = ftsBoost,
  }) {
    final List<(int, double)> scored = <(int, double)>[
      for (final MapEntry<int, double> e in idToVectorScore.entries)
        (e.key, e.value + (ftsIds.contains(e.key) ? boost : 0.0)),
    ];
    scored.sort(((int, double) a, (int, double) b) => b.$2.compareTo(a.$2));
    return scored.take(limit).map<int>(((int, double) r) => r.$1).toList();
  }

  // -------------------------------------------------------------------------
  // Perfect-match shortcutting
  // -------------------------------------------------------------------------

  /// Returns the single result whose score ≥ [threshold] if exactly one such
  /// result exists, otherwise returns the full [results] list unchanged.
  ///
  /// Prevents showing unrelated results alongside a near-perfect match.
  static List<SearchResult> singleIfPerfect(
    List<SearchResult> results, {
    double threshold = perfectScoreThreshold,
  }) {
    final List<SearchResult> perfect =
        results.where((SearchResult r) => r.score >= threshold).toList();
    return perfect.length == 1 ? perfect : results;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /// Checks whether [a] and [b] are within one-character edit distance
  /// (substitution, insertion, or deletion).
  static bool _withinOneTypo(String a, String b) {
    if (a == b) return true;
    final int la = a.length;
    final int lb = b.length;
    if (la == lb) return _diffByOne(a, b);
    if (la == lb + 1) return _canDrop(a, b);
    if (lb == la + 1) return _canDrop(b, a);
    return false;
  }

  /// Returns `true` if [a] and [b] (same length) differ in exactly one char.
  static bool _diffByOne(String a, String b) {
    int diffs = 0;
    for (int i = 0; i < a.length; i++) {
      if (a.codeUnitAt(i) != b.codeUnitAt(i) && ++diffs > 1) return false;
    }
    return true;
  }

  /// Returns `true` if dropping one character from [longer] yields [shorter].
  static bool _canDrop(String longer, String shorter) {
    // Two-pointer scan: O(n) without substring allocations.
    int i = 0;
    int j = 0;
    bool skipped = false;
    while (i < longer.length && j < shorter.length) {
      if (longer.codeUnitAt(i) != shorter.codeUnitAt(j)) {
        if (skipped) return false;
        skipped = true;
        i++; // skip one char in longer
      } else {
        i++;
        j++;
      }
    }
    return j == shorter.length;
  }

  /// Normalises [text] to lowercase words stripped of punctuation.
  static List<String> _normaliseWords(String text) {
    return text
        .toLowerCase()
        .replaceAll(_reNonWord, ' ')
        .split(_reWhitespace)
        .where((String w) => w.isNotEmpty)
        .toList();
  }

  // Cached RegExp instances to avoid re-compilation on every call.
  static final RegExp _reNonWord = RegExp(_Regex.nonWord, unicode: true);
  static final RegExp _reWhitespace = RegExp(_Regex.whitespace);
}

/// Compiled regex patterns.
abstract final class _Regex {
  _Regex._();

  static const String nonWord = r'[^\p{L}\p{N}_\s]';
  static const String whitespace = r'\s+';
}
