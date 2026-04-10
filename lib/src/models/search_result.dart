import 'score_breakdown.dart';
import 'search_entry.dart';

/// A single match returned by [HybridSearchEngine.search].
///
/// Encapsulates the matched [entry], its relevance [score], the [method]
/// identifier describing which signals contributed to the score, and an
/// optional [breakdown] that exposes each signal's individual contribution.
///
/// Results are sorted by [score] in descending order (highest relevance first).
///
/// ```dart
/// final results = await engine.search('What is Flutter?');
/// for (final result in results) {
///   print('${result.score.toStringAsFixed(3)}  ${result.entry.question}');
///   if (result.breakdown case final b?) {
///     print('  vector=${b.vectorScore.toStringAsFixed(3)}'
///           '  fts=${b.ftsScore.toStringAsFixed(3)}');
///   }
/// }
/// ```
final class SearchResult {
  /// Creates a [SearchResult].
  const SearchResult({
    required this.entry,
    required this.score,
    required this.method,
    this.breakdown,
  });

  /// The matched knowledge-base entry.
  final SearchEntry entry;

  /// Composite relevance score (not bounded to 0–1 after boost signals are
  /// applied — can slightly exceed 1.0).
  ///
  /// Higher is more relevant. The base component is cosine similarity in
  /// [0, 1]; boost signals from FTS matches, typo matches, and concise
  /// question detection can raise this above 1.0.
  final double score;

  /// Identifier of the search strategy that produced this result.
  ///
  /// Possible values:
  /// - `"heuristic"` — heuristic reranker applied (default)
  /// - Any custom string returned by a custom [RerankerInterface]
  final String method;

  /// Per-signal score breakdown for this result.
  ///
  /// Populated by [HeuristicReranker] and any custom [RerankerInterface] that
  /// opts in. `null` when the reranker did not provide a breakdown.
  ///
  /// Use [breakdown] to debug why a result ranked where it did or to build
  /// explainability UI.
  final ScoreBreakdown? breakdown;

  /// Returns a copy with the given fields replaced.
  ///
  /// ```dart
  /// final highlighted = result.copyWith(score: result.score * 1.1);
  /// ```
  SearchResult copyWith({
    SearchEntry? entry,
    double? score,
    String? method,
    ScoreBreakdown? breakdown,
  }) =>
      SearchResult(
        entry: entry ?? this.entry,
        score: score ?? this.score,
        method: method ?? this.method,
        breakdown: breakdown ?? this.breakdown,
      );

  @override
  String toString() =>
      'SearchResult(score: ${score.toStringAsFixed(4)}, method: $method, '
      'question: ${entry.question})';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SearchResult &&
          entry == other.entry &&
          score == other.score &&
          method == other.method &&
          breakdown == other.breakdown;

  @override
  int get hashCode => Object.hash(entry, score, method, breakdown);
}
