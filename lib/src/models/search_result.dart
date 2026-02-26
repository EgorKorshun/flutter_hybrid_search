import 'search_entry.dart';

/// A single match returned by [HybridSearchEngine.search].
///
/// Encapsulates the matched [entry], its relevance [score], and the
/// [method] identifier describing which signals contributed to the score.
///
/// Results are sorted by [score] in descending order (highest relevance first).
///
/// ```dart
/// final results = await engine.search('What is Flutter?');
/// for (final result in results) {
///   print('${result.score.toStringAsFixed(3)}  ${result.entry.question}');
/// }
/// ```
final class SearchResult {
  /// Creates a [SearchResult].
  const SearchResult({
    required this.entry,
    required this.score,
    required this.method,
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
  /// - `"hybrid"` — vector similarity + FTS5 + typo-tolerance (default)
  /// - `"heuristic"` — heuristic reranker applied on top
  /// - Any custom string returned by a custom [RerankerInterface]
  final String method;

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
          method == other.method;

  @override
  int get hashCode => Object.hash(entry, score, method);
}
