/// Timing and diagnostic metadata from a [HybridSearchEngine.search] call.
///
/// Returned by [HybridSearchEngine.searchWithMetadata] alongside the results.
/// All durations are in milliseconds.
///
/// ```dart
/// final (:results, :metadata) = await engine.searchWithMetadata('flutter');
/// print('Total: ${metadata.totalMs.toStringAsFixed(1)} ms');
/// print('Vector: ${metadata.vectorMs.toStringAsFixed(1)} ms');
/// print('Candidates: ${metadata.candidateCount}');
/// ```
final class SearchMetadata {
  /// Creates a [SearchMetadata].
  const SearchMetadata({
    required this.embedMs,
    required this.vectorMs,
    required this.ftsMs,
    required this.typoMs,
    required this.rerankMs,
    required this.totalMs,
    required this.candidateCount,
    required this.vectorCandidateCount,
    required this.keywordCandidateCount,
  });

  /// Time spent generating the query embedding.
  final double embedMs;

  /// Time spent on vector scoring (linear scan or HNSW).
  final double vectorMs;

  /// Time spent on FTS5 full-text search.
  final double ftsMs;

  /// Time spent on typo-tolerant keyword scan.
  final double typoMs;

  /// Time spent in the reranker (scoring + dedup + filtering).
  final double rerankMs;

  /// Total wall-clock time for the entire search call.
  final double totalMs;

  /// Total number of candidates in the pool (union of vector + keyword).
  final int candidateCount;

  /// Number of candidates contributed by vector scoring.
  final int vectorCandidateCount;

  /// Number of candidates contributed by keyword matching (FTS + typo).
  final int keywordCandidateCount;

  @override
  String toString() => 'SearchMetadata('
      'total: ${totalMs.toStringAsFixed(1)} ms, '
      'embed: ${embedMs.toStringAsFixed(1)} ms, '
      'vector: ${vectorMs.toStringAsFixed(1)} ms, '
      'fts: ${ftsMs.toStringAsFixed(1)} ms, '
      'typo: ${typoMs.toStringAsFixed(1)} ms, '
      'rerank: ${rerankMs.toStringAsFixed(1)} ms, '
      'candidates: $candidateCount'
      ')';
}
