/// Per-signal score breakdown for a single [SearchResult].
///
/// Each field records the contribution of one ranking signal to the final
/// [totalScore]. Use this to debug why a result ranked where it did or to
/// tune boost constants in [SearchRanking].
///
/// ```dart
/// final result = results.first;
/// print(result.breakdown?.vectorScore); // cosine similarity contribution
/// print(result.breakdown?.ftsScore);    // FTS5 exact-match boost
/// print(result.breakdown?.totalScore);  // sum of all signals
/// ```
final class ScoreBreakdown {
  /// Creates a [ScoreBreakdown].
  const ScoreBreakdown({
    required this.vectorScore,
    required this.ftsScore,
    required this.typoScore,
    required this.conciseScore,
    required this.totalScore,
  });

  /// Cosine similarity between the query embedding and this entry's embedding.
  ///
  /// Range: `[0.0, 1.0]`.
  final double vectorScore;

  /// FTS5 exact-match boost applied when the entry was found by the FTS5
  /// MATCH query.
  ///
  /// Either [SearchRanking.ftsBoost] (0.5) or `0.0`.
  final double ftsScore;

  /// Typo-match boost applied when the entry matched via Levenshtein-1
  /// but **not** via FTS5.
  ///
  /// Either [SearchRanking.typoBoost] (0.7) or `0.0`.
  final double typoScore;

  /// Concise-question boost from [SearchRanking.conciseMatchBoostFor].
  ///
  /// Range: `[0.0, SearchRanking.conciseMatchBoost]`.
  final double conciseScore;

  /// Sum of all signal contributions:
  /// `vectorScore + ftsScore + typoScore + conciseScore`.
  ///
  /// This is equal to [SearchResult.score].
  final double totalScore;

  /// Returns a copy with the given fields replaced.
  ScoreBreakdown copyWith({
    double? vectorScore,
    double? ftsScore,
    double? typoScore,
    double? conciseScore,
    double? totalScore,
  }) =>
      ScoreBreakdown(
        vectorScore: vectorScore ?? this.vectorScore,
        ftsScore: ftsScore ?? this.ftsScore,
        typoScore: typoScore ?? this.typoScore,
        conciseScore: conciseScore ?? this.conciseScore,
        totalScore: totalScore ?? this.totalScore,
      );

  @override
  String toString() =>
      'ScoreBreakdown(total: ${totalScore.toStringAsFixed(4)}, '
      'vector: ${vectorScore.toStringAsFixed(4)}, '
      'fts: ${ftsScore.toStringAsFixed(4)}, '
      'typo: ${typoScore.toStringAsFixed(4)}, '
      'concise: ${conciseScore.toStringAsFixed(4)})';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ScoreBreakdown &&
          vectorScore == other.vectorScore &&
          ftsScore == other.ftsScore &&
          typoScore == other.typoScore &&
          conciseScore == other.conciseScore &&
          totalScore == other.totalScore;

  @override
  int get hashCode =>
      Object.hash(vectorScore, ftsScore, typoScore, conciseScore, totalScore);
}
