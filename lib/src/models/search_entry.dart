/// A single item in the searchable knowledge base.
///
/// [SearchEntry] is an immutable value object representing one Q&A pair.
/// It maps directly to a row in the SQLite `entries` table.
///
/// ## Database schema expected by [HybridSearchEngine]
///
/// ```sql
/// CREATE TABLE entries (
///   id       INTEGER PRIMARY KEY,
///   category TEXT    NOT NULL,
///   question TEXT    NOT NULL,
///   answer   TEXT    NOT NULL
/// );
/// ```
///
/// Column names are configurable via [HybridSearchConfig]:
/// ```dart
/// HybridSearchConfig(
///   tableName:      'my_items',
///   idColumn:       'item_id',
///   categoryColumn: 'tag',
///   questionColumn: 'title',
///   answerColumn:   'body',
/// )
/// ```
final class SearchEntry {
  /// Creates a [SearchEntry].
  const SearchEntry({
    required this.id,
    required this.category,
    required this.question,
    required this.answer,
  });

  /// Creates a [SearchEntry] from a SQLite row map using column names from
  /// [idColumn], [categoryColumn], [questionColumn], and [answerColumn].
  factory SearchEntry.fromMap(
    Map<String, Object?> map, {
    String idColumn = 'id',
    String categoryColumn = 'category',
    String questionColumn = 'question',
    String answerColumn = 'answer',
  }) {
    return SearchEntry(
      id: map[idColumn] as int,
      category: map[categoryColumn] as String? ?? '',
      question: map[questionColumn] as String,
      answer: map[answerColumn] as String,
    );
  }

  /// 1-based row identifier matching the SQLite `id` column and the
  /// corresponding index in the precomputed embeddings list (`id = index + 1`).
  final int id;

  /// Topic or domain label for this entry (e.g. `"Flutter"`, `"Dart"`).
  ///
  /// Used for display purposes; not involved in the search algorithm.
  final String category;

  /// The question or search target text.
  ///
  /// This is the field indexed in FTS5 and matched against the user query.
  final String question;

  /// The answer or document body returned when this entry matches.
  ///
  /// May contain Markdown formatting.
  final String answer;

  /// Serialises the entry to a `Map` for insertion or debugging.
  Map<String, Object> toMap() => <String, Object>{
        'id': id,
        'category': category,
        'question': question,
        'answer': answer,
      };

  @override
  String toString() =>
      'SearchEntry(id: $id, category: $category, question: $question)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SearchEntry &&
          id == other.id &&
          category == other.category &&
          question == other.question &&
          answer == other.answer;

  @override
  int get hashCode => Object.hash(id, category, question, answer);
}
