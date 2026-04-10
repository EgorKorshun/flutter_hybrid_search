import 'dart:convert';

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
///
/// ## Custom metadata
///
/// Attach domain-specific key-value pairs via [metadata]:
/// ```dart
/// const entry = SearchEntry(
///   id: 1,
///   category: 'Flutter',
///   question: 'What is a widget?',
///   answer: 'Everything is a widget.',
///   metadata: {'priority': 1, 'tags': ['ui', 'basics']},
/// );
/// ```
///
/// To persist metadata, add a `TEXT` column to your schema and pass
/// `metadataColumn` to [fromMap] / [toMap].
final class SearchEntry {
  /// Creates a [SearchEntry].
  const SearchEntry({
    required this.id,
    required this.category,
    required this.question,
    required this.answer,
    this.metadata = const <String, Object?>{},
  });

  /// Creates a [SearchEntry] from a SQLite row map using column names from
  /// [idColumn], [categoryColumn], [questionColumn], and [answerColumn].
  ///
  /// If [metadataColumn] is provided and the row contains a non-null value for
  /// that column, the value is JSON-decoded into [metadata].
  factory SearchEntry.fromMap(
    Map<String, Object?> map, {
    String idColumn = 'id',
    String categoryColumn = 'category',
    String questionColumn = 'question',
    String answerColumn = 'answer',
    String? metadataColumn,
  }) {
    Map<String, Object?> meta = const <String, Object?>{};
    if (metadataColumn != null && map[metadataColumn] != null) {
      final Object? raw = map[metadataColumn];
      if (raw is String) {
        meta = (jsonDecode(raw) as Map<String, dynamic>).cast<String, Object?>();
      }
    }
    return SearchEntry(
      id: map[idColumn] as int,
      category: map[categoryColumn] as String? ?? '',
      question: map[questionColumn] as String,
      answer: map[answerColumn] as String,
      metadata: meta,
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

  /// Arbitrary domain-specific key-value pairs attached to this entry.
  ///
  /// Values should be JSON-serialisable: [String], [int], [double], [bool],
  /// [List], [Map<String, Object?>], or `null`.
  ///
  /// This field is **not stored in the SQLite main table** by default.
  /// To persist metadata, add a `TEXT` column (storing JSON) to your schema
  /// and pass `metadataColumn` to [fromMap] and [toMap].
  ///
  /// Default: empty map.
  final Map<String, Object?> metadata;

  /// Serialises the entry to a `Map` for insertion or debugging.
  ///
  /// If [metadataColumn] is provided and [metadata] is non-empty, the
  /// metadata is JSON-encoded and stored under [metadataColumn].
  Map<String, Object> toMap({String? metadataColumn}) {
    final Map<String, Object> m = <String, Object>{
      'id': id,
      'category': category,
      'question': question,
      'answer': answer,
    };
    if (metadataColumn != null && metadata.isNotEmpty) {
      m[metadataColumn] = jsonEncode(metadata);
    }
    return m;
  }

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
          answer == other.answer &&
          _mapEqual(metadata, other.metadata);

  @override
  int get hashCode => Object.hash(id, category, question, answer, metadata.length);

  static bool _mapEqual(Map<String, Object?> a, Map<String, Object?> b) {
    if (identical(a, b)) return true;
    if (a.length != b.length) return false;
    for (final String key in a.keys) {
      if (!b.containsKey(key) || b[key] != a[key]) return false;
    }
    return true;
  }
}
