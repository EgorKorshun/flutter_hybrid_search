// Tests use explicit default values to make expected behaviour obvious.
// ignore_for_file: avoid_redundant_argument_values

import 'dart:typed_data';

import 'package:flutter_hybrid_search/flutter_hybrid_search.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates an in-memory SQLite database with the default schema and seeds it
/// with [entries].
Future<Database> _makeDb(List<SearchEntry> entries) async {
  sqfliteFfiInit();
  databaseFactory = databaseFactoryFfi;
  final Database db = await openDatabase(
    inMemoryDatabasePath,
    version: 1,
    onCreate: (Database db, int version) async {
      await db.execute('''
        CREATE TABLE entries (
          id       INTEGER PRIMARY KEY,
          category TEXT NOT NULL,
          question TEXT NOT NULL,
          answer   TEXT NOT NULL
        )
      ''');
      await db.execute('''
        CREATE VIRTUAL TABLE fts USING fts5(
          question,
          content=entries,
          content_rowid=id
        )
      ''');
      for (final SearchEntry e in entries) {
        await db.insert('entries', e.toMap());
        await db.insert('fts', <String, Object?>{'question': e.question});
      }
    },
  );
  return db;
}

/// Constant dummy embeddings (128-dimensional unit vectors).
///
/// Each entry gets a slightly different vector so cosine similarity can
/// distinguish them.
List<Float32List> _makeEmbeddings(int count, {int dim = 128}) {
  return List<Float32List>.generate(count, (int i) {
    final Float32List v = Float32List(dim);
    v[i % dim] = 1.0; // one-hot-like, guaranteed unit norm
    return v;
  });
}

/// A stub [Embedder] that returns the embedding at [index] from a fixed list.
class _FakeEmbedder implements Embedder {
  _FakeEmbedder(this._embeddings, this._index);

  final List<Float32List> _embeddings;
  final int _index;

  @override
  Future<Float32List> embed(String text) async => _embeddings[_index];

  @override
  List<String> contentWords(String text) =>
      text.toLowerCase().split(RegExp(r'\s+')).toList();
}

// ---------------------------------------------------------------------------
// Test entries
// ---------------------------------------------------------------------------

const SearchEntry _dart = SearchEntry(
  id: 1,
  category: 'Dart',
  question: 'What is Dart?',
  answer: 'Dart is a language.',
);
const SearchEntry _flutter = SearchEntry(
  id: 2,
  category: 'Flutter',
  question: 'What is Flutter?',
  answer: 'Flutter is a UI toolkit.',
);
const SearchEntry _isolate = SearchEntry(
  id: 3,
  category: 'Dart',
  question: 'How do isolates work?',
  answer: 'Isolates are lightweight threads.',
);

// ---------------------------------------------------------------------------
// SearchEntry tests
// ---------------------------------------------------------------------------
void main() {
  group('SearchEntry', () {
    test('fromMap extracts all fields with default columns', () {
      final SearchEntry e = SearchEntry.fromMap(<String, Object?>{
        'id': 42,
        'category': 'Flutter',
        'question': 'Hello?',
        'answer': 'World.',
      });
      expect(e.id, 42);
      expect(e.category, 'Flutter');
      expect(e.question, 'Hello?');
      expect(e.answer, 'World.');
    });

    test('fromMap supports custom column names', () {
      final SearchEntry e = SearchEntry.fromMap(
        <String, Object?>{
          'item_id': 7,
          'section': 'X',
          'title': 'Q',
          'body': 'A'
        },
        idColumn: 'item_id',
        categoryColumn: 'section',
        questionColumn: 'title',
        answerColumn: 'body',
      );
      expect(e.id, 7);
      expect(e.category, 'X');
      expect(e.question, 'Q');
      expect(e.answer, 'A');
    });

    test('equality and hashCode', () {
      const SearchEntry a =
          SearchEntry(id: 1, category: 'X', question: 'Q', answer: 'A');
      const SearchEntry b =
          SearchEntry(id: 1, category: 'X', question: 'Q', answer: 'A');
      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('toMap round-trips correctly', () {
      const SearchEntry e =
          SearchEntry(id: 5, category: 'Cat', question: 'Q?', answer: 'A.');
      final Map<String, Object> m = e.toMap();
      expect(m['id'], 5);
      expect(m['question'], 'Q?');
    });
  });

  // -------------------------------------------------------------------------
  // SearchResult tests
  // -------------------------------------------------------------------------
  group('SearchResult', () {
    test('stores entry, score, and method', () {
      const SearchResult r = SearchResult(
        entry: _dart,
        score: 0.95,
        method: 'hybrid',
      );
      expect(r.entry, _dart);
      expect(r.score, closeTo(0.95, 1e-9));
      expect(r.method, 'hybrid');
    });

    test('equality', () {
      const SearchResult a =
          SearchResult(entry: _dart, score: 0.9, method: 'heuristic');
      const SearchResult b =
          SearchResult(entry: _dart, score: 0.9, method: 'heuristic');
      expect(a, equals(b));
    });
  });

  // -------------------------------------------------------------------------
  // HybridSearchConfig tests
  // -------------------------------------------------------------------------
  group('HybridSearchConfig', () {
    test('default values are correct', () {
      const HybridSearchConfig c = HybridSearchConfig();
      expect(c.candidatePoolSize, 50);
      expect(c.ftsLimit, 50);
      expect(c.hnswThreshold, 1000);
      expect(c.embeddingDim, 128);
      expect(c.tableName, 'entries');
      expect(c.questionColumn, 'question');
    });

    test('custom values are preserved', () {
      const HybridSearchConfig c = HybridSearchConfig(
        candidatePoolSize: 10,
        tableName: 'articles',
        questionColumn: 'title',
      );
      expect(c.candidatePoolSize, 10);
      expect(c.tableName, 'articles');
      expect(c.questionColumn, 'title');
    });
  });

  // -------------------------------------------------------------------------
  // SearchRanking tests
  // -------------------------------------------------------------------------
  group('SearchRanking', () {
    group('queryWordsForFts', () {
      test('lowercases and splits on whitespace', () {
        expect(
          SearchRanking.queryWordsForFts('What IS Flutter'),
          <String>['what', 'is', 'flutter'],
        );
      });

      test('strips punctuation', () {
        expect(
          SearchRanking.queryWordsForFts('Dart?'),
          <String>['dart'],
        );
      });
    });

    group('buildFtsMatchQuery', () {
      test('produces OR-combined column-restricted query', () {
        final String q =
            SearchRanking.buildFtsMatchQuery(<String>['dart', 'flutter']);
        expect(q, 'question: dart OR question: flutter');
      });

      test('uses custom column name', () {
        final String q = SearchRanking.buildFtsMatchQuery(
          <String>['foo'],
          column: 'title',
        );
        expect(q, 'title: foo');
      });

      test('returns empty string for empty word list', () {
        expect(SearchRanking.buildFtsMatchQuery(<String>[]), '');
      });
    });

    group('typo matching', () {
      test('exact match returns true', () {
        expect(
          SearchRanking.questionMatchesWithTypo(
              <String>['dart'], 'What is Dart?'),
          isTrue,
        );
      });

      test('single substitution returns true', () {
        // datt → dart (one char substitution)
        expect(
          SearchRanking.questionMatchesWithTypo(
              <String>['datt'], 'Dart is fast'),
          isTrue,
        );
      });

      test('insertion returns true', () {
        // fluttter → flutter (one deletion from longer)
        expect(
          SearchRanking.questionMatchesWithTypo(
              <String>['fluttter'], 'Flutter is great'),
          isTrue,
        );
      });

      test('two-char difference returns false', () {
        expect(
          SearchRanking.questionMatchesWithTypo(<String>['xyzz'], 'Dart'),
          isFalse,
        );
      });
    });

    group('conciseMatchBoostFor', () {
      test('full boost for exact word match', () {
        // Question 'Dart' normalises to ['dart'], same length as queryWords →
        // zero extra words → full conciseMatchBoost returned.
        expect(
          SearchRanking.conciseMatchBoostFor(<String>['dart'], 'Dart'),
          closeTo(SearchRanking.conciseMatchBoost, 1e-9),
        );
      });

      test('zero boost when question is too long', () {
        expect(
          SearchRanking.conciseMatchBoostFor(
            <String>['dart'],
            'Dart Kotlin Java Python Swift differences compared',
          ),
          0.0,
        );
      });

      test('zero boost for empty query words', () {
        expect(
          SearchRanking.conciseMatchBoostFor(<String>[], 'What is Dart?'),
          0.0,
        );
      });
    });

    group('singleIfPerfect', () {
      test('returns single result when exactly one is perfect', () {
        final List<SearchResult> results = <SearchResult>[
          const SearchResult(entry: _dart, score: 0.9999, method: 'hybrid'),
          const SearchResult(entry: _flutter, score: 0.7, method: 'hybrid'),
        ];
        expect(SearchRanking.singleIfPerfect(results).length, 1);
      });

      test('returns all results when no perfect match', () {
        final List<SearchResult> results = <SearchResult>[
          const SearchResult(entry: _dart, score: 0.8, method: 'hybrid'),
          const SearchResult(entry: _flutter, score: 0.7, method: 'hybrid'),
        ];
        expect(SearchRanking.singleIfPerfect(results).length, 2);
      });

      test('returns all results when multiple are perfect', () {
        final List<SearchResult> results = <SearchResult>[
          const SearchResult(entry: _dart, score: 0.9999, method: 'hybrid'),
          const SearchResult(entry: _flutter, score: 0.9999, method: 'hybrid'),
        ];
        expect(SearchRanking.singleIfPerfect(results).length, 2);
      });
    });
  });

  // -------------------------------------------------------------------------
  // HeuristicReranker tests
  // -------------------------------------------------------------------------
  group('HeuristicReranker', () {
    const HeuristicReranker reranker = HeuristicReranker();

    test('returns empty list for empty candidates', () {
      expect(
        reranker.rerank('query', <({
          SearchEntry entry,
          Float32List? embedding,
          double vectorScore,
        })>[], <int>{}),
        isEmpty,
      );
    });

    test('ranks higher-scored candidate first', () {
      final List<SearchResult> results = reranker.rerank(
        'dart',
        <({
          SearchEntry entry,
          Float32List? embedding,
          double vectorScore,
        })>[
          (entry: _dart, vectorScore: 0.9, embedding: null),
          (entry: _flutter, vectorScore: 0.5, embedding: null),
        ],
        <int>{},
        contentWords: <String>['dart'],
      );
      expect(results.first.entry.id, 1); // _dart has higher score
    });

    test('deduplicates entries with same question', () {
      const SearchEntry dup = SearchEntry(
        id: 99,
        category: 'Dart',
        question: 'What is Dart?', // same as _dart
        answer: 'Another answer.',
      );
      final List<SearchResult> results = reranker.rerank(
        'dart',
        <({
          SearchEntry entry,
          Float32List? embedding,
          double vectorScore,
        })>[
          (entry: _dart, vectorScore: 0.9, embedding: null),
          (entry: dup, vectorScore: 0.85, embedding: null),
        ],
        <int>{},
        limit: 3,
      );
      // Both have the same question → only one should survive.
      expect(results.length, 1);
    });

    test('respects limit', () {
      final List<SearchResult> results = reranker.rerank(
        'query',
        <({
          SearchEntry entry,
          Float32List? embedding,
          double vectorScore,
        })>[
          (entry: _dart, vectorScore: 0.9, embedding: null),
          (entry: _flutter, vectorScore: 0.8, embedding: null),
          (entry: _isolate, vectorScore: 0.7, embedding: null),
        ],
        <int>{},
        limit: 2,
      );
      expect(results.length, lessThanOrEqualTo(2));
    });
  });

  // -------------------------------------------------------------------------
  // Float16Store tests
  // -------------------------------------------------------------------------
  group('Float16Store', () {
    test('peekCount reads header correctly', () {
      // Build a header for 5 vectors of dim 128.
      final ByteData h = ByteData(8);
      h.setUint32(0, 5, Endian.little);
      h.setUint32(4, 128, Endian.little);
      expect(Float16Store.peekCount(h.buffer.asUint8List()), 5);
    });

    test('peekDimension reads header correctly', () {
      final ByteData h = ByteData(8);
      h.setUint32(0, 3, Endian.little);
      h.setUint32(4, 64, Endian.little);
      expect(Float16Store.peekDimension(h.buffer.asUint8List()), 64);
    });

    test('decode throws FormatException for truncated header', () {
      expect(
        () => Float16Store.decode(Uint8List(4)),
        throwsA(isA<FormatException>()),
      );
    });

    test('decode single zero vector returns all zeros', () {
      const int dim = 4;
      final ByteData bd = ByteData(8 + dim * 2);
      bd.setUint32(0, 1, Endian.little); // count = 1
      bd.setUint32(4, dim, Endian.little);
      // All zero bytes → f16(0x0000) = 0.0
      final List<Float32List> vecs =
          Float16Store.decode(bd.buffer.asUint8List());
      expect(vecs.length, 1);
      expect(vecs[0].every((double d) => d == 0.0), isTrue);
    });

    test('decode f16(0x3C00) = 1.0', () {
      // 0x3C00 is the Float16 representation of 1.0.
      const int dim = 1;
      final ByteData bd = ByteData(8 + dim * 2);
      bd.setUint32(0, 1, Endian.little);
      bd.setUint32(4, dim, Endian.little);
      bd.setUint16(8, 0x3C00, Endian.little); // 1.0 in f16
      final List<Float32List> vecs =
          Float16Store.decode(bd.buffer.asUint8List());
      expect(vecs[0][0], closeTo(1.0, 1e-3));
    });
  });

  // -------------------------------------------------------------------------
  // HybridSearchEngine integration tests
  // -------------------------------------------------------------------------
  group('HybridSearchEngine', () {
    late Database db;
    late List<Float32List> embeddings;
    const List<SearchEntry> entries = <SearchEntry>[_dart, _flutter, _isolate];

    setUpAll(() async {
      db = await _makeDb(entries);
      // Each entry gets a unique one-hot-like vector.
      embeddings = _makeEmbeddings(entries.length);
    });

    tearDownAll(() async {
      await db.close();
    });

    test('throws StateError when search called before initialize', () async {
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      expect(() => engine.search('dart'), throwsStateError);
    });

    test('returns results after initialization', () async {
      // Embedder returns vector for _dart (index 0) → should rank _dart first.
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();

      final List<SearchResult> results = await engine.search('dart');
      expect(results, isNotEmpty);
      expect(results.first.entry.id, 1); // _dart
    });

    test('returns at most limit results', () async {
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();

      final List<SearchResult> results = await engine.search('dart', limit: 2);
      expect(results.length, lessThanOrEqualTo(2));
    });

    test('initialize is idempotent', () async {
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();
      await engine.initialize(); // second call must not throw
    });

    test('custom config changes table name (error expected for missing table)',
        () async {
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
        config: const HybridSearchConfig(tableName: 'nonexistent'),
      );
      // initialize() loads questions from tableName; nonexistent table → throws.
      await expectLater(engine.initialize(), throwsA(anything));
    });

    test('custom reranker is used', () async {
      // Reranker that always returns the second candidate first.
      final RerankerInterface custom = _ReverseReranker();
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
        reranker: custom,
      );
      await engine.initialize();
      // Just verifies search completes without error with a custom reranker.
      final List<SearchResult> results = await engine.search('dart');
      expect(results, isA<List<SearchResult>>());
    });
  });
}

/// A reranker that reverses the candidate order (for testing custom rerankers).
class _ReverseReranker implements RerankerInterface {
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
    return candidates.reversed
        .take(limit)
        .map<SearchResult>(
          (({
                    SearchEntry entry,
                    Float32List? embedding,
                    double vectorScore,
                  }) c) =>
              SearchResult(
            entry: c.entry,
            score: c.vectorScore,
            method: 'reversed',
          ),
        )
        .toList();
  }
}
