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
List<Embedding> _makeEmbeddings(int count, {int dim = 128}) {
  return List<Embedding>.generate(count, (int i) {
    final Embedding v = Embedding(dim);
    v[i % dim] = 1.0; // one-hot-like, guaranteed unit norm
    return v;
  });
}

/// A stub [Embedder] that returns the embedding at [index] from a fixed list.
class _FakeEmbedder implements Embedder {
  _FakeEmbedder(this._embeddings, this._index);

  final List<Embedding> _embeddings;
  final int _index;

  @override
  Future<Embedding> embed(String text) async => _embeddings[_index];

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
  // SearchMetadata tests
  // -------------------------------------------------------------------------
  group('SearchMetadata', () {
    test('stores all timing fields', () {
      const SearchMetadata m = SearchMetadata(
        embedMs: 1.5,
        vectorMs: 0.3,
        ftsMs: 2.0,
        typoMs: 0.5,
        rerankMs: 1.0,
        totalMs: 5.3,
        candidateCount: 42,
        vectorCandidateCount: 30,
        keywordCandidateCount: 12,
      );
      expect(m.embedMs, 1.5);
      expect(m.vectorMs, 0.3);
      expect(m.ftsMs, 2.0);
      expect(m.typoMs, 0.5);
      expect(m.rerankMs, 1.0);
      expect(m.totalMs, 5.3);
      expect(m.candidateCount, 42);
      expect(m.vectorCandidateCount, 30);
      expect(m.keywordCandidateCount, 12);
    });

    test('toString includes total', () {
      const SearchMetadata m = SearchMetadata(
        embedMs: 1,
        vectorMs: 2,
        ftsMs: 3,
        typoMs: 4,
        rerankMs: 5,
        totalMs: 15,
        candidateCount: 10,
        vectorCandidateCount: 5,
        keywordCandidateCount: 5,
      );
      expect(m.toString(), contains('15.0'));
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
        hnswSearchK: 10,
        tableName: 'articles',
        questionColumn: 'title',
      );
      expect(c.candidatePoolSize, 10);
      expect(c.tableName, 'articles');
      expect(c.questionColumn, 'title');
    });

    test('copyWith overrides only specified fields', () {
      const HybridSearchConfig base = HybridSearchConfig();
      final HybridSearchConfig tuned = base.copyWith(
        candidatePoolSize: 100,
        hnswSearchK: 200,
        hnswM: 32,
      );
      expect(tuned.candidatePoolSize, 100);
      expect(tuned.hnswM, 32);
      // Unchanged fields keep defaults.
      expect(tuned.ftsLimit, base.ftsLimit);
      expect(tuned.tableName, base.tableName);
      expect(tuned.embeddingDim, base.embeddingDim);
    });

    test('copyWith with no arguments returns equivalent config', () {
      const HybridSearchConfig base = HybridSearchConfig(
        candidatePoolSize: 30,
        hnswSearchK: 30,
        tableName: 'items',
      );
      final HybridSearchConfig copy = base.copyWith();
      expect(copy.candidatePoolSize, base.candidatePoolSize);
      expect(copy.tableName, base.tableName);
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
          Embedding? embedding,
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
          Embedding? embedding,
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
          Embedding? embedding,
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
          Embedding? embedding,
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

    test('decode throws FormatException for zero count', () {
      final ByteData bd = ByteData(8 + 4 * 2);
      bd.setUint32(0, 0, Endian.little); // count = 0
      bd.setUint32(4, 4, Endian.little);
      expect(
        () => Float16Store.decode(bd.buffer.asUint8List()),
        throwsA(isA<FormatException>()),
      );
    });

    test('decode throws FormatException for zero dimension', () {
      final ByteData bd = ByteData(8);
      bd.setUint32(0, 1, Endian.little);
      bd.setUint32(4, 0, Endian.little); // dim = 0
      expect(
        () => Float16Store.decode(bd.buffer.asUint8List()),
        throwsA(isA<FormatException>()),
      );
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
    late List<Embedding> embeddings;
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

    test('entryCount reflects number of embeddings', () {
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      expect(engine.entryCount, entries.length);
    });

    test('isInitialized reflects lifecycle state', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      expect(engine.isInitialized, isFalse);
      await engine.initialize();
      expect(engine.isInitialized, isTrue);
      await engine.dispose();
      expect(engine.isInitialized, isFalse);
    });

    test('dispose is idempotent', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();
      await engine.dispose();
      await engine.dispose(); // second call must not throw
    });

    test('initialize after dispose throws StateError', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.dispose();
      expect(() => engine.initialize(), throwsStateError);
    });

    test('search after dispose throws StateError', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();
      await engine.dispose();
      expect(() => engine.search('dart'), throwsStateError);
    });

    // -----------------------------------------------------------------------
    // New tests for v1.1.0
    // -----------------------------------------------------------------------

    test('throws ArgumentError for mismatched embedding dimensions', () {
      final List<Embedding> badEmbeddings = <Embedding>[
        Embedding(64), // wrong dim, expected 128
      ];
      expect(
        () => HybridSearchEngine(
          db: db,
          embeddings: badEmbeddings,
          embedder: _FakeEmbedder(badEmbeddings, 0),
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('searchWithMetadata returns metadata alongside results', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();

      final (:List<SearchResult> results, :SearchMetadata metadata) =
          await engine.searchWithMetadata('dart');

      expect(results, isNotEmpty);
      expect(metadata.totalMs, greaterThanOrEqualTo(0));
      expect(metadata.embedMs, greaterThanOrEqualTo(0));
      expect(metadata.vectorMs, greaterThanOrEqualTo(0));
      expect(metadata.candidateCount, greaterThan(0));
      await engine.dispose();
    });

    test('searchBatch returns results for each query', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();

      final List<List<SearchResult>> batch =
          await engine.searchBatch(<String>['dart', 'isolates']);
      expect(batch.length, 2);
      // At least the first query should return results (the fake embedder
      // always returns dart's vector, so 'dart' is guaranteed to match).
      expect(batch[0], isNotEmpty);
      await engine.dispose();
    });

    test('embed cache returns same results for repeated queries', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
        embedCacheSize: 4,
      );
      await engine.initialize();

      final List<SearchResult> first = await engine.search('dart');
      final List<SearchResult> second = await engine.search('dart');

      expect(first.length, second.length);
      if (first.isNotEmpty && second.isNotEmpty) {
        expect(first.first.entry.id, second.first.entry.id);
      }
      await engine.dispose();
    });

    test('embed cache disabled with size 0', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
        embedCacheSize: 0,
      );
      await engine.initialize();

      // Should work without cache.
      final List<SearchResult> results = await engine.search('dart');
      expect(results, isNotEmpty);
      await engine.dispose();
    });

    test('concurrent initialize calls are safe', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );

      // Launch two concurrent initializations.
      await Future.wait<void>(<Future<void>>[
        engine.initialize(),
        engine.initialize(),
      ]);

      expect(engine.isInitialized, isTrue);
      await engine.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // ScoreBreakdown tests (v1.2.0)
  // -------------------------------------------------------------------------
  group('ScoreBreakdown', () {
    test('stores all five fields', () {
      const ScoreBreakdown b = ScoreBreakdown(
        vectorScore: 0.8,
        ftsScore: 0.5,
        typoScore: 0.0,
        conciseScore: 0.3,
        totalScore: 1.6,
      );
      expect(b.vectorScore, closeTo(0.8, 1e-9));
      expect(b.ftsScore, closeTo(0.5, 1e-9));
      expect(b.typoScore, closeTo(0.0, 1e-9));
      expect(b.conciseScore, closeTo(0.3, 1e-9));
      expect(b.totalScore, closeTo(1.6, 1e-9));
    });

    test('copyWith overrides only specified fields', () {
      const ScoreBreakdown b = ScoreBreakdown(
        vectorScore: 0.8,
        ftsScore: 0.5,
        typoScore: 0.0,
        conciseScore: 0.3,
        totalScore: 1.6,
      );
      final ScoreBreakdown copy = b.copyWith(ftsScore: 0.0, totalScore: 1.1);
      expect(copy.vectorScore, closeTo(0.8, 1e-9));
      expect(copy.ftsScore, closeTo(0.0, 1e-9));
      expect(copy.typoScore, closeTo(0.0, 1e-9));
      expect(copy.conciseScore, closeTo(0.3, 1e-9));
      expect(copy.totalScore, closeTo(1.1, 1e-9));
    });

    test('equality and hashCode', () {
      const ScoreBreakdown a = ScoreBreakdown(
        vectorScore: 0.5,
        ftsScore: 0.5,
        typoScore: 0.0,
        conciseScore: 0.0,
        totalScore: 1.0,
      );
      const ScoreBreakdown b = ScoreBreakdown(
        vectorScore: 0.5,
        ftsScore: 0.5,
        typoScore: 0.0,
        conciseScore: 0.0,
        totalScore: 1.0,
      );
      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('toString contains all relevant values', () {
      const ScoreBreakdown b = ScoreBreakdown(
        vectorScore: 0.8,
        ftsScore: 0.5,
        typoScore: 0.0,
        conciseScore: 0.3,
        totalScore: 1.6,
      );
      final String s = b.toString();
      expect(s, contains('total'));
      expect(s, contains('vector'));
      expect(s, contains('fts'));
    });
  });

  // -------------------------------------------------------------------------
  // SearchResult.copyWith + breakdown integration tests (v1.2.0)
  // -------------------------------------------------------------------------
  group('SearchResult.copyWith', () {
    test('no-arg copy is equal to original', () {
      const SearchResult r =
          SearchResult(entry: _dart, score: 0.9, method: 'heuristic');
      expect(r.copyWith(), equals(r));
    });

    test('single-field override changes only that field', () {
      const SearchResult r =
          SearchResult(entry: _dart, score: 0.9, method: 'heuristic');
      final SearchResult copy = r.copyWith(score: 0.5);
      expect(copy.score, closeTo(0.5, 1e-9));
      expect(copy.entry, r.entry);
      expect(copy.method, r.method);
    });

    test('breakdown field is null by default', () {
      const SearchResult r =
          SearchResult(entry: _dart, score: 0.9, method: 'heuristic');
      expect(r.breakdown, isNull);
    });

    test('copyWith can attach a breakdown', () {
      const SearchResult r =
          SearchResult(entry: _dart, score: 0.9, method: 'heuristic');
      const ScoreBreakdown bd = ScoreBreakdown(
        vectorScore: 0.9,
        ftsScore: 0.0,
        typoScore: 0.0,
        conciseScore: 0.0,
        totalScore: 0.9,
      );
      final SearchResult withBd = r.copyWith(breakdown: bd);
      expect(withBd.breakdown, equals(bd));
    });
  });

  group('ScoreBreakdown integration with HeuristicReranker', () {
    final List<SearchEntry> entries = <SearchEntry>[_dart, _flutter, _isolate];
    late List<Embedding> embeddings;

    setUpAll(() {
      embeddings = _makeEmbeddings(entries.length);
    });

    test('results from heuristic reranker have non-null breakdown', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();
      final List<SearchResult> results = await engine.search('dart');
      expect(results, isNotEmpty);
      for (final SearchResult r in results) {
        expect(r.breakdown, isNotNull);
      }
      await engine.dispose();
    });

    test('breakdown.totalScore equals result.score', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
      );
      await engine.initialize();
      final List<SearchResult> results = await engine.search('dart');
      expect(results, isNotEmpty);
      for (final SearchResult r in results) {
        expect(r.breakdown, isNotNull);
        expect(r.breakdown!.totalScore, closeTo(r.score, 1e-9));
      }
      await engine.dispose();
    });

    test('custom reranker results have null breakdown', () async {
      final Database freshDb = await _makeDb(entries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
        reranker: _ReverseReranker(),
      );
      await engine.initialize();
      final List<SearchResult> results = await engine.search('dart');
      // _ReverseReranker does not set breakdown.
      for (final SearchResult r in results) {
        expect(r.breakdown, isNull);
      }
      await engine.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // minScore threshold tests (v1.2.0)
  // -------------------------------------------------------------------------
  group('minScore threshold', () {
    test('default minScore is 0.0', () {
      const HybridSearchConfig c = HybridSearchConfig();
      expect(c.minScore, closeTo(0.0, 1e-9));
    });

    test('copyWith propagates minScore', () {
      const HybridSearchConfig base = HybridSearchConfig();
      final HybridSearchConfig tuned = base.copyWith(minScore: 0.6);
      expect(tuned.minScore, closeTo(0.6, 1e-9));
      // Unrelated fields unchanged.
      expect(tuned.candidatePoolSize, base.candidatePoolSize);
    });

    test('minScore 0.0 does not filter any results', () async {
      final List<SearchEntry> entries = <SearchEntry>[_dart, _flutter, _isolate];
      final List<Embedding> embeddings = _makeEmbeddings(entries.length);
      final Database freshDb = await _makeDb(entries);

      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
        config: const HybridSearchConfig(minScore: 0.0),
      );
      await engine.initialize();
      final List<SearchResult> results = await engine.search('dart');
      expect(results, isNotEmpty);
      await engine.dispose();
    });

    test('impossibly high minScore returns empty list', () async {
      final List<SearchEntry> entries = <SearchEntry>[_dart, _flutter, _isolate];
      final List<Embedding> embeddings = _makeEmbeddings(entries.length);
      final Database freshDb = await _makeDb(entries);

      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: embeddings,
        embedder: _FakeEmbedder(embeddings, 0),
        config: const HybridSearchConfig(minScore: 999.0),
      );
      await engine.initialize();
      final List<SearchResult> results = await engine.search('dart');
      expect(results, isEmpty);
      await engine.dispose();
    });
  });

  // -------------------------------------------------------------------------
  // SearchEntry.metadata tests (v1.2.0)
  // -------------------------------------------------------------------------
  group('SearchEntry.metadata', () {
    test('default metadata is empty', () {
      const SearchEntry e =
          SearchEntry(id: 1, category: 'X', question: 'Q', answer: 'A');
      expect(e.metadata, isEmpty);
    });

    test('explicit metadata is stored and accessible', () {
      const SearchEntry e = SearchEntry(
        id: 1,
        category: 'X',
        question: 'Q',
        answer: 'A',
        metadata: <String, Object?>{'priority': 1, 'tag': 'ui'},
      );
      expect(e.metadata['priority'], 1);
      expect(e.metadata['tag'], 'ui');
    });

    test('fromMap without metadataColumn returns empty metadata', () {
      final SearchEntry e = SearchEntry.fromMap(<String, Object?>{
        'id': 1,
        'category': 'X',
        'question': 'Q',
        'answer': 'A',
      });
      expect(e.metadata, isEmpty);
    });

    test('fromMap with metadataColumn decodes JSON correctly', () {
      final SearchEntry e = SearchEntry.fromMap(
        <String, Object?>{
          'id': 1,
          'category': 'X',
          'question': 'Q',
          'answer': 'A',
          'meta': '{"score":42,"active":true,"label":null}',
        },
        metadataColumn: 'meta',
      );
      expect(e.metadata['score'], 42);
      expect(e.metadata['active'], true);
      expect(e.metadata['label'], isNull);
    });

    test('toMap without metadataColumn does not include metadata key', () {
      const SearchEntry e = SearchEntry(
        id: 1,
        category: 'X',
        question: 'Q',
        answer: 'A',
        metadata: <String, Object?>{'k': 'v'},
      );
      final Map<String, Object> m = e.toMap();
      expect(m.containsKey('meta'), isFalse);
    });

    test('toMap with metadataColumn encodes metadata as JSON', () {
      const SearchEntry e = SearchEntry(
        id: 1,
        category: 'X',
        question: 'Q',
        answer: 'A',
        metadata: <String, Object?>{'k': 'v'},
      );
      final Map<String, Object> m = e.toMap(metadataColumn: 'meta');
      expect(m['meta'], isA<String>());
      expect(m['meta'] as String, contains('"k"'));
    });

    test('round-trip fromMap/toMap preserves metadata', () {
      const SearchEntry original = SearchEntry(
        id: 5,
        category: 'Dart',
        question: 'Test?',
        answer: 'Yes.',
        metadata: <String, Object?>{'x': 1, 'y': 'hello'},
      );
      final Map<String, Object> map = original.toMap(metadataColumn: 'meta');
      final SearchEntry restored = SearchEntry.fromMap(
        map,
        metadataColumn: 'meta',
      );
      expect(restored.metadata['x'], original.metadata['x']);
      expect(restored.metadata['y'], original.metadata['y']);
    });

    test('equality considers metadata', () {
      const SearchEntry a = SearchEntry(
        id: 1,
        category: 'X',
        question: 'Q',
        answer: 'A',
        metadata: <String, Object?>{'k': 'v'},
      );
      const SearchEntry b = SearchEntry(
        id: 1,
        category: 'X',
        question: 'Q',
        answer: 'A',
        metadata: <String, Object?>{'k': 'v'},
      );
      const SearchEntry c = SearchEntry(
        id: 1,
        category: 'X',
        question: 'Q',
        answer: 'A',
        metadata: <String, Object?>{'k': 'different'},
      );
      expect(a, equals(b));
      expect(a, isNot(equals(c)));
    });
  });

  // -------------------------------------------------------------------------
  // Incremental index update tests (v1.2.0)
  // -------------------------------------------------------------------------
  group('incremental updates', () {
    late List<SearchEntry> baseEntries;
    late List<Embedding> baseEmbeddings;

    setUp(() {
      baseEntries = <SearchEntry>[_dart, _flutter, _isolate];
      baseEmbeddings = _makeEmbeddings(baseEntries.length);
    });

    Future<HybridSearchEngine> makeEngine(Database db) async {
      final HybridSearchEngine engine = HybridSearchEngine(
        db: db,
        embeddings: baseEmbeddings,
        embedder: _FakeEmbedder(baseEmbeddings, 0),
      );
      await engine.initialize();
      return engine;
    }

    test('addEntries increases entryCount', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);

      final List<SearchEntry> newEntries = <SearchEntry>[
        const SearchEntry(
          id: 0,
          category: 'Dart',
          question: 'What is a mixin?',
          answer: 'A mixin adds behaviour.',
        ),
      ];
      final List<Embedding> newEmbeddings = _makeEmbeddings(1);
      await engine.addEntries(newEntries, newEmbeddings);

      expect(engine.entryCount, baseEntries.length + 1);
      await engine.dispose();
    });

    test('newly added entry is found by search', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: baseEmbeddings,
        // embedder returns the first base embedding for any query
        embedder: _FakeEmbedder(baseEmbeddings, 0),
      );
      await engine.initialize();

      const SearchEntry newEntry = SearchEntry(
        id: 0,
        category: 'Dart',
        question: 'What is a mixin?',
        answer: 'A way to reuse code.',
      );
      // Build an embedding that exactly matches what the fake embedder returns
      // for queries, so the new entry will rank first.
      final List<Embedding> newEmbeddings = <Embedding>[baseEmbeddings[0]];
      await engine.addEntries(<SearchEntry>[newEntry], newEmbeddings);

      final List<SearchResult> results = await engine.search('mixin');
      final bool found = results.any(
        (SearchResult r) => r.entry.question == 'What is a mixin?',
      );
      expect(found, isTrue);
      await engine.dispose();
    });

    test('empty addEntries is a no-op', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);
      final int before = engine.entryCount;

      await engine.addEntries(<SearchEntry>[], <Embedding>[]);
      expect(engine.entryCount, before);
      await engine.dispose();
    });

    test('addEntries throws ArgumentError on length mismatch', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);
      expect(
        () => engine.addEntries(
          <SearchEntry>[
            const SearchEntry(
                id: 0, category: 'X', question: 'Q', answer: 'A'),
          ],
          <Embedding>[], // wrong length
        ),
        throwsA(isA<ArgumentError>()),
      );
      await engine.dispose();
    });

    test('addEntries throws ArgumentError on wrong embedding dim', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);
      expect(
        () => engine.addEntries(
          <SearchEntry>[
            const SearchEntry(
                id: 0, category: 'X', question: 'Q', answer: 'A'),
          ],
          <Embedding>[Embedding(64)], // wrong dim
        ),
        throwsA(isA<ArgumentError>()),
      );
      await engine.dispose();
    });

    test('removeEntries decreases entryCount', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);

      await engine.removeEntries(<int>[1]);
      expect(engine.entryCount, baseEntries.length - 1);
      await engine.dispose();
    });

    test('removed entry no longer appears in search results', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);

      // Confirm entry 1 (_dart) is reachable before removal.
      final List<SearchResult> before = await engine.search('dart');
      expect(before.any((SearchResult r) => r.entry.id == 1), isTrue);

      await engine.removeEntries(<int>[1]);

      // After removal, entry 1 should not appear.
      final List<SearchResult> after = await engine.search('dart');
      expect(after.any((SearchResult r) => r.entry.id == 1), isFalse);
      await engine.dispose();
    });

    test('empty removeEntries is a no-op', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);
      final int before = engine.entryCount;

      await engine.removeEntries(<int>[]);
      expect(engine.entryCount, before);
      await engine.dispose();
    });

    test('removeEntries with unknown ids is silently ignored', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);
      final int before = engine.entryCount;

      await engine.removeEntries(<int>[999, 1000]);
      expect(engine.entryCount, before);
      await engine.dispose();
    });

    test('addEntries then removeEntries leaves corpus consistent', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);

      final List<Embedding> newEmbeddings = <Embedding>[baseEmbeddings[0]];
      await engine.addEntries(
        <SearchEntry>[
          const SearchEntry(
              id: 0, category: 'X', question: 'Extra entry', answer: 'Extra.')
        ],
        newEmbeddings,
      );
      final int afterAdd = engine.entryCount;

      // The new entry has id = baseEntries.length + 1 = 4.
      await engine.removeEntries(<int>[4]);
      expect(engine.entryCount, afterAdd - 1);
      await engine.dispose();
    });

    test('addEntries throws StateError before initialize', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: baseEmbeddings,
        embedder: _FakeEmbedder(baseEmbeddings, 0),
      );
      expect(
        () => engine.addEntries(<SearchEntry>[], <Embedding>[]),
        throwsStateError,
      );
    });

    test('removeEntries throws StateError before initialize', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = HybridSearchEngine(
        db: freshDb,
        embeddings: baseEmbeddings,
        embedder: _FakeEmbedder(baseEmbeddings, 0),
      );
      expect(
        () => engine.removeEntries(<int>[1]),
        throwsStateError,
      );
    });

    test('addEntries throws StateError after dispose', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);
      await engine.dispose();
      expect(
        () => engine.addEntries(<SearchEntry>[], <Embedding>[]),
        throwsStateError,
      );
    });

    test('removeEntries throws StateError after dispose', () async {
      final Database freshDb = await _makeDb(baseEntries);
      final HybridSearchEngine engine = await makeEngine(freshDb);
      await engine.dispose();
      expect(
        () => engine.removeEntries(<int>[1]),
        throwsStateError,
      );
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
    Embedding? queryEmbedding,
    Set<int>? ftsIds,
    List<String>? contentWords,
  }) {
    return candidates.reversed
        .take(limit)
        .map<SearchResult>(
          (({
                    SearchEntry entry,
                    Embedding? embedding,
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
