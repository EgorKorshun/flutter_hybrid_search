import 'dart:math';
import 'dart:typed_data';

import 'package:local_hnsw/local_hnsw.dart';
import 'package:local_hnsw/local_hnsw.item.dart';
import 'package:local_hnsw/local_hnsw.result.dart';
import 'package:sqflite/sqflite.dart';

import 'embedder.dart';
import 'models/search_config.dart';
import 'models/search_entry.dart';
import 'models/search_result.dart';
import 'ranking/search_ranking.dart';
import 'reranker/heuristic_reranker.dart';
import 'reranker/reranker.dart';

/// Offline hybrid search engine combining vector similarity, FTS5, and
/// typo-tolerant matching with optional reranking.
///
/// ## Search pipeline
///
/// ```
/// 1. embed(query)          → Float32 vector
///    ↓
/// 2. vector scoring        → top-N candidates by cosine similarity
///    ├─ HNSW  (corpus ≥ hnswThreshold)
///    └─ O(n)  (corpus <  hnswThreshold)
///    ↓
/// 3. keyword scoring       → FTS5 MATCH + typo-tolerant scan
///    ↓
/// 4. candidate pool        → union of vector-top + keyword matches
///    ↓
/// 5. rerank                → HeuristicReranker (or custom RerankerInterface)
///    ↓
/// 6. keyword-overlap filter → remove entries with zero word overlap
///    ↓
/// 7. return top [limit] SearchResult
/// ```
///
/// ## Usage
///
/// ```dart
/// // 1. Open your SQLite database (with sqflite or sqflite_common_ffi).
/// final db = await openDatabase('kb.db', readOnly: true);
///
/// // 2. Load precomputed embeddings (see Float16Store).
/// final bytes = (await rootBundle.load('assets/embeddings.bin'))
///     .buffer.asUint8List();
/// final embeddings = Float16Store.decode(bytes);
///
/// // 3. Create your Embedder implementation.
/// final embedder = MyBertEmbedder();
///
/// // 4. Assemble and initialise the engine.
/// final engine = HybridSearchEngine(
///   db: db,
///   embeddings: embeddings,
///   embedder: embedder,
/// );
/// await engine.initialize();
///
/// // 5. Search.
/// final results = await engine.search('What is Flutter?', limit: 3);
/// for (final r in results) {
///   print('${r.score.toStringAsFixed(3)}  ${r.entry.question}');
///   print(r.entry.answer);
/// }
///
/// // 6. Clean up.
/// await engine.dispose();
/// ```
///
/// ## Custom configuration
///
/// ```dart
/// final engine = HybridSearchEngine(
///   db: db,
///   embeddings: embeddings,
///   embedder: embedder,
///   config: const HybridSearchConfig(
///     hnswThreshold: 500,
///     candidatePoolSize: 80,
///     tableName: 'articles',
///     questionColumn: 'title',
///     answerColumn: 'body',
///   ),
///   reranker: const HeuristicReranker(),
/// );
/// ```
class HybridSearchEngine {
  /// Creates a [HybridSearchEngine].
  ///
  /// [db] and [embeddings] are accepted directly so the engine works in any
  /// environment — Flutter assets, CLI files, or in-process test databases.
  ///
  /// [embeddings] must be 1-indexed: `embeddings[i]` is the vector for the
  /// database entry with `id = i + 1`.
  ///
  /// Call [initialize] before the first [search] call.
  HybridSearchEngine({
    required Database db,
    required List<Float32List> embeddings,
    required Embedder embedder,
    HybridSearchConfig config = const HybridSearchConfig(),
    RerankerInterface? reranker,
  })  : _db = db,
        _embeddings = embeddings,
        _embedder = embedder,
        _config = config,
        _reranker = reranker ?? const HeuristicReranker();

  // ---------------------------------------------------------------------------
  // Private fields
  // ---------------------------------------------------------------------------

  final Database _db;
  final List<Float32List> _embeddings;
  final Embedder _embedder;
  final HybridSearchConfig _config;
  final RerankerInterface _reranker;

  late final List<double> _norms;
  late final Map<int, String> _idToQuestion;
  LocalHNSW<int>? _hnsw;

  bool _initialized = false;
  bool _disposed = false;

  // ---------------------------------------------------------------------------
  // Public state
  // ---------------------------------------------------------------------------

  /// Whether [initialize] has been called successfully and [dispose] has not.
  bool get isInitialized => _initialized && !_disposed;

  /// Number of entries (embeddings) managed by this engine.
  ///
  /// Available immediately after construction (does not require [initialize]).
  int get entryCount => _embeddings.length;

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /// Initialises the engine: precomputes L2 norms, builds the HNSW index
  /// (when corpus is large enough), and loads the question map for typo
  /// matching.
  ///
  /// This method is idempotent — subsequent calls are no-ops.
  ///
  /// Throws [StateError] if [dispose] has already been called.
  ///
  /// Must be called before [search].
  Future<void> initialize() async {
    if (_disposed) {
      throw StateError(
        'HybridSearchEngine.initialize() called after dispose().',
      );
    }
    if (_initialized) return;

    // Precompute L2 norms for fast cosine similarity.
    _norms = _embeddings.map(_l2Norm).toList();

    // Build HNSW index for large corpora.
    if (_embeddings.length >= _config.hnswThreshold) {
      _hnsw = _buildHnsw();
    }

    // Load question texts for typo-tolerant matching.
    _idToQuestion = await _loadQuestions();

    _initialized = true;
  }

  /// Releases the SQLite database connection.
  ///
  /// Safe to call multiple times — subsequent calls are no-ops.
  /// After [dispose], calls to [search] and [initialize] will throw a
  /// [StateError].
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    if (_initialized) {
      await _db.close();
      _initialized = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Search
  // ---------------------------------------------------------------------------

  /// Searches for the [limit] most relevant entries matching [query].
  ///
  /// Returns an empty list when no candidates pass the keyword-overlap filter.
  ///
  /// Throws [StateError] if [initialize] has not been called.
  Future<List<SearchResult>> search(String query, {int limit = 3}) async {
    if (_disposed) {
      throw StateError(
        'HybridSearchEngine.search() called after dispose().',
      );
    }
    if (!_initialized) {
      throw StateError(
        'HybridSearchEngine.initialize() must be called before search().',
      );
    }

    // Step 1: embed the query.
    final Float32List queryVec = await _embedder.embed(query);
    final double queryNorm = _l2Norm(queryVec);

    // Step 2: vector scoring.
    final List<({int index, double score})> vecScores =
        _scoreByVector(queryVec, queryNorm);
    final Map<int, double> idToScore = _toIdScoreMap(vecScores);

    // Step 3: keyword scoring.
    final List<String> words = _embedder.contentWords(query);
    final List<int> ftsIds = await _ftsSearch(words);
    final Set<int> keywordIds = await _keywordMatches(words, ftsIds);

    // Step 4: build candidate pool.
    final Set<int> poolIds = _buildPool(vecScores, keywordIds);
    if (poolIds.isEmpty) return <SearchResult>[];

    // Supplement idToScore with cosine scores for keyword-only candidates
    // (needed when HNSW only returned top-K, not all entries).
    if (_hnsw != null) {
      for (final int id in poolIds) {
        idToScore.putIfAbsent(id, () {
          final int i = id - 1;
          return _cosine(queryVec, queryNorm, _embeddings[i], _norms[i]);
        });
      }
    }

    // Step 5: fetch entries and rerank.
    final RerankerCandidates candidates =
        await _buildCandidates(poolIds.toList(), idToScore);

    final List<SearchResult> ranked = _reranker.rerank(
      query,
      candidates,
      keywordIds,
      limit: limit,
      queryEmbedding: queryVec,
      ftsIds: ftsIds.toSet(),
      contentWords: words,
    );

    // Step 6: keyword-overlap filter.
    return _filterByOverlap(query, ranked);
  }

  // ---------------------------------------------------------------------------
  // Vector scoring
  // ---------------------------------------------------------------------------

  /// Scores all embeddings against [queryVec].
  ///
  /// Uses HNSW when available (corpus ≥ [HybridSearchConfig.hnswThreshold]),
  /// otherwise performs a linear O(n) cosine scan.
  List<({int index, double score})> _scoreByVector(
    Float32List queryVec,
    double queryNorm,
  ) {
    if (_hnsw != null) {
      final LocalHnswSearchResult<int> result = _hnsw!.search(
        List<double>.from(queryVec),
        _config.hnswSearchK,
      );
      return result.items
          .map((LocalHnswSearchResultItem<int> r) =>
              (index: r.item, score: 1.0 - r.distance))
          .toList();
    }

    return <({int index, double score})>[
      for (int i = 0; i < _embeddings.length; i++)
        (
          index: i,
          score: _cosine(queryVec, queryNorm, _embeddings[i], _norms[i]),
        ),
    ];
  }

  /// Converts vector scores to a `{id: score}` map (IDs are 1-based).
  Map<int, double> _toIdScoreMap(List<({int index, double score})> scores) {
    return <int, double>{
      for (final (:int index, :double score) in scores) index + 1: score,
    };
  }

  // ---------------------------------------------------------------------------
  // Keyword scoring
  // ---------------------------------------------------------------------------

  /// Queries the FTS5 table with [words] and returns matching row IDs.
  ///
  /// Falls back to single-word query if the multi-word query returns nothing.
  Future<List<int>> _ftsSearch(List<String> words) async {
    if (words.isEmpty) return <int>[];
    try {
      final String q = SearchRanking.buildFtsMatchQuery(
        words,
        column: _config.questionColumn,
      );
      List<Map<String, Object?>> rows = await _db.rawQuery(
        'SELECT rowid FROM ${_config.ftsTableName} '
        'WHERE ${_config.ftsTableName} MATCH ? LIMIT ?',
        <Object?>[q, _config.ftsLimit],
      );
      List<int> ids = _rowIds(rows);

      // Retry with just the first word if nothing matched.
      if (ids.isEmpty && words.length > 1) {
        final String q1 = SearchRanking.buildFtsMatchQuery(
          <String>[words.first],
          column: _config.questionColumn,
        );
        rows = await _db.rawQuery(
          'SELECT rowid FROM ${_config.ftsTableName} '
          'WHERE ${_config.ftsTableName} MATCH ? LIMIT ?',
          <Object?>[q1, _config.ftsLimit],
        );
        ids = _rowIds(rows);
      }
      return ids;
    } catch (_) {
      return <int>[];
    }
  }

  /// Returns IDs matching [words] via FTS5 and typo-tolerant scan.
  Future<Set<int>> _keywordMatches(List<String> words, List<int> ftsIds) async {
    final Set<int> ids = ftsIds.toSet();
    if (words.isEmpty) return ids;

    for (final MapEntry<int, String> e in _idToQuestion.entries) {
      final String lower = e.value.toLowerCase();
      final bool exact = words.any(lower.contains);
      final bool typo = SearchRanking.questionMatchesWithTypo(words, e.value);
      if (exact || typo) ids.add(e.key);
    }

    return ids;
  }

  // ---------------------------------------------------------------------------
  // Candidate pool
  // ---------------------------------------------------------------------------

  /// Builds the candidate pool: top-[candidatePoolSize] by vector + all keyword hits.
  Set<int> _buildPool(
    List<({int index, double score})> vecScores,
    Set<int> keywordIds,
  ) {
    final List<({int index, double score})> sorted =
        List<({int index, double score})>.from(vecScores)
          ..sort(
            (({int index, double score}) a, ({int index, double score}) b) =>
                b.score.compareTo(a.score),
          );

    final Set<int> pool = sorted
        .take(_config.candidatePoolSize)
        .map<int>((({int index, double score}) r) => r.index + 1)
        .toSet();

    return pool..addAll(keywordIds);
  }

  /// Fetches entries for [ids] and pairs them with scores and embeddings.
  Future<RerankerCandidates> _buildCandidates(
    List<int> ids,
    Map<int, double> idToScore,
  ) async {
    final List<SearchEntry> entries = await _fetchEntries(ids);
    return <({
      SearchEntry entry,
      double vectorScore,
      Float32List? embedding,
    })>[
      for (final SearchEntry e in entries)
        (
          entry: e,
          vectorScore: idToScore[e.id] ?? 0.0,
          embedding: _embeddingFor(e.id),
        ),
    ];
  }

  // ---------------------------------------------------------------------------
  // Database helpers
  // ---------------------------------------------------------------------------

  /// Loads all question texts into a map for typo-tolerant matching.
  Future<Map<int, String>> _loadQuestions() async {
    final List<Map<String, Object?>> rows = await _db.query(
      _config.tableName,
      columns: <String>[_config.idColumn, _config.questionColumn],
    );
    return <int, String>{
      for (final Map<String, Object?> row in rows)
        row[_config.idColumn] as int: row[_config.questionColumn] as String,
    };
  }

  /// Fetches full [SearchEntry] rows for the given [ids].
  Future<List<SearchEntry>> _fetchEntries(List<int> ids) async {
    if (ids.isEmpty) return <SearchEntry>[];

    final String ph = List<String>.filled(ids.length, '?').join(', ');
    final List<Map<String, Object?>> rows = await _db.query(
      _config.tableName,
      where: '${_config.idColumn} IN ($ph)',
      whereArgs: ids,
    );

    final Map<int, SearchEntry> byId = <int, SearchEntry>{
      for (final Map<String, Object?> row in rows)
        row[_config.idColumn] as int: SearchEntry.fromMap(
          row,
          idColumn: _config.idColumn,
          categoryColumn: _config.categoryColumn,
          questionColumn: _config.questionColumn,
          answerColumn: _config.answerColumn,
        ),
    };

    // Preserve original order.
    return <SearchEntry>[
      for (final int id in ids)
        if (byId.containsKey(id)) byId[id]!,
    ];
  }

  /// Extracts the `rowid` column from FTS query results.
  List<int> _rowIds(List<Map<String, Object?>> rows) =>
      <int>[for (final Map<String, Object?> r in rows) r['rowid'] as int];

  // ---------------------------------------------------------------------------
  // Keyword-overlap filter
  // ---------------------------------------------------------------------------

  /// Removes results with zero keyword overlap between the query and the
  /// matched question text.
  List<SearchResult> _filterByOverlap(
      String query, List<SearchResult> results) {
    final List<String> words = SearchRanking.queryWordsForFts(query);
    return results
        .where((SearchResult r) =>
            SearchRanking.countQueryWordsMatchingQuestion(
                words, r.entry.question) >
            0)
        .toList();
  }

  // ---------------------------------------------------------------------------
  // HNSW
  // ---------------------------------------------------------------------------

  /// Builds an HNSW approximate nearest-neighbour index from [_embeddings].
  LocalHNSW<int> _buildHnsw() {
    final LocalHNSW<int> index = LocalHNSW<int>(
      dim: _config.embeddingDim,
      metric: LocalHnswMetric.cosine,
      M: _config.hnswM,
      ef: _config.hnswEf,
    );
    for (int i = 0; i < _embeddings.length; i++) {
      index.add(LocalHnswItem<int>(
        item: i,
        vector: List<double>.from(_embeddings[i]),
      ));
    }
    return index;
  }

  // ---------------------------------------------------------------------------
  // Math helpers
  // ---------------------------------------------------------------------------

  /// Computes the L2 (Euclidean) norm of [v].
  double _l2Norm(Float32List v) {
    double s = 0.0;
    for (int i = 0; i < v.length; i++) {
      s += v[i] * v[i];
    }
    return sqrt(s);
  }

  /// Computes cosine similarity using precomputed norms.
  double _cosine(
    Float32List a,
    double na,
    Float32List b,
    double nb,
  ) {
    if (na == 0 || nb == 0) return 0.0;
    double dot = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot / (na * nb);
  }

  /// Returns the precomputed embedding for entry [id], or `null` if out of range.
  Float32List? _embeddingFor(int id) =>
      (id >= 1 && id <= _embeddings.length) ? _embeddings[id - 1] : null;
}
