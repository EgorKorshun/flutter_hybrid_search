import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:local_hnsw/local_hnsw.dart';
import 'package:local_hnsw/local_hnsw.item.dart';
import 'package:local_hnsw/local_hnsw.result.dart';
import 'package:logging/logging.dart';
import 'package:sqflite/sqflite.dart';

import 'embedder.dart';
import 'models/search_config.dart';
import 'models/search_entry.dart';
import 'models/search_metadata.dart';
import 'models/search_result.dart';
import 'ranking/search_ranking.dart';
import 'reranker/heuristic_reranker.dart';
import 'reranker/reranker.dart';

/// Type alias for embedding vectors, improving readability across the API.
typedef Embedding = Float32List;

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
///
/// ## Logging
///
/// The engine uses `package:logging` with the logger name
/// `'HybridSearchEngine'`. Attach a handler to see diagnostics:
///
/// ```dart
/// Logger.root.level = Level.FINE;
/// Logger.root.onRecord.listen((record) {
///   print('${record.level.name}: ${record.message}');
/// });
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
  /// [embedCacheSize] controls the LRU cache for [Embedder.embed] results.
  /// Set to `0` to disable caching. Default: `32`.
  ///
  /// Call [initialize] before the first [search] call.
  HybridSearchEngine({
    required Database db,
    required List<Embedding> embeddings,
    required Embedder embedder,
    HybridSearchConfig config = const HybridSearchConfig(),
    RerankerInterface? reranker,
    int embedCacheSize = 32,
  })  : _db = db,
        _embeddings = embeddings,
        _embedder = embedder,
        _config = config,
        _reranker = reranker ?? const HeuristicReranker(),
        _embedCacheSize = embedCacheSize {
    // Validate embedding dimensions eagerly.
    for (int i = 0; i < _embeddings.length; i++) {
      if (_embeddings[i].length != _config.embeddingDim) {
        throw ArgumentError(
          'Embedding at index $i has dimension ${_embeddings[i].length}, '
          'expected ${_config.embeddingDim} (config.embeddingDim). '
          'Ensure all embeddings match the configured dimension.',
        );
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Logger
  // ---------------------------------------------------------------------------

  static final Logger _log = Logger('HybridSearchEngine');

  // ---------------------------------------------------------------------------
  // Private fields
  // ---------------------------------------------------------------------------

  final Database _db;
  final List<Embedding> _embeddings;
  final Embedder _embedder;
  final HybridSearchConfig _config;
  final RerankerInterface _reranker;
  final int _embedCacheSize;

  late final List<double> _norms;
  late final Map<int, String> _idToQuestion;
  LocalHNSW<int>? _hnsw;

  bool _initialized = false;
  bool _disposed = false;

  /// Guards against concurrent [initialize] calls.
  Completer<void>? _initCompleter;

  /// Simple LRU cache for embed results (query → vector).
  final Map<String, Embedding> _embedCache = <String, Embedding>{};

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
  /// Concurrent calls are safe: only one initialization runs, others await it.
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

    // Guard against concurrent initialize() calls.
    if (_initCompleter != null) {
      return _initCompleter!.future;
    }

    _initCompleter = Completer<void>();
    // Prevent unhandled-async-error when completeError is called
    // and no concurrent caller is awaiting the future yet.
    _initCompleter!.future.ignore();
    try {
      final Stopwatch sw = Stopwatch()..start();

      // Precompute L2 norms for fast cosine similarity.
      _norms = _embeddings.map(_l2Norm).toList();

      // Build HNSW index for large corpora.
      if (_embeddings.length >= _config.hnswThreshold) {
        _hnsw = _buildHnsw();
        _log.fine(
          'HNSW index built: ${_embeddings.length} vectors, '
          'M=${_config.hnswM}, ef=${_config.hnswEf}.',
        );
      }

      // Load question texts for typo-tolerant matching.
      _idToQuestion = await _loadQuestions();

      _initialized = true;
      sw.stop();
      _log.fine(
        'Initialized in ${sw.elapsedMilliseconds} ms: '
        '${_embeddings.length} entries, '
        'dim=${_config.embeddingDim}, '
        'HNSW=${_hnsw != null}.',
      );
      _initCompleter!.complete();
    } catch (e, st) {
      _initCompleter!.completeError(e, st);
      _initCompleter = null;
      rethrow;
    }
  }

  /// Releases the SQLite database connection.
  ///
  /// Safe to call multiple times — subsequent calls are no-ops.
  /// After [dispose], calls to [search] and [initialize] will throw a
  /// [StateError].
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _embedCache.clear();
    if (_initialized) {
      await _db.close();
      _initialized = false;
    }
    _log.fine('Disposed.');
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
    final ({List<SearchResult> results, SearchMetadata metadata}) r =
        await searchWithMetadata(query, limit: limit);
    return r.results;
  }

  /// Like [search], but also returns [SearchMetadata] with timing diagnostics.
  ///
  /// ```dart
  /// final (:results, :metadata) = await engine.searchWithMetadata('flutter');
  /// print('Total: ${metadata.totalMs.toStringAsFixed(1)} ms');
  /// ```
  Future<({List<SearchResult> results, SearchMetadata metadata})>
      searchWithMetadata(String query, {int limit = 3}) async {
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

    final Stopwatch total = Stopwatch()..start();
    final Stopwatch phase = Stopwatch();

    // Step 1: embed the query.
    phase.start();
    final Embedding queryVec = await _cachedEmbed(query);
    final double queryNorm = _l2Norm(queryVec);
    phase.stop();
    final double embedMs = phase.elapsedMicroseconds / 1000.0;

    // Step 2: vector scoring.
    phase.reset();
    phase.start();
    final List<({int index, double score})> vecScores =
        _scoreByVector(queryVec, queryNorm);
    final Map<int, double> idToScore = _toIdScoreMap(vecScores);
    phase.stop();
    final double vectorMs = phase.elapsedMicroseconds / 1000.0;

    // Step 3: keyword scoring (FTS + typo).
    final List<String> words = _embedder.contentWords(query);

    phase.reset();
    phase.start();
    final List<int> ftsIds = await _ftsSearch(words);
    phase.stop();
    final double ftsMs = phase.elapsedMicroseconds / 1000.0;

    phase.reset();
    phase.start();
    final Set<int> keywordIds = _keywordMatches(words, ftsIds);
    phase.stop();
    final double typoMs = phase.elapsedMicroseconds / 1000.0;

    // Step 4: build candidate pool.
    final Set<int> poolIds = _buildPool(vecScores, keywordIds);
    if (poolIds.isEmpty) {
      total.stop();
      return (
        results: <SearchResult>[],
        metadata: SearchMetadata(
          embedMs: embedMs,
          vectorMs: vectorMs,
          ftsMs: ftsMs,
          typoMs: typoMs,
          rerankMs: 0,
          totalMs: total.elapsedMicroseconds / 1000.0,
          candidateCount: 0,
          vectorCandidateCount: 0,
          keywordCandidateCount: 0,
        ),
      );
    }

    final int vectorCandidateCount =
        min(vecScores.length, _config.candidatePoolSize);
    final int keywordCandidateCount = keywordIds.length;

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
    phase.reset();
    phase.start();
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
    final List<SearchResult> results = _filterByOverlap(query, ranked);
    phase.stop();
    final double rerankMs = phase.elapsedMicroseconds / 1000.0;

    total.stop();
    final double totalMs = total.elapsedMicroseconds / 1000.0;

    _log.fine(
      'Search "$query": ${results.length} results in '
      '${totalMs.toStringAsFixed(1)} ms '
      '(embed=${embedMs.toStringAsFixed(1)}, '
      'vec=${vectorMs.toStringAsFixed(1)}, '
      'fts=${ftsMs.toStringAsFixed(1)}, '
      'typo=${typoMs.toStringAsFixed(1)}, '
      'rerank=${rerankMs.toStringAsFixed(1)}).',
    );

    return (
      results: results,
      metadata: SearchMetadata(
        embedMs: embedMs,
        vectorMs: vectorMs,
        ftsMs: ftsMs,
        typoMs: typoMs,
        rerankMs: rerankMs,
        totalMs: totalMs,
        candidateCount: poolIds.length,
        vectorCandidateCount: vectorCandidateCount,
        keywordCandidateCount: keywordCandidateCount,
      ),
    );
  }

  /// Searches multiple queries in sequence, reusing the engine state.
  ///
  /// Returns one result list per query, in the same order as [queries].
  ///
  /// ```dart
  /// final batch = await engine.searchBatch(['dart', 'flutter', 'widgets']);
  /// for (final results in batch) {
  ///   print(results.first.entry.question);
  /// }
  /// ```
  Future<List<List<SearchResult>>> searchBatch(
    List<String> queries, {
    int limit = 3,
  }) async {
    final List<List<SearchResult>> results = <List<SearchResult>>[];
    for (final String query in queries) {
      results.add(await search(query, limit: limit));
    }
    return results;
  }

  // ---------------------------------------------------------------------------
  // Embed cache
  // ---------------------------------------------------------------------------

  /// Returns a cached embedding if available, otherwise computes and caches it.
  Future<Embedding> _cachedEmbed(String query) async {
    if (_embedCacheSize <= 0) return _embedder.embed(query);

    final Embedding? cached = _embedCache[query];
    if (cached != null) {
      _log.finest('Embed cache hit for "$query".');
      return cached;
    }

    final Embedding vec = await _embedder.embed(query);

    // Evict oldest entry if cache is full (simple FIFO eviction).
    if (_embedCache.length >= _embedCacheSize) {
      _embedCache.remove(_embedCache.keys.first);
    }
    _embedCache[query] = vec;
    return vec;
  }

  // ---------------------------------------------------------------------------
  // Vector scoring
  // ---------------------------------------------------------------------------

  /// Scores all embeddings against [queryVec].
  ///
  /// Uses HNSW when available (corpus ≥ [HybridSearchConfig.hnswThreshold]),
  /// otherwise performs a linear O(n) cosine scan.
  List<({int index, double score})> _scoreByVector(
    Embedding queryVec,
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
    } on DatabaseException catch (e) {
      _log.warning('FTS query failed (DatabaseException): $e');
      return <int>[];
    } on StateError catch (e) {
      _log.warning('FTS query failed (StateError): $e');
      return <int>[];
    }
  }

  /// Returns IDs matching [words] via FTS5 and typo-tolerant scan.
  Set<int> _keywordMatches(List<String> words, List<int> ftsIds) {
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
      Embedding? embedding,
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
  // Math helpers (loop-unrolled for performance)
  // ---------------------------------------------------------------------------

  /// Computes the L2 (Euclidean) norm of [v].
  ///
  /// Uses 4-element loop unrolling for better throughput on large vectors.
  double _l2Norm(Embedding v) {
    double s = 0.0;
    final int len = v.length;
    final int unrolled = len - (len % 4);
    for (int i = 0; i < unrolled; i += 4) {
      s += v[i] * v[i] +
          v[i + 1] * v[i + 1] +
          v[i + 2] * v[i + 2] +
          v[i + 3] * v[i + 3];
    }
    for (int i = unrolled; i < len; i++) {
      s += v[i] * v[i];
    }
    return sqrt(s);
  }

  /// Computes cosine similarity using precomputed norms.
  ///
  /// Uses 4-element loop unrolling for better throughput on large vectors.
  double _cosine(
    Embedding a,
    double na,
    Embedding b,
    double nb,
  ) {
    if (na == 0 || nb == 0) return 0.0;
    double dot = 0.0;
    final int len = a.length;
    final int unrolled = len - (len % 4);
    for (int i = 0; i < unrolled; i += 4) {
      dot += a[i] * b[i] +
          a[i + 1] * b[i + 1] +
          a[i + 2] * b[i + 2] +
          a[i + 3] * b[i + 3];
    }
    for (int i = unrolled; i < len; i++) {
      dot += a[i] * b[i];
    }
    return dot / (na * nb);
  }

  /// Returns the precomputed embedding for entry [id], or `null` if out of range.
  Embedding? _embeddingFor(int id) =>
      (id >= 1 && id <= _embeddings.length) ? _embeddings[id - 1] : null;
}
