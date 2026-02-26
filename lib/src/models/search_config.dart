/// Configuration for [HybridSearchEngine].
///
/// All parameters have production-ready defaults derived from the
/// [offline_kb_assist](https://github.com/EgorKorshun/offline_kb_assist)
/// reference implementation. Override only what differs from your setup.
///
/// ## Tuning guide
///
/// **Small corpus (< 1 000 entries)**
/// ```dart
/// const config = HybridSearchConfig(); // defaults are optimal
/// ```
///
/// **Large corpus (≥ 1 000 entries)**
/// ```dart
/// const config = HybridSearchConfig(
///   hnswThreshold: 1000,   // enable HNSW (already default)
///   hnswM: 32,             // more connections → better recall, more RAM
///   hnswEf: 128,           // wider search → better recall, slower
///   hnswSearchK: 150,      // return more candidates from HNSW
///   candidatePoolSize: 80, // feed more to reranker
/// );
/// ```
///
/// **Custom database schema**
/// ```dart
/// const config = HybridSearchConfig(
///   tableName:      'articles',
///   ftsTableName:   'articles_fts',
///   idColumn:       'article_id',
///   categoryColumn: 'section',
///   questionColumn: 'title',
///   answerColumn:   'body',
/// );
/// ```
class HybridSearchConfig {
  /// Creates a [HybridSearchConfig] with the given parameters.
  ///
  /// All parameters are optional and have sensible defaults.
  const HybridSearchConfig({
    this.candidatePoolSize = 50,
    this.ftsLimit = 50,
    this.hnswThreshold = 1000,
    this.hnswSearchK = 100,
    this.hnswM = 16,
    this.hnswEf = 64,
    this.embeddingDim = 128,
    this.tableName = 'entries',
    this.ftsTableName = 'fts',
    this.idColumn = 'id',
    this.categoryColumn = 'category',
    this.questionColumn = 'question',
    this.answerColumn = 'answer',
  });

  // -------------------------------------------------------------------------
  // Search tuning
  // -------------------------------------------------------------------------

  /// Maximum number of candidates fed into the reranker.
  ///
  /// Candidates are the union of the top [candidatePoolSize] entries by
  /// cosine similarity and all keyword-matched entries. Larger values
  /// improve recall at the cost of more reranker work.
  ///
  /// Default: `50`.
  final int candidatePoolSize;

  /// Maximum number of entries returned by the FTS5 query per search.
  ///
  /// Default: `50`.
  final int ftsLimit;

  // -------------------------------------------------------------------------
  // HNSW (approximate nearest neighbour)
  // -------------------------------------------------------------------------

  /// Minimum number of entries required to build an HNSW index.
  ///
  /// Below this threshold, a linear O(n) cosine scan is used because it is
  /// faster than HNSW for small corpora. At or above this threshold, HNSW
  /// keeps each query sub-millisecond regardless of corpus size.
  ///
  /// Default: `1000`.
  final int hnswThreshold;

  /// Number of nearest neighbours requested from HNSW per query.
  ///
  /// Must be ≥ [candidatePoolSize] to ensure the pool has enough
  /// vector-top candidates. Increasing this improves recall.
  ///
  /// Default: `100`.
  final int hnswSearchK;

  /// HNSW graph parameter: maximum connections per node.
  ///
  /// Higher values improve recall and graph quality at the expense of
  /// build time and memory usage. The `local_hnsw` default is `8`.
  ///
  /// Default: `16`.
  final int hnswM;

  /// HNSW search parameter: size of the dynamic candidate list during search.
  ///
  /// Higher values improve recall at query time at the expense of latency.
  /// The `local_hnsw` default is `16`.
  ///
  /// Default: `64`.
  final int hnswEf;

  // -------------------------------------------------------------------------
  // Embedding
  // -------------------------------------------------------------------------

  /// Dimensionality of the embedding vectors.
  ///
  /// Must match the output dimension of the model used by the [Embedder]
  /// and the vectors stored in the precomputed embeddings list.
  ///
  /// Default: `128` (BERT-Tiny 2L/128D).
  final int embeddingDim;

  // -------------------------------------------------------------------------
  // Database schema
  // -------------------------------------------------------------------------

  /// Name of the SQLite table containing the searchable entries.
  ///
  /// Default: `'entries'`.
  final String tableName;

  /// Name of the SQLite FTS5 virtual table for full-text search.
  ///
  /// The FTS table should index at least [questionColumn].
  /// Typical setup:
  /// ```sql
  /// CREATE VIRTUAL TABLE fts USING fts5(question, content=entries, content_rowid=id);
  /// ```
  ///
  /// Default: `'fts'`.
  final String ftsTableName;

  /// Name of the primary key column in [tableName].
  ///
  /// The ID must be 1-based and match the index position in the embeddings
  /// list (`id = embeddingIndex + 1`).
  ///
  /// Default: `'id'`.
  final String idColumn;

  /// Name of the category / section column in [tableName].
  ///
  /// Default: `'category'`.
  final String categoryColumn;

  /// Name of the question / title column in [tableName].
  ///
  /// This column is searched by FTS5 and matched by the typo-tolerance logic.
  ///
  /// Default: `'question'`.
  final String questionColumn;

  /// Name of the answer / body column in [tableName].
  ///
  /// Default: `'answer'`.
  final String answerColumn;

  @override
  String toString() => 'HybridSearchConfig('
      'candidatePoolSize: $candidatePoolSize, '
      'ftsLimit: $ftsLimit, '
      'hnswThreshold: $hnswThreshold, '
      'embeddingDim: $embeddingDim, '
      'table: $tableName'
      ')';
}
