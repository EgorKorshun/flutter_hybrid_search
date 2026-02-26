/// Offline hybrid search engine for Flutter.
///
/// Combines vector similarity (cosine / HNSW), FTS5 full-text search,
/// typo-tolerant keyword matching, and heuristic reranking — entirely
/// on-device, no network required.
///
/// ## Quick start
///
/// ```dart
/// import 'package:flutter_hybrid_search/flutter_hybrid_search.dart';
///
/// // Load precomputed Float16 embeddings from assets.
/// final bytes = (await rootBundle.load('assets/embeddings.bin'))
///     .buffer.asUint8List();
/// final embeddings = Float16Store.decode(bytes);
///
/// // Open the SQLite database.
/// final db = await openDatabase('kb.db', readOnly: true);
///
/// // Create and initialise the engine.
/// final engine = HybridSearchEngine(
///   db: db,
///   embeddings: embeddings,
///   embedder: MyEmbedder(),
/// );
/// await engine.initialize();
///
/// // Search.
/// final results = await engine.search('What is Flutter?', limit: 3);
/// for (final r in results) {
///   print('${r.score.toStringAsFixed(3)}  ${r.entry.question}');
/// }
/// ```
///
/// ## Exports
///
/// | Class | Description |
/// |---|---|
/// | `HybridSearchEngine` | Main search engine |
/// | `Embedder` | Abstract interface for embedding generation |
/// | `SearchEntry` | Knowledge-base entry (id, category, question, answer) |
/// | `SearchResult` | Search match (entry + score + method) |
/// | `HybridSearchConfig` | Tunable engine parameters |
/// | `SearchRanking` | Pure ranking utilities (boosts, typo logic) |
/// | `HeuristicReranker` | Rule-based reranker (default) |
/// | `RerankerInterface` | Contract for custom rerankers |
/// | `RerankerCandidates` | Type alias for reranker input |
/// | `Float16Store` | Decoder for the Float16 binary embedding format |
library;

export 'src/embedder.dart';
export 'src/hybrid_search_engine.dart';
export 'src/models/search_config.dart';
export 'src/models/search_entry.dart';
export 'src/models/search_result.dart';
export 'src/ranking/search_ranking.dart';
export 'src/reranker/heuristic_reranker.dart';
export 'src/reranker/reranker.dart';
export 'src/storage/float16_store.dart';
