## 1.0.1

Improvements in documentation and metadata

## 1.0.0

Initial release.

### Added

- `HybridSearchEngine` — offline hybrid search combining vector similarity,
  FTS5 full-text search, typo-tolerant matching, and heuristic reranking.
- `Embedder` — abstract interface for plugging in any embedding model
  (BERT, sentence-transformers, TF-IDF, etc.).
- `SearchEntry` — immutable Q&A entry with configurable column mapping.
- `SearchResult` — search match carrying entry, score, and method identifier.
- `HybridSearchConfig` — all engine parameters in one place (pool sizes,
  HNSW thresholds, database schema column names).
- `SearchRanking` — pure static utilities: FTS query building, typo
  tolerance, concise-match boost, perfect-match shortcutting.
- `HeuristicReranker` — rule-based reranker (FTS + typo + concise boosts,
  deduplication by question text).
- `RerankerInterface` — contract for custom reranker implementations.
- `Float16Store` — decoder for the compact Float16 binary embedding format
  produced by the companion Python training pipeline.
