## 1.2.0

### Added

- **`ScoreBreakdown`** — new model that exposes the individual contribution of
  each ranking signal (`vectorScore`, `ftsScore`, `typoScore`, `conciseScore`,
  `totalScore`). Every `SearchResult` produced by `HeuristicReranker` now
  carries a populated `breakdown` field, making it easy to debug rankings and
  build explainability UI.
- **`SearchResult.breakdown`** — optional `ScoreBreakdown?` field on
  `SearchResult`. `null` for custom rerankers that don't opt in; non-null for
  the default `HeuristicReranker`.
- **`SearchResult.copyWith()`** — create a modified copy of a result without
  touching other fields.
- **`HybridSearchEngine.addEntries()`** — add new entries and their embeddings
  to a live engine without recreating it. Automatically rebuilds the HNSW
  index when the corpus crosses `hnswThreshold`.
- **`HybridSearchEngine.removeEntries()`** — remove entries by id from a live
  engine. Rebuilds or clears the HNSW index as needed. Unknown ids are silently
  ignored.
- **`HybridSearchConfig.minScore`** — minimum composite score threshold
  (default `0.0`). Results below this value are discarded after reranking,
  letting you enforce a quality floor without post-processing the list yourself.
- **`SearchEntry.metadata`** — arbitrary `Map<String, Object?>` for
  domain-specific key-value data (priority, tags, dates, etc.). Defaults to an
  empty map — fully backward compatible. Persists as JSON when you pass a
  `metadataColumn` to `SearchEntry.fromMap()` / `SearchEntry.toMap()`.

### Improved

- **Internal embedding storage** refactored from a contiguous `List` to a
  sparse `Map<int, Embedding>` keyed by SQLite id. This makes incremental
  adds and removes correct in all cases (non-contiguous ids, gaps after
  deletion) without any SQLite ID remapping.
- **HNSW items** are now tagged by their SQLite id instead of a list index,
  removing the `index + 1` offset that would have become incorrect after
  removals.

## 1.1.0

### Added

- **Structured logging** via `package:logging` — the engine logs initialization
  timing, per-query phase breakdown, FTS errors, and cache hits. Attach a
  handler to `Logger('HybridSearchEngine')` to see diagnostics.
- **`SearchMetadata`** — new model with per-phase timing (embed, vector, FTS,
  typo, rerank, total) and candidate counts.
- **`searchWithMetadata()`** — returns `({List<SearchResult> results,
  SearchMetadata metadata})` for performance profiling.
- **`searchBatch()`** — search multiple queries in one call, reusing engine
  state and embed cache.
- **`Embedding` typedef** — `typedef Embedding = Float32List` for improved API
  readability.
- **Embed cache** — LRU cache for `Embedder.embed()` results. Configurable via
  `embedCacheSize` constructor parameter (default: 32, set 0 to disable).

### Improved

- **Embedding dimension validation** — constructor now eagerly validates that
  every embedding vector matches `config.embeddingDim`, throwing a clear
  `ArgumentError` on mismatch instead of crashing later in cosine computation.
- **Config validation** — `HybridSearchConfig` now asserts that numeric
  parameters are positive and that `hnswSearchK >= candidatePoolSize`.
- **Float16Store validation** — `decode()` now throws `FormatException` when the
  header declares zero vectors or zero dimensions.
- **Race-safe initialization** — concurrent `initialize()` calls are now safe:
  uses a `Completer` so only one init runs and others await it.
- **FTS error handling** — `_ftsSearch()` now catches specific `DatabaseException`
  and `StateError` types and logs them via `package:logging` instead of
  silently swallowing all exceptions with `catch (_)`.
- **Loop-unrolled cosine & L2 norm** — inner loops process 4 elements per
  iteration for better throughput on large vectors (128+ dims).

## 1.0.2

### Added

- `HybridSearchEngine.isInitialized` getter — check engine state without
  try/catch.
- `HybridSearchEngine.entryCount` getter — number of embeddings managed by the
  engine, available immediately after construction.
- `HybridSearchConfig.copyWith()` — create a tweaked copy without repeating
  every field.

### Improved

- `dispose()` is now idempotent (safe to call multiple times) and guards
  against use-after-dispose: calling `search()` or `initialize()` after
  `dispose()` throws a clear `StateError`.
- Typo-tolerance `_canDrop` rewritten from O(n²) substring concatenation to
  O(n) two-pointer scan — eliminates temporary string allocations.
- `SearchRanking` RegExp instances are now compiled once and cached as static
  finals instead of being re-created on every call.

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
