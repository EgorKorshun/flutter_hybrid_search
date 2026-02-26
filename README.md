# flutter_hybrid_search

An **offline hybrid search engine** for Flutter that combines vector
similarity, FTS5 full-text search, typo-tolerant keyword matching, and
heuristic reranking — entirely on-device, no cloud, no latency.

---

## Features

- ✅ **Vector similarity search** — cosine distance on precomputed Float32 embeddings
- ✅ **HNSW approximate index** — sub-millisecond vector search for corpora ≥ 1 000 entries
- ✅ **FTS5 full-text search** — exact keyword matching via SQLite FTS5
- ✅ **Typo-tolerant matching** — 1-character edit distance (substitution, insertion, deletion)
- ✅ **Heuristic reranking** — FTS boost + typo boost + concise-question boost + deduplication
- ✅ **Pluggable embedder** — implement `Embedder` with any model (BERT, TF-IDF, …)
- ✅ **Pluggable reranker** — implement `RerankerInterface` for custom ranking logic
- ✅ **Float16 binary format** — compact precomputed embeddings (50 % smaller than Float32)
- ✅ **Configurable schema** — custom table/column names for any SQLite database
- ✅ **Zero cloud dependency** — works fully offline on Android, iOS, macOS, Linux, Windows

---

## Getting started

```yaml
dependencies:
  flutter_hybrid_search: ^1.0.1
  sqflite: ^2.4.2
```

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐    Float32 vector
│   Embedder  │──────────────────────┐
└─────────────┘                      ▼
                              ┌──────────────┐
                              │ Vector scorer│  cosine / HNSW
                              └──────┬───────┘
                                     │ top-N candidates
┌─────────────┐                      ▼
│  SQLite DB  │──► FTS5 MATCH ──► ┌──────────────┐
│    FTS5     │──► Typo scan  ──► │ Candidate    │ union
└─────────────┘                   │    pool      │
                                  └──────┬───────┘
                                         │
                                  ┌──────▼───────┐
                                  │   Reranker   │ boosts + dedup
                                  └──────┬───────┘
                                         │
                                  ┌──────▼───────┐
                                  │ Keyword filter│ overlap check
                                  └──────┬───────┘
                                         │
                                  List<SearchResult>
```

---

## Usage

### 1. Implement `Embedder`

```dart
import 'package:flutter_hybrid_search/flutter_hybrid_search.dart';

class MyEmbedder implements Embedder {
  @override
  Future<Float32List> embed(String text) async {
    // Run your model here (ONNX, TFLite, etc.).
    return myModel.encode(text); // must return Float32List
  }

  @override
  List<String> contentWords(String text) {
    // Return meaningful tokens, stopwords removed.
    return text.toLowerCase()
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty && !_stopwords.contains(w))
        .toList();
  }
}
```

### 2. Load assets and build the engine

```dart
// Load Float16 embeddings from a binary asset.
final bytes = (await rootBundle.load('assets/embeddings.bin')).buffer.asUint8List();
final embeddings = Float16Store.decode(bytes);

// Open the SQLite database.
final db = await openDatabase('kb.db', readOnly: true);

// Create and initialise the engine.
final engine = HybridSearchEngine(
  db: db,
  embeddings: embeddings,
  embedder: MyEmbedder(),
);
await engine.initialize();
```

### 3. Search

```dart
final results = await engine.search('What is Flutter?', limit: 3);

for (final r in results) {
  print('${r.score.toStringAsFixed(3)}  ${r.entry.question}');
  print(r.entry.answer);
}
```

### 4. Custom configuration

```dart
final engine = HybridSearchEngine(
  db: db,
  embeddings: embeddings,
  embedder: MyEmbedder(),
  config: const HybridSearchConfig(
    candidatePoolSize: 80,    // more candidates → better recall
    hnswThreshold: 500,       // enable HNSW at 500+ entries
    embeddingDim: 256,        // match your model's output size
    tableName: 'articles',    // custom DB schema
    questionColumn: 'title',
    answerColumn: 'body',
  ),
  reranker: const HeuristicReranker(), // or your own RerankerInterface
);
```

### 5. Custom reranker

```dart
class CategoryBoostReranker implements RerankerInterface {
  const CategoryBoostReranker(this.preferred);
  final String preferred;

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
    final sorted = candidates.toList()
      ..sort((a, b) {
        final sa = a.vectorScore + (a.entry.category == preferred ? 0.3 : 0.0);
        final sb = b.vectorScore + (b.entry.category == preferred ? 0.3 : 0.0);
        return sb.compareTo(sa);
      });
    return sorted.take(limit).map((c) => SearchResult(
      entry: c.entry, score: c.vectorScore, method: 'category_boost',
    )).toList();
  }
}
```

---

## Database schema

The default schema expected by the engine:

```sql
CREATE TABLE entries (
  id       INTEGER PRIMARY KEY,   -- 1-based, matches embedding index
  category TEXT    NOT NULL,
  question TEXT    NOT NULL,
  answer   TEXT    NOT NULL
);

-- FTS5 virtual table for full-text search over 'question'.
CREATE VIRTUAL TABLE fts USING fts5(
  question,
  content=entries,
  content_rowid=id
);
```

All column and table names are overridable via `HybridSearchConfig`.

---

## Embedding binary format

The `Float16Store.decode` method reads this layout (written by a Python
training script or any compatible tool):

```
Offset  Size     Field
0       4 bytes  count     (uint32, little-endian)
4       4 bytes  dimension (uint32, little-endian)
8+      count × dim × 2 B  Float16 vectors (IEEE 754, little-endian)
```

**Python writer:**
```python
import struct, numpy as np

vectors = embeddings.astype(np.float16)  # shape (N, D)
with open('embeddings.bin', 'wb') as f:
    f.write(struct.pack('<II', N, D))
    vectors.tofile(f)
```

---

## API reference

### `HybridSearchEngine`

| Member                                                               | Description                                                      |
|----------------------------------------------------------------------|------------------------------------------------------------------|
| `HybridSearchEngine({db, embeddings, embedder, config?, reranker?})` | Constructor                                                      |
| `initialize()`                                                       | Builds HNSW index + loads question map. Call once before search. |
| `search(query, {limit})` → `Future<List<SearchResult>>`              | Main search method                                               |
| `dispose()`                                                          | Closes the database connection                                   |

### `Embedder`

| Member                                | Description                                   |
|---------------------------------------|-----------------------------------------------|
| `embed(text)` → `Future<Float32List>` | Convert text to dense vector                  |
| `contentWords(text)` → `List<String>` | Stopword-stripped tokens for keyword matching |

### `HybridSearchConfig`

| Parameter           | Default      | Description                              |
|---------------------|--------------|------------------------------------------|
| `candidatePoolSize` | `50`         | Max candidates fed to reranker           |
| `ftsLimit`          | `50`         | Max FTS5 results per query               |
| `hnswThreshold`     | `1000`       | Min entries to enable HNSW index         |
| `hnswSearchK`       | `100`        | Neighbours to fetch from HNSW            |
| `hnswM`             | `16`         | HNSW graph connections per node          |
| `hnswEf`            | `64`         | HNSW search candidate list size          |
| `embeddingDim`      | `128`        | Vector dimension (must match your model) |
| `tableName`         | `'entries'`  | SQLite table name                        |
| `ftsTableName`      | `'fts'`      | FTS5 virtual table name                  |
| `idColumn`          | `'id'`       | Primary key column                       |
| `categoryColumn`    | `'category'` | Category column                          |
| `questionColumn`    | `'question'` | Question / title column                  |
| `answerColumn`      | `'answer'`   | Answer / body column                     |

### `SearchRanking` — boost constants

| Constant                | Value       | Trigger                            |
|-------------------------|-------------|------------------------------------|
| `ftsBoost`              | `0.5`       | Entry found by FTS5 MATCH          |
| `typoBoost`             | `0.7`       | Entry found by typo tolerance only |
| `conciseMatchBoost`     | `0.5` (max) | Short, on-topic question           |
| `perfectScoreThreshold` | `0.999`     | Returns single result shortcut     |

### `Float16Store`

| Method                                             | Description                   |
|----------------------------------------------------|-------------------------------|
| `Float16Store.decode(bytes)` → `List<Float32List>` | Decode full embedding file    |
| `Float16Store.peekCount(bytes)` → `int`            | Read vector count from header |
| `Float16Store.peekDimension(bytes)` → `int`        | Read dimension from header    |

---

## Search pipeline in detail

1. **Embed query** — `Embedder.embed(query)` → `Float32List`
2. **Vector scoring** — cosine similarity against all precomputed embeddings
   - *Small corpus (< `hnswThreshold`)*: O(n) linear scan
   - *Large corpus (≥ `hnswThreshold`)*: HNSW approximate search (sub-ms)
3. **FTS5 search** — `MATCH` query on the question column; retries with single word on no results
4. **Typo-tolerant scan** — Levenshtein-1 match on all question texts
5. **Candidate pool** — union of top-N by vector score + all keyword matches
6. **Rerank** — apply boost signals, sort, deduplicate by question text
7. **Keyword-overlap filter** — discard results with zero word overlap to query

---

## Pair with `dart_wordpiece`

For a complete offline pipeline, pair this package with
[`dart_wordpiece`](https://pub.dev/packages/dart_wordpiece) for BERT
tokenization:

```dart
class BertEmbedder implements Embedder {
  BertEmbedder(this._session, this._tokenizer);

  final OrtSession _session;
  final WordPieceTokenizer _tokenizer;

  @override
  Future<Float32List> embed(String text) async {
    final output = _tokenizer.encode(text);
    final inputs = {
      'input_ids':      OrtValueTensor.createTensorWithDataList(output.inputIdsInt64, [1, 64]),
      'attention_mask': OrtValueTensor.createTensorWithDataList(output.attentionMaskInt64, [1, 64]),
      'token_type_ids': OrtValueTensor.createTensorWithDataList(output.tokenTypeIdsInt64, [1, 64]),
    };
    final outputs = _session.run(OrtRunOptions(), inputs);
    final raw = outputs[0]!.value as List<List<List<double>>>;
    return Float32List.fromList(_meanPool(raw[0], output.realLength));
  }

  @override
  List<String> contentWords(String text) => _tokenizer.contentWords(text);
}
```

---

## Performance

Benchmarked on a mid-range Android device with 500 entries and 128-dim BERT-Tiny:

| Step                                 | Time                 |
|--------------------------------------|----------------------|
| Embedding generation                 | 10–50 ms             |
| Vector search (linear, 500 entries)  | < 1 ms               |
| Vector search (HNSW, 10 000 entries) | < 2 ms               |
| FTS5 query                           | < 5 ms               |
| Typo-tolerant scan                   | < 10 ms              |
| Reranking                            | < 5 ms               |
| **Total**                            | **< 100 ms typical** |

---

## Contributing

Issues and pull requests are welcome!

1. Run `flutter analyze` — zero warnings required
2. Run `flutter test` — all tests must pass
3. Follow the [Dart style guide](https://dart.dev/effective-dart)

---

## License

MIT
