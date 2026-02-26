// ignore_for_file: avoid_print
// ignore_for_file: unreachable_from_main
// ignore_for_file: avoid_redundant_argument_values

/// flutter_hybrid_search — usage example.
///
/// This file demonstrates how to wire up [HybridSearchEngine] in a real
/// Flutter application. For a runnable unit-test version, see the package's
/// test file which uses an in-memory SQLite database.
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_hybrid_search/flutter_hybrid_search.dart';
import 'package:sqflite/sqflite.dart';

// ---------------------------------------------------------------------------
// Step 1 — Implement Embedder for your model
// ---------------------------------------------------------------------------

/// Example [Embedder] that wraps a BERT/ONNX session.
///
/// Replace the body of [embed] with your actual model call.
/// If you use `package:dart_wordpiece` for tokenization, combine it here.
class BertEmbedder implements Embedder {
  BertEmbedder({required Set<String> stopwords}) : _stopwords = stopwords;

  final Set<String> _stopwords;

  @override
  Future<Float32List> embed(String text) async {
    // In a real implementation, tokenize and run your ONNX session.
    // Example with package:onnxruntime:
    //
    // final tokens = _tokenizer.encode(text);
    // final inputs = {
    //   'input_ids':      OrtValueTensor.createTensorWithDataList(tokens.inputIdsInt64, [1, 64]),
    //   'attention_mask': OrtValueTensor.createTensorWithDataList(tokens.attentionMaskInt64, [1, 64]),
    //   'token_type_ids': OrtValueTensor.createTensorWithDataList(tokens.tokenTypeIdsInt64, [1, 64]),
    // };
    // final outputs = _session.run(OrtRunOptions(), inputs);
    // final raw = outputs[0]!.value as List<List<List<double>>>;
    // return Float32List.fromList(_meanPool(raw[0], tokens.realLength));
    throw UnimplementedError('Replace with your model call.');
  }

  @override
  List<String> contentWords(String text) {
    return text
        .toLowerCase()
        .split(RegExp(r'[^\p{L}\p{N}]+', unicode: true))
        .where((String w) => w.isNotEmpty && !_stopwords.contains(w))
        .toList();
  }
}

// ---------------------------------------------------------------------------
// Step 2 — Load assets and build the engine
// ---------------------------------------------------------------------------

Future<HybridSearchEngine> buildEngine() async {
  // Load precomputed Float16 embeddings from Flutter asset.
  //   Training script produces: [count: uint32][dim: uint32][float16 vectors]
  final ByteData embeddingData =
      await rootBundle.load('assets/embeddings.bin');
  final List<Float32List> embeddings =
      Float16Store.decode(embeddingData.buffer.asUint8List());

  // Open the SQLite knowledge-base (copy from assets first if needed).
  final Database db = await openDatabase('kb.db', readOnly: true);

  // Load stopwords (optional — omit if your embedder handles it internally).
  final String stopwordsRaw = await rootBundle.loadString('assets/stopwords.txt');
  final Set<String> stopwords = stopwordsRaw
      .split('\n')
      .map((String s) => s.trim())
      .where((String s) => s.isNotEmpty)
      .toSet();

  final HybridSearchEngine engine = HybridSearchEngine(
    db: db,
    embeddings: embeddings,
    embedder: BertEmbedder(stopwords: stopwords),
    // Optional: override defaults.
    config: const HybridSearchConfig(
      candidatePoolSize: 50,
      ftsLimit: 50,
      hnswThreshold: 1000, // enable HNSW for large corpora
    ),
    // Optional: custom reranker (default is HeuristicReranker).
    reranker: const HeuristicReranker(),
  );

  await engine.initialize();
  return engine;
}

// ---------------------------------------------------------------------------
// Step 3 — Search
// ---------------------------------------------------------------------------

Future<void> runSearch(HybridSearchEngine engine, String query) async {
  print('Query: $query');
  print('---');

  final List<SearchResult> results = await engine.search(query, limit: 3);

  if (results.isEmpty) {
    print('No results found.');
    return;
  }

  for (int i = 0; i < results.length; i++) {
    final SearchResult r = results[i];
    print('${i + 1}. [${r.score.toStringAsFixed(3)}] ${r.entry.question}');
    print('   Category: ${r.entry.category}');
    print('   ${r.entry.answer}');
    print('');
  }
}

// ---------------------------------------------------------------------------
// Step 4 — Custom reranker example
// ---------------------------------------------------------------------------

/// A reranker that boosts entries in a preferred category.
///
/// Demonstrates how to implement [RerankerInterface] for domain-specific logic.
class CategoryBoostReranker implements RerankerInterface {
  const CategoryBoostReranker({required this.preferredCategory, this.boost = 0.3});

  final String preferredCategory;
  final double boost;

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
    final List<({SearchEntry entry, double vectorScore, Float32List? embedding})>
        sorted = candidates.toList()
          ..sort(
            (
              ({
                SearchEntry entry,
                Float32List? embedding,
                double vectorScore,
              }) a,
              ({
                SearchEntry entry,
                Float32List? embedding,
                double vectorScore,
              }) b,
            ) {
              final double sa = a.vectorScore +
                  (a.entry.category == preferredCategory ? boost : 0.0);
              final double sb = b.vectorScore +
                  (b.entry.category == preferredCategory ? boost : 0.0);
              return sb.compareTo(sa);
            },
          );

    return sorted
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
                method: 'category_boost',
              ),
        )
        .toList();
  }
}

// ---------------------------------------------------------------------------
// Step 5 — Float16Store standalone usage
// ---------------------------------------------------------------------------

void float16StoreExample() {
  // Build a small 2-vector, 4-dim file in memory (for illustration).
  // Normally you'd do: Float16Store.decode(await File(...).readAsBytes());
  print('Float16Store.peekCount / peekDimension work on any Uint8List.');
  print('See the test file for a complete decode example.');
}
