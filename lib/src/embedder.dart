import 'dart:typed_data';

/// Contract for converting text into a dense vector embedding.
///
/// Implement this interface to plug in any embedding model — BERT via ONNX,
/// a local sentence-transformer, a simple TF-IDF encoder, etc.
///
/// [HybridSearchEngine] calls [embed] once per query to obtain the vector
/// used for cosine similarity, and [contentWords] to obtain the
/// stopword-stripped tokens used for FTS5 query building and typo matching.
///
/// ## Minimal implementation
///
/// ```dart
/// class MyEmbedder implements Embedder {
///   @override
///   Future<Float32List> embed(String text) async {
///     // Call your model here.
///     return myModel.encode(text);
///   }
///
///   @override
///   List<String> contentWords(String text) {
///     // Return meaningful tokens after stopword removal.
///     return text.toLowerCase().split(RegExp(r'\s+')).toList();
///   }
/// }
/// ```
///
/// ## BERT + ONNX implementation
///
/// For a production setup, pair this library with a BERT-based tokenizer
/// (e.g. `package:dart_wordpiece`) and an ONNX runtime session:
///
/// ```dart
/// class BertEmbedder implements Embedder {
///   BertEmbedder(this._session, this._tokenizer);
///
///   final OrtSession _session;
///   final WordPieceTokenizer _tokenizer;
///
///   @override
///   Future<Float32List> embed(String text) async {
///     final output = _tokenizer.encode(text);
///     final inputs = {
///       'input_ids':      OrtValueTensor.createTensorWithDataList(output.inputIdsInt64, [1, 64]),
///       'attention_mask': OrtValueTensor.createTensorWithDataList(output.attentionMaskInt64, [1, 64]),
///       'token_type_ids': OrtValueTensor.createTensorWithDataList(output.tokenTypeIdsInt64, [1, 64]),
///     };
///     final results = _session.run(OrtRunOptions(), inputs);
///     final raw = results[0]!.value as List<List<List<double>>>;
///     return Float32List.fromList(_meanPool(raw[0], output.realLength));
///   }
///
///   @override
///   List<String> contentWords(String text) => _tokenizer.contentWords(text);
/// }
/// ```
abstract interface class Embedder {
  /// Converts [text] into a dense Float32 vector.
  ///
  /// The returned vector must have the same dimensionality as the
  /// precomputed embeddings supplied to [HybridSearchEngine] (see
  /// [HybridSearchConfig.embeddingDim]).
  ///
  /// This method is called once per [HybridSearchEngine.search] call and
  /// may be asynchronous (e.g. when running an ONNX model on a background
  /// isolate).
  Future<Float32List> embed(String text);

  /// Returns the meaningful (non-stopword) words from [text].
  ///
  /// These tokens are used to:
  /// - Build the FTS5 `MATCH` query (keyword recall).
  /// - Drive typo-tolerant matching against stored questions.
  /// - Filter final results by keyword overlap.
  ///
  /// The list should be lowercase and have stopwords removed. Empty lists
  /// are safe — the engine will skip keyword-based signals in that case.
  List<String> contentWords(String text);
}
