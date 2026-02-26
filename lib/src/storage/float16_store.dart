import 'dart:math';
import 'dart:typed_data';

/// Loads precomputed Float16 embeddings from a compact binary file.
///
/// ## Binary format
///
/// ```
/// Offset  Size    Field
/// 0       4 B     count     (uint32 LE) — number of vectors
/// 4       4 B     dimension (uint32 LE) — floats per vector
/// 8       count × dimension × 2 B — Float16 vectors (IEEE 754 half-precision, LE)
/// ```
///
/// This layout is written by the companion Python training script:
///
/// ```python
/// import struct, numpy as np
///
/// vectors = np.array([...], dtype=np.float16)   # shape (N, D)
/// with open('embeddings.bin', 'wb') as f:
///     f.write(struct.pack('<II', N, D))
///     vectors.tofile(f)
/// ```
///
/// ## Usage
///
/// ```dart
/// // From a Flutter asset byte array:
/// final bytes = (await rootBundle.load('assets/embeddings.bin'))
///     .buffer.asUint8List();
/// final embeddings = Float16Store.decode(bytes);
///
/// // From a dart:io File:
/// final bytes = await File('/path/to/embeddings.bin').readAsBytes();
/// final embeddings = Float16Store.decode(bytes);
/// ```
abstract final class Float16Store {
  Float16Store._();

  // ---------------------------------------------------------------------------
  // Binary format constants
  // ---------------------------------------------------------------------------

  /// Header size: 2 × uint32 (count + dimension).
  static const int _headerBytes = 8;

  /// Bytes per Float16 value.
  static const int _f16Bytes = 2;

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Decodes [bytes] and returns a list of Float32 vectors.
  ///
  /// Each vector in the returned list corresponds to one row in the
  /// embeddings file. Index `i` maps to database entry with `id = i + 1`
  /// (IDs are 1-based).
  ///
  /// Throws [FormatException] if [bytes] is too short to contain the header
  /// or the declared data.
  ///
  /// All IEEE 754 half-precision special cases are handled correctly:
  /// subnormals, ±infinity, and NaN are converted to their Float32
  /// equivalents.
  static List<Float32List> decode(Uint8List bytes) {
    if (bytes.length < _headerBytes) {
      throw const FormatException(
        'Float16Store: byte array is too short to contain the 8-byte header.',
      );
    }

    final ByteData header = ByteData.sublistView(bytes, 0, _headerBytes);
    final int count = header.getUint32(0, Endian.little);
    final int dim = header.getUint32(4, Endian.little);

    final int expectedBytes = _headerBytes + count * dim * _f16Bytes;
    if (bytes.length < expectedBytes) {
      throw FormatException(
        'Float16Store: expected $expectedBytes bytes '
        '($count vectors × $dim dims × 2 B/f16 + 8 B header), '
        'got ${bytes.length}.',
      );
    }

    final List<Float32List> result = <Float32List>[];
    int offset = _headerBytes;

    for (int i = 0; i < count; i++) {
      result.add(_decodeVector(bytes, offset, dim));
      offset += dim * _f16Bytes;
    }

    return result;
  }

  /// Returns the number of vectors declared in the [bytes] header without
  /// decoding the full payload.
  ///
  /// Useful for a quick sanity check before full decoding.
  static int peekCount(Uint8List bytes) {
    if (bytes.length < _headerBytes) return 0;
    return ByteData.sublistView(bytes, 0, _headerBytes)
        .getUint32(0, Endian.little);
  }

  /// Returns the vector dimensionality declared in the [bytes] header.
  static int peekDimension(Uint8List bytes) {
    if (bytes.length < _headerBytes) return 0;
    return ByteData.sublistView(bytes, 0, _headerBytes)
        .getUint32(4, Endian.little);
  }

  // ---------------------------------------------------------------------------
  // Float16 → Float32 conversion
  // ---------------------------------------------------------------------------

  /// Decodes [dim] Float16 values starting at [offset] in [bytes] into a
  /// [Float32List].
  static Float32List _decodeVector(Uint8List bytes, int offset, int dim) {
    final Float32List out = Float32List(dim);
    for (int i = 0; i < dim; i++) {
      out[i] = _f16ToF32(bytes, offset + i * _f16Bytes);
    }
    return out;
  }

  /// Converts a single IEEE 754 half-precision value (little-endian at
  /// [byteOffset] in [bytes]) to a Dart [double].
  ///
  /// Handles all special cases:
  /// - Subnormals (exponent == 0, mantissa != 0)
  /// - Signed zero (exponent == 0, mantissa == 0)
  /// - ±Infinity (exponent == 31, mantissa == 0)
  /// - NaN (exponent == 31, mantissa != 0)
  /// - Normal values
  static double _f16ToF32(Uint8List bytes, int byteOffset) {
    final int raw = (bytes[byteOffset + 1] << 8) | bytes[byteOffset];
    final int sign = (raw >> 15) & 0x1;
    final int exp = (raw >> 10) & 0x1F;
    final int mant = raw & 0x3FF;
    final double sign_ = sign == 1 ? -1.0 : 1.0;

    if (exp == 0) {
      // Subnormal or signed zero.
      return sign_ * (mant / 1024.0) * pow(2, -14).toDouble();
    }
    if (exp == 31) {
      // Infinity or NaN.
      return mant == 0
          ? (sign == 1 ? double.negativeInfinity : double.infinity)
          : double.nan;
    }
    // Normal.
    return sign_ * (1.0 + mant / 1024.0) * pow(2, exp - 15).toDouble();
  }
}
