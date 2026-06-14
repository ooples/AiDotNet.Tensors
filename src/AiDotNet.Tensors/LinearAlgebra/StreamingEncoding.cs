// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Streaming-store encoding ids used by <see cref="WeightRegistry"/> on the
/// serialize side and <see cref="TensorBase{T}"/> on the restore side. The
/// id travels in <c>Tensor&lt;T&gt;.StreamingStoreEncoding</c> (a single byte
/// alongside each registered weight), so the encode and decode paths must
/// agree on the integer value. Centralising the ids here keeps the two
/// sides in sync — a previous CodeRabbit review on PR #604 flagged the
/// hardcoded <c>1</c>/<c>2</c>/<c>3</c> magic numbers as a blocking
/// production-readiness issue because a one-sided edit could silently
/// corrupt restore.
/// </summary>
internal static class StreamingEncoding
{
    /// <summary>Raw native bytes (float / double / int / long); no compression,
    /// no quantisation. <c>SerializeToBytes</c> writes this when the type
    /// isn't streamable to a narrower format or when the policy explicitly
    /// requests full precision.</summary>
    internal const byte Native = 0;

    /// <summary>bf16: narrow float/double → 2-byte bf16. Halves the resident,
    /// disk, and eviction footprint at the cost of 16-bit mantissa rounding
    /// on restore. Float/double only; the policy never picks this
    /// automatically when the model is in training mode.</summary>
    internal const byte Bf16 = 1;

    /// <summary>int8: per-tensor symmetric quantisation (1 byte/element +
    /// 4-byte scale prefix). Lossier than bf16; explicit opt-in only —
    /// <c>StreamingStoreDtype.Auto</c> never picks int8.</summary>
    internal const byte Int8 = 2;

    /// <summary>Lossless: byte-shuffle + LZ4 (variable size). Exact restore,
    /// modest compression ratio (~0.65× for typical weights). Explicit
    /// opt-in.</summary>
    internal const byte Lossless = 3;
}
