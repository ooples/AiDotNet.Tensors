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

    /// <summary>int8: PER-ROW symmetric quantisation — [int32 rows][rows × fp32
    /// scale][1 byte/element]. One scale per output channel preserves far more SNR
    /// than a single per-tensor scale on varied-magnitude rows, and the layout feeds
    /// the int8 weight-only GEMM directly (no upcast). Lossier than bf16; explicit
    /// opt-in only — <c>StreamingStoreDtype.Auto</c> never picks int8.</summary>
    internal const byte Int8 = 2;

    /// <summary>Lossless: SIMD byte-plane shuffle + Deflate (variable size). Exact
    /// restore, ~1.18× on typical fp weights. The <c>Auto</c> default in training
    /// (bit-exact masters); otherwise explicit opt-in.</summary>
    internal const byte Lossless = 3;

    /// <summary>int4: AWQ/GPTQ-style GROUP-symmetric quantization — [int32 count][int32
    /// groupSize][numGroups × fp32 scale][ceil(count/2) packed nibbles]. 8x compression at
    /// 4 bits/weight; one scale per contiguous group bounds the int4 RMSE. Feeds the no-upcast
    /// int4 weight-only GEMM directly. Lossier than int8; explicit opt-in only —
    /// <c>StreamingStoreDtype.Auto</c> never picks int4.</summary>
    internal const byte Int4 = 4;
}
