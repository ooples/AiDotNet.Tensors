// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// The no-upcast resident form of a streaming int8 weight: the per-row-quantized weight kept
/// as int8 + per-row scales so it feeds the int8 weight-only GEMM directly (no dequant to
/// fp32). Attached to a <see cref="TensorBase{T}"/> when an int8-stored weight is materialized
/// for inference; the engine's matmul fast path reads it instead of the fp32 backing array.
/// If any non-matmul path needs an fp32 view, the tensor lazily dequantizes from this.
/// </summary>
/// <remarks>
/// Layout matches <c>SgemmWithInt8RowScaledCachedB</c>: <see cref="Data"/> is the weight in
/// row-major [<see cref="Rows"/>, <see cref="K"/>] order (output channels × input features),
/// <see cref="Scales"/> has one fp32 scale per row, and dequant is <c>Data[r*K+j] * Scales[r]</c>.
/// </remarks>
internal sealed class StreamingInt8Weight
{
    public StreamingInt8Weight(sbyte[] data, float[] scales, int rows, int k, bool transposedFromLogical)
    {
        Data = data;
        Scales = scales;
        Rows = rows;
        K = k;
        TransposedFromLogical = transposedFromLogical;
    }

    /// <summary>Signed int8 weights in the kernel's [Rows, K] (= [out, in]) row-major layout —
    /// i.e. Wᵀ of a Linear weight whose logical shape is [in, out].</summary>
    public sbyte[] Data { get; }

    /// <summary>Per-row (per-output-channel) fp32 scales, length <see cref="Rows"/>.</summary>
    public float[] Scales { get; }

    /// <summary>Rows of <see cref="Data"/> = output channels (the matmul's N).</summary>
    public int Rows { get; }

    /// <summary>Columns of <see cref="Data"/> = input features (the matmul's K).</summary>
    public int K { get; }

    /// <summary>
    /// True when <see cref="Data"/> is the TRANSPOSE of the tensor's logical layout: a Linear
    /// weight is logically [in, out] = [K, Rows] but stored here as [out, in] = [Rows, K] so it
    /// feeds the int8 GEMM (c = x·Wᵀᵀ = x·W) directly. The fp32 lazy fallback must transpose
    /// back to the logical [K, Rows]. False for a 1-D / per-leading store (no transpose).
    /// </summary>
    public bool TransposedFromLogical { get; }
}
