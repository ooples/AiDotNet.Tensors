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
    public StreamingInt8Weight(sbyte[] data, float[] scales, int rows, int k)
    {
        Data = data;
        Scales = scales;
        Rows = rows;
        K = k;
    }

    /// <summary>Signed int8 weights, row-major [Rows, K].</summary>
    public sbyte[] Data { get; }

    /// <summary>Per-row (per-output-channel) fp32 scales, length <see cref="Rows"/>.</summary>
    public float[] Scales { get; }

    /// <summary>Number of rows (output channels) = the weight's leading dim.</summary>
    public int Rows { get; }

    /// <summary>Columns per row (input features) = total elements / <see cref="Rows"/>.</summary>
    public int K { get; }
}
