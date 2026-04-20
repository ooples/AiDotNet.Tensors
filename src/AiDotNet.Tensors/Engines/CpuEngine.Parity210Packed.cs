using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Sub-byte packed-storage Gather / Scatter overloads for #210 item #2.
/// PyTorch requires dequant → gather → requant for quantized tensors. We
/// gather directly over the packed byte storage, preserving the compact
/// format end-to-end. Useful for KV-cache retrieval in quantized LLMs.
/// </summary>
/// <remarks>
/// <para>
/// The packed tensor is held as <see cref="Tensor{Byte}"/> — a raw byte
/// buffer that encodes <c>valuesPerByte</c> sub-byte values per byte:
///   <list type="bullet">
///     <item>int1 (BitNet) — 8 values per byte</item>
///     <item>int2            — 4 values per byte</item>
///     <item>int4 / NF4 / FP4 — 2 values per byte</item>
///   </list>
/// The gather / scatter ops treat each byte as an atomic unit, so the
/// gather axis must not straddle the packing boundary. For common
/// workloads (KV-cache gathered on sequence axis, packed on feature axis
/// with a multiple-of-<c>valuesPerByte</c> feature dim) this is exactly
/// the degree of freedom we need.
/// </para>
/// <para>
/// The "beat PyTorch" angle here is that the storage format stays
/// packed. PyTorch has no packed-storage gather — it forces dequant →
/// gather → requant, which blows up transient memory by 8× / 4× / 2×
/// depending on the width.
/// </para>
/// </remarks>
public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<byte> TensorGatherPacked(
        Tensor<byte> packed, Tensor<int> indices, int axis, int valuesPerByte)
    {
        if (packed == null) throw new ArgumentNullException(nameof(packed));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (valuesPerByte != 1 && valuesPerByte != 2 && valuesPerByte != 4 && valuesPerByte != 8)
            throw new ArgumentOutOfRangeException(nameof(valuesPerByte),
                "valuesPerByte must be 1, 2, 4 or 8");

        int rank = packed.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        // The gather axis cannot be the packing axis — per the module-level
        // contract. Callers packing on the last axis should gather on any
        // earlier axis, which is byte-aligned.
        if (axis == rank - 1 && valuesPerByte > 1)
            throw new ArgumentException(
                "Packed gather does not support axis == last-dim when valuesPerByte > 1 " +
                "(would cross the packing boundary). Pack on a different axis.");

        if (!packed.IsContiguous) packed = packed.Contiguous();
        var src = packed.AsSpan();
        var idx = indices.AsSpan();

        var outShape = (int[])packed._shape.Clone();
        int idxLen = indices.Length;
        outShape[axis] = idxLen;

        var result = new Tensor<byte>(outShape);
        var dst = result.AsWritableSpan();

        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= packed._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= packed._shape[k];
        int srcAxis = packed._shape[axis];

        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < idxLen; i++)
            {
                int target = idx[i];
                if (target < 0 || target >= srcAxis)
                    throw new IndexOutOfRangeException(
                        $"indices[{i}]={target} out of range for axis size {srcAxis}");
                int srcOffset = (outer * srcAxis + target) * innerSize;
                int dstOffset = (outer * idxLen + i) * innerSize;
                for (int inner = 0; inner < innerSize; inner++)
                    dst[dstOffset + inner] = src[srcOffset + inner];
            }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<byte> TensorScatterPacked(
        Tensor<byte> packed, Tensor<int> indices, Tensor<byte> source, int axis, int valuesPerByte)
    {
        if (packed == null) throw new ArgumentNullException(nameof(packed));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (valuesPerByte != 1 && valuesPerByte != 2 && valuesPerByte != 4 && valuesPerByte != 8)
            throw new ArgumentOutOfRangeException(nameof(valuesPerByte));

        int rank = packed.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (axis == rank - 1 && valuesPerByte > 1)
            throw new ArgumentException(
                "Packed scatter does not support axis == last-dim when valuesPerByte > 1.");

        if (!packed.IsContiguous) packed = packed.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<byte>)packed.Clone();
        var dst = result.AsWritableSpan();
        var srcData = source.AsSpan();
        var idx = indices.AsSpan();

        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= packed._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= packed._shape[k];
        int dstAxis = packed._shape[axis];
        int srcAxis = source._shape[axis];
        if (srcAxis != idx.Length)
            throw new ArgumentException(
                $"source.shape[{axis}]={srcAxis} must match indices.length={idx.Length}");

        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < idx.Length; i++)
            {
                int target = idx[i];
                if (target < 0 || target >= dstAxis)
                    throw new IndexOutOfRangeException(
                        $"indices[{i}]={target} out of range for axis size {dstAxis}");
                int dstOffset = (outer * dstAxis + target) * innerSize;
                int srcOffset = (outer * srcAxis + i) * innerSize;
                for (int inner = 0; inner < innerSize; inner++)
                    dst[dstOffset + inner] = srcData[srcOffset + inner];
            }
        return result;
    }
}
