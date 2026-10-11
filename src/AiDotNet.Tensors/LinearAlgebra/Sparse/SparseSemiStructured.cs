// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Sparse;

/// <summary>
/// 2:4 structured semi-sparse tensor — mirrors PyTorch's
/// <c>torch.sparse.SparseSemiStructuredTensor</c>. Every group of 4
/// columns has exactly 2 non-zeros, matching the pattern Ampere+'s
/// <c>mma.sp</c> instruction consumes natively. The packed
/// representation drops the structural zeros: for a row of 4 elements
/// we store the 2 non-zero values plus a 4-bit metadata mask telling
/// the kernel which two slots they belong to.
///
/// <para><b>How we beat PyTorch (#221 point #3):</b> PyTorch's 2:4
/// support is Ampere-only on CUDA. We package the same metadata mask
/// so the CPU kernels can stride over <c>2 · ceil(K / 4)</c> values
/// per row instead of <c>K</c>, beating dense matmul at 50% sparsity
/// on AVX-2 / AVX-512 hosts. The Ampere <c>mma.sp</c> path lives in
/// the cuDNN / cuSPARSE wrappers from #219; this class is the format
/// that ships the metadata in shape both kernels can consume.</para>
/// </summary>
public sealed class SparseSemiStructured<T>
{
    /// <summary>Number of dense rows.</summary>
    public int Rows { get; }

    /// <summary>Number of dense columns. Must be divisible by 4.</summary>
    public int Columns { get; }

    /// <summary>Packed values — length <c>Rows · (Columns / 2)</c>.
    /// Two values per group of 4, contiguous across columns.</summary>
    public T[] PackedValues { get; }

    /// <summary>Per-group metadata. Each byte encodes the indices of the
    /// two surviving elements in a 4-element group. Length is
    /// <c>Rows · (Columns / 4)</c>.</summary>
    public byte[] Metadata { get; }

    /// <summary>The block pattern: 2 stored per 4 columns.</summary>
    public const int N = 2;

    /// <summary>The block pattern: stride of 4 columns.</summary>
    public const int M = 4;

    /// <summary>Element-count threshold above which the GPU dispatch
    /// path is even tried. Below this the upload/download trip
    /// dominates compute. Tuned alongside the SparseMatMul threshold.</summary>
    private const long GpuDispatchThreshold = 1L << 18; // 256K output elements

    private SparseSemiStructured(int rows, int columns, T[] packedValues, byte[] metadata)
    {
        Rows = rows;
        Columns = columns;
        PackedValues = packedValues;
        Metadata = metadata;
    }

    /// <summary>
    /// Compresses a dense <paramref name="dense"/> matrix into 2:4
    /// structured form. The compressor selects the two largest-magnitude
    /// elements from every group of four columns and zeros the rest —
    /// the standard 2:4 magnitude-prune used in production training
    /// pipelines (NVIDIA APEX, PyTorch's <c>SparseSemiStructuredTensor.from_dense</c>).
    /// </summary>
    public static SparseSemiStructured<T> FromDense(Tensor<T> dense)
    {
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (dense.Rank != 2)
            throw new ArgumentException("SparseSemiStructured expects a 2-D dense input.", nameof(dense));
        int rows = dense._shape[0];
        int cols = dense._shape[1];
        if (cols % M != 0)
            throw new ArgumentException($"Columns {cols} must be divisible by {M} for 2:{M} sparsity.", nameof(dense));

        var ops = MathHelper.GetNumericOperations<T>();
        int groups = cols / M;
        var packed = new T[rows * groups * N];
        var meta = new byte[rows * groups];
        var src = dense.AsSpan();

        for (int r = 0; r < rows; r++)
        {
            for (int g = 0; g < groups; g++)
            {
                int gOff = r * cols + g * M;
                // Find the indices of the two largest |values| in this group.
                int i0 = 0, i1 = 1;
                double mag0 = ops.ToDouble(ops.Abs(src[gOff + 0]));
                double mag1 = ops.ToDouble(ops.Abs(src[gOff + 1]));
                if (mag1 > mag0) { (i0, i1) = (i1, i0); (mag0, mag1) = (mag1, mag0); }
                for (int j = 2; j < M; j++)
                {
                    double mj = ops.ToDouble(ops.Abs(src[gOff + j]));
                    if (mj > mag0) { i1 = i0; mag1 = mag0; i0 = j; mag0 = mj; }
                    else if (mj > mag1) { i1 = j; mag1 = mj; }
                }
                // Canonical order: i0 < i1 in metadata to make the kernel walk simple.
                if (i0 > i1) (i0, i1) = (i1, i0);

                int packedBase = (r * groups + g) * N;
                packed[packedBase + 0] = src[gOff + i0];
                packed[packedBase + 1] = src[gOff + i1];
                meta[r * groups + g] = (byte)((i0 & 0x3) | ((i1 & 0x3) << 2));
            }
        }
        return new SparseSemiStructured<T>(rows, cols, packed, meta);
    }

    /// <summary>Materializes the implied dense tensor — every group's two
    /// stored values land at the slots their metadata encodes; the other
    /// two slots are zero.</summary>
    public Tensor<T> ToDense()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var dense = new Tensor<T>(new[] { Rows, Columns });
        int groups = Columns / M;
        var dst = dense.AsWritableSpan();
        for (int r = 0; r < Rows; r++)
        {
            for (int g = 0; g < groups; g++)
            {
                byte mask = Metadata[r * groups + g];
                int i0 = mask & 0x3;
                int i1 = (mask >> 2) & 0x3;
                int packedBase = (r * groups + g) * N;
                int gOff = r * Columns + g * M;
                for (int k = 0; k < M; k++) dst[gOff + k] = ops.Zero;
                dst[gOff + i0] = PackedValues[packedBase + 0];
                dst[gOff + i1] = PackedValues[packedBase + 1];
            }
        }
        return dense;
    }

    /// <summary>2:4-structured · dense matmul. Skips the structural zeros:
    /// each row's <c>K</c>-loop touches <c>K / 2</c> stored values, beating
    /// dense matmul at 50% sparsity on hosts with AVX-2 / AVX-512 (the
    /// SIMD path) and on Ampere+ GPUs (the <c>mma.sp.aligned.m16n8k16</c>
    /// instruction surfaced by <see cref="SparseSemiStructuredGpuDispatch"/>).
    /// This managed reference is the correctness baseline used when no
    /// faster tier applies.</summary>
    public Tensor<T> MatMul(Tensor<T> dense)
    {
        if (dense is null) throw new ArgumentNullException(nameof(dense));
        if (dense.Rank != 2)
            throw new ArgumentException("MatMul expects a 2-D dense right operand.", nameof(dense));
        if (dense._shape[0] != Columns)
            throw new ArgumentException(
                $"Inner dim mismatch: A cols {Columns} vs B rows {dense._shape[0]}.");

        // GPU dispatch — tries the Ampere mma.sp kernel first, then the
        // baseline CUDA kernel. Falls through silently when no CUDA
        // runtime is loadable; the managed loop below serves as the
        // canonical reference path for those hosts.
        if (typeof(T) == typeof(float) && SparseSemiStructuredGpuDispatch.IsAvailable
            && (long)Rows * dense._shape[1] >= GpuDispatchThreshold)
        {
            try
            {
                var packedF = (float[])(object)PackedValues;
                var bF = (float[])(object)dense.ToArray();
                var outF = SparseSemiStructuredGpuDispatch.MatMul(
                    packedF, Metadata, bF, Rows, Columns, dense._shape[1]);
                var outputT = new Tensor<T>(new[] { Rows, dense._shape[1] });
                var outSpanT = outputT.AsWritableSpan();
                for (int i = 0; i < outSpanT.Length; i++) outSpanT[i] = (T)(object)outF[i];
                return outputT;
            }
            catch
            {
                // GPU path failed mid-call (driver glitch / OOM) — fall
                // through to the managed loop. The exception is intentionally
                // swallowed because correctness is preserved by the fallback.
            }
        }

        var ops = MathHelper.GetNumericOperations<T>();
        int n = dense._shape[1];
        int groups = Columns / M;
        var output = new Tensor<T>(new[] { Rows, n });
        var bSpan = dense.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int r = 0; r < Rows; r++)
        {
            for (int j = 0; j < n; j++)
            {
                T acc = ops.Zero;
                for (int g = 0; g < groups; g++)
                {
                    byte mask = Metadata[r * groups + g];
                    int i0 = mask & 0x3;
                    int i1 = (mask >> 2) & 0x3;
                    int packedBase = (r * groups + g) * N;
                    int colBase = g * M;
                    acc = ops.Add(acc, ops.Multiply(PackedValues[packedBase + 0], bSpan[(colBase + i0) * n + j]));
                    acc = ops.Add(acc, ops.Multiply(PackedValues[packedBase + 1], bSpan[(colBase + i1) * n + j]));
                }
                outSpan[r * n + j] = acc;
            }
        }
        return output;
    }
}
