// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.Simd.Sparse;

/// <summary>
/// SIMD-specialised CSR × dense matmul. Inner column loop uses
/// <see cref="Vector{T}"/> for portable AVX-2 / AVX-512 vectorisation —
/// the FMA per non-zero touches <see cref="Vector{T}.Count"/> output
/// columns at once, which is the dominant cost in CSR · dense.
///
/// <para><b>Why this matters (#221 point #2):</b> PyTorch's CPU sparse
/// matmul is famously slow because it falls through to a scalar
/// triple-loop. We hit the AVX-2 path on net471 and AVX-512 on
/// hardware that supports <see cref="Vector{T}.Count"/> = 16, closing
/// most of the perf gap that drives users to densify "sparse-enough"
/// matrices.</para>
///
/// <para><b>Span ↔ Vector bridge:</b> on net10 you can construct a
/// <see cref="Vector{T}"/> directly from a span; on net471 you can't.
/// Both targets cleanly support <see cref="MemoryMarshal.Cast{TFrom,TTo}(Span{TFrom})"/>
/// to reinterpret a span as a span of vectors, so that's the portable
/// pattern used here.</para>
/// </summary>
internal static class CsrDenseSimd
{
    /// <summary>True when <see cref="Vector{T}"/> is hardware-accelerated
    /// for floats. Hot-path callers gate on this so non-SIMD targets
    /// fall through to the scalar reference path without overhead.</summary>
    public static bool IsHardwareAccelerated => Vector.IsHardwareAccelerated && Vector<float>.Count >= 4;

    /// <summary>
    /// CSR (<paramref name="rowPtr"/>, <paramref name="colIdx"/>,
    /// <paramref name="values"/>) × dense <paramref name="b"/> →
    /// <paramref name="output"/>. Output is row-major contiguous
    /// (row-stride = <paramref name="n"/>). Every output row is
    /// cleared on entry — caller doesn't need to zero ahead.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(
        int[] rowPtr,
        int[] colIdx,
        float[] values,
        float[] b,
        float[] output,
        int rows,
        int n)
    {
        int width = Vector<float>.Count;
        var bSpan = b.AsSpan();
        var outSpan = output.AsSpan();

        for (int r = 0; r < rows; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            int outRowOff = r * n;

            // Clear output row up-front so FMA accumulators add into zero.
            outSpan.Slice(outRowOff, n).Clear();
            if (re == rs) continue;

            for (int p = rs; p < re; p++)
            {
                float v = values[p];
                int bRowOff = colIdx[p] * n;
                int j = 0;

                if (IsHardwareAccelerated)
                {
                    var vScalar = new Vector<float>(v);
                    int vecCount = n / width;
                    var bRowVecs = MemoryMarshal.Cast<float, Vector<float>>(
                        bSpan.Slice(bRowOff, vecCount * width));
                    var outRowVecs = MemoryMarshal.Cast<float, Vector<float>>(
                        outSpan.Slice(outRowOff, vecCount * width));

                    int vi = 0;
                    // Unrolled body of 4 vectors per iter for FMA pipelining.
                    while (vi + 4 <= vecCount)
                    {
                        outRowVecs[vi + 0] += vScalar * bRowVecs[vi + 0];
                        outRowVecs[vi + 1] += vScalar * bRowVecs[vi + 1];
                        outRowVecs[vi + 2] += vScalar * bRowVecs[vi + 2];
                        outRowVecs[vi + 3] += vScalar * bRowVecs[vi + 3];
                        vi += 4;
                    }
                    while (vi < vecCount)
                    {
                        outRowVecs[vi] += vScalar * bRowVecs[vi];
                        vi++;
                    }
                    j = vecCount * width;
                }
                // Scalar tail — covers j..n when n isn't a multiple of width.
                for (; j < n; j++)
                    output[outRowOff + j] += v * b[bRowOff + j];
            }
        }
    }

    /// <summary>Double-precision companion. Same loop shape; smaller
    /// SIMD width (4 doubles in AVX-512, 2 in AVX-2).</summary>
    public static void MultiplyDouble(
        int[] rowPtr,
        int[] colIdx,
        double[] values,
        double[] b,
        double[] output,
        int rows,
        int n)
    {
        int width = Vector<double>.Count;
        var bSpan = b.AsSpan();
        var outSpan = output.AsSpan();

        for (int r = 0; r < rows; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            int outRowOff = r * n;
            outSpan.Slice(outRowOff, n).Clear();
            if (re == rs) continue;

            for (int p = rs; p < re; p++)
            {
                double v = values[p];
                int bRowOff = colIdx[p] * n;
                int j = 0;

                if (Vector.IsHardwareAccelerated && Vector<double>.Count >= 2)
                {
                    var vScalar = new Vector<double>(v);
                    int vecCount = n / width;
                    var bRowVecs = MemoryMarshal.Cast<double, Vector<double>>(
                        bSpan.Slice(bRowOff, vecCount * width));
                    var outRowVecs = MemoryMarshal.Cast<double, Vector<double>>(
                        outSpan.Slice(outRowOff, vecCount * width));
                    for (int vi = 0; vi < vecCount; vi++)
                        outRowVecs[vi] += vScalar * bRowVecs[vi];
                    j = vecCount * width;
                }
                for (; j < n; j++)
                    output[outRowOff + j] += v * b[bRowOff + j];
            }
        }
    }
}
