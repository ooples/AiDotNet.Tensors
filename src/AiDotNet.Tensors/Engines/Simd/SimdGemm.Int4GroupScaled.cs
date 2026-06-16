// Copyright (c) AiDotNet. All rights reserved.
//
// Weight-only int4 GROUP-quant GEMM (Phase A / #1622, L5a + L2 batch=1).
//
// Mirrors the int8 no-upcast path (SgemmWithInt8RowScaledCachedB) but for the 8x int4
// store: the weight stays as sign-extended 4-bit values + one fp32 scale per contiguous
// `groupSize` block (the layout StreamingStoreCodec produces), and the kernel dequantizes
// one output-feature row into a k-float scratch on the fly, accumulating in fp32. The whole
// weight is NEVER upcast to fp32 at once — only a single k-length row at a time — so an
// int4-RESIDENT model keeps its ~8x-smaller footprint through the matmul.
//
// The kernel is parallelized over N (output features), not M (batch). That is the key for the
// foundation-model latency case: at batch=1 (m=1) an M-parallel GEMM has a single row and
// leaves every other core idle, but N is huge (vocab / FFN width) so N-partitioning saturates
// the machine. Correct for any m.

using System;
using System.Numerics;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>
    /// Weight-only int4 group-quant GEMM: <c>C[m,n] = A[m,k] · Wᵀ</c>, where the weight
    /// <c>W[n,k]</c> (row-major: row j = output feature j) is supplied as sign-extended int4
    /// values (<paramref name="wData"/>, one <see cref="sbyte"/> per element in <c>[-7,7]</c>)
    /// plus per-group fp32 scales (<paramref name="groupScales"/>) over the FLAT weight array —
    /// group of a flat index is <c>flatIndex / groupSize</c>, matching
    /// <see cref="LinearAlgebra.StreamingStoreCodec"/>'s encoder.
    /// </summary>
    /// <param name="a">Activations <c>A[m,k]</c> row-major (lda = k).</param>
    /// <param name="wData">Sign-extended int4 weights for <c>W[n,k]</c> row-major (length n·k).</param>
    /// <param name="groupScales">One fp32 scale per <paramref name="groupSize"/>-element group of the flat weight.</param>
    /// <param name="groupSize">Quantization group size (e.g. 128).</param>
    /// <param name="c">Output <c>C[m,n]</c> row-major (ldc = n); fully overwritten.</param>
    internal static void SgemmWithInt4GroupScaled(
        float[] a,
        sbyte[] wData, float[] groupScales, int groupSize,
        float[] c,
        int m, int k, int n)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (wData is null) throw new ArgumentNullException(nameof(wData));
        if (groupScales is null) throw new ArgumentNullException(nameof(groupScales));
        if (c is null) throw new ArgumentNullException(nameof(c));
        if (m <= 0 || k <= 0 || n <= 0) return;
        if (groupSize <= 0) throw new ArgumentOutOfRangeException(nameof(groupSize));
        long need = (long)n * k;
        if (wData.Length < need)
            throw new ArgumentException($"int4 weight has {wData.Length} elements, need n*k = {need}.", nameof(wData));
        if ((long)m * k > a.Length)
            throw new ArgumentException($"activation buffer too small: need m*k = {(long)m * k}, got {a.Length}.", nameof(a));
        if ((long)m * n > c.Length)
            throw new ArgumentException($"output buffer too small: need m*n = {(long)m * n}, got {c.Length}.", nameof(c));

        // One output-feature row j: dequant W[j,:] into wf[k], then dot it with every A row.
        void ComputeRange(int j0, int j1, float[] wf)
        {
            for (int j = j0; j < j1; j++)
            {
                long baseFlat = (long)j * k;
                for (int p = 0; p < k; p++)
                {
                    long fi = baseFlat + p;
                    wf[p] = wData[fi] * groupScales[(int)(fi / groupSize)];
                }
                for (int i = 0; i < m; i++)
                    c[(long)i * n + j] = Dot(a, (long)i * k, wf, k);
            }
        }

        bool parallel = UseParallelGemm && (long)m * k * n >= ParallelWorkThreshold && n >= 2;
        if (!parallel)
        {
            ComputeRange(0, n, new float[k]);
            return;
        }

        int maxThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        int numChunks = Math.Min(n, Math.Max(1, maxThreads * 4)); // 4x oversubscribe for load balance
        int per = (n + numChunks - 1) / numChunks;
        CpuParallelSettings.LightweightParallel(numChunks, chunk =>
        {
            int j0 = chunk * per;
            if (j0 >= n) return;
            int j1 = Math.Min(n, j0 + per);
            ComputeRange(j0, j1, new float[k]); // per-chunk scratch (one alloc per task, not per row)
        });
    }

    // fp32 dot of a[aOff .. aOff+len) and w[0 .. len), SIMD-accelerated via System.Numerics.Vector.
    private static float Dot(float[] a, long aOff, float[] w, int len)
    {
        int vw = Vector<float>.Count;
        int p = 0;
        int aBase = checked((int)aOff);
        var acc = Vector<float>.Zero;
        for (; p + vw <= len; p += vw)
            acc += new Vector<float>(a, aBase + p) * new Vector<float>(w, p);
        float sum = Vector.Dot(acc, Vector<float>.One);
        for (; p < len; p++) sum += a[aBase + p] * w[p];
        return sum;
    }
}
