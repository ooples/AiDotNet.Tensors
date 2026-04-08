using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Flash Attention: tiled fused attention kernel using AVX2/FMA SIMD.
/// O = softmax(Q @ K^T / sqrt(d)) @ V without materializing the full N×N attention matrix.
///
/// Standard attention: O(N^2) memory, 4 separate kernels.
/// Flash Attention: O(N) memory, single tiled pass with online softmax.
///
/// SIMD optimization:
/// - Score tile: AVX2 FMA dot products (8 floats at a time)
/// - Online softmax: AVX2 FastExp256 + vectorized max/sum reduction
/// - V accumulation: AVX2 FMA multiply-add
///
/// Tile sizes tuned for L1 cache (32KB): Br=32 queries, Bc=32 keys.
/// Each tile is Br*Bc*4 = 4KB, fits comfortably in L1 with room for K/V tiles.
/// </summary>
internal static class FusedAttention
{
    private const int DefaultBr = 32;  // Query tile — tuned for L1
    private const int DefaultBc = 32;  // Key tile — tuned for L1

    // Thread-local work buffers to avoid per-call allocation in compiled plans
    [ThreadStatic] private static float[]? _tlRowMax;
    [ThreadStatic] private static float[]? _tlRowSum;
    [ThreadStatic] private static float[]? _tlScores;

    /// <summary>
    /// Flash Attention forward: O = softmax(Q @ K^T / sqrt(d)) @ V
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void FlashAttentionForward(
        float[] q, float[] k, float[] v, float[] output,
        int seqQ, int seqK, int headDim,
        float scale, bool isCausal = false)
    {
        int br = Math.Min(DefaultBr, seqQ);
        int bc = Math.Min(DefaultBc, seqK);

        // Reuse thread-local work buffers to avoid per-call allocation
        var rowMax = _tlRowMax;
        if (rowMax is null || rowMax.Length < br) _tlRowMax = rowMax = new float[br];
        var rowSum = _tlRowSum;
        if (rowSum is null || rowSum.Length < br) _tlRowSum = rowSum = new float[br];
        var localScores = _tlScores;
        if (localScores is null || localScores.Length < br * bc) _tlScores = localScores = new float[br * bc];

        Array.Clear(output, 0, seqQ * headDim);

        fixed (float* pQ = q, pK = k, pV = v, pO = output, pScores = localScores,
               pRowMax = rowMax, pRowSum = rowSum)
        {
            for (int qi = 0; qi < seqQ; qi += br)
            {
                int actualBr = Math.Min(br, seqQ - qi);

                for (int r = 0; r < actualBr; r++)
                {
                    pRowMax[r] = float.NegativeInfinity;
                    pRowSum[r] = 0f;
                }

                for (int kj = 0; kj < seqK; kj += bc)
                {
                    int actualBc = Math.Min(bc, seqK - kj);

                    // Step 1: Score tile — Q_tile @ K_tile^T with AVX2 FMA
                    ComputeScoreTileSimd(pQ, pK, pScores, qi, kj, actualBr, actualBc, headDim, scale);

                    if (isCausal)
                        ApplyCausalMask(pScores, qi, kj, actualBr, actualBc);

                    // Step 2: Online softmax + V accumulation with AVX2
                    OnlineSoftmaxAndAccumulate(
                        pScores, pV, pO, pRowMax, pRowSum,
                        qi, kj, actualBr, actualBc, headDim);
                }

                // Step 3: Normalize output by row sum
                NormalizeOutput(pO, pRowSum, qi, actualBr, headDim);
            }
        }
    }

    /// <summary>AVX2 FMA dot product for score tile computation.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ComputeScoreTileSimd(
        float* q, float* k, float* scores,
        int qi, int kj, int br, int bc, int headDim, float scale)
    {
        for (int r = 0; r < br; r++)
        {
            float* qRow = q + (qi + r) * headDim;
            for (int c = 0; c < bc; c++)
            {
                float* kRow = k + (kj + c) * headDim;
                float dot = DotProductSimd(qRow, kRow, headDim);
                scores[r * bc + c] = dot * scale;
            }
        }
    }

    /// <summary>AVX2 FMA vectorized dot product — 8 floats per cycle.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotProductSimd(float* a, float* b, int length)
    {
        float sum = 0f;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 32)
        {
            // 4x unrolled AVX2 FMA accumulation for maximum throughput
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;

            int simdLength = length & ~31;
            for (; i < simdLength; i += 32)
            {
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0);
                acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), acc1);
                acc2 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16), acc2);
                acc3 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24), acc3);
            }

            // Reduce 4 accumulators and horizontal sum via SimdKernels helper
            acc0 = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));
            sum = SimdKernels.HorizontalSum(acc0);
        }
        else if (Fma.IsSupported && length >= 8)
        {
            var acc = Vector256<float>.Zero;
            int simdLength = length & ~7;
            for (; i < simdLength; i += 8)
                acc = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc);

            sum = SimdKernels.HorizontalSum(acc);
        }
#endif

        // Scalar tail
        for (; i < length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    /// <summary>Online softmax with AVX2 exp + V accumulation via FMA.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void OnlineSoftmaxAndAccumulate(
        float* scores, float* v, float* output,
        float* rowMax, float* rowSum,
        int qi, int kj, int br, int bc, int headDim)
    {
        for (int r = 0; r < br; r++)
        {
            int globalRow = qi + r;
            float* scoreRow = scores + r * bc;
            float* oRow = output + globalRow * headDim;

            // Find local row max
            float localMax = float.NegativeInfinity;
            for (int cm = 0; cm < bc; cm++)
                if (scoreRow[cm] > localMax) localMax = scoreRow[cm];

            // Update global max and compute rescale factor
            float prevMax = rowMax[r];
            float newMax = Math.Max(prevMax, localMax);
            rowMax[r] = newMax;

            float rescale = prevMax == float.NegativeInfinity ? 0f : MathF.Exp(prevMax - newMax);

            // Rescale previous output with AVX2
            if (rescale > 0f && rescale < 1f)
            {
                RescaleRowSimd(oRow, rescale, headDim);
                rowSum[r] *= rescale;
            }

            // Vectorized exp computation for all columns at once
            float localSum = 0f;

            // Batch exp: compute exp(score[c] - max) for all c using AVX2 FastExp256
            int c = 0;
#if NET5_0_OR_GREATER
            if (Fma.IsSupported && bc >= 8)
            {
                var vNewMax = Vector256.Create(newMax);
                var vLocalSum = Vector256<float>.Zero;
                int simdBc = bc & ~7;
                // Pre-compute all exp values into scores (in-place)
                for (; c < simdBc; c += 8)
                {
                    var s = Avx.Subtract(Avx.LoadVector256(scoreRow + c), vNewMax);
                    var expS = SimdKernels.FastExp256(s);
                    Avx.Store(scoreRow + c, expS);
                    vLocalSum = Avx.Add(vLocalSum, expS);
                }
                localSum = SimdKernels.HorizontalSum(vLocalSum);
            }
#endif
            for (; c < bc; c++)
            {
                float expVal = MathF.Exp(scoreRow[c] - newMax);
                scoreRow[c] = expVal;
                localSum += expVal;
            }

            // Accumulate O += exp_weights @ V (each column has different weight)
            for (c = 0; c < bc; c++)
            {
                float expVal = scoreRow[c];
                float* vRow = v + (kj + c) * headDim;
                AccumulateRowSimd(oRow, vRow, expVal, headDim);
            }

            rowSum[r] += localSum;
        }
    }

    /// <summary>AVX2 row rescale: row[i] *= scale</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void RescaleRowSimd(float* row, float scale, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && length >= 8)
        {
            var vScale = Vector256.Create(scale);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
                Avx.Store(row + i, Avx.Multiply(Avx.LoadVector256(row + i), vScale));
        }
#endif
        for (; i < length; i++)
            row[i] *= scale;
    }

    /// <summary>AVX2 FMA row accumulate: dst[i] += weight * src[i]</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AccumulateRowSimd(float* dst, float* src, float weight, int length)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vWeight = Vector256.Create(weight);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
                Avx.Store(dst + i, Fma.MultiplyAdd(vWeight, Avx.LoadVector256(src + i), Avx.LoadVector256(dst + i)));
        }
#endif
        for (; i < length; i++)
            dst[i] += weight * src[i];
    }

    /// <summary>Normalize output rows by their softmax sum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void NormalizeOutput(float* output, float* rowSum, int qi, int br, int headDim)
    {
        for (int r = 0; r < br; r++)
        {
            float invSum = rowSum[r] > 0f ? 1f / rowSum[r] : 0f;
            float* oRow = output + (qi + r) * headDim;
            RescaleRowSimd(oRow, invSum, headDim);
        }
    }

    /// <summary>Apply causal mask: scores[r,c] = -inf where qi+r < kj+c</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ApplyCausalMask(float* scores, int qi, int kj, int br, int bc)
    {
        for (int r = 0; r < br; r++)
            for (int c = 0; c < bc; c++)
                if (qi + r < kj + c)
                    scores[r * bc + c] = float.NegativeInfinity;
    }

    /// <summary>
    /// Batched Flash Attention: processes multiple batch*head combinations in parallel.
    /// Q, K, V are [batchHeads, seq, headDim].
    /// </summary>
    internal static void BatchedFlashAttention(
        float[] q, float[] k, float[] v, float[] output,
        int batchHeads, int seqQ, int seqK, int headDim,
        float scale, bool isCausal = false)
    {
        int qStride = seqQ * headDim;
        int kStride = seqK * headDim;

        System.Threading.Tasks.Parallel.For(0, batchHeads, bh =>
        {
            int qOffset = bh * qStride;
            int kOffset = bh * kStride;

            var qSlice = new float[qStride];
            var kSlice = new float[kStride];
            var vSlice = new float[kStride];
            var oSlice = new float[qStride];

            Array.Copy(q, qOffset, qSlice, 0, qStride);
            Array.Copy(k, kOffset, kSlice, 0, kStride);
            Array.Copy(v, kOffset, vSlice, 0, kStride);

            FlashAttentionForward(qSlice, kSlice, vSlice, oSlice,
                seqQ, seqK, headDim, scale, isCausal);

            Array.Copy(oSlice, 0, output, qOffset, qStride);
        });
    }
}
