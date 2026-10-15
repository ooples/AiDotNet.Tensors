#if NET5_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// #1307: the per-head attention of the fused multi-head attention node, on AVX2 + FMA. Profiling the training step
// against PyTorch showed the per-head loop spending its time on everything but the math: SgemmSequential clears C
// before each of its five products per head (ZeroMemory was ~10% of all managed samples), Q, K, V, O and dO were
// copied out of the packed buffers per head and dQ, dK, dV copied back, and the softmax ran scalar MathF.Exp. Here the
// direct kernel reads Q, V, K and dO in place through their row strides and stores straight into the concat and dQKV
// buffers, only Kᵀ and Vᵀ are gathered, and the row softmax, its recompute and its gradient are vectorized. Each head
// still runs serially in a fixed order, so the result does not depend on the thread count.
public partial class CpuEngine
{
    private static bool MhaDirectHeads => Avx2.IsSupported && Fma.IsSupported;

    private static unsafe void MhaForwardHeadsDirect(
        float[] qkv, float[] concat, float[] lse, float[] kTBuf, float[] pBuf, float[] rowsBuf, int chunk, int chunks,
        int batchHeads, int numHeads, int seq, int dHead, int dModel, float scale, Tensor<bool>? mask)
    {
        int d3 = 3 * dModel;
        fixed (float* pQkv = qkv, pCat = concat, kT = kTBuf, p = pBuf, rowScratch = rowsBuf)
        {
            for (int s = chunk; s < batchHeads; s += chunks)
            {
                int b = s / numHeads, h = s % numHeads;
                long rowBase = (long)b * seq;
                float* q = pQkv + rowBase * d3 + h * dHead, k = q + dModel, v = q + 2 * dModel;
                TransposeRows(k, d3, seq, dHead, kT, rowScratch);
                SimdGemm.SgemmDirectSerialStrided(q, d3, kT, seq, p, seq, seq, dHead, seq);
                if (mask is null) SoftmaxRowsVec(p, seq, scale, lse, s * seq);
                else SoftmaxRowsExact(pBuf, seq, scale, mask, b, h, lse, s * seq);
                SimdGemm.SgemmDirectSerialStrided(p, seq, v, d3, pCat + rowBase * dModel + h * dHead, dModel, seq, seq, dHead);
            }
        }
    }

    private static unsafe void MhaBackwardHeadsDirect(
        float[] qkv, float[] concat, float[] dConcat, float[] dQkv, float[] lse,
        float[] kTBuf, float[] vTBuf, float[] pBuf, float[] pTBuf, float[] dPBuf, float[] rowsBuf, int chunk, int chunks,
        int batchHeads, int numHeads, int seq, int dHead, int dModel, float scale, Tensor<bool>? mask)
    {
        int d3 = 3 * dModel, ss = seq * seq;
        fixed (float* pQkv = qkv, pCat = concat, pDCat = dConcat, pDQkv = dQkv)
        fixed (float* kT = kTBuf, vT = vTBuf, p = pBuf, pT = pTBuf, dP = dPBuf, rowScratch = rowsBuf)
        {
            for (int s = chunk; s < batchHeads; s += chunks)
            {
                int b = s / numHeads, h = s % numHeads, col = h * dHead;
                long rowBase = (long)b * seq;
                float* q = pQkv + rowBase * d3 + col, k = q + dModel, v = q + 2 * dModel;
                float* o = pCat + rowBase * dModel + col, dO = pDCat + rowBase * dModel + col;
                float* dq = pDQkv + rowBase * d3 + col, dk = dq + dModel, dv = dq + 2 * dModel;

                TransposeRows(k, d3, seq, dHead, kT, rowScratch);
                TransposeRows(v, d3, seq, dHead, vT, rowScratch);
                SimdGemm.SgemmDirectSerialStrided(q, d3, kT, seq, p, seq, seq, dHead, seq);
                if (mask is null) RecomputeRowsVec(p, seq, scale, lse, s * seq);
                else RecomputeProbabilities(pBuf, seq, scale, mask, b, h, lse, s * seq);

                // dV = Pᵀ·dO
                SimdTranspose.Transpose(new ReadOnlySpan<float>(p, ss), new Span<float>(pT, ss), seq, seq);
                SimdGemm.SgemmDirectSerialStrided(pT, seq, dO, dModel, dv, d3, seq, seq, dHead);

                // dS = P ⊙ (dO·Vᵀ - rowsum(dO ⊙ O)) · scale
                SimdGemm.SgemmDirectSerialStrided(dO, dModel, vT, seq, dP, seq, seq, dHead, seq);
                for (int i = 0; i < seq; i++)
                {
                    float di = Dot(dO + (long)i * dModel, o + (long)i * dModel, dHead);
                    SoftmaxGradRow(p + i * seq, dP + i * seq, seq, di, scale);
                }

                // dQ = dS·K, dK = dSᵀ·Q
                SimdGemm.SgemmDirectSerialStrided(dP, seq, k, d3, dq, d3, seq, seq, dHead);
                SimdTranspose.Transpose(new ReadOnlySpan<float>(dP, ss), new Span<float>(pT, ss), seq, seq);
                SimdGemm.SgemmDirectSerialStrided(pT, seq, q, d3, dk, d3, seq, seq, dHead);
            }
        }
    }

    // dst [cols x rows] = src [rows x cols] read with row stride ld: each row is copied whole into the contiguous scratch
    // (one cache line per row at dHead = 32) and the scratch transposed in 8x8 SIMD tiles. The column-strided scalar
    // gather it replaces took ~17 us per head for Kᵀ and Vᵀ at 128 x 32. The transpose stays serial below its 64K-element
    // parallel threshold, as it must inside a parallel task.
    private static unsafe void TransposeRows(float* src, int ld, int rows, int cols, float* dst, float* scratch)
    {
        long rowBytes = (long)cols * sizeof(float);
        for (int r = 0; r < rows; r++)
            Buffer.MemoryCopy(src + (long)r * ld, scratch + (long)r * cols, rowBytes, rowBytes);
        SimdTranspose.Transpose(new ReadOnlySpan<float>(scratch, rows * cols), new Span<float>(dst, rows * cols), rows, cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float Dot(float* a, float* b, int n)
    {
        var acc = Vector256<float>.Zero;
        int i = 0;
        for (; i + 8 <= n; i += 8) acc = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc);
        float sum = HorizontalSum(acc);
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSum(Vector256<float> v)
    {
        var s = Sse.Add(v.GetLower(), v.GetUpper());
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        s = Sse.AddScalar(s, Sse.Shuffle(s, s, 0x55));
        return s.ToScalar();
    }

    // Row-wise exact softmax of scale·scores in place, recording each row's log-sum-exp.
    private static unsafe void SoftmaxRowsVec(float* scores, int seq, float scale, float[] lse, int lseOffset)
    {
        var vScale = Vector256.Create(scale);
        for (int i = 0; i < seq; i++)
        {
            float* row = scores + i * seq;
            int j = 0;
            var vMax = Vector256.Create(float.NegativeInfinity);
            for (; j + 8 <= seq; j += 8) vMax = Avx.Max(vMax, Avx.Multiply(Avx.LoadVector256(row + j), vScale));
            float max = float.NegativeInfinity;
            for (int t = 0; t < 8; t++) max = Math.Max(max, vMax.GetElement(t));
            for (; j < seq; j++) max = Math.Max(max, row[j] * scale);

            var vNegMax = Vector256.Create(-max);
            var vSum = Vector256<float>.Zero;
            j = 0;
            for (; j + 8 <= seq; j += 8)
            {
                var e = SimdKernels.FastExp256(Fma.MultiplyAdd(Avx.LoadVector256(row + j), vScale, vNegMax));
                Avx.Store(row + j, e);
                vSum = Avx.Add(vSum, e);
            }
            float sum = HorizontalSum(vSum);
            for (; j < seq; j++) { float e = MathF.Exp(row[j] * scale - max); row[j] = e; sum += e; }

            var vInv = Vector256.Create(1f / sum);
            j = 0;
            for (; j + 8 <= seq; j += 8) Avx.Store(row + j, Avx.Multiply(Avx.LoadVector256(row + j), vInv));
            for (; j < seq; j++) row[j] *= 1f / sum;
            lse[lseOffset + i] = max + MathF.Log(sum);
        }
    }

    // P = exp(scale·scores - lse) in place, from the forward's log-sum-exp.
    private static unsafe void RecomputeRowsVec(float* scores, int seq, float scale, float[] lse, int lseOffset)
    {
        var vScale = Vector256.Create(scale);
        for (int i = 0; i < seq; i++)
        {
            float* row = scores + i * seq;
            float l = lse[lseOffset + i];
            var vNegL = Vector256.Create(-l);
            int j = 0;
            for (; j + 8 <= seq; j += 8)
                Avx.Store(row + j, SimdKernels.FastExp256(Fma.MultiplyAdd(Avx.LoadVector256(row + j), vScale, vNegL)));
            for (; j < seq; j++) row[j] = MathF.Exp(row[j] * scale - l);
        }
    }

    // dS = P ⊙ (dP - di) · scale, written over dP.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SoftmaxGradRow(float* p, float* dP, int seq, float di, float scale)
    {
        var vDi = Vector256.Create(di);
        var vScale = Vector256.Create(scale);
        int j = 0;
        for (; j + 8 <= seq; j += 8)
            Avx.Store(dP + j, Avx.Multiply(Avx.Multiply(Avx.LoadVector256(p + j), Avx.Subtract(Avx.LoadVector256(dP + j), vDi)), vScale));
        for (; j < seq; j++) dP[j] = p[j] * (dP[j] - di) * scale;
    }
}
#endif