#pragma warning disable CS0618 // SimdGemm.Sgemm (no-trans shim) is [Obsolete] — pending migration to BlasManaged.Gemm<T> in a later K task.
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Fused multi-layer GEMM: executes GEMM1 → Activation → GEMM2 in a single tiled pass,
/// keeping inter-layer data in L1 cache instead of writing to DRAM between operations.
///
/// For a 2-layer MLP: input[M,K] @ W1[K,H] → activation → @ W2[H,N] = output[M,N]
///
/// Traditional: 3 separate kernel dispatches with 2 full-array store/load round-trips.
/// Fused: 1 kernel dispatch. Inter-layer tile [Mr,H] stays in L1 (H≤512 → 12KB ≤ 32KB L1).
///
/// Uses SimdGemm's BLIS tiling (Mr=6, Nr=16) and packing infrastructure.
/// </summary>
internal static class FusedMultiLayerGemm
{
    // Same tiling as SimdGemm for consistency
    private const int Mr = 6;
    private const int Nr = 16;
    private const int Kc = 512;

    /// <summary>
    /// Computes output[M,N] = W2[H,N]^T @ activation(input[M,K] @ W1[K,H])
    /// in a single fused pass. Also stores the activated intermediate for backward.
    /// </summary>
    /// <param name="input">Input tensor data [M,K] row-major</param>
    /// <param name="w1">First weight matrix [K,H] row-major</param>
    /// <param name="w2">Second weight matrix [H,N] row-major</param>
    /// <param name="output">Output buffer [M,N] row-major (will be cleared)</param>
    /// <param name="activated">Buffer to store activated intermediate [M,H] for backward</param>
    /// <param name="m">Batch size (rows of input)</param>
    /// <param name="k">Input features</param>
    /// <param name="h">Hidden features</param>
    /// <param name="n">Output features</param>
    /// <param name="applyActivation">Activation function to apply element-wise</param>
    /// <summary>
    /// Issue #338 Phase G.2 — fused MatMul → GELU → MatMul with SIMD GELU
    /// on the [Mr, H] tile (matches SimdKernels.GELUUnsafe's tanh-approx
    /// kernel for bit-equivalence with the unfused engine.GELU path).
    /// Saves the GEMM1 output (pre-activation) into <paramref name="preActivation"/>
    /// before applying GELU — backward needs it for derivative computation.
    /// </summary>
    [MethodImpl(Hot)]
    internal static unsafe void FusedGemmGeluGemm(
        float[] input, float[] w1, float[] w2,
        float[] output, float[] activated, float[] preActivation,
        int m, int k, int h, int n)
    {
        int gtileBytes = Mr * h * sizeof(float);
        bool ghasFma = false;
#if NET5_0_OR_GREATER
        ghasFma = Fma.IsSupported;
#endif
        if (gtileBytes > 28_000 || !ghasFma)
        {
            // Fallback path: separate GEMM + SIMD GELU + GEMM. No L1 win,
            // but at least vectorized GELU.
            Array.Clear(activated, 0, m * h);
            BlasProvider.Gemm(m, h, k, input, 0, k, w1, 0, h, activated, 0, h);
            Buffer.BlockCopy(activated, 0, preActivation, 0, m * h * sizeof(float));
            fixed (float* pAct = activated)
                SimdKernels.GELUUnsafe(pAct, pAct, m * h);
            Array.Clear(output, 0, m * n);
            BlasProvider.Gemm(m, n, h, activated, 0, h, w2, 0, n, output, 0, n);
            return;
        }

        Array.Clear(output, 0, m * n);
        var gtile = ArrayPool<float>.Shared.Rent(Mr * h);

        try
        {
            for (int ic = 0; ic < m; ic += Mr)
            {
                int mr = Math.Min(Mr, m - ic);

                // GEMM1 strip → tile[mr, h]
                Array.Clear(gtile, 0, mr * h);
                BlasProvider.Gemm(mr, h, k, input, ic * k, k, w1, 0, h, gtile, 0, h);

                // Save pre-activation rows to preActivation[ic..ic+mr, :]
                Buffer.BlockCopy(gtile, 0, preActivation, ic * h * sizeof(float), mr * h * sizeof(float));

                // SIMD GELU on the tile in-place. Tile is contiguous mr*h
                // floats — matches the contract of SimdKernels.GELUUnsafe
                // (input pointer, output pointer, length).
                fixed (float* pt = gtile)
                    SimdKernels.GELUUnsafe(pt, pt, mr * h);

                // Save post-activation rows to activated[ic..ic+mr, :]
                Buffer.BlockCopy(gtile, 0, activated, ic * h * sizeof(float), mr * h * sizeof(float));

                // GEMM2 strip: output[ic..ic+mr] += tile @ W2
                BlasProvider.Gemm(mr, n, h, gtile, 0, h, w2, 0, n, output, ic * n, n);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gtile);
        }
    }

    [MethodImpl(Hot)]
    internal static unsafe void FusedGemmActivationGemm(
        float[] input, float[] w1, float[] w2,
        float[] output, float[] activated,
        int m, int k, int h, int n,
        Func<float, float> applyActivation,
        float[]? preActivation = null)
    {
        // Issue #338 Phase G.2: optional preActivation buffer captures the
        // GEMM1 output BEFORE activation is applied. Required for GELU
        // backward (whose derivative is not recoverable from post-activation
        // values), unused for ReLU (where pre and post give the same
        // derivative). Passing null restores the original ReLU-only
        // behaviour — existing callers in tests + DataflowFusionPass +
        // FusedGemmActivationGemmJit are unaffected.

        // Check if hidden dim fits in L1 for the fused path
        // Mr rows * H columns * 4 bytes must fit in L1 (32KB)
        int tileBytes = Mr * h * sizeof(float);
        bool hasFma = false;
#if NET5_0_OR_GREATER
        hasFma = Fma.IsSupported;
#endif
        if (tileBytes > 28_000 || !hasFma || m >= 64)
        {
            // Separate full-matrix GEMM + activation + GEMM via FullGemm (managed-first).
            //
            // This BEATS the Mr-row strip loop below for any real batch: the strip loop
            // issues m/Mr (e.g. 85 for m=512, Mr=6) tiny per-strip GEMM calls per layer
            // whose per-call overhead dominates — measured 25 GF/s for the forward vs the
            // backward's full-matrix GEMMs at 150+ GF/s on the same shape
            // (CompiledTrainingPlanGemmPerfBench: forward 42ms -> ~1ms). FullGemm picks
            // the managed M-axis-parallel BlasManaged in its sweet spot (matches native at
            // square shapes) or managed SimdGemm for tall-skinny (6x native), so this path
            // needs no native BLAS. The L1-resident strip fusion below is kept only for
            // tiny batches (m < 64) where parallel dispatch can't amortise.
            FallbackSeparate(input, w1, w2, output, activated, m, k, h, n, applyActivation, preActivation);
            return;
        }

        // Clear output
        Array.Clear(output, 0, m * n);

        // Allocate inter-layer tile buffer [Mr, H] — stays in L1
        var tile = ArrayPool<float>.Shared.Rent(Mr * h);

        try
        {
            // Process Mr rows at a time
            for (int ic = 0; ic < m; ic += Mr)
            {
                int mr = Math.Min(Mr, m - ic);

                // Step 1: Compute GEMM1 tile = input[ic:ic+mr, :] @ W1[:, :] → tile[mr, H]
                // Use BLAS for maximum performance on the first GEMM
                Array.Clear(tile, 0, mr * h);
                BlasProvider.Gemm(mr, h, k, input, ic * k, k, w1, 0, h, tile, 0, h);

                // Step 2: Apply activation in-place on tile AND save to activated buffer.
                // When preActivation is provided (Phase G.2 for GELU), capture the
                // pre-activation GEMM1 output into preActivation BEFORE applying
                // the activation — backward needs it for derivative computation.
                for (int row = 0; row < mr; row++)
                {
                    int tileRowOff = row * h;
                    int actRowOff = (ic + row) * h;
                    if (preActivation is not null)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            float pre = tile[tileRowOff + j];
                            preActivation[actRowOff + j] = pre;
                            float val = applyActivation(pre);
                            tile[tileRowOff + j] = val;
                            activated[actRowOff + j] = val;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < h; j++)
                        {
                            float val = applyActivation(tile[tileRowOff + j]);
                            tile[tileRowOff + j] = val;
                            activated[actRowOff + j] = val;
                        }
                    }
                }

                // Step 3: Compute GEMM2: output[ic:ic+mr, :] += tile[mr, H] @ W2[H, N]
                // Use BLAS with beta=0 (output was pre-cleared) for the strip
                BlasProvider.Gemm(mr, n, h, tile, 0, h, w2, 0, n, output, ic * n, n);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tile);
        }
    }

    /// <summary>
    /// Computes tile[mr,H] = input[ic:ic+mr, :K] @ W1[K,H] using tiled GEMM.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | Hot)]
    private static unsafe void ComputeGemm1Tile(
        float[] input, float[] w1, float[] tile,
        int ic, int mr, int k, int h)
    {
        // Simple but correct: for each row in the strip, dot with W1 columns
        // Uses FMA for the inner loop when possible
        fixed (float* pIn = input, pW1 = w1, pTile = tile)
        {
            for (int row = 0; row < mr; row++)
            {
                float* inRow = pIn + (ic + row) * k;
                float* tileRow = pTile + row * h;

                // Tile the K dimension for cache efficiency
                for (int pc = 0; pc < k; pc += Kc)
                {
                    int kc = Math.Min(Kc, k - pc);

                    // For each output column of the hidden layer
                    int j = 0;
#if NET5_0_OR_GREATER
                    if (Fma.IsSupported)
                    {
                        // Vectorized: process 8 output columns at a time
                        for (; j + 7 < h; j += 8)
                        {
                            var acc = Avx.LoadVector256(tileRow + j);
                            for (int p = 0; p < kc; p++)
                            {
                                var a = Vector256.Create(inRow[pc + p]);
                                var b = Avx.LoadVector256(pW1 + (pc + p) * h + j);
                                acc = Fma.MultiplyAdd(a, b, acc);
                            }
                            Avx.Store(tileRow + j, acc);
                        }
                    }
#endif
                    // Scalar tail
                    for (; j < h; j++)
                    {
                        float sum = tileRow[j];
                        for (int p = 0; p < kc; p++)
                            sum += inRow[pc + p] * pW1[(pc + p) * h + j];
                        tileRow[j] = sum;
                    }
                }
            }
        }
    }

    /// <summary>Fallback: separate GEMM + activation + GEMM when H is too large for L1.</summary>
    [MethodImpl(Hot)]
    private static void FallbackSeparate(
        float[] input, float[] w1, float[] w2,
        float[] output, float[] activated,
        int m, int k, int h, int n,
        Func<float, float> applyActivation,
        float[]? preActivation = null)
    {
        // GEMM1: hidden = input @ W1 (reuse activated buffer as scratch — same size [m*h])
        Array.Clear(activated, 0, m * h);
        FullGemm(input, k, w1, h, activated, h, m, h, k);

        // Capture pre-activation if requested (Phase G.2 GELU support)
        if (preActivation is not null)
            Buffer.BlockCopy(activated, 0, preActivation, 0, m * h * sizeof(float));

        // Activation in-place on activated buffer
        for (int i = 0; i < m * h; i++)
            activated[i] = applyActivation(activated[i]);

        // GEMM2: output = activated @ W2 (clear first since the managed fallback accumulates)
        Array.Clear(output, 0, m * n);
        FullGemm(activated, h, w2, n, output, n, m, n, h);
    }

    /// <summary>
    /// Full-matrix GEMM C[mm,nn] = A[mm,kk] @ B[kk,nn] (no transpose, row-major).
    /// <para>
    /// Managed-first by design (the library is replacing native BLAS with managed
    /// kernels that match or beat it). Head-to-head (raw GEMM, Ryzen, GF/s):
    /// </para>
    /// <list type="bullet">
    /// <item>square K=N=256: <see cref="BlasManaged"/> 518-551 ≈ native 526-580,
    /// both ~30x single-threaded SimdGemm — so the M-axis-parallel BlasManaged
    /// ThinMDirect path (64&lt;=m&lt;=1024, n&lt;=512, k&lt;=1024) replaces native at
    /// parity here;</item>
    /// <item>tall-skinny m=2048,k=n=128: SimdGemm 600 >> native 96 >> BlasManaged 5.8
    /// — SimdGemm (also managed) beats native 6x where BlasManaged's packed path
    /// collapses.</item>
    /// </list>
    /// <para>
    /// Every other shape (e.g. large-k moderate-m) goes through <see cref="BlasProvider.Gemm"/>.
    /// </para>
    /// </summary>
    private static void FullGemm(
        float[] a, int lda, float[] b, int ldb, float[] c, int ldc, int mm, int nn, int kk)
    {
        // Managed parallel path — replaces native at parity for the square K=N
        // batch shapes that dominate MLP/FFN training.
        if (mm >= 64 && mm <= 1024 && nn <= 512 && kk <= 1024)
        {
            BlasManaged.BlasManaged.Gemm<float>(
                a.AsSpan(0, mm * kk), lda, false, b.AsSpan(0, kk * nn), ldb, false,
                c.AsSpan(0, mm * nn), ldc, mm, nn, kk,
                new BlasManaged.BlasOptions<float> { PackingMode = BlasManaged.PackingMode.DisableAutotune });
            return;
        }

        // Tall-skinny large-m / small-K·N — SimdGemm (managed) beats native here.
        // Use the 10-arg lda/ldb kernel (the 3-arg no-trans shim is [Obsolete] AND a
        // far slower path — ~5 GF/s vs this one's 600 at [2048,128]x[128,128]).
        if (mm > 1024 && nn <= 256 && kk <= 256)
        {
            SimdGemm.Sgemm(a.AsSpan(0, mm * kk), lda, false, b.AsSpan(0, kk * nn), ldb, false,
                           c.AsSpan(0, mm * nn), mm, kk, nn);
            return;
        }

        // Long-tail shapes.
        BlasProvider.Gemm(mm, nn, kk, a, 0, lda, b, 0, ldb, c, 0, ldc);
    }
}
