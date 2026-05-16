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
            if (!BlasProvider.TryGemm(m, h, k, input, 0, k, w1, 0, h, activated, 0, h))
                SimdGemm.Sgemm(input.AsSpan(0, m * k), w1.AsSpan(0, k * h), activated.AsSpan(0, m * h), m, k, h);
            Buffer.BlockCopy(activated, 0, preActivation, 0, m * h * sizeof(float));
            fixed (float* pAct = activated)
                SimdKernels.GELUUnsafe(pAct, pAct, m * h);
            Array.Clear(output, 0, m * n);
            if (!BlasProvider.TryGemm(m, n, h, activated, 0, h, w2, 0, n, output, 0, n))
                SimdGemm.Sgemm(activated.AsSpan(0, m * h), w2.AsSpan(0, h * n), output.AsSpan(0, m * n), m, h, n);
            return;
        }

        Array.Clear(output, 0, m * n);
        var gtile = ArrayPool<float>.Shared.Rent(Mr * h);
        int gnrBlocks = (n + Nr - 1) / Nr;
        float[] gpackedW2 = Array.Empty<float>();
        bool gw2Packed = false;

        try
        {
            for (int ic = 0; ic < m; ic += Mr)
            {
                int mr = Math.Min(Mr, m - ic);

                // GEMM1 strip → tile[mr, h]
                Array.Clear(gtile, 0, mr * h);
                if (!BlasProvider.TryGemm(mr, h, k, input, ic * k, k, w1, 0, h, gtile, 0, h))
                    SimdGemm.Sgemm(input.AsSpan(ic * k, mr * k), w1.AsSpan(0, k * h), gtile.AsSpan(0, mr * h), mr, k, h);

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
                if (!BlasProvider.TryGemm(mr, n, h, gtile, 0, h, w2, 0, n, output, ic * n, n))
                {
                    if (!gw2Packed)
                    {
                        gpackedW2 = ArrayPool<float>.Shared.Rent(h * ((gnrBlocks * Nr) + Nr));
                        PackBRowMajor(w2, gpackedW2, h, n);
                        gw2Packed = true;
                    }
                    ComputeGemm2FromTile(gtile, gpackedW2, output, ic, mr, h, n);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gtile);
            if (gw2Packed)
                ArrayPool<float>.Shared.Return(gpackedW2);
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
        if (tileBytes > 28_000 || !hasFma)
        {
            // Fallback: separate GEMM + activation + GEMM
            FallbackSeparate(input, w1, w2, output, activated, m, k, h, n, applyActivation, preActivation);
            return;
        }

        // Clear output
        Array.Clear(output, 0, m * n);

        // Allocate inter-layer tile buffer [Mr, H] — stays in L1
        var tile = ArrayPool<float>.Shared.Rent(Mr * h);

        // Defer W2 packing until we know BLAS isn't available (avoids wasted work)
        int nrBlocks = (n + Nr - 1) / Nr;
        float[] packedW2 = Array.Empty<float>();
        bool w2Packed = false;

        try
        {
            // Process Mr rows at a time
            for (int ic = 0; ic < m; ic += Mr)
            {
                int mr = Math.Min(Mr, m - ic);

                // Step 1: Compute GEMM1 tile = input[ic:ic+mr, :] @ W1[:, :] → tile[mr, H]
                // Use BLAS for maximum performance on the first GEMM
                Array.Clear(tile, 0, mr * h);
                if (!BlasProvider.TryGemm(mr, h, k, input, ic * k, k, w1, 0, h, tile, 0, h))
                    SimdGemm.Sgemm(input.AsSpan(ic * k, mr * k), w1.AsSpan(0, k * h), tile.AsSpan(0, mr * h), mr, k, h);

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
                if (!BlasProvider.TryGemm(mr, n, h, tile, 0, h, w2, 0, n, output, ic * n, n))
                {
                    // Pack W2 on first BLAS miss (deferred to avoid wasted work when BLAS is available)
                    if (!w2Packed)
                    {
                        packedW2 = ArrayPool<float>.Shared.Rent(h * ((nrBlocks * Nr) + Nr));
                        PackBRowMajor(w2, packedW2, h, n);
                        w2Packed = true;
                    }
                    ComputeGemm2FromTile(tile, packedW2, output, ic, mr, h, n);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tile);
            if (w2Packed)
                ArrayPool<float>.Shared.Return(packedW2);
        }
    }

    /// <summary>
    /// Computes tile[mr,H] = input[ic:ic+mr, :K] @ W1[K,H] using tiled GEMM.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

    /// <summary>
    /// Computes output[ic:ic+mr, :N] += tile[mr,H] @ W2[H,N] using packed B.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ComputeGemm2FromTile(
        float[] tile, float[] packedW2, float[] output,
        int ic, int mr, int h, int n)
    {
        fixed (float* pTile = tile, pW2 = packedW2, pOut = output)
        {
            int nrBlocks = (n + Nr - 1) / Nr;

            for (int jr = 0; jr < nrBlocks; jr++)
            {
                int jLocal = jr * Nr;
                int nc = Math.Min(Nr, n - jLocal);
                int bOff = jr * h * Nr;

                for (int row = 0; row < mr; row++)
                {
                    float* tileRow = pTile + row * h;
                    float* outRow = pOut + (ic + row) * n + jLocal;

#if NET5_0_OR_GREATER
                    if (Fma.IsSupported && nc == Nr)
                    {
                        // Full Nr=16 tile: 2 Vector256 accumulators
                        var acc0 = Avx.LoadVector256(outRow);
                        var acc1 = Avx.LoadVector256(outRow + 8);

                        for (int p = 0; p < h; p++)
                        {
                            var a = Vector256.Create(tileRow[p]);
                            var b0 = Avx.LoadVector256(pW2 + bOff + p * Nr);
                            var b1 = Avx.LoadVector256(pW2 + bOff + p * Nr + 8);
                            acc0 = Fma.MultiplyAdd(a, b0, acc0);
                            acc1 = Fma.MultiplyAdd(a, b1, acc1);
                        }

                        Avx.Store(outRow, acc0);
                        Avx.Store(outRow + 8, acc1);
                    }
                    else
#endif
                    {
                        // Scalar path for edge tiles
                        for (int p = 0; p < h; p++)
                        {
                            float a = tileRow[p];
                            for (int jj = 0; jj < nc; jj++)
                                outRow[jj] += a * pW2[bOff + p * Nr + jj];
                        }
                    }
                }
            }
        }
    }

    /// <summary>Pack B[H,N] into column-panel layout for GEMM2 micro-kernel access.</summary>
    private static unsafe void PackBRowMajor(float[] b, float[] packed, int h, int n)
    {
        int nrBlocks = (n + Nr - 1) / Nr;
        fixed (float* pB = b, pPacked = packed)
        {
            for (int jr = 0; jr < nrBlocks; jr++)
            {
                int jLocal = jr * Nr;
                int nc = Math.Min(Nr, n - jLocal);
                int packOff = jr * h * Nr;

                for (int p = 0; p < h; p++)
                {
                    int srcRowOff = p * n + jLocal;
                    int dstOff = packOff + p * Nr;
                    int j = 0;
                    for (; j < nc; j++)
                        pPacked[dstOff + j] = pB[srcRowOff + j];
                    // Zero-pad partial tiles
                    for (; j < Nr; j++)
                        pPacked[dstOff + j] = 0f;
                }
            }
        }
    }

    /// <summary>Fallback: separate GEMM + activation + GEMM when H is too large for L1.</summary>
    private static void FallbackSeparate(
        float[] input, float[] w1, float[] w2,
        float[] output, float[] activated,
        int m, int k, int h, int n,
        Func<float, float> applyActivation,
        float[]? preActivation = null)
    {
        // GEMM1: hidden = input @ W1 (reuse activated buffer as scratch — same size [m*h])
        Array.Clear(activated, 0, m * h);
        SimdGemm.Sgemm(input.AsSpan(0, m * k), w1.AsSpan(0, k * h), activated.AsSpan(0, m * h), m, k, h);

        // Capture pre-activation if requested (Phase G.2 GELU support)
        if (preActivation is not null)
            Buffer.BlockCopy(activated, 0, preActivation, 0, m * h * sizeof(float));

        // Activation in-place on activated buffer
        for (int i = 0; i < m * h; i++)
            activated[i] = applyActivation(activated[i]);

        // GEMM2: output = activated @ W2 (clear first since SimdGemm accumulates)
        Array.Clear(output, 0, m * n);
        SimdGemm.Sgemm(activated.AsSpan(0, m * h), w2.AsSpan(0, h * n), output.AsSpan(0, m * n), m, h, n);
    }
}
