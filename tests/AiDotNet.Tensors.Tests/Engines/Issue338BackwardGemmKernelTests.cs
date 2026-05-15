// Copyright (c) AiDotNet. All rights reserved.
// Issue #338 Phase F.2 follow-up: at d=128 backward shapes, is the
// compiled-spec path's choice of BlasProvider.TryGemmEx (OpenBLAS) the
// right one, or would SimdGemm.Sgemm beat it on this hardware?

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue338BackwardGemmKernelTests
{
    private readonly ITestOutputHelper _output;
    public Issue338BackwardGemmKernelTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    [Trait("Category", "Perf")]
    public void Issue338_BackwardGemm_KernelAB_AtTransformerShapes()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") != "1")
        {
            _output.WriteLine("Skip: AIDOTNET_RUN_PERF_GATES != 1.");
            return;
        }
        if (Environment.ProcessorCount < 16) { _output.WriteLine("Skip: <16 cores."); return; }
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64) { _output.WriteLine("Skip: not X64."); return; }

        // From the Phase F.2 compiled-spec breakdown: for an Issue #327 d=128
        // forward at B=32, ctx=64, L=4, every backward MatMul becomes two
        // GEMMs against these shapes (M = B*ctx*..., K = N = D = 128).
        const int M = 2048;     // B*ctx = 32*64
        const int K = 128;      // d
        const int N = 128;      // d
        const int iters = 200;

        var rng = new Random(42);
        var dC = NewArr(M * N, rng);   // gradOut: [M, N]
        var Ag = NewArr(M * K, rng);   // A:       [M, K]
        var Bg = NewArr(K * N, rng);   // B:       [K, N]
        var gradA = new float[M * K];
        var gradB = new float[K * N];

        bool blasAvail = BlasProvider.IsAvailable;
        _output.WriteLine($"# BlasProvider.IsAvailable={blasAvail}, BackendName={BlasProvider.BackendName}");

        // Warmup both kernels
        for (int w = 0; w < 5; w++)
        {
            if (blasAvail)
            {
                BlasProvider.TryGemmEx(M, K, N, dC, 0, N, false, Bg, 0, N, true, gradA, 0, K);
                BlasProvider.TryGemmEx(K, N, M, Ag, 0, K, true, dC, 0, N, false, gradB, 0, N);
            }
            // SimdGemm with transposed flags (same as MatMulBackward fast path)
            SimdGemm.Sgemm(dC, N, false, Bg, N, true,  gradA.AsSpan(0, M * K), M, N, K);
            SimdGemm.Sgemm(Ag, K, true,  dC, N, false, gradB.AsSpan(0, K * N), K, M, N);
        }

        // ----- OpenBLAS measurement -----
        double blasMsPerIter = double.NaN;
        if (blasAvail)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                BlasProvider.TryGemmEx(M, K, N, dC, 0, N, false, Bg, 0, N, true, gradA, 0, K);
                BlasProvider.TryGemmEx(K, N, M, Ag, 0, K, true, dC, 0, N, false, gradB, 0, N);
            }
            sw.Stop();
            blasMsPerIter = sw.Elapsed.TotalMilliseconds / iters;
        }

        // ----- SimdGemm (transposed-flag) measurement -----
        double simdTransMsPerIter;
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                SimdGemm.Sgemm(dC, N, false, Bg, N, true,  gradA.AsSpan(0, M * K), M, N, K);
                SimdGemm.Sgemm(Ag, K, true,  dC, N, false, gradB.AsSpan(0, K * N), K, M, N);
            }
            sw.Stop();
            simdTransMsPerIter = sw.Elapsed.TotalMilliseconds / iters;
        }

        // ----- SimdGemm via materialized-transpose (parallel NoTrans path) -----
        // Mirrors the ND×2D collapsed fast path in BackwardFunctions.cs:475-522 —
        // explicit transpose buffer + SimdGemm.Sgemm NoTrans×NoTrans which engages
        // SgemmTiledParallel2D.
        var BT = new float[N * K];
        var AT = new float[K * M];
        double simdMaterializeMsPerIter;
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                // Transpose B: [K,N] → [N,K]
                for (int r = 0; r < K; r++)
                    for (int c = 0; c < N; c++)
                        BT[c * K + r] = Bg[r * N + c];
                // gradA = dC @ B^T = dC @ BT (NoTrans × NoTrans)
                SimdGemm.Sgemm(dC.AsSpan(0, M * N), BT.AsSpan(0, N * K), gradA.AsSpan(0, M * K), M, N, K);

                // Transpose A: [M,K] → [K,M]
                for (int r = 0; r < M; r++)
                    for (int c = 0; c < K; c++)
                        AT[c * M + r] = Ag[r * K + c];
                // gradB = A^T @ dC (NoTrans × NoTrans)
                SimdGemm.Sgemm(AT.AsSpan(0, K * M), dC.AsSpan(0, M * N), gradB.AsSpan(0, K * N), K, M, N);
            }
            sw.Stop();
            simdMaterializeMsPerIter = sw.Elapsed.TotalMilliseconds / iters;
        }

        _output.WriteLine($"# Issue #338 backward-GEMM kernel A/B at M={M}, K={K}, N={N}");
        _output.WriteLine($"# OpenBLAS (TryGemmEx trans): {blasMsPerIter:F3} ms/2-gemm");
        _output.WriteLine($"# SimdGemm  (Sgemm trans):    {simdTransMsPerIter:F3} ms/2-gemm");
        _output.WriteLine($"# SimdGemm  (materialize+NoTrans): {simdMaterializeMsPerIter:F3} ms/2-gemm");
        if (blasAvail)
        {
            _output.WriteLine($"# ratio SimdTrans/OpenBLAS:      {simdTransMsPerIter / blasMsPerIter:F2}×");
            _output.WriteLine($"# ratio SimdMaterialize/OpenBLAS: {simdMaterializeMsPerIter / blasMsPerIter:F2}×");
        }
    }

    private static float[] NewArr(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
