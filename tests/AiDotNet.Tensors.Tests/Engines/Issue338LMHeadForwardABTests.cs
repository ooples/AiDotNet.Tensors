// Copyright (c) AiDotNet. All rights reserved.
// Issue #338: micro-bench the LM head forward in isolation — MKL GEMM vs
// analytic col_sum+dot. Decides whether Phase G.8 should be enabled by
// default or stay opt-in.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue338LMHeadForwardABTests
{
    private readonly ITestOutputHelper _output;
    public Issue338LMHeadForwardABTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    [Trait("Category", "Perf")]
    public unsafe void LMHeadForward_GemmVsAnalytic_AtIssue327Shape()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") != "1")
        {
            _output.WriteLine("Skip: AIDOTNET_RUN_PERF_GATES != 1.");
            return;
        }
        if (Environment.ProcessorCount < 16) return;
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64) return;

        // LM head shape from Issue #327: x [B*Ctx, D] @ W [D, V] → y [B*Ctx, V]
        // → loss = sum(y)
        const int M = 32 * 64;   // 2048
        const int K = 128;       // D
        const int V = 8192;      // Vocab
        const int iters = 200;

        var rng = new Random(42);
        var x = new float[M * K];
        var w = new float[K * V];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);
        var y = new float[M * V];

        // Pre-compute row_sum_W for analytic path
        var rowSumW = new float[K];
        for (int k = 0; k < K; k++)
        {
            float s = 0;
            int baseIdx = k * V;
            for (int v = 0; v < V; v++) s += w[baseIdx + v];
            rowSumW[k] = s;
        }

        bool blasAvail = BlasProvider.IsAvailable;
        _output.WriteLine($"# BlasProvider.IsAvailable={blasAvail}, BackendName={BlasProvider.BackendName}");

        // Warmup both paths
        for (int wi = 0; wi < 5; wi++)
        {
            BlasProvider.TryGemm(M, V, K, x, 0, K, w, 0, V, y, 0, V);
            // Sum y to scalar
            float ls = 0;
            for (int i = 0; i < y.Length; i++) ls += y[i];

            // Analytic: col_sum_x then dot
            float[] cs = new float[K];
            for (int m = 0; m < M; m++)
            {
                int baseIdx = m * K;
                for (int k = 0; k < K; k++) cs[k] += x[baseIdx + k];
            }
            float la = 0;
            for (int k = 0; k < K; k++) la += rowSumW[k] * cs[k];
        }

        // ─── GEMM path: x @ W → y, then sum(y) ───
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < iters; it++)
        {
            BlasProvider.TryGemm(M, V, K, x, 0, K, w, 0, V, y, 0, V);
            float ls = 0;
            for (int i = 0; i < y.Length; i++) ls += y[i];
        }
        sw.Stop();
        double gemmMs = sw.Elapsed.TotalMilliseconds / iters;

        // ─── Analytic path: col_sum_x then dot with cached row_sum_W ───
        sw = Stopwatch.StartNew();
        Span<float> colSum = stackalloc float[K];
        for (int it = 0; it < iters; it++)
        {
            for (int k = 0; k < K; k++) colSum[k] = 0f;
            for (int m = 0; m < M; m++)
            {
                int baseIdx = m * K;
                for (int k = 0; k < K; k++) colSum[k] += x[baseIdx + k];
            }
            float la = 0;
            for (int k = 0; k < K; k++) la += rowSumW[k] * colSum[k];
        }
        sw.Stop();
        double analyticMs = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"# LM head forward at M={M}, K={K}, V={V}");
        _output.WriteLine($"# GEMM+sum: {gemmMs:F3} ms/iter");
        _output.WriteLine($"# Analytic: {analyticMs:F3} ms/iter");
        _output.WriteLine($"# ratio: {gemmMs / analyticMs:F2}× speedup from analytic");
    }
}
