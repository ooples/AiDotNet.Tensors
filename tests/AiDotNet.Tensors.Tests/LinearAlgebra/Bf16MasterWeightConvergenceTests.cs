// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Earns (or rejects) the bf16-during-training default with DATA. The streaming
/// pool's stored copy is the canonical master weight, so a bf16 store re-quantizes
/// the masters every optimizer step. This measures the classic mixed-precision
/// failure mode — small updates lost to bf16 truncation — on a controlled convex
/// problem (linear least squares, known optimum), comparing three store modes:
/// fp32 (no quantization), bf16 deterministic (RNE), bf16 stochastic.
///
/// Expectation: bf16 DETERMINISTIC stalls in the small-update regime (each step's
/// update is below the bf16 grid spacing, so it's repeatedly rounded away), while
/// bf16 STOCHASTIC stays close to fp32 (unbiased rounding lets sub-grid updates
/// accumulate probabilistically). That is the evidence for making Bf16Stochastic —
/// not deterministic Bf16 — the opt-in training store.
/// </summary>
public class Bf16MasterWeightConvergenceTests
{
    private struct Rng
    {
        private ulong _s;
        public Rng(ulong seed) { _s = seed | 1UL; }
        public double NextUnit() { ulong x = _s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; _s = x; return (x >> 11) * (1.0 / (1UL << 53)); }
        public float NextGaussian(double std) { double u1 = Math.Max(1e-12, NextUnit()), u2 = NextUnit(); return (float)(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)); }
    }

    private enum Store { Fp32, Bf16Det, Bf16Stoch }

    // Quantize a parameter vector through the streaming store (in place).
    private static void ApplyStore(float[] x, Store store)
    {
        if (store == Store.Fp32) return;
        var enc = new byte[x.Length * 2];
        StreamingStoreCodec.EncodeFloat(x, enc, stochastic: store == Store.Bf16Stoch);
        StreamingStoreCodec.DecodeFloat(enc, x);
    }

    private static double RunLeastSquares(Store store, int steps, ulong stochSeed = 1)
    {
        const int m = 256, n = 48;
        var rng = new Rng(12345);
        // A (m x n), x_true small-std (realistic trained-weight scale where bf16
        // grid spacing matters), b = A x_true.
        var A = new float[m * n];
        for (int i = 0; i < m * n; i++) A[i] = rng.NextGaussian(1.0 / Math.Sqrt(n));
        var xTrue = new float[n];
        for (int j = 0; j < n; j++) xTrue[j] = rng.NextGaussian(0.05);
        var b = new double[m];
        for (int i = 0; i < m; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i * n + j] * xTrue[j]; b[i] = s; }

        var x = new float[n]; // start at 0
        double lr = 0.15;     // small enough that late-stage updates approach bf16 grid spacing
        var grad = new double[n];
        var resid = new double[m];
        // Pin the stochastic-rounding sequence so this run is reproducible regardless of which
        // xUnit pool thread it lands on (otherwise the thread-static RNG state leaks across tests
        // and the final loss varies run to run — the intermittent-failure root cause).
        if (store == Store.Bf16Stoch) StreamingStoreCodec.SeedStochasticRng(stochSeed);
        for (int step = 0; step < steps; step++)
        {
            // resid = A x - b
            for (int i = 0; i < m; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i * n + j] * x[j]; resid[i] = s - b[i]; }
            // grad = A^T resid / m
            Array.Clear(grad, 0, n);
            for (int i = 0; i < m; i++) { double r = resid[i]; int bse = i * n; for (int j = 0; j < n; j++) grad[j] += A[bse + j] * r; }
            for (int j = 0; j < n; j++) x[j] = (float)(x[j] - lr * grad[j] / m);
            // Master weights live in the streaming store → re-quantize each step.
            ApplyStore(x, store);
        }
        // Final loss 0.5/m * ||A x - b||^2
        double loss = 0;
        for (int i = 0; i < m; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i * n + j] * x[j]; double e = s - b[i]; loss += e * e; }
        return 0.5 * loss / m;
    }

    [Fact]
    public void Bf16Stochastic_TracksFp32_WhileDeterministicStalls()
    {
        const int steps = 4000;
        const int stochTrials = 8;
        double fp32 = RunLeastSquares(Store.Fp32, steps);
        double det = RunLeastSquares(Store.Bf16Det, steps);

        // Stochastic rounding is randomized by construction; assert on its EXPECTED behavior by
        // averaging several independently-seeded runs rather than a single noisy draw. Each run is
        // deterministic (seeded), so the mean is reproducible and the gate is stable across machines.
        double stochSum = 0, stochWorst = 0;
        for (ulong seed = 1; seed <= stochTrials; seed++)
        {
            double v = RunLeastSquares(Store.Bf16Stoch, steps, seed);
            stochSum += v;
            stochWorst = Math.Max(stochWorst, v);
        }
        double stoch = stochSum / stochTrials;

        var outLines = new[]
        {
            $"Final least-squares loss after {steps} steps (lower = better convergence):",
            $"  fp32 store           : {fp32:E3}",
            $"  bf16 deterministic   : {det:E3}  ({det / fp32:F1}x fp32 loss)",
            $"  bf16 stochastic mean : {stoch:E3}  ({stoch / fp32:F1}x fp32 loss; worst of {stochTrials}: {stochWorst / fp32:F1}x)",
        };
        foreach (var l in outLines) _output.WriteLine(l);

        // Stochastic must converge MUCH closer to fp32 than deterministic does —
        // that's the data justifying Bf16Stochastic as the training store.
        Assert.True(stoch < det, "stochastic rounding must beat deterministic in the small-update regime");
        Assert.True(stoch < fp32 * 5.0, "stochastic store should stay within a small factor of fp32 convergence");
        Assert.True(det > fp32 * 5.0, "deterministic bf16 should visibly stall (validates the masters concern)");
    }

    private readonly ITestOutputHelper _output;
    public Bf16MasterWeightConvergenceTests(ITestOutputHelper output) => _output = output;
}
