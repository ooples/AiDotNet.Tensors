// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #653: the float LayerNorm runs its rows in parallel from the elementwise grain up (it used to stay serial below
/// 512K elements). Each row is normalized by one thread in a fixed order, so every thread count must give the same
/// bits, and the result must match a double-precision reference.
/// </summary>
[Collection("CpuParallelSettings")]
public class LayerNormParallelGateTests
{
    [Theory]
    [InlineData(256, 1152)]   // DiT-XL/2 token LayerNorm: 295K elements, now parallel
    [InlineData(77, 1024)]    // a 77-token context
    [InlineData(1024, 64)]
    [InlineData(64, 64)]      // below the grain: serial
    public void Float_SameBitsAtEveryThreadCount_AndMatchesDoubleReference(int rows, int features)
    {
        var rng = new Random(rows * 31 + features);
        var x = new Tensor<float>(new[] { rows, features });
        var g = new Tensor<float>(new[] { features });
        var b = new Tensor<float>(new[] { features });
        var xs = x.AsWritableSpan(); var gs = g.AsWritableSpan(); var bs = b.AsWritableSpan();
        for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 6 - 3 + 0.5 * (i / features));
        for (int i = 0; i < features; i++) { gs[i] = (float)(rng.NextDouble() + 0.5); bs[i] = (float)(rng.NextDouble() - 0.5); }
        const double eps = 1e-5;

        var engine = new CpuEngine();
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        float[] serial, parallel;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            serial = engine.LayerNorm(x, g, b, eps, out _, out _).AsSpan().ToArray();
            CpuParallelSettings.MaxDegreeOfParallelism = 16;
            parallel = engine.LayerNorm(x, g, b, eps, out _, out _).AsSpan().ToArray();
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }

        for (int i = 0; i < serial.Length; i++)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(serial[i]) == TestHelpers.MathCompat.SingleToInt32Bits(parallel[i]),
                $"element {i}: {serial[i]} at 1 thread vs {parallel[i]} at 16");

        for (int r = 0; r < rows; r++)
        {
            double mean = 0, sq = 0;
            for (int f = 0; f < features; f++) mean += xs[r * features + f];
            mean /= features;
            for (int f = 0; f < features; f++) { double d = xs[r * features + f] - mean; sq += d * d; }
            double inv = 1.0 / Math.Sqrt(sq / features + eps);
            for (int f = 0; f < features; f++)
            {
                double expected = (xs[r * features + f] - mean) * inv * gs[f] + bs[f];
                Assert.True(Math.Abs(expected - parallel[r * features + f]) < 1e-4,
                    $"row {r} feature {f}: {parallel[r * features + f]} vs reference {expected}");
            }
        }
    }
}