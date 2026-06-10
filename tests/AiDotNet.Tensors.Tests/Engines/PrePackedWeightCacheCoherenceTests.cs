using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression for the transparent prepacked-B GEMM cache serving STALE packed
/// weights after the weight buffer was mutated in place (e.g. an optimizer step
/// or any raw Data.Span write that doesn't notify the cache). This produced the
/// post-training / clone inference divergence in AiDotNet #1221 / #1331: a model
/// predicted with weights from before its last training update. The cache now
/// validates a content fingerprint of the live weight on every hit.
/// </summary>
public class PrePackedWeightCacheCoherenceTests
{
    private static double MaxAbs(Tensor<float> a, Tensor<float> b)
    { double d = 0; int n = Math.Min(a.Length, b.Length); for (int i = 0; i < n; i++) d = Math.Max(d, Math.Abs(a[i] - b[i])); return d; }
    private static double MaxAbs(Tensor<double> a, Tensor<double> b)
    { double d = 0; int n = Math.Min(a.Length, b.Length); for (int i = 0; i < n; i++) d = Math.Max(d, Math.Abs(a[i] - b[i])); return d; }

    [Theory]
    [InlineData(8, 32, 64)]
    [InlineData(128, 64, 128)]
    [InlineData(64, 256, 256)]
    public void FusedLinear_Float_ReflectsInPlaceWeightMutation(int m, int k, int n)
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { m, k });
        var w = new Tensor<float>(new[] { k, n });
        var b = new Tensor<float>(new[] { n });
        var r = new Random(7);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(r.NextDouble() - 0.5);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(r.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(r.NextDouble() - 0.5);

        _ = engine.FusedLinear(x, w, b, FusedActivationType.ReLU); // packs/caches w
        var ws = w.Data.Span;
        for (int i = 0; i < w.Length; i++) ws[i] += 3.0f;          // in-place mutation
        var stale = engine.FusedLinear(x, w, b, FusedActivationType.ReLU);

        var wFresh = new Tensor<float>(new[] { k, n });
        w.Data.Span.CopyTo(wFresh.Data.Span);
        var truth = engine.FusedLinear(x, wFresh, b, FusedActivationType.ReLU);

        Assert.True(MaxAbs(stale, truth) < 1e-3,
            $"Prepacked-B cache served stale weights after in-place mutation (max|delta|={MaxAbs(stale, truth):E3}).");
    }

    [Theory]
    [InlineData(8, 32, 64)]
    [InlineData(128, 64, 128)]
    public void FusedLinear_Double_ReflectsInPlaceWeightMutation(int m, int k, int n)
    {
        var engine = new CpuEngine();
        var x = new Tensor<double>(new[] { m, k });
        var w = new Tensor<double>(new[] { k, n });
        var b = new Tensor<double>(new[] { n });
        var r = new Random(11);
        for (int i = 0; i < x.Length; i++) x[i] = r.NextDouble() - 0.5;
        for (int i = 0; i < w.Length; i++) w[i] = r.NextDouble() - 0.5;
        for (int i = 0; i < b.Length; i++) b[i] = r.NextDouble() - 0.5;

        _ = engine.FusedLinear(x, w, b, FusedActivationType.ReLU);
        var ws = w.Data.Span;
        for (int i = 0; i < w.Length; i++) ws[i] += 3.0;
        var stale = engine.FusedLinear(x, w, b, FusedActivationType.ReLU);

        var wFresh = new Tensor<double>(new[] { k, n });
        w.Data.Span.CopyTo(wFresh.Data.Span);
        var truth = engine.FusedLinear(x, wFresh, b, FusedActivationType.ReLU);

        Assert.True(MaxAbs(stale, truth) < 1e-9,
            $"Prepacked-B double cache served stale weights after in-place mutation (max|delta|={MaxAbs(stale, truth):E3}).");
    }
}
