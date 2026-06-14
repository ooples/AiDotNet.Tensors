using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase E (Tensors public API, #558): an external caller — exactly as AiDotNet's training path will —
/// builds and trains a mixed-precision model using ONLY the public surface: ordinary engine ops inside a
/// forward delegate, <see cref="MixedPrecisionCompiledPlan.Trace"/> to compile, and
/// <see cref="MixedPrecisionCompiledPlan.Step"/> to train. No GraphMode / LazyTensorScope / MixedPrecisionEmit
/// internals are touched (Trace manages them). The loss must descend — proving the API is sufficient to
/// drive FP16 activation-storage training from outside the assembly.
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public class MixedPrecisionPublicApiTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int r, int c, int seed, double s)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * s);
        return new Tensor<float>(d, new[] { r, c });
    }

    [Fact]
    public void PublicApi_TraceAndStep_TrainsMixedPrecision()
    {
        const int B = 8, d = 6;
        var x = Rand(B, d, 1, 1.0);
        var wTrue = Rand(d, d, 2, 0.4);
        var t = _engine.TensorMatMul(x, wTrue);
        var W = Rand(d, d, 3, 0.1);

        // Forward built from PUBLIC engine ops only: loss = sum((x·W - t)^2).
        // Trace activates the FP16 autocast + activation-storage internally; matmul auto-emits Half.
        Func<Tensor<float>> forward = () =>
        {
            var y = _engine.TensorMatMul(x, W);
            var diff = _engine.TensorSubtract(y, t);
            var sq = _engine.TensorMultiply(diff, diff);
            return _engine.ReduceSum(sq);
        };

        var plan = MixedPrecisionCompiledPlan.Trace(forward, _engine);
        var pars = new[] { W };

        float first = 0, last = 0; int bad = 0;
        const int steps = 50;
        for (int s = 0; s < steps; s++)
        {
            var r = plan.Step(pars, learningRate: 0.02f, scaler: null);
            if (s == 0) first = r.Loss;
            if (s == steps - 1) last = r.Loss;
            if (float.IsNaN(r.Loss) || float.IsInfinity(r.Loss)) bad++;
        }

        Assert.Equal(0, bad);
        Assert.True(first > 0, "first loss positive");
        Assert.True(last < 0.5f * first, $"public-API training did not descend: first {first}, last {last}");
    }
}
