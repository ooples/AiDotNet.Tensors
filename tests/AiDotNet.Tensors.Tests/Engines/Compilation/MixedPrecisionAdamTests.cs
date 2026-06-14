using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Fused-Adam path gate (Tensors #558): the compiled mixed-dtype plan trains with the ADAM optimizer —
/// the optimizer Adam-configured models (the cortex) use, so this is what makes the fused training path
/// eligible for the FP16 activation-storage win. Learns y = x·W toward a target via
/// <see cref="MixedPrecisionCompiledPlan.StepAdam"/> with FP32 master moments + a GradScaler, and asserts
/// the loss descends with zero overflow.
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public class MixedPrecisionAdamTests
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
    public void CompiledAdam_MixedPrecision_DescendsLoss()
    {
        const int B = 8, d = 6;
        var x = Rand(B, d, 1, 1.0);
        var wTrue = Rand(d, d, 2, 0.4);
        var t = _engine.TensorMatMul(x, wTrue);
        var W = Rand(d, d, 3, 0.1);

        Func<Tensor<float>> forward = () =>
        {
            var y = _engine.TensorMatMul(x, W);
            var diff = _engine.TensorSubtract(y, t);
            var sq = _engine.TensorMultiply(diff, diff);
            return _engine.ReduceSum(sq);
        };

        var plan = MixedPrecisionCompiledPlan.Trace(forward, _engine);
        var scaler = new AiDotNet.Tensors.Engines.Autodiff.GradScaler(
            new AiDotNet.Tensors.Engines.Autodiff.MixedPrecisionConfig { LossScale = 128f, DynamicLossScale = false });
        var pars = new[] { W };

        float first = 0, last = 0; int overflows = 0;
        const int steps = 60;
        for (int s = 0; s < steps; s++)
        {
            var r = plan.StepAdam(pars, learningRate: 0.05f, scaler: scaler);
            if (s == 0) first = r.Loss;
            if (s == steps - 1) last = r.Loss;
            if (r.FoundInfNan) overflows++;
            Assert.False(float.IsNaN(r.Loss) || float.IsInfinity(r.Loss), $"loss non-finite at step {s}");
        }

        Assert.Equal(0, overflows);
        Assert.True(first > 0, "first loss positive");
        Assert.True(last < 0.25f * first, $"Adam mixed-precision did not descend: first {first}, last {last}");
    }
}
