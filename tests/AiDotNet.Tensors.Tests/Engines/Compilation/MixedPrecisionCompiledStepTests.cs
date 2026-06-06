using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase C gate (Tensors #558): a complete compiled mixed-precision TRAINING step. The graph is compiled
/// ONCE (forward + backward + SGD wired into <see cref="MixedPrecisionCompiledPlan.Step"/>); each step
/// replays forward, runs the loss-scaled mixed-dtype backward, and updates the FP32 master weight — no
/// per-step retrace. Learns y = x·W toward a target with a GradScaler active and asserts the loss
/// descends with zero FP16 overflow.
/// </summary>
public class MixedPrecisionCompiledStepTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int r, int c, int seed, double scale)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(d, new[] { r, c });
    }

    [Fact]
    public void CompiledStep_DescendsLoss_WithLossScaling_NoOverflow()
    {
        const int B = 8, d = 6;
        var x = Rand(B, d, 1, 1.0);
        var wTrue = Rand(d, d, 2, 0.4);
        var t = _engine.TensorMatMul(x, wTrue); // FP32 target
        var W = Rand(d, d, 3, 0.1);

        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        try
        {
            // Trace y = x·W (FP16 activation) + squared-error loss, once; then compile.
            Tensor<float> lossT;
            var scope = new LazyTensorScope(null);
            using (new AutocastScope(PrecisionMode.Float16))
            {
                GraphMode.SetCurrent(scope);
                try
                {
                    var y = _engine.TensorMatMul(x, W);
                    lossT = scope.RecordUnary<float>(LazyNodeType.Sum, "sqerr", y, new[] { 1 },
                        (e, o) => { var ya = y.ToArray(); var ta = t.ToArray(); float s = 0; for (int i = 0; i < ya.Length; i++) { float dd = ya[i] - ta[i]; s += dd * dd; } o.AsWritableSpan()[0] = s; },
                        (gradOut, inputs, output, state, e, grads) =>
                        {
                            var yt = inputs[0]; var ya = yt.ToArray(); var ta = t.ToArray(); float go = gradOut.ToArray()[0];
                            var g = new float[yt.Length];
                            for (int i = 0; i < g.Length; i++) g[i] = 2f * (ya[i] - ta[i]) * go;
                            grads[yt] = grads.TryGetValue(yt, out var ex) ? e.TensorAdd(ex, new Tensor<float>(g, yt._shape)) : new Tensor<float>(g, yt._shape);
                        });
                }
                finally { GraphMode.SetCurrent(null); }
            }

            var plan = MixedPrecisionCompiledPlan.Compile(lossT, _engine);
            var scaler = new AiDotNet.Tensors.Engines.Autodiff.GradScaler(
                new AiDotNet.Tensors.Engines.Autodiff.MixedPrecisionConfig { LossScale = 256f, DynamicLossScale = false });
            var pars = new[] { W };

            float first = 0, last = 0;
            const int steps = 60;
            int overflows = 0;
            for (int s = 0; s < steps; s++)
            {
                var r = plan.Step(pars, learningRate: 0.03f, scaler: scaler);
                if (s == 0) first = r.Loss;
                if (s == steps - 1) last = r.Loss;
                if (r.FoundInfNan) overflows++;
                Assert.False(float.IsNaN(r.Loss) || float.IsInfinity(r.Loss), $"loss non-finite at step {s}");
            }

            Assert.Equal(0, overflows);
            Assert.True(first > 0, "first loss positive");
            Assert.True(last < 0.5f * first, $"compiled training did not descend: first {first}, last {last}");
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }
    }
}
