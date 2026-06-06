using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase B gate (Tensors #558): the compiled mixed-dtype BACKWARD. After <see cref="MixedPrecisionCompiledPlan.Compile"/>
/// detaches the lazy sources, the reference <see cref="MixedPrecisionGraphBackward.Backward"/> can no longer
/// re-topo from <c>loss.LazySource</c> — so the plan drives the backward over its captured order via the
/// shared dispatch. This test builds the same FP16-activation matmul+loss graph twice: once traced fresh
/// and backprop'd with the non-compiled walk (reference), once compiled and backprop'd via the plan. The
/// FP32 parameter/input gradients must match.
/// </summary>
public class MixedPrecisionCompiledBackwardTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int r, int c, int seed)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 0.3);
        return new Tensor<float>(d, new[] { r, c });
    }

    // y = x @ W (FP16 activation); loss = sum(y^2), dL/dy = 2y.
    private Tensor<float> BuildGraph(LazyTensorScope scope, Tensor<float> x, Tensor<float> W)
    {
        GraphMode.SetCurrent(scope);
        try
        {
            var y = _engine.TensorMatMul(x, W);
            return scope.RecordUnary<float>(LazyNodeType.Sum, "sqloss", y, new[] { 1 },
                (e, o) => { var ya = y.ToArray(); float s = 0; foreach (var v in ya) s += v * v; o.AsWritableSpan()[0] = s; },
                (gradOut, inputs, output, state, e, grads) =>
                {
                    var yt = inputs[0]; var ya = yt.ToArray(); float go = gradOut.ToArray()[0];
                    var g = new float[yt.Length];
                    for (int i = 0; i < g.Length; i++) g[i] = 2f * ya[i] * go;
                    grads[yt] = grads.TryGetValue(yt, out var ex) ? e.TensorAdd(ex, new Tensor<float>(g, yt._shape)) : new Tensor<float>(g, yt._shape);
                });
        }
        finally { GraphMode.SetCurrent(null); }
    }

    [Fact]
    public void CompiledBackward_MatchesNonCompiled()
    {
        const int B = 4, d = 5;
        var x = Rand(B, d, 1);
        var W = Rand(d, d, 2);

        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        try
        {
            // Reference: fresh trace + non-compiled backward.
            float[] refGx, refGw;
            using (new AutocastScope(PrecisionMode.Float16))
            {
                var sr = new LazyTensorScope(null);
                var lossR = BuildGraph(sr, x, W);
                _ = lossR.ToArray(); // realize
                var gr = MixedPrecisionGraphBackward.Backward(lossR, _engine);
                Assert.True(gr.Fp32.TryGetValue(x, out var gx0), "reference grad for x missing");
                Assert.True(gr.Fp32.TryGetValue(W, out var gw0), "reference grad for W missing");
                refGx = gx0.ToArray(); refGw = gw0.ToArray();
            }

            // Compiled: same graph, plan-driven backward over the captured order.
            float[] gotGx, gotGw;
            using (new AutocastScope(PrecisionMode.Float16))
            {
                var sp = new LazyTensorScope(null);
                var lossP = BuildGraph(sp, x, W);
                var plan = MixedPrecisionCompiledPlan.Compile(lossP, _engine);
                plan.Forward();
                var gp = plan.Backward();
                Assert.True(gp.Fp32.TryGetValue(x, out var gx1), "compiled grad for x missing");
                Assert.True(gp.Fp32.TryGetValue(W, out var gw1), "compiled grad for W missing");
                gotGx = gx1.ToArray(); gotGw = gw1.ToArray();
            }

            AssertClose(refGx, gotGx, "dL/dx");
            AssertClose(refGw, gotGw, "dL/dW");
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }
    }

    private static void AssertClose(float[] expected, float[] got, string what)
    {
        // Both paths run the SAME ops on the SAME data; the only divergence is FP16 GEMM reduction-order
        // noise (GPU FP16 matmul is not bit-deterministic run-to-run). FP16 has ~10 mantissa bits, so a
        // few e-3 relative is the right bound for an algorithmic-equivalence check.
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 2e-4f + 4e-3f * Math.Abs(expected[i]),
                $"{what}: mismatch at {i}: expected {expected[i]}, got {got[i]}");
    }
}
