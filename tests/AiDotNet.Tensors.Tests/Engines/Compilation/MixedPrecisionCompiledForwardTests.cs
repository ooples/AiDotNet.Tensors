using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase A gate (Tensors #558): compile-once / replay FORWARD over a mixed-dtype graph. A 2-layer matmul
/// stack with FP16 activations is compiled into a <see cref="MixedPrecisionCompiledPlan"/>; replaying its
/// forward must match a freshly-traced-and-realized graph — both initially AND after a parameter is
/// mutated in place (proving the plan reuses stable buffers and reads current leaf data on each pass,
/// i.e. it is usable as a training-step forward).
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)] // serializes MixedPrecisionEmit.TestOverrideEnabled mutators
public class MixedPrecisionCompiledForwardTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int r, int c, int seed)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 0.25);
        return new Tensor<float>(d, new[] { r, c });
    }

    [Fact]
    public void CompiledForward_Replays_AndTracksParamMutation()
    {
        const int B = 4, d = 5;
        var x = Rand(B, d, 1);
        var W1 = Rand(d, d, 2);
        var W2 = Rand(d, d, 3);

        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        try
        {
            // Fresh trace + realize (the reference), FP16 activations.
            float[] FreshTrace()
            {
                var scope = new LazyTensorScope(null);
                using (new AutocastScope(PrecisionMode.Float16))
                {
                    GraphMode.SetCurrent(scope);
                    try
                    {
                        var y1 = _engine.TensorMatMul(x, W1);
                        var y2 = _engine.TensorMatMul(y1, W2);
                        return y2.ToArray(); // realize
                    }
                    finally { GraphMode.SetCurrent(null); }
                }
            }

            var ref1 = FreshTrace();

            // Build the plan's own graph, compile it.
            Tensor<float> yPlan;
            var planScope = new LazyTensorScope(null);
            using (new AutocastScope(PrecisionMode.Float16))
            {
                GraphMode.SetCurrent(planScope);
                try
                {
                    var y1 = _engine.TensorMatMul(x, W1);
                    yPlan = _engine.TensorMatMul(y1, W2);
                }
                finally { GraphMode.SetCurrent(null); }
            }
            var plan = MixedPrecisionCompiledPlan.Compile(yPlan, _engine);

            var out1 = plan.Forward().ToArray();
            AssertClose(ref1, out1, "initial replay vs fresh trace");

            // Mutate a parameter in place; both the plan (shared W2 leaf) and a fresh trace must agree.
            var w2span = W2.AsWritableSpan();
            for (int i = 0; i < w2span.Length; i++) w2span[i] *= 1.7f;
            W2.IncrementVersion();

            var ref2 = FreshTrace();
            var out2 = plan.Forward().ToArray();
            AssertClose(ref2, out2, "post-mutation replay vs fresh trace");

            // And the mutation actually changed the output (guards against a frozen/stale buffer).
            Assert.False(MaxAbsDiff(ref1, ref2) < 1e-6f, "param mutation should change the output");
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0; for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i])); return m;
    }

    private static void AssertClose(float[] expected, float[] got, string what)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-5f + 1e-4f * Math.Abs(expected[i]),
                $"{what}: mismatch at {i}: expected {expected[i]}, got {got[i]}");
    }
}
