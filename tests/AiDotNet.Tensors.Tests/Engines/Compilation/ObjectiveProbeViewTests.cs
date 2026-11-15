using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// View ops (Narrow, Reshape, Transpose) on a probe value: the probe records them shape-only like any other op, so an
/// objective that slices or reshapes an intermediate can be signed and rebound. They used to abort the probe with
/// "View exceeds storage bounds ... outside storage [0, -1]" because a view tried to share the probe value's storage.
/// Also: an op's saved shape record (LayerNorm's) compares by value instead of making the objective unsignable.
/// </summary>
public class ObjectiveProbeViewTests
{
    private static Tensor<float> Filled(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows * cols; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5));
        return t;
    }

    private static ObjectiveSignature Probe(Func<Tensor<float>> objective, Tensor<float>[] parameters, out object[] leaves)
    {
        var result = ClosureObjectiveGuard<float>.Probe(objective, parameters, out var signature, out leaves, out var reason);
        Assert.True(result == ClosureGuardResult.Matched, reason);
        return signature!;
    }

    [Fact]
    public void Narrow_then_reshape_of_a_probe_value_signs_and_matches_across_fresh_data()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 4, 1);
        int seed = 10;
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.TensorMatMul(Filled(3, 6, seed), w);           // [3, 4], a probe value
            var part = engine.Reshape(engine.TensorNarrow(y, 1, 1, 2), new[] { 6 });
            return engine.ReduceSum(engine.TensorMultiply(part, part));
        };

        var first = Probe(objective, new[] { w }, out var leaves1);
        seed = 11;
        var second = Probe(objective, new[] { w }, out var leaves2);

        Assert.True(first.Signable, first.UnsignableReason);
        Assert.Equal(first, second);
        Assert.Single(leaves1);
        Assert.NotSame(leaves1[0], leaves2[0]);
    }

    [Fact]
    public void Transpose_of_a_probe_value_keeps_its_layout()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 4, 1);
        var x = Filled(3, 6, 2);
        Tensor<float>? transposed = null;

        using (GraphMode.EnableProbe())
        {
            var y = engine.TensorMatMul(x, w);
            transposed = y.Transpose();
            Assert.Equal(new[] { 4, 3 }, transposed.Shape.ToArray());
            Assert.False(transposed.IsContiguous);   // downstream ops see the same layout as in a real trace
            Assert.Throws<ObjectiveProbeAbortedException>(() => transposed.ToArray());
        }
    }

    [Fact]
    public void Plan_with_a_sliced_intermediate_rebinds_and_trains_like_eager()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 4, 1);
        var parameters = new[] { w };
        int step = 0;
        Tensor<float>? current = null;
        Func<Tensor<float>> objective = () =>
        {
            current = Filled(3, 6, 100 + step);
            var y = engine.TensorMatMul(current, w);
            var part = engine.Reshape(engine.TensorNarrow(y, 1, 1, 2), new[] { 6 });
            return engine.ReduceSum(engine.TensorMultiply(part, part));
        };

        CompiledTrainingPlan<float> plan;
        ClosureObjectiveGuard<float> guard;
        using (ClosureObjectiveGuard<float>.BeginRecording())
        {
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var loss = objective();
                plan = scope.CompileTraining(parameters, loss);
            }
            guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason)
                    ?? throw new InvalidOperationException($"no guard was recorded: {reason}");
        }
        guard.AttachTo(plan);
        var expectedWeights = w.ToArray();

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
            for (step = 1; step <= 3; step++)
            {
                var result = guard.TryRebind(objective, parameters, out _);
                Assert.True(result == ClosureGuardResult.Matched, guard.LastReason);
                var x = current!;

                // Eager loss on this step's data with the current weights: sum over rows of y[:, 1..2]^2.
                double expected = 0;
                for (int r = 0; r < 3; r++)
                    for (int c = 1; c <= 2; c++)
                    {
                        double y = 0;
                        for (int k = 0; k < 6; k++) y += x[r, k] * expectedWeights[k * 4 + c];
                        expected += y * y;
                    }

                var gradients = new double[24];
                for (int r = 0; r < 3; r++)
                    for (int c = 1; c <= 2; c++)
                    {
                        double y = 0;
                        for (int k = 0; k < 6; k++) y += x[r, k] * expectedWeights[k * 4 + c];
                        for (int k = 0; k < 6; k++) gradients[k * 4 + c] += 2 * x[r, k] * y;
                    }
                for (int i = 0; i < expectedWeights.Length; i++)
                    expectedWeights[i] -= (float)(0.01 * gradients[i]);

                float actual = plan.Step().ToArray()[0];
                Assert.True(Math.Abs(actual - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                    $"step {step}: plan loss {actual}, eager {expected}");
                for (int i = 0; i < expectedWeights.Length; i++)
                    Assert.True(Math.Abs(w.GetFlat(i) - expectedWeights[i]) <= 1e-5,
                        $"step {step}, weight {i}: compiled {w.GetFlat(i)}, eager {expectedWeights[i]}");
            }
        }
    }

    [Fact]
    public void LayerNorm_saved_shape_record_signs_by_value()
    {
        var engine = new CpuEngine();
        var gamma = Filled(1, 8, 1).Reshape(8);
        var beta = Filled(1, 8, 2).Reshape(8);
        var parameters = new[] { gamma, beta };
        int rows = 4;
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.LayerNorm(Filled(rows, 8, rows), gamma, beta, 1e-5, out _, out _);
            return engine.ReduceSum(engine.TensorMultiply(y, y));
        };

        var a = Probe(objective, parameters, out _);
        var b = Probe(objective, parameters, out _);
        rows = 5;
        var c = Probe(objective, parameters, out _);

        Assert.True(a.Signable, a.UnsignableReason);
        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
    }
}
