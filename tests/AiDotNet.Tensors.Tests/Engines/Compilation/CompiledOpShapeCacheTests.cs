using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class CompiledOpShapeCacheTests
{
    [Fact]
    public void Moe_with_materialized_operands_aborts_before_eager_forward()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 1, 2 });
        var scores = new Tensor<float>(new[] { 1, 1 });
        var weight = new Tensor<float>(new[] { 1, 2, 2 });
        using var scope = GraphMode.EnableProbe();
        Assert.Throws<ObjectiveProbeAbortedException>(() =>
            engine.MoeSwiGlu(x, scores, null, weight, weight, weight, 1, 1, out _));
        Assert.Empty(scope.Nodes);
    }

    private static Tensor<float> Values(int[] shape, float[] values) => new(values, shape);

    private sealed class CountingEngine : CpuEngine
    {
        internal int EagerCalls { get; private set; }
        public override Tensor<T> PadNd<T>(Tensor<T> input, int[] pad, PadMode mode, T value = default!)
        {
            if (!GraphMode.IsActive) EagerCalls++;
            return base.PadNd(input, pad, mode, value);
        }
    }

    private static (CompiledTrainingPlan<float> Plan, ClosureObjectiveGuard<float> Guard) Compile(
        Func<Tensor<float>> objective, Tensor<float>[] parameters)
    {
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        using var scope = GraphMode.EnableTraining(parameters);
        var loss = objective();
        var plan = scope.CompileTraining(parameters, loss);
        var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
        Assert.True(guard is not null, reason);
        guard!.AttachTo(plan);
        return (plan, guard);
    }

    [Fact]
    public void Masked_replay_uses_fresh_data_and_matches_loss_and_sgd_weights()
    {
        var engine = new CpuEngine();
        var w = Values(new[] { 2, 2 }, new float[] { 1, 2, 3, 4 });
        var parameters = new[] { w };
        var expected = w.ToArray();
        var mask = new[] { true, false, false, true };
        int step = 0;
        Func<Tensor<float>> objective = () =>
        {
            var x = Values(new[] { 2, 2 }, new float[] { 1, step + 1, step + 2, 1 });
            var y = engine.TensorMaskedFill(engine.TensorMultiply(x, w), mask, 0f);
            return engine.ReduceSum(engine.TensorMultiply(y, y));
        };
        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
            for (step = 0; step < 4; step++)
            {
                Assert.True(guard.TryRebind(objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
                float a = step + 1, b = step + 2;
                float loss = a * a * expected[1] * expected[1] + b * b * expected[2] * expected[2];
                Assert.Equal(loss, plan.Step().ToArray()[0], precision: 3);
                expected[1] -= 0.02f * a * a * expected[1];
                expected[2] -= 0.02f * b * b * expected[2];
                var actual = w.ToArray();
                for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], precision: 4);
            }
        }
    }

    [Fact]
    public void Changed_options_miss_the_cache_and_do_not_poison_the_original_entry()
    {
        var engine = new CountingEngine();
        var w = Values(new[] { 2, 2 }, new float[] { 1, 2, 3, 4 });
        var parameters = new[] { w };
        int[] pad = { 0, 0, 0, 0 };
        Func<Tensor<float>> objective = () => engine.ReduceSum(engine.PadNd(w, pad, PadMode.Constant));
        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            int executions = engine.EagerCalls;
            Assert.True(executions > 0);
            Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(objective, parameters, out _));
            pad[0] = 1;
            Assert.Equal(ClosureGuardResult.Aborted, guard.TryRebind(objective, parameters, out _));
            Assert.Contains("no matching trace-time shape", guard.LastReason);
            pad[0] = 0;
            Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(objective, parameters, out _));
            Assert.Equal(executions, engine.EagerCalls);
        }
    }

    [Fact]
    public void Tensor_indices_are_produced_values_and_host_reads_still_abort()
    {
        var engine = new CpuEngine();
        var w = Values(new[] { 1, 1, 2, 2 }, new float[] { 1, 2, 3, 4 });
        var parameters = new[] { w };
        bool read = false;
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.MaxPool2DWithTensorIndices(w, new[] { 2, 2 }, new[] { 2, 2 }, out var indices);
            if (read) _ = indices.ToArray();
            return engine.ReduceSum(y);
        };
        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            Assert.Equal(0, guard.LeafCount);
            Assert.True(guard.TryRebind(objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
            read = true;
            Assert.Equal(ClosureGuardResult.Aborted, guard.TryRebind(objective, parameters, out _));
            Assert.Contains("MaxPool2DWithTensorIndices", guard.LastReason);
        }
    }

    [Fact]
    public void Pool_unpool_replay_refreshes_indices_when_the_winner_changes()
    {
        var engine = new CpuEngine();
        var w = Values(new[] { 1, 1, 2, 2 }, new float[] { 1, 2, 3, 4 });
        var weights = Values(new[] { 1, 1, 2, 2 }, new float[] { 1, 2, 3, 4 });
        var parameters = new[] { w };
        Func<Tensor<float>> objective = () =>
        {
            var pooled = engine.MaxPool2DWithTensorIndices(w, new[] { 2, 2 }, new[] { 2, 2 }, out var indices);
            var restored = engine.TensorMaxUnpool(pooled, indices, new[] { 2, 2 });
            return engine.ReduceSum(engine.TensorMultiply(restored, weights));
        };
        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
            for (int winner = 0; winner < 4; winner++)
            {
                for (int i = 0; i < 4; i++) w.SetFlat(i, i == winner ? 5f : 1f);
                Assert.True(guard.TryRebind(objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
                Assert.Equal(5f * (winner + 1), plan.Step().ToArray()[0], precision: 4);
                for (int i = 0; i < 4; i++)
                    Assert.Equal(i == winner ? 5f - 0.1f * (winner + 1) : 1f, w.GetFlat(i), precision: 4);
            }
        }
    }

    [Fact]
    public void Unseeded_probe_and_data_dependent_scatter_decline_without_execution()
    {
        var engine = new CpuEngine();
        var w = Values(new[] { 2 }, new float[] { 1, 2 });
        var parameters = new[] { w };
        var result = ClosureObjectiveGuard<float>.Probe(
            () => engine.ReduceSum(engine.TensorMaskedFill(w, new[] { false, true }, 0f)),
            parameters, out _, out _, out var reason);
        Assert.Equal(ClosureGuardResult.Aborted, result);
        Assert.Contains("no matching trace-time shape", reason);

        var indices = new Tensor<int>(new[] { 0, 1 }, new[] { 2 });
        Func<Tensor<float>> objective = () => engine.ReduceSum(engine.ScatterAdd(w, indices));
        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            indices[1] = 3;
            Assert.Equal(ClosureGuardResult.Aborted, guard.TryRebind(objective, parameters, out _));
            Assert.Contains("shape cache unsupported", guard.LastReason);
        }
    }

    [Theory]
    [InlineData("FFT")]
    [InlineData("STFT")]
    [InlineData("TensorWhere")]
    [InlineData("BoxIou")]
    public void Audited_shape_stable_ops_probe_upstream_computed_values(string operation)
    {
        var engine = new CpuEngine();
        var w = Values(new[] { 2, 4 }, new float[] { 0, 0, 1, 1, 0, 0, 2, 2 });
        var parameters = new[] { w };
        Func<Tensor<float>> objective = () =>
        {
            var x = engine.TensorMultiplyScalar(w, 2f);
            Tensor<float> y;
            switch (operation)
            {
                case "FFT":
                    engine.FFT(x, x, out var real, out var imaginary);
                    y = engine.TensorAdd(real, imaginary);
                    break;
                case "STFT":
                    engine.STFT(x, 4, 2, Values(new[] { 4 }, new float[] { 1, 1, 1, 1 }), true,
                        out var magnitude, out var phase);
                    y = engine.TensorAdd(magnitude, phase);
                    break;
                case "TensorWhere":
                    y = engine.TensorWhere(new[] { true, false, true, false, true, false, true, false }, x, w);
                    break;
                default:
                    y = engine.BoxIou(x, w);
                    break;
            }
            return engine.ReduceSum(y);
        };
        var (plan, guard) = Compile(objective, parameters);
        using (plan)
            Assert.True(guard.TryRebind(objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
    }
}
