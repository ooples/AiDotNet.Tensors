using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class SearchSortedReplayTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Replay_refreshes_both_operands_and_preserves_exact_boundary_and_NaN_indices(bool useDouble, bool right)
    {
        if (useDouble) CheckBoundaries<double>(right);
        else CheckBoundaries<float>(right);
    }

    private static void CheckBoundaries<T>(bool right)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var ops = MathHelper.GetNumericOperations<T>();
        var p = new Tensor<T>(new[] { ops.FromDouble(2) }, new[] { 1 });
        var sequence = new Tensor<T>(new[] { 4 });
        var values = new Tensor<T>(new[] { 7 });
        Tensor<T>[] parameters = [p, sequence, values];
        Tensor<int> indices;
        CompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            // Both operands are unrealized producers, rather than host constants available at trace time.
            indices = engine.TensorSearchSorted(engine.TensorAddScalar(sequence, ops.Zero),
                engine.TensorAddScalar(values, ops.Zero), right);
            var coefficient = engine.ReduceSum(ForwardOnlyHeterogeneousTrainingTests.DataCast<int, T>(engine, indices), null, false);
            plan = scope.CompileTraining(parameters, engine.ReduceSum(engine.TensorMultiply(p, coefficient), null, false));
            Assert.Null(scope.HostReadOp);
        }
        using (plan)
        {
            Assert.True(plan.HasForwardOnlyHeterogeneousReplay);
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0, weightDecay: 0);
            double[][] sequences = [[-2, 1, 1, 4], [-3, 0, 2, 2], [-2, 1, 4, double.NaN]];
            double[][] queries = [
                [double.NegativeInfinity, -2, 1, 2, 4, double.PositiveInfinity, double.NaN],
                [-3, 0, 1, 2, 3, double.NaN, double.NegativeInfinity],
                [double.NegativeInfinity, -2, 1, 2, 4, double.PositiveInfinity, double.NaN]];
            // torch.searchsorted lower/upper insertion indices, including duplicate edges and non-numeric queries.
            int[][] expected = right
                ? [[0, 1, 3, 3, 4, 4, 4], [1, 2, 2, 4, 4, 4, 0], [0, 1, 2, 2, 4, 4, 4]]
                : [[0, 0, 1, 3, 3, 4, 4], [0, 1, 2, 2, 4, 4, 0], [0, 0, 1, 2, 2, 4, 4]];
            for (int step = 0; step < sequences.Length; step++)
            {
                for (int i = 0; i < 4; i++) sequence[i] = ops.FromDouble(sequences[step][i]);
                for (int i = 0; i < 7; i++) values[i] = ops.FromDouble(queries[step][i]);
                int sum = expected[step].Sum();
                Assert.Equal(ops.FromDouble(2 * sum), plan.Step().ToArray()[0]);
                Assert.Equal(expected[step], indices.ToArray());
                Assert.Equal(ops.FromDouble(sum), plan.Gradients[0].ToArray()[0]);
                Assert.All(plan.Gradients[1].ToArray(), value => Assert.Equal(ops.Zero, value));
                Assert.All(plan.Gradients[2].ToArray(), value => Assert.Equal(ops.Zero, value));
                // A connected zero gradient would still advance AdamW's per-parameter age.
                Assert.Equal(new long[] { step + 1, 0, 0 }, plan.ParameterAcceptedUpdates);
            }
        }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void Ordinary_gpu_replay_refreshes_both_search_operands()
    {
        DirectGpuTensorEngine? device = null;
        try { device = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        using (device)
        {
            bool available = device?.IsGpuAvailable == true;
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                Assert.True(available, "GPU tests were required but no direct GPU backend resolved.");
            Skip.If(!available, "No direct GPU backend available.");
            var engine = device!;
            AiDotNetEngine.Current = engine;
            var p = new Tensor<float>(new[] { 2f }, new[] { 1 });
            p.Gpu();
            var sequence = new Tensor<float>(new[] { 0f, 2f, 4f }, new[] { 3 });
            var values = new Tensor<float>(new[] { 1f, 3f }, new[] { 2 });
            Tensor<int> indices;
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { p }))
            {
                indices = engine.TensorSearchSorted(engine.TensorAddScalar(sequence, 0f), engine.TensorAddScalar(values, 0f));
                var coefficient = engine.ReduceSum(ForwardOnlyHeterogeneousTrainingTests.DataCast<int, float>(engine, indices), null, false);
                plan = scope.CompileTraining(new[] { p }, engine.TensorMultiply(p, coefficient));
                Assert.Null(scope.HostReadOp);
            }
            using (plan)
            {
                Assert.True(plan.HasForwardOnlyHeterogeneousReplay);
                Assert.False(plan.IsGraphStepEligible);
                float[][] sequences = [[0f, 2f, 4f], [10f, 20f, 30f], [0f, 0f, 3f]];
                float[][] queries = [[1f, 3f], [5f, 25f], [0f, 4f]];
                int[][] expected = [[1, 2], [0, 2], [0, 3]];
                for (int step = 0; step < sequences.Length; step++)
                {
                    for (int i = 0; i < 3; i++) sequence[i] = sequences[step][i];
                    for (int i = 0; i < 2; i++) values[i] = queries[step][i];
                    Assert.Equal(2f * expected[step].Sum(), plan.Step().ToArray()[0]);
                    Assert.Equal(expected[step], indices.ToArray());
                    Assert.Equal((float)expected[step].Sum(), plan.Gradients[0].ToArray()[0]);
                }
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Batched_replay_uses_each_current_row(bool right)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var p = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var sequence = new Tensor<double>(new[] { 0.0, 2, 2, 10, 20, 30 }, new[] { 2, 3 });
        var values = new Tensor<double>(new[] { 2.0, 3, double.NaN, 5, 20, 40 }, new[] { 2, 3 });
        Tensor<int> indices;
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(new[] { p }))
        {
            indices = engine.TensorSearchSorted(engine.TensorAddScalar(sequence, 0.0), engine.TensorAddScalar(values, 0.0), right);
            var coefficient = engine.ReduceSum(ForwardOnlyHeterogeneousTrainingTests.DataCast<int, double>(engine, indices), null, false);
            plan = scope.CompileTraining(new[] { p }, engine.TensorMultiply(p, coefficient));
            Assert.Null(scope.HostReadOp);
        }
        using (plan)
        {
            int[] first = right ? [3, 3, 3, 0, 2, 3] : [1, 3, 3, 0, 1, 3];
            Assert.Equal(2.0 * first.Sum(), plan.Step().ToArray()[0]);
            Assert.Equal(first, indices.ToArray());
            sequence[0] = -1; sequence[1] = 0; sequence[2] = 3;
            sequence[3] = -10; sequence[4] = 0; sequence[5] = 100;
            values[0] = -2; values[1] = 0; values[2] = 3;
            values[3] = -10; values[4] = 0; values[5] = double.PositiveInfinity;
            int[] second = right ? [0, 2, 3, 1, 2, 3] : [0, 1, 2, 0, 1, 3];
            Assert.Equal(2.0 * second.Sum(), plan.Step().ToArray()[0]);
            Assert.Equal(second, indices.ToArray());
            Assert.Equal((double)second.Sum(), plan.Gradients[0].ToArray()[0]);
        }
    }

    private sealed class SearchData(float[] sequence, float[] values)
    {
        private readonly Tensor<float> _sequence = new(sequence, new[] { sequence.Length });
        private readonly Tensor<float> _values = new(values, new[] { values.Length });
        internal Func<Tensor<float>> Objective(IEngine engine, Tensor<float> p) => () =>
        {
            var indices = engine.TensorSearchSorted(engine.TensorAddScalar(_sequence, 0f), engine.TensorAddScalar(_values, 0f));
            var coefficient = engine.ReduceSum(ForwardOnlyHeterogeneousTrainingTests.DataCast<int, float>(engine, indices), null, false);
            return engine.TensorMultiply(p, coefficient);
        };
    }

    [Fact]
    public void Closure_guard_rebinds_both_fresh_operands_without_recompilation()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var p = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var initial = new SearchData([0, 2, 4], [1, 3]);
        CompiledTrainingPlan<float> plan;
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        using (var scope = GraphMode.EnableTraining(new[] { p }))
        {
            plan = scope.CompileTraining(new[] { p }, initial.Objective(engine, p)());
            Assert.Null(scope.HostReadOp);
        }
        var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
        Assert.True(guard is not null, reason);
        guard!.AttachTo(plan);
        using (plan)
        {
            foreach (var example in new[] {
                (Data: initial, Coefficient: 3f),
                (Data: new SearchData([10, 20, 30], [5, 25]), Coefficient: 2f),
                (Data: new SearchData([0, 0, 3], [0, 4]), Coefficient: 3f) })
            {
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(example.Data.Objective(engine, p), new[] { p }, out _));
                Assert.Equal(2 * example.Coefficient, plan.Step().ToArray()[0]);
                Assert.Equal(example.Coefficient, plan.Gradients[0].ToArray()[0]);
            }
        }
    }

    [Fact]
    public void Second_operand_is_read_before_a_later_in_place_write()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var p = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var sequence = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var values = new Tensor<float>(new[] { 0f }, new[] { 1 });
        var increment = new Tensor<float>(new[] { 2f }, new[] { 1 });
        Tensor<int> indices;
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { p }))
        {
            indices = engine.TensorSearchSorted(sequence, values);
            engine.TensorAddInPlace(values, increment);
            var coefficient = engine.TensorAdd(ForwardOnlyHeterogeneousTrainingTests.DataCast<int, float>(engine, indices), values);
            plan = scope.CompileTraining(new[] { p }, engine.TensorMultiply(p, coefficient));
            Assert.Null(scope.HostReadOp);
        }
        using (plan)
        {
            int[] expectedIndices = [0, 1, 3];
            for (int step = 0; step < 3; step++)
            {
                float updated = 2 * (step + 1), coefficient = expectedIndices[step] + updated;
                Assert.Equal(2 * coefficient, plan.Step().ToArray()[0]);
                Assert.Equal(expectedIndices[step], indices.ToArray()[0]);
                Assert.Equal(updated, values.ToArray()[0]);
                Assert.Equal(coefficient, plan.Gradients[0].ToArray()[0]);
            }
        }
    }

    [Fact]
    public void Plan_leases_second_operand_after_its_caller_arena_is_reset()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var p = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var sequence = new Tensor<float>(new[] { 0f, 1f }, new[] { 2 });
        Tensor<float> values;
        CompiledTrainingPlan<float> plan;
        using (var arena = TensorArena.Create())
        {
            values = arena.TryRentTensor<float>(1024, new[] { 1024 })!;
            Assert.NotNull(values);
            values.AsWritableSpan().Fill(0.5f);
            using (var scope = GraphMode.EnableTraining(new[] { p }))
            {
                var indices = engine.TensorSearchSorted(sequence, values);
                var coefficient = engine.ReduceSum(ForwardOnlyHeterogeneousTrainingTests.DataCast<int, float>(engine, indices), null, false);
                plan = scope.CompileTraining(new[] { p }, engine.TensorMultiply(p, coefficient));
                Assert.Null(scope.HostReadOp);
            }
            arena.Reset();
        }
        using (plan)
        {
            Assert.True(values._storage.RefCount >= 2);
            using (var reuse = TensorArena.Create())
                for (int i = 0; i < 4; i++) reuse.TryRentTensor<float>(1024, new[] { 1024 })!.AsWritableSpan().Fill(-99f);
            Assert.Equal(2048f, plan.Step().ToArray()[0]);
            Assert.Equal(1024f, plan.Gradients[0].ToArray()[0]);
            Assert.All(values.ToArray(), value => Assert.Equal(0.5f, value));
        }
    }
}
