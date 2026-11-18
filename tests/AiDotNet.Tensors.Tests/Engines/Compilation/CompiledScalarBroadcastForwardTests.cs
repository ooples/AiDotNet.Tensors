using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledScalarBroadcastForwardTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledScalarBroadcastForwardTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    private static CompiledStep<T> Step<T>(Tensor<T> left, Tensor<T> right, Tensor<T> output)
        => new("TensorBroadcastMultiply", (engine, destination) =>
        {
            var value = engine.TensorMultiply(left, right);
            value.AsSpan().CopyTo(destination.AsWritableSpan());
        }, output, new[] { left, right });

    private static Action<IEngine> Replay<T>(Tensor<T> left, Tensor<T> right, Tensor<T> output)
        => CompiledTrainingPlan<T>.TryBuildSpecializedForward(Step(left, right, output))
            ?? throw new InvalidOperationException("Dense scalar-right multiply must have a planned replay.");

    [Theory]
    [InlineData(7, 19)]
    [InlineData(1024, 5120)]
    public void SpectralWeightRowsFloatOverwriteTheWholeOutputOnEveryReplay(int rows, int columns)
    {
        var left = new Tensor<float>(Enumerable.Range(0, rows * columns).Select(i => (i % 127 - 63) * .03125f).ToArray(), new[] { rows, columns });
        var scalar = new Tensor<float>(new[] { .75f }, new[] { 1, 1 });
        var output = new Tensor<float>(left.Shape);
        var replay = Replay(left, scalar, output);
        foreach (float scale in new[] { .75f, -.125f, 0f })
        {
            scalar[0] = scale;
            left[0] += .25f;
            output.Fill(float.NaN);
            replay(new CpuEngine());
            for (int i = 0; i < left.Length; i++)
                Assert.Equal(left[i] * scale, output[i]);
        }
    }

    [Fact]
    public void DoubleScalarAndSourceChangesRemainVisible()
    {
        var left = new Tensor<double>(new[] { 2.5, -4.0, 1e-100, -7.25, 0.0, 9.0 }, new[] { 2, 3 });
        var scalar = new Tensor<double>(new[] { -1.125 }, new[] { 1, 1 });
        var output = new Tensor<double>(left.Shape);
        var replay = Replay(left, scalar, output);
        foreach (double scale in new[] { -1.125, .375 })
        {
            scalar[0] = scale;
            left[3] += 1;
            replay(new CpuEngine());
            Assert.Equal(left.ToArray().Select(value => value * scale), output.ToArray());
        }
    }

    [Fact]
    public void ScalarAliasingTheInPlaceDestinationIsReadBeforeAnyWrite()
    {
        float[] shared = { 2, -3, 4, -5, 6, -7, 8, -9 };
        var dense = Tensor<float>.FromMemory(shared.AsMemory(), new[] { 2, 4 });
        var scalar = Tensor<float>.FromMemory(shared.AsMemory(0, 1), new[] { 1, 1 });
        var expected = shared.Select(value => value * shared[0]).ToArray();
        Replay(dense, scalar, dense)(new CpuEngine());
        Assert.Equal(expected, shared);
    }

    [Fact]
    public void PartialOverlapAndStridesRetainTheExistingFallback()
    {
        float[] shared = Enumerable.Range(0, 14).Select(i => i + .5f).ToArray();
        var source = Tensor<float>.FromMemory(shared.AsMemory(0, 12), new[] { 3, 4 });
        var target = Tensor<float>.FromMemory(shared.AsMemory(1, 12), new[] { 3, 4 });
        var scalar = new Tensor<float>(new[] { -.75f }, new[] { 1, 1 });
        var original = shared.ToArray();
        var step = Step(source, scalar, target);
        Assert.Null(CompiledTrainingPlan<float>.TryBuildSpecializedForward(step));
        Assert.Equal(original, shared);
        step.Execute(new CpuEngine(), target);
        Assert.Equal(original.Take(12).Select(value => value * -.75f), target.ToArray());

        var strided = source.Transpose(new[] { 1, 0 });
        var output = new Tensor<float>(strided.Shape);
        output.Fill(777);
        Assert.Null(CompiledTrainingPlan<float>.TryBuildSpecializedForward(Step(strided, scalar, output)));
        Assert.All(output.ToArray(), value => Assert.Equal(777f, value));
    }

    [Fact]
    public void CustomEngineMultiplyOverrideRemainsAuthoritative()
    {
        var engine = new CountingMultiplyEngine();
        var left = new Tensor<float>(new[] { 1f, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var scalar = new Tensor<float>(new[] { -.5f }, new[] { 1, 1 });
        var output = new Tensor<float>(left.Shape);
        var replay = Replay(left, scalar, output);
        replay(engine);
        Assert.Equal(1, engine.MultiplyCalls);
        Assert.Equal(new[] { -.5f, -1, -1.5f, -2, -2.5f, -3 }, output.ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void CompiledForwardAndGradientsRemainCorrectForOtherBroadcastOrientations(int pattern)
    {
        int[] leftShape = pattern == 0 ? new[] { 1, 1 } : pattern == 1 ? new[] { 4, 5 } : new[] { 4, 1 };
        int[] rightShape = pattern == 0 ? new[] { 4, 5 } : pattern == 1 ? new[] { 4, 1 } : new[] { 1, 5 };
        var left = new Tensor<float>(leftShape);
        var right = new Tensor<float>(rightShape);
        var probe = new Tensor<float>(new[] { 4, 5 });
        var engine = new CpuEngine();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { left, right, probe }))
        {
            var value = engine.TensorMultiply(left, right);
            var loss = engine.ReduceSum(engine.TensorMultiply(value, probe), null);
            plan = scope.CompileTraining(new[] { left, right, probe }, loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int step = 0; step < 3; step++)
            {
                left.Fill(step + 1);
                right.Fill(-step - .5f);
                probe.Fill(.25f);
                plan.Step();
                Assert.All(plan.Gradients[2].ToArray(), value => Assert.Equal((step + 1) * (-step - .5f), value));
                Assert.All(plan.Gradients[0].ToArray(), value => Assert.Equal((20f / left.Length) * (-step - .5f) * .25f, value));
                Assert.All(plan.Gradients[1].ToArray(), value => Assert.Equal((20f / right.Length) * (step + 1) * .25f, value));
            }
        }
    }

    [Fact]
    public void ScalarRightCompiledTrainingRebindsBothParametersAndInitializesGradientsEachStep()
    {
        var engine = new CpuEngine();
        var dense = new Tensor<float>(new[] { 3, 7 });
        var scalar = new Tensor<float>(new[] { 1, 1 });
        var probe = new Tensor<float>(dense.Shape);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { dense, scalar, probe }))
        {
            var value = engine.TensorMultiply(dense, scalar);
            var loss = engine.ReduceSum(engine.TensorMultiply(value, probe), null);
            plan = scope.CompileTraining(new[] { dense, scalar, probe }, loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int step = 0; step < 3; step++)
            {
                dense.Fill(step + 1);
                scalar[0] = -.5f - step;
                probe.Fill(.25f);
                Assert.Equal(21 * (step + 1) * scalar[0] * .25f, plan.Step()[0]);
                Assert.All(plan.Gradients[0].ToArray(), value => Assert.Equal(scalar[0] * .25f, value));
                Assert.Equal(21 * (step + 1) * .25f, plan.Gradients[1][0]);
                Assert.All(plan.Gradients[2].ToArray(), value => Assert.Equal((step + 1) * scalar[0], value));
            }
        }
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void LargeScalarReplayAllocationsDoNotScaleWithTheWeightMatrix()
    {
        static long Allocated(int columns)
        {
            var source = new Tensor<float>(new[] { 32, columns });
            var scalar = new Tensor<float>(new[] { .75f }, new[] { 1, 1 });
            var output = new Tensor<float>(source.Shape);
            var replay = Replay(source, scalar, output);
            var engine = new CpuEngine();
            for (int warm = 0; warm < 4; warm++) replay(engine);
            long before = GC.GetTotalAllocatedBytes(true);
            for (int step = 0; step < 8; step++) replay(engine);
            return GC.GetTotalAllocatedBytes(true) - before;
        }
        long small = Allocated(64), large = Allocated(65536);
        Assert.True(large <= small + 128 * 1024,
            $"Eight scalar replays of an8MiB weight allocated {large:N0}B versus {small:N0}B for the small matrix.");
    }
#endif

    private sealed class CountingMultiplyEngine : CpuEngine
    {
        internal int MultiplyCalls;
        public override Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
        {
            MultiplyCalls++;
            return base.TensorMultiply(a, b);
        }
    }
}
