using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledCpuScalarBackwardTests : IDisposable
{
    private readonly IEngine _previous = AiDotNetEngine.Current;
    public CompiledCpuScalarBackwardTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _previous;

    [Theory]
    [InlineData(17, 37, false, 0)]
    [InlineData(32, 512, true, 0)]
    [InlineData(17, 37, false, 4)]
    [InlineData(17, 37, true, 4)]
    public void ScalarMatrixMatchesCanonicalProductsAndReductionWithOffsetsAndOverlap(int rows, int cols, bool add, int overlap)
    {
        int length = rows * cols;
        var engine = new CpuEngine();
        var matrix = Tensor<float>.FromMemory(new float[length + 2].AsMemory(1, length), new[] { rows, cols });
        var scalar = new Tensor<float>(new[] { -.1734567f }, new[] { 1, 1 });
        var backing = Enumerable.Repeat(.27f, length + 16).ToArray();
        var upstream = Tensor<float>.FromMemory(backing.AsMemory(1, length), new[] { rows, cols });
        var target = overlap == 0 ? new Tensor<float>(new[] { rows, cols })
            : Tensor<float>.FromMemory(backing.AsMemory(1 + overlap, length), new[] { rows, cols });
        for (int i = 0; i < length; i++) { matrix[i] = (float)(Math.Sin(i * .17) * (i % 7 + 1)); upstream[i] = (float)(Math.Cos(i * .13) * .003); }
        if (overlap == 0) target.Fill(.27f);
        var scalarTarget = new Tensor<float>(new[] { .31f }, new[] { 1, 1 });
        var original = target.ToArray();
        var product = engine.TensorMultiply(upstream, matrix);
        var expectedScalar = engine.ReduceSum(product, new[] { 0, 1 }, true).ToArray()[0];
        var expectedMatrix = engine.TensorMultiply(upstream, scalar).ToArray();
        var output = new Tensor<float>(new[] { rows, cols });
        var step = new CompiledStep<float>("TensorBroadcastMultiply", (_, _) => { }, output, new[] { matrix, scalar });
        var map = new Dictionary<Tensor<float>, Tensor<float>> { [matrix] = target, [scalar] = scalarTarget, [output] = upstream };
        var action = CompiledTrainingPlan<float>.BuildCpuScalarMultiplyBackward(step, map,
            new Dictionary<Tensor<float>, int> { [matrix] = add ? 2 : 1, [scalar] = add ? 2 : 1 }, engine, null);
        Assert.NotNull(action);
        action!(engine);
        for (int i = 0; i < length; i++) AssertBits(add ? original[i] + expectedMatrix[i] : expectedMatrix[i], target[i]);
        AssertBits(add ? .31f + expectedScalar : expectedScalar, scalarTarget[0]);
        Assert.Equal(.27f, backing[0]);
        Assert.Equal(.27f, backing[^1]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FullMeanPreservesCanonicalScaleRoundingAndOffsetAccumulation(bool add)
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 3, 7, 13 });
        var output = new Tensor<float>(Array.Empty<int>());
        var upstream = new Tensor<float>(new[] { .31415927f }, Array.Empty<int>());
        var backing = Enumerable.Repeat(.27f, input.Length + 2).ToArray();
        var target = Tensor<float>.FromMemory(backing.AsMemory(1, input.Length), input.Shape.ToArray());
        var expected = engine.ReduceMeanBackward(upstream, input.Shape.ToArray(), new[] { -1, 0, 1 }).ToArray();
        var step = new CompiledStep<float>("ReduceMean", (_, _) => { }, output, new[] { input }, savedState: new object[] { new[] { -1, 0, 1 }, false });
        var map = new Dictionary<Tensor<float>, Tensor<float>> { [input] = target, [output] = upstream };
        var action = CompiledTrainingPlan<float>.BuildCpuFullMeanBackward(step, map,
            new Dictionary<Tensor<float>, int> { [input] = add ? 2 : 1 }, engine, null);
        Assert.NotNull(action); action!(engine);
        for (int i = 0; i < input.Length; i++) AssertBits(add ? .27f + expected[i] : expected[i], target[i]);
        Assert.Equal(.27f, backing[0]); Assert.Equal(.27f, backing[^1]);
    }

    [Fact]
    public void FullMeanDoubleMatchesCanonicalAndRejectsPartialReduction()
    {
        var engine = new CpuEngine(); var input = new Tensor<double>(new[] { 3, 7 });
        var output = new Tensor<double>(Array.Empty<int>()); var upstream = new Tensor<double>(new[] { .123456789 }, Array.Empty<int>());
        var target = new Tensor<double>(input.Shape.ToArray());
        var map = new Dictionary<Tensor<double>, Tensor<double>> { [input] = target, [output] = upstream };
        var step = new CompiledStep<double>("ReduceMean", (_, _) => { }, output, new[] { input }, savedState: new object[] { new[] { 0, 1 }, false });
        var action = CompiledTrainingPlan<double>.BuildCpuFullMeanBackward(step, map, new(), engine, null);
        Assert.NotNull(action); action!(engine);
        var expected = engine.ReduceMeanBackward(upstream, input.Shape.ToArray(), new[] { 0, 1 }).ToArray();
        Assert.Equal(expected, target.ToArray());
        var partial = new CompiledStep<double>("ReduceMean", (_, _) => { }, output, new[] { input }, savedState: new object[] { new[] { 0 }, false });
        Assert.Null(CompiledTrainingPlan<double>.BuildCpuFullMeanBackward(partial, map, new(), engine, null));
    }

    [Fact]
    public void CompiledSharedInputsRemainBitwiseCorrectOnRepeatedReplays()
    {
        var engine = new CpuEngine();
        var matrix = new Tensor<float>(new[] { 17, 37 }); var scalar = new Tensor<float>(new[] { .37f }, new[] { 1, 1 });
        for (int i = 0; i < matrix.Length; i++) matrix[i] = (float)Math.Sin(i * .07);
        ICompiledTrainingPlan<float> plan;
        using (var graph = GraphMode.Enable())
        {
            var scaled = engine.TensorBroadcastMultiply(matrix, scalar);
            var first = engine.ReduceMean(scaled, Array.Empty<int>(), false);
            var second = engine.ReduceMean(matrix, Array.Empty<int>(), false);
            plan = graph.CompileTraining(new[] { matrix, scalar }, engine.TensorAdd(first, second));
        }
        using (plan) for (int replay = 0; replay < 5; replay++)
        {
            scalar[0] += .013f; matrix[replay] -= .019f;
            plan.Step();
            var seed = new Tensor<float>(new[] { 1f }, Array.Empty<int>());
            var dy = engine.ReduceMeanBackward(seed, matrix.Shape.ToArray(), new[] { 0, 1 });
            var expectedMatrix = engine.TensorAdd(engine.TensorMultiply(dy, scalar), dy).ToArray();
            var expectedScalar = engine.ReduceSum(engine.TensorMultiply(dy, matrix), new[] { 0, 1 }, true)[0];
            for (int i = 0; i < matrix.Length; i++) AssertBits(expectedMatrix[i], plan.Gradients[0][i]);
            AssertBits(expectedScalar, plan.Gradients[1][0]);
        }
    }

    [Fact]
    public void ReplayingScalarVjpDoesNotAllocateMatrixSizedProducts()
    {
        var engine = new CpuEngine(); var shape = new[] { 128, 4096 };
        var matrix = new Tensor<float>(shape); var scalar = new Tensor<float>(new[] { .75f }, new[] { 1, 1 });
        var output = new Tensor<float>(shape); var upstream = new Tensor<float>(shape); upstream.Fill(.125f);
        var target = new Tensor<float>(shape); var scalarTarget = new Tensor<float>(new[] { 1, 1 });
        var step = new CompiledStep<float>("TensorBroadcastMultiply", (_, _) => { }, output, new[] { matrix, scalar });
        var map = new Dictionary<Tensor<float>, Tensor<float>> { [matrix] = target, [scalar] = scalarTarget, [output] = upstream };
        var action = CompiledTrainingPlan<float>.BuildCpuScalarMultiplyBackward(step, map, new(), engine, null)!;
        action(engine); action(engine);
        long before = GC.GetAllocatedBytesForCurrentThread();
        action(engine);
        Assert.InRange(GC.GetAllocatedBytesForCurrentThread() - before, 0, 65536);
        AssertBits(.09375f, target[0]);
    }

    private static void AssertBits(float expected, float actual) => Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual));
}
