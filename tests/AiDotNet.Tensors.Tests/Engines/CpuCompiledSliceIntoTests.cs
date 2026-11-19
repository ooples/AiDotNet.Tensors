using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public class CpuCompiledSliceIntoTests
{
    [Theory]
    [MemberData(nameof(CpuSliceConcatBlockCopyTests.SliceBoxes), MemberType = typeof(CpuSliceConcatBlockCopyTests))]
    public void PlannedDestinationMatchesExistingSliceForEveryBox(int[] shape, int[] start, int[] length)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var input = new Tensor<float>(Enumerable.Range(0, count).Select(i => i * .25f - 17).ToArray(), shape);
        var expected = new CpuEngine().TensorSlice(input, start, length).ToArray();
        var destination = new Tensor<float>(length);
        destination.Fill(float.NaN);
        Assert.True(CpuEngine.TryCopyCompiledSliceInto(input, start, length, destination));
        Assert.Equal(expected, destination.ToArray());
    }

    [Fact]
    public void OffsetMemoryAndPartialRowsPreserveOutputSentinels()
    {
        var storage = new Tensor<double>(Enumerable.Range(0, 64).Select(i => (double)i).ToArray(), new[] { 64 });
        var input = storage.Slice(0, 2, 62).Reshape(new[] { 3, 4, 5 });
        var targetStorage = new Tensor<double>(new[] { 16 });
        targetStorage.Fill(999);
        var target = targetStorage.Slice(0, 2, 14).Reshape(new[] { 2, 2, 3 });
        Assert.Equal(2, input.StorageOffset);
        Assert.True(CpuEngine.TryCopyCompiledSliceInto(input, new[] { 1, 1, 1 }, new[] { 2, 2, 3 }, target));
        Assert.Equal(new double[] { 28, 29, 30, 33, 34, 35, 48, 49, 50, 53, 54, 55 }, target.ToArray());
        Assert.Equal(999, targetStorage[0]);
        Assert.Equal(999, targetStorage[1]);
        Assert.Equal(999, targetStorage[14]);
        Assert.Equal(999, targetStorage[15]);
    }

    [Fact]
    public void OverlapAndStridedLayoutLeaveDestinationUntouchedForFallback()
    {
        var input = new Tensor<float>(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), new[] { 3, 4 });
        var original = input.ToArray();
        Assert.False(CpuEngine.TryCopyCompiledSliceInto(input, new[] { 0, 0 }, new[] { 3, 4 }, input));
        Assert.Equal(original, input.ToArray());
        var strided = input.Transpose(new[] { 1, 0 });
        var target = new Tensor<float>(new[] { 2, 2 });
        target.Fill(777);
        Assert.False(CpuEngine.TryCopyCompiledSliceInto(strided, new[] { 0, 0 }, new[] { 2, 2 }, target));
        Assert.All(target.ToArray(), value => Assert.Equal(777, value));
    }

    [Fact]
    public void RepeatedCopiesDoNotAllocateAnotherSlicePayload()
    {
        static long Allocated(int columns)
        {
            var input = new Tensor<float>(new[] { 32, columns });
            var output = new Tensor<float>(new[] { 16, columns / 2 });
            int[] start = { 3, 1 };
            int[] length = { 16, columns / 2 };
            for (int i = 0; i < 4; i++) Assert.True(CpuEngine.TryCopyCompiledSliceInto(input, start, length, output));
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 8; i++)
                if (!CpuEngine.TryCopyCompiledSliceInto(input, start, length, output)) throw new InvalidOperationException();
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }
        long small = Allocated(64), large = Allocated(65536);
        Assert.True(large <= small + 4096,
            $"Eight copies of a2MiB slice allocated {large:N0}B, versus {small:N0}B for a small slice.");
    }

    [Fact]
    public void CompiledReplayUsesChangingInputAndPublishesSliceGradient()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 4, 6 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            _ = engine.ReduceSum(engine.TensorSlice(input, new[] { 1, 2 }, new[] { 2, 3 }), null);
            plan = scope.CompileTraining(new[] { input });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int step = 0; step < 3; step++)
            {
                input.Fill(step + 1);
                Assert.Equal(6 * (step + 1), plan.Step()[0]);
                var gradient = input.Grad ?? throw new InvalidOperationException("Input gradient was not published.");
                for (int i = 0; i < 24; i++)
                {
                    bool selected = i / 6 >= 1 && i / 6 < 3 && i % 6 >= 2 && i % 6 < 5;
                    Assert.Equal(selected ? 1f : 0f, gradient[i]);
                }
            }
        }
    }

    [Fact]
    public void CompiledReplayHonorsCustomEngineSliceOverride()
    {
        var engine = new CountingSliceEngine();
        var input = new Tensor<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 4, 6 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            _ = engine.ReduceSum(engine.TensorSlice(input, new[] { 1, 2 }, new[] { 2, 3 }), null);
            plan = scope.CompileTraining(new[] { input });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            int before = engine.Calls;
            for (int step = 0; step < 3; step++) Assert.Equal(72, plan.Step()[0]);
            Assert.True(engine.Calls >= before + 3, "Compiled replay bypassed the custom virtual slice implementation.");
        }
    }

    private sealed class CountingSliceEngine : CpuEngine
    {
        internal int Calls;
        public override Tensor<T> TensorSlice<T>(Tensor<T> tensor, int[] start, int[] length)
        {
            Calls++;
            return base.TensorSlice(tensor, start, length);
        }
    }
}
