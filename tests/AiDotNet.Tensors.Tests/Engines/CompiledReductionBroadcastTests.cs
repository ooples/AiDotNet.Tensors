using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class CompiledReductionBroadcastTests
{
    public static TheoryData<int[], int[], bool> Shapes => new()
    {
        { new[] { 17, 1003 }, new[] { 1 }, true },
        { new[] { 3, 5, 7 }, new[] { -2 }, false },
        { new[] { 3, 5, 7 }, new[] { 0 }, true },
        { new[] { 2, 3, 4, 5 }, new[] { 1, 2 }, false },
        { new[] { 3, 5, 7 }, new[] { 0, 2 }, true }, // non-contiguous axes retain the canonical fallback
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SharedInputGradientsMatchAnalyticProjectionOnEveryReplay(int[] shape, int[] axes, bool keepDims)
    {
        int length = shape.Aggregate(1, (a, b) => a * b);
        var storage = new Tensor<float>(new[] { length + 2 });
        storage.Fill(-123);
        var input = storage.Slice(0, 1, length + 1).Reshape(shape);
        for (int i = 0; i < length; i++) input[i] = (float)(Math.Sin(i * .13) * .2);
        Assert.Equal(1, input.StorageOffset);
        var engine = new CpuEngine();
        ICompiledTrainingPlan<float> plan;
        Tensor<float> coefficients;
        using (var scope = GraphMode.Enable())
        {
            var summed = engine.ReduceSum(input, axes, keepDims);
            coefficients = new Tensor<float>(summed.Shape.ToArray());
            for (int i = 0; i < coefficients.Length; i++) coefficients[i] = .2f + .01f * i;
            var linear = engine.ReduceSum(engine.TensorMultiply(summed, coefficients));
            var quadratic = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(input, input)), .125f);
            var loss = engine.TensorAdd(linear, quadratic);
            plan = scope.CompileTraining(new[] { input }, loss);
        }
        using (plan)
        {
            for (int replay = 0; replay < 3; replay++)
            {
                if (replay > 0)
                    for (int i = 0; i < length; i++) input[i] += .01f;
                plan.Step();
                var actual = plan.Gradients[0].ToArray();
                var reduced = new HashSet<int>(axes.Select(a => a < 0 ? a + shape.Length : a));
                for (int index = 0; index < length; index++)
                {
                    // Project the coordinate onto retained dimensions, independently of the block-broadcast kernel.
                    int remaining = index, projected = 0, retainedStride = 1;
                    for (int axis = shape.Length - 1; axis >= 0; axis--)
                    {
                        int coordinate = remaining % shape[axis];
                        remaining /= shape[axis];
                        if (!reduced.Contains(axis)) { projected += coordinate * retainedStride; retainedStride *= shape[axis]; }
                    }
                    double expected = coefficients[projected] + .25 * input[index];
                    Assert.InRange(Math.Abs(actual[index] - expected), 0, 2e-5 * Math.Max(1, Math.Abs(expected)));
                }
            }
        }
        Assert.Equal(-123f, storage[0]);
        Assert.Equal(-123f, storage[length + 1]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BroadcastWritesOffsetAccumulatorsAndPreservesSentinels(bool accumulate)
    {
        var input = new Tensor<double>(new[] { 3, 5, 7 });
        var output = new Tensor<double>(new[] { 3, 7 });
        var sourceStorage = new Tensor<double>(new[] { 23 });
        var destinationStorage = new Tensor<double>(new[] { 107 });
        sourceStorage.Fill(-50);
        destinationStorage.Fill(.25);
        var upstream = sourceStorage.Slice(0, 1, 22).Reshape(new[] { 3, 7 });
        var destination = destinationStorage.Slice(0, 1, 106).Reshape(input.Shape.ToArray());
        for (int i = 0; i < upstream.Length; i++) upstream[i] = i * .1 - .75;
        var map = new Dictionary<Tensor<double>, Tensor<double>> { [input] = destination, [output] = upstream };
        var writers = new Dictionary<Tensor<double>, int> { [input] = accumulate ? 2 : 1 };
        var step = new CompiledStep<double>("ReduceSum", (_, _) => { }, output, new[] { input },
            savedState: new object[] { new[] { 1 }, false });
        var engine = new CpuEngine();
        var action = CompiledTrainingPlan<double>.BuildCpuReducedGradientBackward(step, map, writers, engine);
        Assert.NotNull(action);
        action!(engine);
        for (int o = 0; o < 3; o++)
            for (int r = 0; r < 5; r++)
                for (int i = 0; i < 7; i++)
                    Assert.Equal(upstream[o * 7 + i] + (accumulate ? .25 : 0), destination[(o * 5 + r) * 7 + i], 12);
        Assert.Equal(.25, destinationStorage[0]);
        Assert.Equal(.25, destinationStorage[106]);
        Assert.Equal(-50, sourceStorage[0]);
        Assert.Equal(-50, sourceStorage[22]);
    }

    [Fact]
    public void SteadyReductionReplayDoesNotAllocateInputSizedBroadcasts()
    {
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = 1;
        try
        {
            var engine = new CpuEngine();
            var input = new Tensor<float>(new[] { 256, 4096 });
            input.Fill(.1f);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var reduced = engine.ReduceSum(input, new[] { 1 });
                var coefficients = new Tensor<float>(new[] { 256 });
                coefficients.Fill(.2f);
                var loss = engine.ReduceSum(engine.TensorMultiply(reduced, coefficients));
                plan = scope.CompileTraining(new[] { input }, loss);
            }
            using (plan)
            {
                for (int i = 0; i < 4; i++) plan.Step();
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 3; i++) plan.Step();
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                Assert.True(allocated < 512 * 1024, $"Three steady steps allocated {allocated:N0} bytes; an input-sized broadcast is4MiB.");
                Assert.InRange(Math.Abs(plan.Gradients[0][12345] - .2f), 0, 1e-6);
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previous; }
    }
}
