using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class SliceMetadataSnapshotGradientTests
{
    [Fact]
    public async Task TensorSlice_BackwardUsesMetadataFromEachForwardCall()
    {
        await Task.Yield();

        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, [2, 2]);
        var start = new[] { 0, 0 };
        var length = new[] { 1, 2 };

        Tensor<float> gradient;
        using (var tape = new GradientTape<float>())
        {
            var firstRow = engine.TensorSlice(input, start, length);
            start[0] = 1;
            var secondRow = engine.TensorSlice(input, start, length);
            var objective = engine.TensorAdd(
                engine.ReduceSum(firstRow, [0, 1], keepDims: false),
                engine.TensorMultiplyScalar(
                    engine.ReduceSum(secondRow, [0, 1], keepDims: false),
                    2f));
            gradient = tape.ComputeGradients(objective, [input])[input];
        }

        Assert.Equal(new[] { 1f, 1f, 2f, 2f }, gradient.ToArray());
    }
}
