using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// index_copy and scatter are last-write-wins along the axis, so with duplicate indices only the LAST entry per
/// target reaches the output. Its gradient is the upstream gradient there; every overwritten entry's is zero.
/// </summary>
public class IndexCopyDuplicateIndicesGradientTests : IDisposable
{
    private readonly IEngine _previousEngine = AiDotNetEngine.Current;

    public void Dispose() => AiDotNetEngine.Current = _previousEngine;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Overwritten_entries_get_zero_gradient(bool scatter)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var destination = new Tensor<double>(new[] { 2, 5 });
        var source = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < source.Length; i++) source[i] = i + 1;
        var indices = new Tensor<int>(new[] { 1, 4, 1 }, new[] { 3 });

        using var tape = new GradientTape<double>();
        var y = scatter
            ? engine.Scatter(destination, indices, source, 1)
            : engine.TensorIndexCopy(destination, 1, indices, source);
        // The forward is last-write-wins: column 1 holds source column 2.
        Assert.Equal(source[0, 2], y[0, 1]);

        var weight = new Tensor<double>(y.Shape.ToArray());
        for (int i = 0; i < weight.Length; i++) weight[i] = i + 1;
        var loss = engine.ReduceSum(engine.TensorMultiply(y, weight), null);
        var g = tape.ComputeGradients(loss, new[] { source })[source];

        for (int r = 0; r < 2; r++)
        {
            Assert.Equal(0.0, g[r, 0]);              // overwritten by the later write to column 1
            Assert.Equal(weight[r, 4], g[r, 1]);     // column 4
            Assert.Equal(weight[r, 1], g[r, 2]);     // the winning write to column 1
        }
    }
}