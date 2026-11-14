using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public class ScatterProductZeroDerivativeTests : IDisposable
{
    private readonly IEngine _previous = AiDotNetEngine.Current;
    private readonly CpuEngine _engine = new();

    public ScatterProductZeroDerivativeTests() => AiDotNetEngine.Current = _engine;
    public void Dispose() => AiDotNetEngine.Current = _previous;

    private Tensor<double> Component(Tensor<double> tensor, int index)
    {
        var selector = new Tensor<double>((int[])tensor._shape.Clone());
        selector[index] = 1;
        return _engine.ReduceSum(_engine.TensorMultiply(tensor, selector), null);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ThreeZeroSources_PreserveThirdMixedDerivative(bool includeSelf)
    {
        var self = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var source = new Tensor<double>(new[] { 0.0, 0.0, 0.0 }, new[] { 3 });
        var index = new Tensor<int>(new[] { 0, 0, 0 }, new[] { 3 });
        using var tape = new GradientTape<double>();
        var output = _engine.TensorScatterReduce(self, 0, index, source, ScatterReduceMode.Prod, includeSelf);
        var first = tape.ComputeGradients(_engine.ReduceSum(output, null), new[] { source }, createGraph: true);
        Assert.Equal(0, first[source][0]);
        var second = tape.ComputeGradients(Component(first[source], 0), new[] { source }, createGraph: true);
        Assert.Equal(0, second[source][1]);
        var third = tape.ComputeGradients(Component(second[source], 1), new[] { source }, createGraph: true);
        Assert.Equal(includeSelf ? 2.0 : 1.0, third[source][2], 12);
    }

    [Fact]
    public void ZeroSelfAndZeroSource_PreserveMixedDerivative()
    {
        var self = new Tensor<double>(new[] { 0.0 }, new[] { 1 });
        var source = new Tensor<double>(new[] { 0.0, 3.0 }, new[] { 2 });
        var index = new Tensor<int>(new[] { 0, 0 }, new[] { 2 });
        using var tape = new GradientTape<double>();
        var output = _engine.TensorScatterReduce(self, 0, index, source, ScatterReduceMode.Prod, true);
        var first = tape.ComputeGradients(_engine.ReduceSum(output, null), new[] { source }, createGraph: true);
        var mixed = tape.ComputeGradients(Component(first[source], 0), new[] { self }, createGraph: true);
        Assert.Equal(3.0, mixed[self][0], 12);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MixedCollisionSizesAndUntouchedSlots_NonLeadingAxisMatchFiniteDifferences(bool includeSelf)
    {
        var self = new Tensor<double>(new[] { 0.0, 2.0, 4.0, 1.0, 0.0, 3.0 }, new[] { 2, 3 });
        var source = new Tensor<double>(new[] { 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 4.0, 5.0 }, new[] { 2, 4 });
        var index = new Tensor<int>(new[] { 0, 0, 1, 0, 1, 1, 0, 1 }, new[] { 2, 4 });
        var result = GradCheck.GradGradCheckMany<double>(
            x => _engine.TensorScatterReduce(x[0], 1, index, x[1], ScatterReduceMode.Prod, includeSelf),
            new[] { self, source }, new GradCheckOptions { Engine = _engine });
        Assert.True(result.Passed, result.ToString());
    }
}
