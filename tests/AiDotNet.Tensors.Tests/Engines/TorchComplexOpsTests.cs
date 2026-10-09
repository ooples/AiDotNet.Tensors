using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Real/complex conversions against PyTorch float64.</summary>
[Collection("EngineCurrentGlobalState")]
public class TorchComplexOpsTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    [Fact]
    public void ComplexConversions_RoundTripLikeTorch()
    {
        var re = new Tensor<double>(new[] { 1.0, -2.0, 0.5 }, new[] { 3 });
        var im = new Tensor<double>(new[] { 3.0, 0.25, -1.0 }, new[] { 3 });
        var z = _engine.TensorComplex(re, im);
        Assert.Equal(new[] { 3 }, z.Shape.ToArray());
        var pairs = _engine.TensorViewAsReal(z);
        Assert.Equal(new[] { 3, 2 }, pairs.Shape.ToArray());
        Assert.Equal(new[] { 1.0, 3.0, -2.0, 0.25, 0.5, -1.0 }, pairs.ToArray());
        var back = _engine.TensorViewAsComplex(pairs);
        Assert.Equal(re.ToArray(), _engine.TensorReal(back).ToArray());
        Assert.Equal(im.ToArray(), _engine.TensorImag(back).ToArray());
        Assert.Throws<ArgumentException>(() => _engine.TensorViewAsComplex(new Tensor<double>(new[] { 3, 3 })));
    }

    [Fact]
    public void Angle_OfRealInput_MatchesTorch()
    {
        var x = new Tensor<double>(new[] { 1.5, -2.0, 0.0, -0.0, double.NaN, double.NegativeInfinity, double.PositiveInfinity }, new[] { 7 });
        var angle = _engine.TensorAngle(x).ToArray();
        Assert.Equal(new[] { 0.0, Math.PI, 0.0, 0.0 }, angle.Take(4).ToArray());
        Assert.True(double.IsNaN(angle[4]));
        Assert.Equal(new[] { Math.PI, 0.0 }, angle.Skip(5).ToArray());
    }

    [Fact]
    public void Polar_MatchesTorch()
    {
        var z = _engine.NativeComplexFromPolar(new Tensor<double>(new[] { 2.0 }, new[] { 1 }), new Tensor<double>(new[] { 0.5 }, new[] { 1 }));
        Assert.Equal(1.7551651237807455, _engine.TensorReal(z).GetFlat(0), 14);
        Assert.Equal(0.958851077208406, _engine.TensorImag(z).GetFlat(0), 14);
    }
}
