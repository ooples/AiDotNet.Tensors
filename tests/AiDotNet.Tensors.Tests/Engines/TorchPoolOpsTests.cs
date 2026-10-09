using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Adaptive, Lp, indexed and fractional pooling and max unpooling against PyTorch float64.</summary>
[Collection("EngineCurrentGlobalState")]
public class TorchPoolOpsTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    // Distinct values (i * mul mod m) / 4 + shift, so no window holds a tie.
    private static Tensor<double> Perm(int[] shape, int mod, int mul, double shift)
        => new Tensor<double>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => (i * mul % mod) * 0.25 + shift).ToArray(), shape);

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-12)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: torch {expected[i]:R}, ours {actual.GetFlat(i):R}");
    }

    private static void Same(int[] expected, Tensor<int> actual, string what)
        => Assert.True(expected.SequenceEqual(actual.ToArray()), $"{what}: torch [{string.Join(", ", expected)}], ours [{string.Join(", ", actual.ToArray())}]");

    [Fact]
    public void AdaptivePooling_MatchesTorch()
    {
        Close(new[] { 1.25, 2.5833333333333335, 3.9166666666666665, 2.3333333333333335, 3.6666666666666665, 1.4166666666666667, 3.4166666666666665, 1.1666666666666667, 2.5, 0.9166666666666666, 2.25, 3.5833333333333335, 2.0, 3.3333333333333335, 1.0833333333333333, 3.0833333333333335, 0.8333333333333334, 2.1666666666666665 }, _engine.TensorAdaptiveAvgPool1D(Perm(new[] { 2, 3, 7 }, 43, 17, -3), 3), "adaptive_avg_pool1d");
        Close(new[] { 5.5, 5.5, 7.5, 5.25, 7.25, 5.0, 7.0, 4.75, 6.75, 4.5, 6.5, 6.5, 6.25, 6.25, 4.0, 6.0, 3.75, 5.75 }, _engine.TensorAdaptiveMaxPool1D(Perm(new[] { 2, 3, 7 }, 43, 17, -3), 3), "adaptive_max_pool1d");
        Close(new[] { 3.75, 6.375, 5.020833333333333, 5.0, 6.291666666666667, 6.270833333333333, 6.1875, 8.8125, 4.8125, 7.4375, 3.4375, 6.0625, 5.770833333333333, 5.75, 7.041666666666667, 7.020833333333333, 8.3125, 5.645833333333333, 5.5625, 5.541666666666667, 4.1875, 6.8125, 5.458333333333333, 5.4375 }, _engine.TensorAdaptiveAvgPool3D(Perm(new[] { 1, 2, 3, 4, 5 }, 127, 37, -10), new[] { 2, 3, 2 }), "adaptive_avg_pool3d");
        Close(new[] { 17.5, 21.5, 19.0, 18.75, 20.25, 20.25, 20.0, 21.5, 17.5, 21.25, 17.25, 18.75, 19.75, 19.5, 21.0, 21.0, 21.0, 21.0, 18.25, 19.5, 18.0, 19.5, 20.75, 19.25 }, _engine.TensorAdaptiveMaxPool3D(Perm(new[] { 1, 2, 3, 4, 5 }, 127, 37, -10), new[] { 2, 3, 2 }), "adaptive_max_pool3d");
        var (o1, i1) = _engine.TensorAdaptiveMaxPoolWithIndices(Perm(new[] { 2, 3, 7 }, 43, 17, -3), new[] { 3 });
        Close(new[] { 5.5, 5.5, 7.5, 5.25, 7.25, 5.0, 7.0, 4.75, 6.75, 4.5, 6.5, 6.5, 6.25, 6.25, 4.0, 6.0, 3.75, 5.75 }, o1, "adaptive_max_pool1d"); Same(new[] { 2, 2, 5, 0, 3, 5, 1, 3, 6, 1, 4, 4, 2, 2, 4, 0, 2, 5 }, i1, "adaptive_max_pool1d indices");
        var (o2, i2) = _engine.TensorAdaptiveMaxPoolWithIndices(Perm(new[] { 1, 2, 5, 7 }, 71, 29, -5), new[] { 3, 4 });
        Close(new[] { 10.25, 9.5, 11.0, 11.0, 12.5, 12.5, 11.75, 11.0, 12.5, 12.5, 9.25, 10.75, 7.5, 12.25, 11.5, 8.25, 10.5, 12.25, 11.25, 11.25, 10.5, 12.0, 12.0, 11.25 }, o2, "adaptive_max_pool2d"); Same(new[] { 7, 2, 12, 12, 22, 22, 17, 12, 22, 22, 24, 34, 1, 9, 4, 6, 21, 9, 26, 26, 21, 31, 31, 26 }, i2, "adaptive_max_pool2d indices");
        var (o3, i3) = _engine.TensorAdaptiveMaxPoolWithIndices(Perm(new[] { 1, 2, 3, 4, 5 }, 127, 37, -10), new[] { 2, 3, 2 });
        Close(new[] { 17.5, 21.5, 19.0, 18.75, 20.25, 20.25, 20.0, 21.5, 17.5, 21.25, 17.25, 18.75, 19.75, 19.5, 21.0, 21.0, 21.0, 21.0, 18.25, 19.5, 18.0, 19.5, 20.75, 19.25 }, o3, "adaptive_max_pool3d"); Same(new[] { 27, 24, 10, 34, 17, 17, 41, 24, 27, 48, 51, 34, 5, 29, 12, 12, 12, 12, 22, 29, 46, 29, 36, 53 }, i3, "adaptive_max_pool3d indices");
    }

    [Fact]
    public void MaxPool1DWithIndices_MatchesTorch()
    {
        var (o, i) = _engine.TensorMaxPool1DWithIndices(Perm(new[] { 2, 3, 7 }, 43, 17, -3), 3, 2, 1, 1, ceilMode: true);
        Close(new[] { 1.25, 5.5, 7.5, 7.5, 5.25, 7.25, 7.25, 5.0, 7.0, 7.0, 4.75, 6.75, 4.5, 4.5, 6.5, 4.25, 2.0, 6.25, 4.0, 1.75, 6.0, 3.75, 5.75, 5.75 }, o, "max_pool1d ceil"); Same(new[] { 1, 2, 5, 5, 0, 3, 3, 5, 1, 1, 3, 6, 1, 1, 4, 6, 1, 2, 4, 6, 0, 2, 5, 5 }, i, "max_pool1d ceil indices");
        var (ob, ib) = _engine.TensorMaxPool1DWithIndices(Perm(new[] { 2, 3, 7 }, 43, 17, -3), 2, 3, 0, 2);
        Close(new[] { 5.5, 7.5, 5.25, 7.25, 2.75, 4.75, 0.25, 2.25, 6.25, -0.25, 6.0, 5.75 }, ob, "max_pool1d dilated"); Same(new[] { 2, 5, 0, 3, 0, 3, 0, 3, 2, 3, 0, 5 }, ib, "max_pool1d dilated indices");
    }

    [Fact]
    public void LpPool_MatchesTorch()
    {
        Close(new[] { 4.751846004557954, 9.063846411131628, 11.789253058642961, 8.799313393399654, 11.490027586642153, 8.840356247970591, 11.191590267010803, 8.552230304021698, 6.049218987130381, 8.265733672633749, 5.783827071176623, 10.140921452623033, 5.521438365715662, 9.868914158595903, 7.505921249868998, 9.598694122139385, 7.252674399783512, 9.713373430095206 }, _engine.TensorLpPool(Perm(new[] { 2, 3, 7 }, 43, 17, 0.5), 3.0, new[] { 2 }), "lp_pool1d", 1e-11);
        Close(new[] { 17.583728273605686, 20.27775875189366, 23.23117517475171, 21.10983183258455, 22.172618248641726, 19.587942719948924, 22.93877721239735, 25.479158149358074 }, _engine.TensorLpPool(Perm(new[] { 1, 2, 5, 6 }, 61, 23, 0.5), 2.0, new[] { 2, 3 }, new[] { 2, 2 }), "lp_pool2d", 1e-11);
    }

    [Fact]
    public void MaxUnpool_InvertsMaxPoolIndices_LikeTorch()
    {
        var pooled = new Tensor<double>(new[] { 10.25, 9.5, 11.0, 12.5, 11.75, 8.5, 7.5, 12.25, 11.5, 10.5, 9.75, 11.25 }, new[] { 1, 2, 2, 3 });
        var indices = new Tensor<int>(new[] { 7, 2, 12, 22, 17, 19, 1, 9, 4, 21, 16, 26 }, new[] { 1, 2, 2, 3 });
        Close(new[] { 0.0, 0.0, 9.5, 0.0, 0.0, 0.0, 0.0, 10.25, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0, 11.75, 0.0, 8.5, 0.0, 0.0, 12.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 11.5, 0.0, 0.0, 0.0, 0.0, 12.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.75, 0.0, 0.0, 0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 0.0, 11.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, _engine.TensorMaxUnpool(pooled, indices, new[] { 5, 7 }), "max_unpool2d");
    }

    [Fact]
    public void PoolingGradients_MatchTorch()
    {
        var x = Perm(new[] { 2, 3, 7 }, 43, 17, -3);
        using (var tape = new GradientTape<double>())
        {
            var loss = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorAdaptiveAvgPool1D(x, 3), Perm(new[] { 2, 3, 3 }, 7, 3, 0.5)), null, false);
            Close(new[] { 0.16666666666666666, 0.16666666666666666, 0.5833333333333334, 0.4166666666666667, 1.0833333333333333, 0.6666666666666666, 0.6666666666666666, 0.3333333333333333, 0.3333333333333333, 0.9166666666666667, 0.5833333333333334, 0.8333333333333334, 0.25, 0.25, 0.5, 0.5, 0.6666666666666666, 0.16666666666666666, 0.5833333333333334, 0.4166666666666667, 0.4166666666666667, 0.6666666666666666, 0.6666666666666666, 1.0, 0.3333333333333333, 0.9166666666666667, 0.5833333333333334, 0.5833333333333334, 0.25, 0.25, 0.75, 0.5, 0.6666666666666666, 0.16666666666666666, 0.16666666666666666, 0.4166666666666667, 0.4166666666666667, 1.0833333333333333, 0.6666666666666666, 1.0, 0.3333333333333333, 0.3333333333333333 }, tape.ComputeGradients(loss, new[] { x })[x], "adaptive_avg_pool1d gradient");
        }
        var xp = Perm(new[] { 1, 2, 5, 6 }, 61, 23, 0.5);
        using (var tape = new GradientTape<double>())
        {
            var loss = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLpPool(xp, 2.0, new[] { 2, 3 }, new[] { 2, 2 }), Perm(new[] { 1, 2, 2, 2 }, 5, 3, 0.5)), null, false);
            Close(new[] { 0.014217690134308215, 0.1777211266788527, 1.0809512846787062, 0.15410973363652275, 0.5085621210005251, 0.0, 0.12795921120877393, 0.2914626477533184, 0.06755945529241913, 0.4006853074549591, 0.7551376948189614, 0.0, 0.274415734548312, 0.4600499079192289, 0.4908704221696128, 0.7460978431712912, 0.07105693744488488, 0.0, 0.40355255080634117, 0.09685261219352187, 0.9042349882071814, 1.0303255929508308, 0.3552846872244244, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23677853202210838, 0.4961074004272747, 0.10593986932825625, 0.1850628241988984, 0.33183678821871443, 0.0, 0.4171812230865719, 0.6765100914917382, 0.3884461875369396, 0.28716645134311825, 0.04467033687559617, 0.0, 0.7220306403712197, 0.2043482944446848, 0.7973226683513757, 0.44889630705040223, 0.1692559846255615, 0.0, 0.10898575703716523, 0.4223198085190153, 1.1330374760782707, 0.11774329365256452, 0.286999278278126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, tape.ComputeGradients(loss, new[] { xp })[xp], "lp_pool2d gradient", 1e-11);
        }
    }

    [Fact]
    public void FractionalMaxPool_PicksEachWindowMax_AndIsSeeded()
    {
        var x = Perm(new[] { 2, 3, 9, 8 }, 211, 97, -20);
        var (output, indices) = _engine.TensorFractionalMaxPool(x, new[] { 2, 2 }, new[] { 5, 4 }, seed: 7);
        Assert.Equal(new[] { 2, 3, 5, 4 }, output.Shape.ToArray());
        var values = x.ToArray();
        for (int i = 0; i < output.Length; i++)
        {
            int plane = i / 20, at = indices.GetFlat(i);
            Assert.InRange(at, 0, 71);
            Assert.Equal(values[plane * 72 + at], output.GetFlat(i));
        }
        // PyTorch's interval sequence ends flush with the input, so each plane's last output row reads input row 7 or 8.
        for (int plane = 0; plane < 6; plane++)
            Assert.InRange(indices.GetFlat(plane * 20 + 19) / 8, 7, 8);
        var (again, againIndices) = _engine.TensorFractionalMaxPool(x, new[] { 2, 2 }, new[] { 5, 4 }, seed: 7);
        Assert.Equal(output.ToArray(), again.ToArray());
        Assert.Equal(indices.ToArray(), againIndices.ToArray());
    }

    [Fact]
    public void PoolingArguments_AreValidated()
    {
        var x = Perm(new[] { 1, 2, 5 }, 11, 3, 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorMaxPool1DWithIndices(x, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorMaxPool1DWithIndices(x, 2, dilation: 0));
        // torch: padding at most half the effective kernel (kernel 2 allows 1, not 2).
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorMaxPool1DWithIndices(x, 2, padding: 2));
        Assert.Throws<ArgumentException>(() => _engine.TensorMaxPool1DWithIndices(x, 3, dilation: 3));
        Assert.Throws<ArgumentException>(() => _engine.TensorLpPool(x, 2.0, new[] { 2 }, new[] { 1, 1 }));
        Assert.Throws<ArgumentException>(() => _engine.TensorLpPool(x, 2.0, new[] { 6 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => _engine.TensorLpPool(x, 0.0, new[] { 2 }));
        Assert.Throws<ArgumentException>(() => _engine.TensorMaxUnpool(
            new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 }), new Tensor<int>(new[] { 0, 1 }, new[] { 2 }), new[] { 2, 2 }));
        // A leading (batch or channel) axis is required, as for every pooling op and PyTorch's max_unpool.
        Assert.Throws<ArgumentException>(() => _engine.TensorMaxUnpool(
            new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 }), new Tensor<int>(new[] { 0, 2, 4 }, new[] { 3 }), new[] { 7 }));
        Assert.Equal(new[] { 1, 7 }, _engine.TensorMaxUnpool(
            new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 1, 3 }), new Tensor<int>(new[] { 0, 2, 4 }, new[] { 1, 3 }), new[] { 7 })._shape);
    }
}
