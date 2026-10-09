using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NN.Losses;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Random sampling (by its moments), dropout variants and the composed ops against PyTorch float64.</summary>
[Collection("EngineCurrentGlobalState")]
public class TorchRandomOpsTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static Tensor<double> Range(int[] shape, double scale, double shift)
        => new Tensor<double>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => i * scale + shift).ToArray(), shape);

    private static Tensor<double> Values(int[] shape, params double[] values) => new Tensor<double>(values, shape);

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-12)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: torch {expected[i]:R}, ours {actual.GetFlat(i):R}");
    }

    private static (double Mean, double Variance) Moments(Tensor<double> t)
    {
        var v = t.ToArray();
        double mean = v.Average();
        return (mean, v.Select(x => (x - mean) * (x - mean)).Sum() / (v.Length - 1));
    }

    [Fact]
    public void ComposedOps_MatchTorch()
    {
        Close(new[] { 0.685, -1.535, -1.96, 5.18 }, _engine.TensorBilinear(Range(new[] { 2, 3 }, 0.25, 0), Range(new[] { 2, 4 }, 0.2, -0.5),
            Range(new[] { 2, 3, 4 }, 0.1, -1), Values(new[] { 2 }, 0.5, -1)), "bilinear");
        Close(new[] { 0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0, 16.0, 17.0, 20.0, 21.0, 14.0, 15.0, 18.0, 19.0, 22.0, 23.0 }, _engine.TensorChannelShuffle(Range(new[] { 2, 6, 2 }, 1, 0), 3), "channel_shuffle");
        Close(new[] { 0.0, 2.0, 8.0, 10.0, 1.0, 3.0, 9.0, 11.0, 4.0, 6.0, 12.0, 14.0, 5.0, 7.0, 13.0, 15.0, 16.0, 18.0, 24.0, 26.0, 17.0, 19.0, 25.0, 27.0, 20.0, 22.0, 28.0, 30.0, 21.0, 23.0, 29.0, 31.0 }, _engine.TensorPixelUnshuffle(Range(new[] { 1, 2, 4, 4 }, 1, 0), 2), "pixel_unshuffle");
        Close(new[] { 0.0, 0.08438429007574101, 0.168173575863148, 0.25112358021672115, 0.3330003923139149, 0.41348106357568853, 0.49218477356181667, 0.5688221113164209, 0.6431356619318523, 0.7149023711957149, 0.7839348829651037, 0.8500818966069793, 0.91322763167554, 0.9732905166212885, 1.0302212371280608, 1.0840002880757684, 1.2350942321880667, 1.295452076631822, 1.3531794446518963, 1.4082567158612156 }, _engine.TensorLocalResponseNorm(Range(new[] { 1, 5, 2, 2 }, 1.0 / 7, 0), 3, 0.1, 0.75, 2.0), "local_response_norm", 1e-10);
        Close(new[] { 0.42224059573077544 }, _engine.TensorSoftMarginLoss(Values(new[] { 4 }, 0.3, -1.2, 2.0, -0.1), Values(new[] { 4 }, 1, -1, 1, 1)), "soft_margin_loss");
        Close(new[] { -3.4020690409025627, -2.7216552327220502, -2.041241424541538, -1.3608276163610251, -1.0, 0.0, 1.0, 2.0, 1.617491580609716, 2.156655440812955, 2.6958193010161935, 3.234983161219432 }, _engine.TensorRenorm(Range(new[] { 3, 4 }, 1, -5), 2, 0, 5.0), "renorm p=2", 1e-10);
        Close(new[] { -2.2222221975308645, -1.9999999750000004, -1.3333333185185188, -0.7999999920000002, -0.4444444395061729, 0.0, 0.4444444395061729, 0.7999999920000002, 1.3333333185185188, 1.9999999750000004, 2.2222221975308645, 2.3999999760000006 }, _engine.TensorRenorm(Range(new[] { 3, 4 }, 1, -5), 1, 1, 4.0), "renorm p=1", 1e-10);
        Close(new[] { 5.916079783099616, 5.656854249492381, 5.916079783099616, 6.6332495807108 }, _engine.TensorNormExceptDim(Range(new[] { 3, 4 }, 1, -5), 2, 1), "norm_except_dim");
        var gradients = _engine.TensorGradient(Values(new[] { 2, 4 }, 1, 4, 9, 16, 2, 3, 5, 8), 0.5);
        Close(new[] { 2.0, -2.0, -8.0, -16.0, 2.0, -2.0, -8.0, -16.0 }, gradients[0], "gradient dim 0");
        Close(new[] { 6.0, 8.0, 12.0, 14.0, 2.0, 3.0, 5.0, 6.0 }, gradients[1], "gradient dim 1");
        var padded = _engine.TensorPadSequence(new[] { Values(new[] { 3, 2 }, 1, 2, 3, 4, 5, 6), Values(new[] { 1, 2 }, 7, 8) }, batchFirst: true, paddingValue: -1);
        Close(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -1.0, -1.0, -1.0 }, padded, "pad_sequence");
        var nz = _engine.TensorNonzeroStatic(Values(new[] { 2, 3 }, 0, 2, 0, 3, 0, 4), 4);
        Assert.Equal(new[] { 0, 1, 1, 0, 1, 2, -1, -1 }, nz.ToArray());
        Close(new[] { 0.0, 5.0, 1.0, 0.0, 2.0, 1.0 }, _engine.TensorUniqueDim(Values(new[] { 4, 2 }, 2, 1, 1, 0, 2, 1, 0, 5), 0), "unique dim");
        var reduced = _engine.TensorIndexReduce(Values(new[] { 3, 2 }, 1, 1, 1, 1, 1, 1), 0,
            new Tensor<int>(new[] { 0, 2, 0 }, new[] { 3 }), Values(new[] { 3, 2 }, 2, 3, 4, 5, 6, 7), ScatterReduceMode.Prod);
        Close(new[] { 12.0, 21.0, 1.0, 1.0, 4.0, 5.0 }, reduced, "index_reduce prod");
        Close(new[] { 0.7142857142857144, 1.1904761904761907, 0.5714285714285715, 0.904761904761905, 0.4285714285714288, 0.6190476190476192, 0.2857142857142856, 0.3333333333333336, 0.14285714285714302, 0.7619047619047619, 1.4285714285714288, 2.9047619047619047, 1.5714285714285716, 3.1904761904761902, 1.7142857142857149, 3.4761904761904763, 1.857142857142857, 3.7619047619047614, 2.0, 2.619047619047619, 2.142857142857143, 4.619047619047619, 2.571428571428571, 5.476190476190475, 3.0, 6.333333333333333, 3.428571428571428, 7.19047619047619, 3.8571428571428568, 4.476190476190475 }, _engine.TensorConvTranspose1D(Range(new[] { 1, 2, 5 }, 1.0 / 3, 0), Range(new[] { 2, 3, 3 }, 1.0 / 7, -1), 2, 1, 1), "conv_transpose1d", 1e-10);
        Close(new[] { 4.677777777777778, 4.911111111111111, 7.077777777777778, 7.711111111111112, 9.366666666666667, 10.466666666666667, 11.166666666666668, 12.866666666666667, 12.966666666666667, 15.266666666666666, 14.76666666666667, 17.666666666666668, 16.566666666666666, 20.066666666666666, 18.366666666666667, 22.46666666666667, 1.4777777777777783, 4.111111111111111, 1.4777777777777776, 4.511111111111111 }, _engine.TensorConvTbc(Range(new[] { 5, 2, 3 }, 1.0 / 9, 0), Range(new[] { 3, 3, 2 }, 0.2, -1), Values(new[] { 2 }, 0.1, -0.2), 1), "conv_tbc", 1e-10);
    }

    [Fact]
    public void Samplers_HaveTheirDistributionsMoments()
    {
        const int N = 20000;
        var shape = new[] { N };
        void Check(string what, Tensor<double> t, double mean, double variance)
        {
            var (m, v) = Moments(t);
            double se = Math.Sqrt(variance / N);
            Assert.True(Math.Abs(m - mean) <= 6 * se, $"{what} mean {m}, expected {mean}");
            Assert.True(Math.Abs(v - variance) <= 0.1 * variance, $"{what} variance {v}, expected {variance}");
        }
        Tensor<double> Fill(double value) => new Tensor<double>(Enumerable.Repeat(value, N).ToArray(), shape);
        Check("bernoulli", _engine.TensorBernoulli(Fill(0.3), seed: 1), 0.3, 0.21);
        Check("binomial", _engine.TensorBinomial(Fill(40), Fill(0.25), seed: 2), 10, 7.5);
        Check("poisson small", _engine.TensorPoisson(Fill(3.5), seed: 3), 3.5, 3.5);
        Check("poisson large", _engine.TensorPoisson(Fill(55), seed: 4), 55, 55);
        Check("normal", _engine.TensorNormal(Fill(2), Fill(3), seed: 5), 2, 9);
        Check("uniform", _engine.TensorUniform<double>(shape, -1, 3, seed: 6), 1, 16.0 / 12);
        Check("exponential", _engine.TensorExponential<double>(shape, 2, seed: 7), 0.5, 0.25);
        Check("geometric", _engine.TensorGeometric<double>(shape, 0.4, seed: 8), 2.5, 0.6 / 0.16);
        Check("log-normal", _engine.TensorLogNormal<double>(shape, 0.1, 0.4, seed: 9), Math.Exp(0.1 + 0.08), (Math.Exp(0.16) - 1) * Math.Exp(0.2 + 0.16));
        var cauchy = _engine.TensorCauchy<double>(shape, 1, 2, seed: 10).ToArray().OrderBy(v => v).ToArray();
        Assert.InRange(cauchy[N / 2], 0.9, 1.1);                      // median
        Assert.InRange(cauchy[3 * N / 4] - cauchy[N / 4], 3.8, 4.2);  // interquartile range = 2σ
        var picks = _engine.TensorMultinomial(Values(new[] { 4 }, 1, 3, 0, 6), N, replacement: true, seed: 11).ToArray();
        Assert.DoesNotContain(2.0, picks);
        Assert.InRange(picks.Count(v => v == 3) / (double)N, 0.58, 0.62);
        var distinct = _engine.TensorMultinomial(Values(new[] { 5 }, 1, 1, 1, 1, 1), 5, seed: 12).ToArray();
        Assert.Equal(new[] { 0.0, 1, 2, 3, 4 }, distinct.OrderBy(v => v).ToArray());
    }

    [Fact]
    public void DropoutVariants_KeepTheirInvariants()
    {
        var x = new Tensor<double>(Enumerable.Range(0, 4 * 8 * 50).Select(i => Math.Sin(i)).ToArray(), new[] { 4, 8, 50 });
        Assert.Same(x, _engine.TensorAlphaDropout(x, 0.3, training: false));
        var channels = _engine.TensorChannelDropout(x, 0.5, true, 1, seed: 3);
        for (int c = 0; c < 32; c++)
        {
            var row = Enumerable.Range(0, 50).Select(i => channels.GetFlat(c * 50 + i)).ToArray();
            bool dropped = row.All(v => v == 0);
            bool kept = Enumerable.Range(0, 50).All(i => Math.Abs(row[i] - 2 * x.GetFlat(c * 50 + i)) < 1e-12);
            Assert.True(dropped || kept, $"channel {c} was partly dropped");
        }
        // Alpha dropout of standard-normal input keeps mean 0 and variance 1.
        var normal = _engine.TensorNormal(new Tensor<double>(new[] { 40000 }), _engine.TensorOnesLike(new Tensor<double>(new[] { 40000 })), seed: 13);
        var (m, v) = Moments(_engine.TensorAlphaDropout(normal, 0.2, true, seed: 6));
        Assert.InRange(m, -0.03, 0.03);
        Assert.InRange(v, 0.95, 1.05);
    }

    [Fact]
    public void RenormAndNormExceptDim_ZeroSlices_HaveTorchsFiniteGradients()
    {
        // A zero row has a zero norm and |x|^p has zero elements: PyTorch's gradients are finite there, not NaN.
        var w = Values(new[] { 3, 3 }, 0, 0, 0, 1, -2, 0.5, 0, 3, -1);
        var weights = Values(new[] { 3, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var y = _engine.TensorRenorm(w, 3.0, 0, 2.0);
            Close(new[] { 0.0, 0.0, 0.0, 0.9570890565775739, -1.9141781131551479, 0.47854452828878696, 0.0, 1.9759012029552872, -0.6586337343184291 }, y, "renorm");
            var loss = _engine.ReduceSum(_engine.TensorMultiply(y, weights), null, false);
            Close(new[] { 1.0, 2.0, 3.0, 4.143015627113561, 3.526807679674805, 5.82119918966626, 4.6104361402290035, 2.0935144743742624, 6.280543097773992 },
                tape.ComputeGradients(loss, new[] { w })[w], "renorm gradient", 1e-11);
        }
        var v = Values(new[] { 3, 3 }, 0, 0, 0, 1, -2, 0.5, 0, 3, -1);
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var n = _engine.TensorNormExceptDim(v, 3, 0);
            Close(new[] { 0.0, 2.089669598190616, 3.0365889718756622 }, n, "norm_except_dim");
            var loss = _engine.ReduceSum(_engine.TensorMultiply(n, Values(new[] { 3, 1 }, 1, 2, 3)), null, false);
            Close(new[] { 0.0, 0.0, 0.0, 0.4580097749458884, -1.8320390997835536, 0.1145024437364721, 0.0, 2.9281393657372465, -0.3253488184152496 },
                tape.ComputeGradients(loss, new[] { v })[v], "norm_except_dim gradient", 1e-11);
        }
    }

    [Fact]
    public void Binomial_RejectsAFractionalCount()
        => Assert.Throws<ArgumentOutOfRangeException>(() =>
            _engine.TensorBinomial(Values(new[] { 2 }, 3, 2.5), Values(new[] { 2 }, 0.5, 0.5), seed: 1));

    [Fact]
    public void BilinearAndLocalResponseNorm_ValidateShapes()
    {
        var w = Range(new[] { 2, 3, 4 }, 0.1, -1);
        Assert.Throws<ArgumentException>(() => _engine.TensorBilinear(Range(new[] { 2, 2 }, 1, 0), Range(new[] { 2, 4 }, 1, 0), w));
        Assert.Throws<ArgumentException>(() => _engine.TensorBilinear(Range(new[] { 2, 3 }, 1, 0), Range(new[] { 2, 5 }, 1, 0), w));
        Assert.Throws<ArgumentException>(() => _engine.TensorBilinear(Range(new[] { 2, 3 }, 1, 0), Range(new[] { 3, 4 }, 1, 0), w));
        Assert.Throws<ArgumentException>(() => _engine.TensorBilinear(Range(new[] { 2, 3 }, 1, 0), Range(new[] { 2, 4 }, 1, 0), w, Values(new[] { 3 }, 1, 2, 3)));
        Assert.Throws<ArgumentException>(() => _engine.TensorLocalResponseNorm(Range(new[] { 4 }, 1, 0), 2));
    }
}
