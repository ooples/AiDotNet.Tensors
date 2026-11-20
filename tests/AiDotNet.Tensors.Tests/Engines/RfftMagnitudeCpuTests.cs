using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class RfftMagnitudeCpuTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(17)]
    public void FloatAndDoubleMatchIndependentDftAndAdjoint(int n)
    {
        CheckDft<float>(new CpuEngine(), n, 3e-6);
        CheckDft<double>(new CpuEngine(), n, 2e-11);
    }

    internal static void CheckDft<T>(CpuEngine engine, int n, double tolerance)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int bins = n / 2 + 1;
        var values = Enumerable.Range(0, 6 * n).Select(i => .2 + .3 * Math.Sin(.31 * i)).Select(ops.FromDouble).ToArray();
        var seeds = Enumerable.Range(0, 6 * bins).Select(i => .4 * Math.Cos(.23 * i)).Select(ops.FromDouble).ToArray();
        var input = new Tensor<T>(values, new[] { 2, 3, n });
        var upstream = new Tensor<T>(seeds, new[] { 2, 3, bins });
        var output = new Tensor<T>(new[] { 2, 3, bins });
        var gradient = new Tensor<T>(new[] { 2, 3, n });
        output.Fill(ops.FromDouble(123));
        gradient.Fill(ops.FromDouble(123));
        engine.TensorRfftMagnitudeInto(output, input, .125);
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .125);
        var expected = Oracle(values.Select(ops.ToDouble).ToArray(), seeds.Select(ops.ToDouble).ToArray(), n, .125);
        AssertClose(expected.Magnitude, output.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
        AssertClose(expected.Gradient, gradient.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .125, accumulate: true);
        AssertClose(expected.Gradient.Select(v => 2 * v).ToArray(), gradient.ToArray().Select(ops.ToDouble).ToArray(), 2 * tolerance);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(8)]
    public void IntoAdjointMatchesFiniteDifferencesOfActualForward(int n)
    {
        var engine = new CpuEngine();
        var values = Enumerable.Range(0, 2 * n).Select(i => .4 * Math.Sin(.2 + .39 * i)).ToArray();
        var weights = Enumerable.Range(0, 2 * (n / 2 + 1)).Select(i => .3 * Math.Cos(.27 * i)).ToArray();
        var input = new Tensor<double>(values, new[] { 2, n });
        var upstream = new Tensor<double>(weights, new[] { 2, n / 2 + 1 });
        var destination = new Tensor<double>(new[] { 2, n });
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, destination, .3);
        var actual = destination.ToArray();
        double Loss()
        {
            var output = engine.TensorRfftMagnitude(input, .3).ToArray();
            return output.Select((value, i) => value * weights[i]).Sum();
        }
        for (int i = 0; i < input.Length; i++)
        {
            double original = input[i], step = 1e-6;
            input[i] = original + step; double positive = Loss();
            input[i] = original - step; double negative = Loss();
            input[i] = original;
            Assert.InRange(Math.Abs(actual[i] - (positive - negative) / (2 * step)), 0, 3e-8);
        }
    }

    [Fact]
    public void StridedInputUpstreamAndOutputPreserveLogicalRowsAndAccumulation()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(Enumerable.Range(0, 16).Select(i => .3 * Math.Sin(.2 + i * .17)).ToArray(), new[] { 8, 2 }).Transpose(new[] { 1, 0 });
        var upstream = new Tensor<double>(Enumerable.Range(0, 10).Select(i => .4 * Math.Cos(.3 + i * .29)).ToArray(), new[] { 5, 2 }).Transpose(new[] { 1, 0 });
        var output = new Tensor<double>(new[] { 5, 2 }).Transpose(new[] { 1, 0 });
        var gradient = new Tensor<double>(new[] { 8, 2 }).Transpose(new[] { 1, 0 });
        Assert.False(input.IsContiguous); Assert.False(output.IsContiguous);
        Assert.False(upstream.IsContiguous); Assert.False(gradient.IsContiguous);
        var expected = Oracle(input.ToArray(), upstream.ToArray(), 8, .125);
        engine.TensorRfftMagnitudeInto(output, input, .125);
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .125);
        AssertClose(expected.Magnitude, output.ToArray(), 2e-11);
        AssertClose(expected.Gradient, gradient.ToArray(), 2e-11);
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .125, accumulate: true);
        AssertClose(expected.Gradient.Select(v => v * 2).ToArray(), gradient.ToArray(), 4e-11);
    }

    [Fact]
    public void ForwardSnapshotsCrossRowOverlapsAndPreservesSentinels()
    {
        var engine = new CpuEngine();
        var storage = new Tensor<double>(Enumerable.Range(0, 32).Select(i => .3 * Math.Sin(.7 + .19 * i)).ToArray(), new[] { 32 });
        var input = storage.Slice(0, 1, 17).Reshape(new[] { 2, 8 });
        var output = storage.Slice(0, 7, 17).Reshape(new[] { 2, 5 });
        double first = storage[0], last = storage[31];
        var expected = Oracle(input.ToArray(), new double[10], 8, .25).Magnitude;
        engine.TensorRfftMagnitudeInto(output, input, .25);
        AssertClose(expected, output.ToArray(), 2e-11);
        Assert.Equal(first, storage[0]); Assert.Equal(last, storage[31]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BackwardSnapshotsInputAndUpstreamWhenBothOverlapOffsetDestination(bool accumulate)
    {
        var engine = new CpuEngine();
        var storage = new Tensor<double>(Enumerable.Range(0, 32).Select(i => .3 * Math.Sin(.2 + .17 * i)).ToArray(), new[] { 32 });
        var input = storage.Slice(0, 1, 17).Reshape(new[] { 2, 8 });
        var upstream = storage.Slice(0, 7, 17).Reshape(new[] { 2, 5 });
        var gradient = storage.Slice(0, 4, 20).Reshape(new[] { 2, 8 });
        double first = storage[0], last = storage[31];
        var initial = gradient.ToArray();
        var expected = Oracle(input.ToArray(), upstream.ToArray(), 8, .25).Gradient;
        if (accumulate) expected = expected.Select((value, i) => value + initial[i]).ToArray();
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .25, accumulate);
        AssertClose(expected, gradient.ToArray(), 2e-11);
        Assert.Equal(first, storage[0]); Assert.Equal(last, storage[31]);
    }

    [Fact]
    public void ZeroSignalPreservesPositiveEpsilonAndUnsmoothedSqrtDerivative()
    {
        var engine = new CpuEngine();
        foreach (int n in new[] { 1, 8 })
        {
            var input = new Tensor<double>(new[] { 2, n });
            var upstream = new Tensor<double>(Enumerable.Repeat(1.0, 2 * (n / 2 + 1)).ToArray(), new[] { 2, n / 2 + 1 });
            var gradient = new Tensor<double>(new[] { 2, n });
            Assert.All(engine.TensorRfftMagnitude(input, .25).ToArray(), value => Assert.Equal(.5, value));
            engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .25);
            Assert.All(gradient.ToArray(), value => Assert.Equal(0, value));
            Assert.All(engine.TensorRfftMagnitude(input, 0).ToArray(), value => Assert.Equal(0, value));
            engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, 0);
            Assert.All(gradient.ToArray(), value => Assert.True(double.IsNaN(value)));
        }
    }

    [Fact]
    public void StridedForwardDestinationSnapshotsDistinctMemoryWrappersOverTheSameArray()
    {
        var engine = new CpuEngine();
        var shared = Enumerable.Range(0, 40).Select(i => .3 * Math.Sin(.2 + i * .17)).ToArray();
        var input = Tensor<double>.FromMemory(shared.AsMemory(1, 16), new[] { 2, 8 });
        var output = Tensor<double>.FromMemory(shared.AsMemory(7, 10), new[] { 5, 2 }).Transpose(new[] { 1, 0 });
        Assert.NotSame(input.StorageIdentity, output.StorageIdentity);
        Assert.False(output.IsContiguous);
        double first = shared[0], last = shared[39];
        var expected = Oracle(input.ToArray(), new double[10], 8, .25).Magnitude;
        engine.TensorRfftMagnitudeInto(output, input, .25);
        AssertClose(expected, output.ToArray(), 2e-11);
        Assert.Equal(first, shared[0]); Assert.Equal(last, shared[39]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StridedAdjointSnapshotsDistinctMemoryWrappersForBothPrimalAndUpstream(bool accumulate)
    {
        var engine = new CpuEngine();
        var shared = Enumerable.Range(0, 40).Select(i => .3 * Math.Sin(.4 + i * .19)).ToArray();
        var input = Tensor<double>.FromMemory(shared.AsMemory(1, 16), new[] { 2, 8 });
        var upstream = Tensor<double>.FromMemory(shared.AsMemory(7, 10), new[] { 2, 5 });
        var gradient = Tensor<double>.FromMemory(shared.AsMemory(4, 16), new[] { 8, 2 }).Transpose(new[] { 1, 0 });
        Assert.NotSame(input.StorageIdentity, gradient.StorageIdentity);
        Assert.NotSame(upstream.StorageIdentity, gradient.StorageIdentity);
        Assert.False(gradient.IsContiguous);
        double first = shared[0], last = shared[39];
        var initial = gradient.ToArray();
        var expected = Oracle(input.ToArray(), upstream.ToArray(), 8, .25).Gradient;
        if (accumulate) expected = expected.Select((value, i) => value + initial[i]).ToArray();
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .25, accumulate);
        AssertClose(expected, gradient.ToArray(), 2e-11);
        Assert.Equal(first, shared[0]); Assert.Equal(last, shared[39]);
    }

    [Fact]
    public void DecimalUsesExistingCpuFftConversionContract()
    {
        var engine = new CpuEngine();
        var input = new Tensor<decimal>(new[] { 3m, -1m }, new[] { 1, 2 });
        var upstream = new Tensor<decimal>(new[] { .25m, -.5m }, new[] { 1, 2 });
        var output = new Tensor<decimal>(new[] { 1, 2 });
        var gradient = new Tensor<decimal>(new[] { 1, 2 });
        engine.TensorRfftMagnitudeInto(output, input, 0);
        engine.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, 0);
        Assert.Equal(new[] { 2m, 4m }, output.ToArray());
        Assert.Equal(new[] { -.25m, .75m }, gradient.ToArray());
    }

    [Fact]
    public void ValidateGeometryAndAllowEmptyLeadingBatches()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 2, 8 });
        foreach (double epsilon in new[] { -1.0, double.NaN, double.PositiveInfinity })
            Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorRfftMagnitude(input, epsilon));
        Assert.Throws<ArgumentException>(() => engine.TensorRfftMagnitude(new Tensor<double>(new[] { 0 })));
        Assert.Throws<ArgumentException>(() => engine.TensorRfftMagnitudeInto(new Tensor<double>(new[] { 1, 5 }), input));
        Assert.Throws<ArgumentException>(() => engine.TensorRfftMagnitudeBackwardInto(new Tensor<double>(new[] { 2, 4 }), input, new Tensor<double>(new[] { 2, 8 })));
        Assert.Throws<ArgumentException>(() => engine.TensorRfftMagnitudeBackwardInto(new Tensor<double>(new[] { 2, 5 }), input, new Tensor<double>(new[] { 4, 4 })));
        var empty = new Tensor<double>(new[] { 0, 8 });
        var magnitudes = engine.TensorRfftMagnitude(empty);
        Assert.Equal(new[] { 0, 5 }, magnitudes.Shape.ToArray());
        engine.TensorRfftMagnitudeBackwardInto(magnitudes, empty, new Tensor<double>(new[] { 0, 8 }));
    }

    [Fact]
    public void CreateGraphPreservesHessianVectorProductAgainstIndependentDft()
    {
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            const int n = 5;
            var values = new[] { .2, -.3, .5, .7, -.1 };
            var weights = new[] { .3, -.2, .4 };
            var vector = new[] { -.3, .2, .1, -.4, .5 };
            var input = new Tensor<double>(values, new[] { 1, n });
            var upstream = new Tensor<double>(weights, new[] { 1, n / 2 + 1 });
            using var outer = new GradientTape<double>();
            Tensor<double> first;
            using (var inner = new GradientTape<double>())
            {
                var loss = engine.ReduceSum(engine.TensorMultiply(engine.TensorRfftMagnitude(input, .3), upstream));
                first = inner.ComputeGradients(loss, new[] { input }, createGraph: true)[input];
            }
            var pairing = engine.ReduceSum(engine.TensorMultiply(first, new Tensor<double>(vector, new[] { 1, n })));
            var second = outer.ComputeGradients(pairing, new[] { input })[input].ToArray();
            const double step = 1e-5;
            var positive = Oracle(values.Select((v, i) => v + step * vector[i]).ToArray(), weights, n, .3).Gradient;
            var negative = Oracle(values.Select((v, i) => v - step * vector[i]).ToArray(), weights, n, .3).Gradient;
            AssertClose(positive.Select((v, i) => (v - negative[i]) / (2 * step)).ToArray(), second, 2e-8);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private static (double[] Magnitude, double[] Gradient) Oracle(double[] input, double[] upstream, int n, double epsilon)
    {
        int bins = n / 2 + 1, rows = input.Length / n;
        var magnitude = new double[rows * bins];
        var gradient = new double[input.Length];
        for (int row = 0; row < rows; row++)
            for (int bin = 0; bin < bins; bin++)
            {
                double re = 0, im = 0;
                for (int t = 0; t < n; t++)
                {
                    double angle = 2 * Math.PI * bin * t / n;
                    re += input[row * n + t] * Math.Cos(angle);
                    im -= input[row * n + t] * Math.Sin(angle);
                }
                double value = Math.Sqrt(re * re + im * im + epsilon);
                magnitude[row * bins + bin] = value;
                for (int t = 0; t < n; t++)
                {
                    double angle = 2 * Math.PI * bin * t / n;
                    gradient[row * n + t] += upstream[row * bins + bin] / value * (re * Math.Cos(angle) - im * Math.Sin(angle));
                }
            }
        return (magnitude, gradient);
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, tolerance * Math.Max(1, Math.Abs(expected[i])));
    }
}

[Collection("DirectGpuSerial")]
public sealed class GpuRfftMagnitudeTests
{
    [SkippableFact]
    public void OddLengthMagnitudeAndAdjointMatchDft()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "A GPU backend is required.");
        RfftMagnitudeCpuTests.CheckDft<float>(gpu, 129, 3e-4);
    }
}
