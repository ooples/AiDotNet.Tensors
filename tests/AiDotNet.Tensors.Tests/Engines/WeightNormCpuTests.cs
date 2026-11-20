using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class WeightNormCpuTests
{
    [Theory]
    [InlineData(1e-24)]
    [InlineData(.001)]
    public void FloatForwardAndBothGradientsMatchScalarFiniteDifferences(double epsilon) => CheckFiniteDifferences<float>(epsilon, 3e-5);

    [Theory]
    [InlineData(1e-24)]
    [InlineData(.001)]
    public void DoubleForwardAndBothGradientsMatchScalarFiniteDifferences(double epsilon) => CheckFiniteDifferences<double>(epsilon, 2e-8);

    private static void CheckFiniteDifferences<T>(double epsilon, double tolerance)
    {
        var engine = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();
        var values = Enumerable.Range(0, 30).Select(i => .13 * Math.Sin(i * .7) + .03 * (i % 3 - 1)).ToArray();
        var gains = new[] { -.7, 0.0, 1.2 };
        var upstream = Enumerable.Range(0, 30).Select(i => .2 * Math.Cos(i * .3)).ToArray();
        // Rank4 deliberately normalizes by its FIRST dimension, also the ConvTranspose weight-norm convention.
        var direction = Tensor<T>(values, new[] { 3, 2, 1, 5 });
        var gain = Tensor<T>(gains, new[] { 3 });
        var h = Tensor<T>(upstream, direction.Shape.ToArray());
        var output = new Tensor<T>(direction.Shape.ToArray());
        var dv = new Tensor<T>(direction.Shape.ToArray());
        var dg = new Tensor<T>(new[] { 3 });
        engine.TensorWeightNormInto(output, direction, gain, epsilon);
        engine.TensorWeightNormBackwardInto(h, direction, gain, dv, dg, epsilon);
        Close(Forward(values, gains, epsilon), output.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
        const double step = 1e-6;
        double Objective() => Forward(values, gains, epsilon).Zip(upstream, (a, b) => a * b).Sum();
        var expectedDirection = new double[values.Length];
        var expectedGain = new double[gains.Length];
        for (int i = 0; i < values.Length; i++)
        {
            double original = values[i];
            values[i] = original + step; double plus = Objective();
            values[i] = original - step; double minus = Objective();
            values[i] = original;
            expectedDirection[i] = (plus - minus) / (2 * step);
        }
        for (int i = 0; i < gains.Length; i++)
        {
            double original = gains[i];
            gains[i] = original + step; double plus = Objective();
            gains[i] = original - step; double minus = Objective();
            gains[i] = original;
            expectedGain[i] = (plus - minus) / (2 * step);
        }
        Close(expectedDirection, dv.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
        Close(expectedGain, dg.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
        // Unrequested gradients must not force their counterpart to be allocated or written.
        dg.Fill(ops.FromDouble(19));
        engine.TensorWeightNormBackwardInto(h, direction, gain, dv, null, epsilon);
        Assert.All(dg.ToArray(), x => Assert.Equal(ops.FromDouble(19), x));
        engine.TensorWeightNormBackwardInto(h, direction, gain, null, dg, epsilon);
        Close(expectedGain, dg.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OffsetDestinationsPreserveSentinelsAndIndependentAccumulation(bool addDirection, bool addGain)
    {
        var engine = new CpuEngine();
        var values = Enumerable.Range(0, 21).Select(i => .15 + .03 * Math.Sin(i)).ToArray();
        var gains = new[] { -.3, .7, 0.0 };
        var upstream = Enumerable.Range(0, 21).Select(i => .2 * Math.Cos(i)).ToArray();
        var v = Offset(values, new[] { 3, 7 }, out var vStorage);
        var g = Offset(gains, new[] { 3 }, out var gStorage);
        var h = Offset(upstream, new[] { 3, 7 }, out var hStorage);
        var w = Offset(new double[21], new[] { 3, 7 }, out var wStorage);
        var dv = Offset(Enumerable.Repeat(.25, 21).ToArray(), new[] { 3, 7 }, out var dvStorage);
        var dg = Offset(Enumerable.Repeat(-.75, 3).ToArray(), new[] { 3 }, out var dgStorage);
        var expected = Backward(values, gains, upstream, 1e-24);
        engine.TensorWeightNormInto(w, v, g);
        for (int call = 1; call <= 2; call++)
        {
            engine.TensorWeightNormBackwardInto(h, v, g, dv, dg, accumulateDirection: addDirection, accumulateGain: addGain);
            Close(expected.Direction.Select(x => addDirection ? .25 + call * x : x).ToArray(), dv.ToArray(), 2e-12);
            Close(expected.Gain.Select(x => addGain ? -.75 + call * x : x).ToArray(), dg.ToArray(), 2e-12);
        }
        Close(Forward(values, gains, 1e-24), w.ToArray(), 2e-12);
        foreach (var storage in new[] { vStorage, gStorage, hStorage, wStorage, dvStorage, dgStorage })
        {
            Assert.Equal(123.0, storage[0]);
            Assert.Equal(123.0, storage[1]);
            Assert.Equal(123.0, storage[storage.Length - 1]);
        }
    }

    [Fact]
    public void ShiftedForwardOverlapSnapshotsOriginalDirection()
    {
        var engine = new CpuEngine();
        var storage = Tensor<double>(Enumerable.Range(0, 13).Select(i => .1 + i * .03).ToArray(), new[] { 13 });
        var v = storage.Slice(0, 0, 12).Reshape(new[] { 3, 4 });
        var w = storage.Slice(0, 1, 13).Reshape(new[] { 3, 4 });
        var gains = new[] { .4, -.6, .8 };
        var expected = Forward(v.ToArray(), gains, .001);
        double originalFirst = storage[0];
        engine.TensorWeightNormInto(w, v, Tensor<double>(gains, new[] { 3 }), .001);
        Close(expected, w.ToArray(), 1e-12);
        Assert.Equal(originalFirst, storage[0]);
    }

    [Fact]
    public void BackwardMayOverwriteItsDirectionAndGainSources()
    {
        var engine = new CpuEngine();
        var values = new[] { .2, -.3, .7, .1, -.4, .9 };
        var gains = new[] { -.8, .3 };
        var upstream = new[] { .2, -.1, .4, -.7, .1, .3 };
        var v = Tensor<double>(values, new[] { 2, 3 });
        var g = Tensor<double>(gains, new[] { 2 });
        var expected = Backward(values, gains, upstream, .001);
        engine.TensorWeightNormBackwardInto(Tensor<double>(upstream, new[] { 2, 3 }), v, g, v, g, .001);
        Close(expected.Direction, v.ToArray(), 1e-12);
        Close(expected.Gain, g.ToArray(), 1e-12);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OverlappingGradientDestinationsCommitWholeDirectionBeforeGain(bool addDirection, bool addGain)
    {
        var engine = new CpuEngine();
        var values = new[] { .2, -.3, .7, .1 };
        var gains = new[] { -.8, .3 };
        var upstream = new[] { .2, -.1, .4, -.7 };
        var original = new[] { 9.0, 10, 11, 12 };
        var storage = Tensor<double>(original, new[] { 4 });
        var dv = storage.Reshape(new[] { 2, 2 });
        var dg = storage.Slice(0, 0, 2);
        var expected = Backward(values, gains, upstream, .001);
        var final = expected.Direction.Select((x, i) => addDirection ? x + original[i] : x).ToArray();
        for (int i = 0; i < 2; i++) final[i] = addGain ? final[i] + expected.Gain[i] : expected.Gain[i];
        engine.TensorWeightNormBackwardInto(Tensor<double>(upstream, new[] { 2, 2 }),
            Tensor<double>(values, new[] { 2, 2 }), Tensor<double>(gains, new[] { 2 }), dv, dg,
            .001, addDirection, addGain);
        Close(final, storage.ToArray(), 1e-12);
    }

    [Fact]
    public void TiedDirectionAndGainClaimFirstWriteOnceThenAccumulateBothContributions()
    {
        var engine = new CpuEngine();
        var values = new[] { -.4, .2, .9 };
        var upstream = new[] { .3, -.1, .4 };
        var tied = Tensor<double>(values, new[] { 3 });
        var h = Tensor<double>(upstream, new[] { 3 });
        var gradient = new Tensor<double>(new[] { 3 }); gradient.Fill(19);
        var expected = new double[3];
        const double step = 1e-6;
        double Objective() => Forward(values, values, .001).Zip(upstream, (a, b) => a * b).Sum();
        for (int i = 0; i < 3; i++)
        {
            double original = values[i];
            values[i] = original + step; double plus = Objective();
            values[i] = original - step; double minus = Objective();
            values[i] = original;
            expected[i] = (plus - minus) / (2 * step);
        }
        engine.TensorWeightNormBackwardInto(h, tied, tied, gradient, gradient, .001,
            accumulateDirection: false, accumulateGain: true);
        Close(expected, gradient.ToArray(), 1e-9);
        engine.TensorWeightNormBackwardInto(h, tied, tied, gradient, gradient, .001,
            accumulateDirection: true, accumulateGain: true);
        Close(expected.Select(x => 2 * x).ToArray(), gradient.ToArray(), 2e-9);
    }

    [Fact]
    public void StridedSourcesAndDestinationsUseLogicalElementOrder()
    {
        var engine = new CpuEngine();
        var v = Tensor<double>(new[] { .2, .3, -.7, -.1, .4, .9 }, new[] { 3, 2 }).Transpose(new[] { 1, 0 });
        var h = Tensor<double>(new[] { .3, -.2, .4, .1, -.6, .8 }, new[] { 3, 2 }).Transpose(new[] { 1, 0 });
        var w = new Tensor<double>(new[] { 3, 2 }).Transpose(new[] { 1, 0 });
        var dv = new Tensor<double>(new[] { 3, 2 }).Transpose(new[] { 1, 0 });
        var gains = new[] { -.5, .9 };
        var g = Tensor<double>(gains, new[] { 2 });
        var dg = new Tensor<double>(new[] { 2 });
        Assert.False(v.IsContiguous);
        Assert.False(w.IsContiguous);
        var expected = Backward(v.ToArray(), gains, h.ToArray(), .001);
        engine.TensorWeightNormInto(w, v, g, .001);
        engine.TensorWeightNormBackwardInto(h, v, g, dv, dg, .001);
        Close(Forward(v.ToArray(), gains, .001), w.ToArray(), 1e-12);
        Close(expected.Direction, dv.ToArray(), 1e-12);
        Close(expected.Gain, dg.ToArray(), 1e-12);
    }

    [Fact]
    public void EpsilonIsInsideSquaredNormAndZeroRowsHaveCorrectDerivative()
    {
        var engine = new CpuEngine();
        var v = Tensor<double>(new[] { 0.0, 0, 3, 4 }, new[] { 2, 2 });
        var g = Tensor<double>(new[] { -2.0, 0 }, new[] { 2 });
        var h = Tensor<double>(new[] { .3, -.7, .2, -.1 }, new[] { 2, 2 });
        var w = new Tensor<double>(new[] { 2, 2 });
        var dv = new Tensor<double>(new[] { 2, 2 });
        var dg = new Tensor<double>(new[] { 2 });
        engine.TensorWeightNormInto(w, v, g, .25);
        engine.TensorWeightNormBackwardInto(h, v, g, dv, dg, .25);
        Close(new double[4], w.ToArray(), 0);
        Close(new[] { -1.2, 2.8, 0, 0 }, dv.ToArray(), 1e-12);
        Close(new[] { 0, .2 / Math.Sqrt(25.25) }, dg.ToArray(), 1e-12);
        g[1] = 2;
        engine.TensorWeightNormInto(w, v, g, .25);
        Close(new[] { 0, 0, 6 / Math.Sqrt(25.25), 8 / Math.Sqrt(25.25) }, w.ToArray(), 1e-12);
    }

    [Fact]
    public void EmptyRowsAndColumnsAvoidDivisionAndPreserveAccumulation()
    {
        var engine = new CpuEngine();
        foreach (var shape in new[] { new[] { 0, 3 }, new[] { 2, 0 } })
        {
            var v = new Tensor<double>(shape);
            var g = new Tensor<double>(new[] { shape[0] });
            var w = new Tensor<double>(shape);
            var dv = new Tensor<double>(shape);
            var dg = new Tensor<double>(new[] { shape[0] });
            dg.Fill(7);
            engine.TensorWeightNormInto(w, v, g, 0);
            engine.TensorWeightNormBackwardInto(w, v, g, dv, dg, 0, accumulateGain: true);
            Assert.All(dg.ToArray(), x => Assert.Equal(7, x));
            engine.TensorWeightNormBackwardInto(w, v, g, dv, dg, 0);
            Assert.All(dg.ToArray(), x => Assert.Equal(0, x));
        }
    }

    [Fact]
    public void DecimalUsesNumericOperationsWithoutFloatConversion()
    {
        var engine = new CpuEngine();
        var v = new Tensor<decimal>(new[] { 3m, 4m }, new[] { 1, 2 });
        var g = new Tensor<decimal>(new[] { -2m }, new[] { 1 });
        var h = new Tensor<decimal>(new[] { .2m, -.1m }, new[] { 1, 2 });
        var w = new Tensor<decimal>(new[] { 1, 2 });
        var dv = new Tensor<decimal>(new[] { 1, 2 });
        var dg = new Tensor<decimal>(new[] { 1 });
        engine.TensorWeightNormInto(w, v, g, 0);
        engine.TensorWeightNormBackwardInto(h, v, g, dv, dg, 0);
        Assert.Equal(new[] { -1.2m, -1.6m }, w.ToArray());
        Assert.Equal(new[] { -.0704m, .0528m }, dv.ToArray());
        Assert.Equal(new[] { .04m }, dg.ToArray());
    }

    [Fact]
    public void InvalidShapesAndEpsilonAreRejectedBeforeWrites()
    {
        var engine = new CpuEngine();
        var v = new Tensor<float>(new[] { 2, 3 });
        var g = new Tensor<float>(new[] { 2 });
        var output = new Tensor<float>(new[] { 2, 3 });
        output.Fill(19);
        foreach (double epsilon in new[] { -1.0, double.NaN, double.PositiveInfinity })
            Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorWeightNormInto(output, v, g, epsilon));
        Assert.Throws<ArgumentException>(() => engine.TensorWeightNormInto(output, v, new Tensor<float>(new[] { 2, 1 })));
        Assert.Throws<ArgumentException>(() => engine.TensorWeightNormInto(new Tensor<float>(new[] { 3, 2 }), v, g));
        Assert.Throws<ArgumentException>(() => engine.TensorWeightNormBackwardInto(output, v, g,
            new Tensor<float>(new[] { 3, 2 }), null));
        Assert.Throws<ArgumentException>(() => engine.TensorWeightNormBackwardInto(output, v, g,
            null, new Tensor<float>(new[] { 1, 2 })));
        Assert.All(output.ToArray(), x => Assert.Equal(19, x));
    }

    [Fact]
    public void OrdinaryIntoAllocationDoesNotScaleWithWeightPayload()
    {
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
#if NETFRAMEWORK
            // The 4.7.1 reference assemblies omit the per-thread counter exposed by newer desktop CLRs.
            var counter = typeof(GC).GetMethod("GetAllocatedBytesForCurrentThread", Type.EmptyTypes);
            Func<long> allocatedBytes;
            if (counter != null)
                allocatedBytes = (Func<long>)Delegate.CreateDelegate(typeof(Func<long>), counter);
            else
            {
                AppDomain.MonitoringIsEnabled = true;
                allocatedBytes = () => AppDomain.CurrentDomain.MonitoringTotalAllocatedMemorySize;
            }
#else
            Func<long> allocatedBytes = GC.GetAllocatedBytesForCurrentThread;
#endif
            long Allocated(int rows, int columns)
            {
                var engine = new CpuEngine();
                var v = new Tensor<float>(new[] { rows, columns }); v.Fill(.2f);
                var g = new Tensor<float>(new[] { rows }); g.Fill(-.3f);
                var h = new Tensor<float>(new[] { rows, columns }); h.Fill(.1f);
                var w = new Tensor<float>(v.Shape.ToArray());
                var dv = new Tensor<float>(v.Shape.ToArray());
                var dg = new Tensor<float>(g.Shape.ToArray());
                void Run()
                {
                    engine.TensorWeightNormInto(w, v, g);
                    engine.TensorWeightNormBackwardInto(h, v, g, dv, dg);
                    engine.TensorWeightNormBackwardInto(h, v, g, dv, dg, accumulateDirection: true, accumulateGain: true);
                }
                for (int i = 0; i < 3; i++) Run();
                long before = allocatedBytes();
                for (int i = 0; i < 3; i++) Run();
                return allocatedBytes() - before;
            }
            long small = Allocated(3, 17), large = Allocated(257, 4097);
            Assert.True(large <= small + 65536, $"Into allocated {large:N0}B for a 4 MiB weight payload, versus {small:N0}B for the small case.");
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previous; }
    }

    private static Tensor<T> Tensor<T>(double[] data, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(data.Select(ops.FromDouble).ToArray(), shape);
    }

    private static Tensor<double> Offset(double[] data, int[] shape, out Tensor<double> storage)
    {
        storage = new Tensor<double>(new[] { data.Length + 3 }); storage.Fill(123);
        var result = storage.Slice(0, 2, data.Length + 2).Reshape(shape);
        Assert.Equal(2, result.StorageOffset);
        for (int i = 0; i < data.Length; i++) result[i] = data[i];
        return result;
    }

    private static double[] Forward(double[] direction, double[] gain, double epsilon)
    {
        int columns = direction.Length / gain.Length;
        var result = new double[direction.Length];
        for (int row = 0; row < gain.Length; row++)
        {
            double norm = Math.Sqrt(direction.Skip(row * columns).Take(columns).Sum(x => x * x) + epsilon);
            for (int i = 0; i < columns; i++) result[row * columns + i] = gain[row] * direction[row * columns + i] / norm;
        }
        return result;
    }

    private static (double[] Direction, double[] Gain) Backward(double[] direction, double[] gain, double[] upstream, double epsilon)
    {
        int columns = direction.Length / gain.Length;
        var dv = new double[direction.Length]; var dg = new double[gain.Length];
        for (int row = 0; row < gain.Length; row++)
        {
            double squared = 0, dot = 0;
            for (int i = 0; i < columns; i++) { int at = row * columns + i; squared += direction[at] * direction[at]; dot += upstream[at] * direction[at]; }
            double norm = Math.Sqrt(squared + epsilon);
            dg[row] = dot / norm;
            for (int i = 0; i < columns; i++)
            {
                int at = row * columns + i;
                dv[at] = gain[row] / norm * upstream[at] - gain[row] * dot / (norm * norm * norm) * direction[at];
            }
        }
        return (dv, dg);
    }

    private static void Close(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(!double.IsNaN(actual[i]) && !double.IsInfinity(actual[i])
                && Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"Index{i}: expected {expected[i]:R}, actual {actual[i]:R}, tolerance {tolerance:R}");
    }
}
