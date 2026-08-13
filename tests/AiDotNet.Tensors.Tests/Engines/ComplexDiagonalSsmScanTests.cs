using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class ComplexDiagonalSsmScanTests
{
    private static double[] Values(int length, int seed, double scale = 0.35)
    {
        var values = new double[length];
        for (int i = 0; i < length; i++)
            values[i] = Math.Sin((i + 1) * 0.71 + seed * 0.37) * scale;
        return values;
    }

    private static Tensor<double> Tensor(int[] shape, int seed, double scale = 0.35)
    {
        int length = 1;
        foreach (int dimension in shape) length *= dimension;
        return new Tensor<double>(Values(length, seed, scale), shape);
    }

    private static Tensor<double>[] Inputs(int batch, int time, int groups, int width, int state)
        => new[]
        {
            Tensor(new[] { batch, time, groups, width }, 1),
            Tensor(new[] { groups, state }, 2, 0.25),
            Tensor(new[] { groups, state }, 3, 0.18),
            Tensor(new[] { groups, state, width }, 4),
            Tensor(new[] { groups, state, width }, 5),
            Tensor(new[] { groups, width, state }, 6),
            Tensor(new[] { groups, width, state }, 7),
            Tensor(new[] { groups, width }, 8)
        };

    [Fact]
    public void Forward_MatchesIndependentGroupedReference()
    {
        const int batch = 2, time = 4, groups = 3, width = 2, state = 3;
        Tensor<double>[] p = Inputs(batch, time, groups, width, state);
        var engine = new CpuEngine();

        Tensor<double> actual = Scan(engine, p);
        double[] expected = Reference(p, batch, time, groups, width, state);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual.GetFlat(i), precision: 11);
    }

    [Fact]
    public void Backward_AllEightInputsMatchCentralFiniteDifferences()
    {
        const int batch = 1, time = 3, groups = 2, width = 2, state = 2;
        Tensor<double>[] p = Inputs(batch, time, groups, width, state);
        var engine = new CpuEngine();
        Dictionary<Tensor<double>, Tensor<double>> gradients;

        using (var tape = new GradientTape<double>())
            gradients = tape.ComputeGradients(Scan(engine, p), p);

        const double epsilon = 1e-6;
        foreach (Tensor<double> parameter in p)
        {
            double[] data = parameter.GetDataArray()!;
            Tensor<double> analytic = gradients[parameter];
            for (int i = 0; i < data.Length; i++)
            {
                double original = data[i];
                data[i] = original + epsilon;
                double plus = Sum(Scan(engine, p));
                data[i] = original - epsilon;
                double minus = Sum(Scan(engine, p));
                data[i] = original;

                double numeric = (plus - minus) / (2 * epsilon);
                double calculated = analytic.GetFlat(i);
                double tolerance = 2e-6 + 2e-5 * Math.Abs(numeric);
                Assert.True(Math.Abs(numeric - calculated) <= tolerance,
                    $"gradient mismatch for input {Array.IndexOf(p, parameter)}, element {i}: " +
                    $"analytic={calculated:R}, numeric={numeric:R}");
            }
        }
    }

    [Fact]
    public void Backward_AllEightInputsMatchProjectedCentralFiniteDifferences()
    {
        const int batch = 1, time = 4, groups = 3, width = 2, state = 2;
        Tensor<double>[] p = Inputs(batch, time, groups, width, state);
        var projection = Tensor(new[] { batch, time, groups, width }, 19, 0.7);
        var engine = new CpuEngine();
        Dictionary<Tensor<double>, Tensor<double>> gradients;

        using (var tape = new GradientTape<double>())
        {
            Tensor<double> projected = engine.TensorMultiply(Scan(engine, p), projection);
            Tensor<double> objective = engine.ReduceSum(projected, new[] { 0, 1, 2, 3 }, keepDims: false);
            gradients = tape.ComputeGradients(objective, p);
        }

        const double epsilon = 1e-6;
        foreach (Tensor<double> parameter in p)
        {
            double[] data = parameter.GetDataArray()!;
            Tensor<double> analytic = gradients[parameter];
            for (int i = 0; i < data.Length; i++)
            {
                double original = data[i];
                data[i] = original + epsilon;
                double plus = Project(engine, p, projection);
                data[i] = original - epsilon;
                double minus = Project(engine, p, projection);
                data[i] = original;

                double numeric = (plus - minus) / (2 * epsilon);
                double calculated = analytic.GetFlat(i);
                double tolerance = 2e-6 + 2e-5 * Math.Abs(numeric);
                Assert.True(Math.Abs(numeric - calculated) <= tolerance,
                    $"projected gradient mismatch for input {Array.IndexOf(p, parameter)}, element {i}: " +
                    $"analytic={calculated:R}, numeric={numeric:R}");
            }
        }
    }

    [Fact]
    public void ComplexEmaSpecialization_ComposedInputGradientMatchesFiniteDifferences()
    {
        const int batch = 1, time = 4, channels = 5;
        var input = Tensor(new[] { batch, time, channels }, 23, 0.4);
        var alphaReal = Tensor(new[] { channels }, 24, 0.45);
        var alphaImag = Tensor(new[] { channels }, 25, 0.25);
        var gamma = Tensor(new[] { channels }, 26, 0.3);
        var beta = Tensor(new[] { channels }, 27, 0.2);
        var projection = Tensor(new[] { batch, time, channels }, 28, 0.7);
        var engine = new CpuEngine();
        Tensor<double> inputGradient;

        using (var tape = new GradientTape<double>())
        {
            Tensor<double> objective = ComplexEmaObjective(
                engine, input, alphaReal, alphaImag, gamma, beta, projection);
            inputGradient = tape.ComputeGradients(objective, new[] { input })[input];
        }

        const double epsilon = 1e-6;
        int[] samples = { 0, input.Length / 3, input.Length - 1 };
        foreach (int index in samples)
        {
            double original = input[index];
            input[index] = original + epsilon;
            double plus = ComplexEmaObjective(
                engine, input, alphaReal, alphaImag, gamma, beta, projection)[0];
            input[index] = original - epsilon;
            double minus = ComplexEmaObjective(
                engine, input, alphaReal, alphaImag, gamma, beta, projection)[0];
            input[index] = original;

            double numeric = (plus - minus) / (2 * epsilon);
            double calculated = inputGradient[index];
            double tolerance = 2e-6 + 2e-5 * Math.Abs(numeric);
            Assert.True(Math.Abs(numeric - calculated) <= tolerance,
                $"complex EMA input[{index}] mismatch: analytic={calculated:R}, numeric={numeric:R}");
        }
    }

    [Fact]
    public void Contract_RejectsMismatchedGroupedMaps()
    {
        Tensor<double>[] p = Inputs(batch: 1, time: 2, groups: 2, width: 3, state: 4);
        p[5] = Tensor(new[] { 2, 2, 4 }, 9);

        var error = Assert.Throws<ArgumentException>(() => Scan(new CpuEngine(), p));
        Assert.Contains("outputMapReal must have shape [2,3,4]", error.Message);
    }

    private static Tensor<double> Scan(CpuEngine engine, Tensor<double>[] p)
        => engine.ComplexDiagonalSsmScanForward(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);

    private static double Sum(Tensor<double> tensor)
    {
        double sum = 0;
        foreach (double value in tensor.GetDataArray()!) sum += value;
        return sum;
    }

    private static double Project(
        CpuEngine engine,
        Tensor<double>[] inputs,
        Tensor<double> projection)
    {
        Tensor<double> output = Scan(engine, inputs);
        double sum = 0;
        for (int i = 0; i < output.Length; i++)
            sum += output.GetFlat(i) * projection.GetFlat(i);
        return sum;
    }

    private static Tensor<double> ComplexEmaObjective(
        CpuEngine engine,
        Tensor<double> input,
        Tensor<double> alphaReal,
        Tensor<double> alphaImag,
        Tensor<double> gamma,
        Tensor<double> beta,
        Tensor<double> projection)
    {
        int batch = input.Shape[0];
        int time = input.Shape[1];
        int channels = input.Shape[2];
        var one = new Tensor<double>(new[] { channels });
        one.Fill(1.0);
        var zero = new Tensor<double>(new[] { channels });
        var scanned = engine.ComplexDiagonalSsmScanForward(
            engine.Reshape(input, new[] { batch, time, channels, 1 }),
            engine.Reshape(alphaReal, new[] { channels, 1 }),
            engine.Reshape(alphaImag, new[] { channels, 1 }),
            engine.Reshape(engine.TensorSubtract(one, alphaReal), new[] { channels, 1, 1 }),
            engine.Reshape(engine.TensorNegate(alphaImag), new[] { channels, 1, 1 }),
            engine.Reshape(one, new[] { channels, 1, 1 }),
            engine.Reshape(zero, new[] { channels, 1, 1 }),
            engine.Reshape(zero, new[] { channels, 1 }));
        var normalized = engine.LayerNorm(
            engine.Reshape(scanned, new[] { batch, time, channels }),
            gamma,
            beta,
            1e-5,
            out _,
            out _);
        return engine.ReduceSum(
            engine.TensorMultiply(normalized, projection),
            new[] { 0, 1, 2 },
            keepDims: false);
    }

    private static double[] Reference(
        Tensor<double>[] p, int batch, int time, int groups, int width, int state)
    {
        var result = new double[batch * time * groups * width];
        for (int b = 0; b < batch; b++)
        {
            for (int g = 0; g < groups; g++)
            {
                var real = new double[state];
                var imag = new double[state];
                for (int t = 0; t < time; t++)
                {
                    for (int n = 0; n < state; n++)
                    {
                        double oldReal = real[n];
                        double oldImag = imag[n];
                        double nextReal = p[1][g, n] * oldReal - p[2][g, n] * oldImag;
                        double nextImag = p[1][g, n] * oldImag + p[2][g, n] * oldReal;
                        for (int w = 0; w < width; w++)
                        {
                            nextReal += p[3][g, n, w] * p[0][b, t, g, w];
                            nextImag += p[4][g, n, w] * p[0][b, t, g, w];
                        }
                        real[n] = nextReal;
                        imag[n] = nextImag;
                    }

                    for (int w = 0; w < width; w++)
                    {
                        double y = p[7][g, w] * p[0][b, t, g, w];
                        for (int n = 0; n < state; n++)
                            y += p[5][g, w, n] * real[n] - p[6][g, w, n] * imag[n];
                        result[((b * time + t) * groups + g) * width + w] = y;
                    }
                }
            }
        }
        return result;
    }
}
