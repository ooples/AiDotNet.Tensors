// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class PreActivatedConvCpuTests
{
    private sealed class Fixture
    {
        public readonly Tensor<double> X = new(Enumerable.Range(0, 18).Select(i => Math.Sin(i + .4)).ToArray(), new[] { 1, 2, 3, 3 });
        public readonly Tensor<double> Kernel = new(Enumerable.Range(0, 16).Select(i => .2 * Math.Cos(i + .7)).ToArray(), new[] { 2, 2, 2, 2 });
        public readonly Tensor<double> PreGamma = V(1.1, -.8), PreBeta = V(.2, -.1), PreMean = V(.15, -.25), PreVariance = V(.6, 1.3);
        public readonly Tensor<double> Bias = V(.13, -.17), Gamma = V(-.9, 1.2), Beta = V(.24, -.3), Mean = V(.1, -.2), Variance = V(.7, 1.4);
        public const double Epsilon = .07;
        private static Tensor<double> V(params double[] values) => new(values, new[] { values.Length });

        public Tensor<double> Run(CpuEngine engine, bool post, bool bias = true, int stride = 1, int pad = 0, int dilation = 1,
            FusedActivationType activation = FusedActivationType.None) => post
            ? engine.BatchNormActivationConv2DBatchNorm(X, PreGamma, PreBeta, PreMean, PreVariance, Epsilon,
                Kernel, bias ? Bias : null, Gamma, Beta, Mean, Variance, Epsilon,
                stride, stride, pad, pad, dilation, dilation, activation)
            : engine.BatchNormActivationConv2D(X, PreGamma, PreBeta, PreMean, PreVariance, Epsilon,
                Kernel, bias ? Bias : null, stride, stride, pad, pad, dilation, dilation, activation);

        // Independent NCHW cross-correlation, with scalar frozen-statistic normalization before and after it.
        public double[] Reference(bool post, bool bias = true, int stride = 1, int pad = 0, int dilation = 1,
            FusedActivationType activation = FusedActivationType.None)
        {
            int outputSize = (3 + 2 * pad - dilation - 1) / stride + 1;
            var result = new double[2 * outputSize * outputSize];
            for (int oc = 0; oc < 2; oc++)
                for (int y = 0; y < outputSize; y++)
                    for (int x = 0; x < outputSize; x++)
                    {
                        double sum = bias ? Bias[oc] : 0;
                        for (int ic = 0; ic < 2; ic++)
                            for (int ky = 0; ky < 2; ky++)
                                for (int kx = 0; kx < 2; kx++)
                                {
                                    int iy = y * stride - pad + ky * dilation, ix = x * stride - pad + kx * dilation;
                                    if (iy < 0 || iy >= 3 || ix < 0 || ix >= 3) continue;
                                    double normalized = (X[0, ic, iy, ix] - PreMean[ic]) * PreGamma[ic]
                                        / Math.Sqrt(PreVariance[ic] + Epsilon) + PreBeta[ic];
                                    sum += Math.Max(0, normalized) * Kernel[oc, ic, ky, kx];
                                }
                        if (post) sum = (sum - Mean[oc]) * Gamma[oc] / Math.Sqrt(Variance[oc] + Epsilon) + Beta[oc];
                        result[(oc * outputSize + y) * outputSize + x] = activation == FusedActivationType.ReLU ? Math.Max(0, sum) : sum;
                    }
            return result;
        }
    }

    [Theory]
    [InlineData(false, false, FusedActivationType.None)]
    [InlineData(true, true, FusedActivationType.None)]
    [InlineData(true, false, FusedActivationType.ReLU)]
    public void InferenceWithPaddingStrideAndDilationMatchesIndependentCorrelation(bool post, bool bias, FusedActivationType activation)
    {
        var fixture = new Fixture();
        var actual = fixture.Run(new CpuEngine(), post, bias, stride: 2, pad: 1, dilation: 2, activation: activation);
        Assert.Equal(new[] { 1, 2, 2, 2 }, actual.Shape.ToArray());
        Close(fixture.Reference(post, bias, stride: 2, pad: 1, dilation: 2, activation: activation), actual.ToArray(), 1e-12);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void EagerAndCompiledGradientsMatchIndependentFiniteDifferences(bool post, bool compiled)
    {
        var fixture = new Fixture();
        var engine = new CpuEngine();
        var parameters = post
            ? new[] { fixture.X, fixture.PreGamma, fixture.PreBeta, fixture.Kernel, fixture.Bias, fixture.Gamma, fixture.Beta }
            : new[] { fixture.X, fixture.PreGamma, fixture.PreBeta, fixture.Kernel, fixture.Bias };
        Tensor<double> Loss() => engine.ReduceSum(fixture.Run(engine, post), null);
        double[][] gradients;
        if (compiled)
        {
            using var scope = GraphMode.EnableTraining(parameters);
            var loss = Loss();
            using var plan = scope.CompileTraining(parameters, loss);
            Assert.Equal(fixture.Reference(post).Sum(), plan.Step().ToArray()[0], 10);
            gradients = plan.Gradients.Select(g => g.ToArray()).ToArray();
        }
        else
        {
            using var tape = new GradientTape<double>();
            var loss = Loss();
            Assert.Equal(fixture.Reference(post).Sum(), loss.ToArray()[0], 10);
            var recorded = tape.ComputeGradients(loss, parameters);
            gradients = parameters.Select(p => recorded[p].ToArray()).ToArray();
        }
        const double h = 1e-5;
        for (int p = 0; p < parameters.Length; p++)
            for (int i = 0; i < parameters[p].Length; i++)
            {
                double original = parameters[p].GetFlat(i);
                parameters[p].SetFlat(i, original + h);
                double plus = fixture.Reference(post).Sum();
                parameters[p].SetFlat(i, original - h);
                double minus = fixture.Reference(post).Sum();
                parameters[p].SetFlat(i, original);
                Assert.InRange(Math.Abs(gradients[p][i] - (plus - minus) / (2 * h)), 0, 1e-8);
            }
    }

    [Fact]
    public void PostNormalizationFoldReusesThenInvalidatesForEachChangedParameter()
    {
        var fixture = new Fixture();
        var engine = new CpuEngine();
        Close(fixture.Reference(true), fixture.Run(engine, true).ToArray(), 1e-12);
        Close(fixture.Reference(true), fixture.Run(engine, true).ToArray(), 1e-12);
        foreach (var parameter in new[] { fixture.Kernel, fixture.Bias, fixture.Gamma, fixture.Beta, fixture.Mean, fixture.Variance })
        {
            parameter.SetFlat(0, parameter.GetFlat(0) + .125);
            Close(fixture.Reference(true), fixture.Run(engine, true).ToArray(), 1e-12);
        }
        // A surrounding tape with recording suppressed must retain inference semantics too.
        using var tape = new GradientTape<double>();
        using var suppressed = new NoGradScope<double>();
        Close(fixture.Reference(true), fixture.Run(engine, true).ToArray(), 1e-12);
    }

    private static void Close(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, tolerance * (1 + Math.Abs(expected[i])));
    }
}
