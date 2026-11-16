using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("DirectGpuSerial")]
public sealed class BatchNormReluDestinationGradientTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public BatchNormReluDestinationGradientTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static readonly double[] X = { -1.2, 0.4, 2.0, -0.8, 0.6, 1.3, -0.1, 1.8 };
    private static readonly double[] Gamma = { 1.1, -0.7 }, Beta = { 0.15, 0.4 };
    private static readonly float[] Seed = { 0.3f, -0.9f, 0.2f, 0.7f, -0.4f, 1.2f, 0.8f, -0.1f };
    private const double Epsilon = 1e-3;

    // Independent scalar forward, differentiated by central differences rather than repeating the backward formula.
    private static double Objective(double[] x, double[] gamma, double[] beta)
    {
        double loss = 0;
        for (int c = 0; c < 2; c++)
        {
            double mean = 0, variance = 0;
            for (int n = 0; n < 2; n++) for (int s = 0; s < 2; s++) mean += x[(n * 2 + c) * 2 + s] / 4;
            for (int n = 0; n < 2; n++) for (int s = 0; s < 2; s++)
            {
                double centered = x[(n * 2 + c) * 2 + s] - mean;
                variance += centered * centered / 4;
            }
            for (int n = 0; n < 2; n++) for (int s = 0; s < 2; s++)
            {
                int i = (n * 2 + c) * 2 + s;
                loss += Seed[i] * Math.Max(0, gamma[c] * (x[i] - mean) / Math.Sqrt(variance + Epsilon) + beta[c]);
            }
        }
        return loss;
    }

    private static double[] NumericalGradient(double[] values, Func<double> objective)
    {
        const double h = 1e-5;
        var gradient = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            double original = values[i];
            values[i] = original + h;
            double plus = objective();
            values[i] = original - h;
            double minus = objective();
            values[i] = original;
            gradient[i] = (plus - minus) / (2 * h);
        }
        return gradient;
    }

    private static Tensor<float> Tensor(double[] values, int[] shape)
        => new Tensor<float>(Array.ConvertAll(values, value => (float)value), shape);

    private static void Verify(IEngine engine, bool resident)
    {
        var x = (double[])X.Clone();
        var gamma = (double[])Gamma.Clone();
        var beta = (double[])Beta.Clone();
        double ObjectiveNow() => Objective(x, gamma, beta);
        var expectedX = NumericalGradient(x, ObjectiveNow);
        var expectedGamma = NumericalGradient(gamma, ObjectiveNow);
        var expectedBeta = NumericalGradient(beta, ObjectiveNow);
        var input = Tensor(x, new[] { 2, 2, 1, 2 });
        var scale = Tensor(gamma, new[] { 2 });
        var shift = Tensor(beta, new[] { 2 });
        var dy = new Tensor<float>((float[])Seed.Clone(), new[] { 2, 2, 1, 2 });
        // The CPU forward supplies saved statistics; the derivative oracle above uses neither engine implementation.
        var cpu = new CpuEngine();
        var output = cpu.ReLU(cpu.BatchNorm(input, scale, shift, Epsilon, out var mean, out var variance));
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();
        DirectGpuTensorEngine? gpu = engine as DirectGpuTensorEngine;
        IDisposable? capturePath = null;
        int priorGeneration = DifferentiableOps.GradWriteGeneration;
        try
        {
            if (resident && gpu is not null)
            {
                gpu.SuspendActivationEviction();
                capturePath = gpu.EnterCompiledCapturePath();
                var destination = new Tensor<float>(new[] { 2, 2, 1, 2 });
                destination.AsWritableSpan().Fill(123f); // A previous step's contents must be overwritten.
                grads[input] = destination;
                foreach (var tensor in new[] { input, scale, shift, output, mean, variance, dy, destination })
                    gpu.EnsureResidentInput(tensor);
            }
            var saved = new object[] { mean, variance, Epsilon };
            DifferentiableOps.GradWriteGeneration = DifferentiableOps.NextGradWriteGeneration();
            for (int contribution = 1; contribution <= 2; contribution++)
            {
                BackwardFunctions<float>.BatchNormReluBackward(dy, new[] { input, scale, shift }, output, saved, engine, grads);
                Check(expectedX, grads[input], contribution);
                Check(expectedGamma, grads[scale], contribution);
                Check(expectedBeta, grads[shift], contribution);
            }
            // Starting a new step changes accumulation back to overwrite, even with the same destination dictionary.
            DifferentiableOps.GradWriteGeneration = DifferentiableOps.NextGradWriteGeneration();
            BackwardFunctions<float>.BatchNormReluBackward(dy, new[] { input, scale, shift }, output, saved, engine, grads);
            Check(expectedX, grads[input], 1);
            Check(expectedGamma, grads[scale], 1);
            Check(expectedBeta, grads[shift], 1);
        }
        finally
        {
            DifferentiableOps.GradWriteGeneration = priorGeneration;
            capturePath?.Dispose();
            if (resident) gpu?.ResumeActivationEviction();
        }
    }

    private static void Check(double[] expected, Tensor<float> actual, int contributions)
    {
        var values = actual.ToArray();
        Assert.Equal(expected.Length, values.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(contributions * expected[i] - values[i]) <= 2e-4 * (1 + Math.Abs(expected[i])),
                $"gradient[{i}], {contributions} contributions: expected {contributions * expected[i]}, got {values[i]}");
    }

    [Fact]
    public void Cpu_fallback_matches_finite_differences_and_resets_gradient_generation() => Verify(new CpuEngine(), false);

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gpu_eager_and_resident_destinations_match_finite_differences(bool resident)
    {
        Skip.IfNot(_fixture.IsAvailable && _fixture.Engine is not null, "No direct GPU backend available.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException("The fixture reported a GPU without an engine.");
        Verify(gpu, resident);
    }
}
