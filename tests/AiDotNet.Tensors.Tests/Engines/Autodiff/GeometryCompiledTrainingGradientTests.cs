using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A compiled training step must give AffineGrid, AffineGrid3D and the mode/padding GridSample the same gradients as
/// the tape. They had no GraphMode branch, so a compiled plan ran them once at trace time and held their outputs as
/// constants: a spatial transformer's localization head (theta) and everything feeding a grid got zero gradient.
/// </summary>
public class GeometryCompiledTrainingGradientTests
{
    private static Tensor<float> Rnd(int[] shape, int seed, double scale = 1.0)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static Tensor<float> Theta2D(int seed)
    {
        // Near-identity warps, so the sample points stay inside the image and every gradient term is live.
        var theta = Rnd(new[] { 2, 2, 3 }, seed, 0.1);
        for (int b = 0; b < 2; b++) { theta[b, 0, 0] += 1f; theta[b, 1, 1] += 1f; }
        return theta;
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double norm = 0;
        foreach (float v in expected) norm += Math.Abs(v);
        Assert.True(norm > 0, $"{what}: the tape's own gradient is zero, so the test proves nothing.");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * (1f + Math.Abs(expected[i])),
                $"{what}: element {i} tape {expected[i]:R} compiled {actual[i]:R}");
    }

    private static void AssertCompiledMatchesTape(Func<IEngine, Tensor<float>> forward, Tensor<float>[] parameters, string what)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var expected = new float[parameters.Length][];
            using (var tape = new GradientTape<float>())
            {
                var loss = forward(AiDotNetEngine.Current);
                var grads = tape.ComputeGradients(loss, parameters);
                for (int i = 0; i < parameters.Length; i++) expected[i] = grads[parameters[i]].GetFlattenedData();
            }

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                forward(AiDotNetEngine.Current);
                plan = scope.CompileTraining(parameters);
            }

            try
            {
                for (int step = 0; step < 2; step++)
                {
                    plan.Step();
                    for (int i = 0; i < parameters.Length; i++)
                        AssertClose(expected[i], plan.Gradients[i].AsSpan(), $"{what}, parameter {i}, step {step}");
                }
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [Fact]
    public void AffineGrid_ThenGridSample_CompiledThetaGradient_MatchesTape()
    {
        var theta = Theta2D(1);
        var image = Rnd(new[] { 2, 3, 8, 8 }, 2);
        var weights = Rnd(new[] { 2, 3, 8, 8 }, 3);
        AssertCompiledMatchesTape(e =>
            e.ReduceSum(e.TensorMultiply(e.GridSample(image, e.AffineGrid(theta, 8, 8)), weights), null),
            new[] { theta }, "AffineGrid -> GridSample");
    }

    [Theory]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Zeros, false)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Zeros, true)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Border, false)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Reflection, true)]
    [InlineData(GridSampleMode.Bicubic, GridSamplePadding.Reflection, false)]
    [InlineData(GridSampleMode.Bicubic, GridSamplePadding.Zeros, true)]
    public void GridSampleWithModes_CompiledInputAndGridGradients_MatchTape(GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        var image = Rnd(new[] { 2, 3, 8, 8 }, 4);
        var grid = Rnd(new[] { 2, 6, 6, 2 }, 5, 0.8);
        var weights = Rnd(new[] { 2, 3, 6, 6 }, 6);
        AssertCompiledMatchesTape(e =>
            e.ReduceSum(e.TensorMultiply(e.GridSample(image, grid, mode, padding, alignCorners), weights), null),
            new[] { image, grid }, $"GridSample({mode}, {padding}, {alignCorners})");
    }

    /// <summary>Nearest is piecewise constant, so only its input gradient is non-zero; the grid's is 0 by definition.</summary>
    [Theory]
    [InlineData(GridSamplePadding.Border, false)]
    [InlineData(GridSamplePadding.Reflection, true)]
    public void GridSampleNearest_CompiledInputGradient_MatchesTape(GridSamplePadding padding, bool alignCorners)
    {
        var image = Rnd(new[] { 2, 3, 8, 8 }, 4);
        var grid = Rnd(new[] { 2, 6, 6, 2 }, 5, 1.3);
        var weights = Rnd(new[] { 2, 3, 6, 6 }, 6);
        AssertCompiledMatchesTape(e =>
            e.ReduceSum(e.TensorMultiply(e.GridSample(image, grid, GridSampleMode.Nearest, padding, alignCorners), weights), null),
            new[] { image }, $"GridSample(Nearest, {padding}, {alignCorners})");
    }

    [Fact]
    public void AffineGrid3D_CompiledThetaGradient_MatchesTape()
    {
        var theta = Rnd(new[] { 2, 3, 4 }, 7, 0.1);
        var weights = Rnd(new[] { 2, 3, 4, 5, 3 }, 8);
        AssertCompiledMatchesTape(e =>
            e.ReduceSum(e.TensorMultiply(e.AffineGrid3D(theta, 3, 4, 5), weights), null),
            new[] { theta }, "AffineGrid3D");
    }
}
