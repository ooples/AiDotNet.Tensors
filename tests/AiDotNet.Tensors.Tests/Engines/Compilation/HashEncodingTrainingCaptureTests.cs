using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled training plan records the multiresolution hash encoding (Müller et al. 2022) as one node, so it trains
/// the hash tables and replays the encoding on each step's positions. Untraced, the encoding was invisible to the
/// plan, and Instant-NGP had to train on the eager tape.
/// </summary>
public class HashEncodingTrainingCaptureTests
{
    private const int Points = 37, Levels = 3, FeaturesPerLevel = 2, TableSize = 64;
    private static readonly int[] Resolutions = { 4, 9, 17 };

    private static Tensor<double> Positions(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(new[] { Points, 3 });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble();
        return t;
    }

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    [Fact]
    public void CompiledPlan_TrainsTheTables_OnEachStepsPositions()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = new CpuEngine();
            var tables = new Tensor<double>[Levels];
            for (int level = 0; level < Levels; level++) tables[level] = Random(new[] { TableSize, FeaturesPerLevel }, 10 + level);
            var coef = Random(new[] { Points, Levels * FeaturesPerLevel }, 20);
            var positions = Positions(1);

            ICompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.Enable())
            {
                var features = engine.MultiresolutionHashEncoding(positions, tables, Resolutions, FeaturesPerLevel);
                engine.ReduceSum(engine.TensorMultiply(features, coef), null);
                plan = scope.CompileTraining(tables);
            }

            try
            {
                for (int step = 0; step < 3; step++)
                {
                    // Each step samples new points into the traced positions buffer.
                    Positions(100 + step).AsSpan().CopyTo(positions.AsWritableSpan());
                    plan.Step();

                    // d(sum(features * coef)) / dfeatures = coef, scattered into the tables by the encoding backward.
                    var expected = engine.MultiresolutionHashEncodingBackward(positions, tables, Resolutions, FeaturesPerLevel, coef);
                    bool anyNonZero = false;
                    for (int level = 0; level < Levels; level++)
                    {
                        var actual = plan.Gradients[level].AsSpan();
                        var reference = expected[level].AsSpan();
                        Assert.Equal(reference.Length, actual.Length);
                        for (int i = 0; i < reference.Length; i++)
                        {
                            Assert.True(Math.Abs(reference[i] - actual[i]) <= 1e-12,
                                $"step {step}, level {level}, element {i}: plan {actual[i]:R} vs backward {reference[i]:R}");
                            anyNonZero |= reference[i] != 0.0;
                        }
                    }
                    Assert.True(anyNonZero, $"step {step}: the reference gradient is all zero, so the check proves nothing.");
                }
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }
}