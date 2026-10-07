using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compiled plan's specialized LogSoftmax backward (float, rank 2, last axis): dX = dY - softmax(X) * rowsum(dY),
/// written straight into the input's gradient. Pinned against a double-precision reference, including a ragged
/// column count, the multi-row case, and an input with a second consumer (which must accumulate, not overwrite).
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class CompiledLogSoftmaxBackwardTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledLogSoftmaxBackwardTests() => AiDotNetEngine.Current = new CpuEngine();

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    private static Tensor<float> Rand(int rows, int cols, int seed, float scale)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return new Tensor<float>(data, new[] { rows, cols });
    }

    /// <summary>d/dP of sum(logsoftmax(P) * Y) (+ sum(tanh(P) * Z) when <paramref name="z"/> is given).</summary>
    private static double[] Reference(Tensor<float> p, Tensor<float> y, Tensor<float>? z, int rows, int cols)
    {
        var pd = p.ToArray(); var yd = y.ToArray(); var zd = z?.ToArray();
        var g = new double[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity;
            for (int c = 0; c < cols; c++) max = Math.Max(max, pd[r * cols + c]);
            double sumExp = 0, sumY = 0;
            for (int c = 0; c < cols; c++) { sumExp += Math.Exp(pd[r * cols + c] - max); sumY += yd[r * cols + c]; }
            for (int c = 0; c < cols; c++)
            {
                int i = r * cols + c;
                double th = Math.Tanh(pd[i]);
                g[i] = yd[i] - Math.Exp(pd[i] - max) / sumExp * sumY + (zd is null ? 0 : zd[i] * (1 - th * th));
            }
        }
        return g;
    }

    private static void AssertClose(double[] expected, Tensor<float> actual, string what)
    {
        var got = actual.ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-5 * (1 + Math.Abs(expected[i])),
                $"{what}[{i}] = {got[i]:G9}, expected {expected[i]:G9}");
    }

    [Theory]
    [InlineData(64, 10)]
    [InlineData(5, 37)]
    [InlineData(1, 8)]
    public void GradientMatchesReference(int rows, int cols)
    {
        var engine = new CpuEngine();
        var p = Rand(rows, cols, rows * 100 + cols, 3f);
        var y = Rand(rows, cols, 7, 1f);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { p }))
        {
            var lsm = engine.TensorLogSoftmax(p, 1);
            var loss = engine.ReduceSum(engine.TensorMultiply(lsm, y), null);
            plan = scope.CompileTraining(new[] { p }, loss);
        }
        var expected = Reference(p, y, null, rows, cols);
        using (plan)
        {
            for (int step = 0; step < 2; step++)   // the second step overwrites the first step's buffer
            {
                plan.Step();
                AssertClose(expected, plan.Gradients[0], $"step {step} dP");
            }
        }
    }

    // Both recording orders, so in one of them the other consumer's backward runs before the LogSoftmax backward,
    // whose write must then add to it rather than overwrite it.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void InputWithASecondConsumer_AccumulatesBothContributions(bool logSoftmaxBranchFirst)
    {
        const int rows = 6, cols = 10;
        var engine = new CpuEngine();
        var p = Rand(rows, cols, 1, 2f);
        var y = Rand(rows, cols, 2, 1f);
        var z = Rand(rows, cols, 3, 1f);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { p }))
        {
            Tensor<float> a, b;
            if (logSoftmaxBranchFirst)
            {
                a = engine.ReduceSum(engine.TensorMultiply(engine.TensorLogSoftmax(p, 1), y), null);
                b = engine.ReduceSum(engine.TensorMultiply(engine.Tanh(p), z), null);
            }
            else
            {
                b = engine.ReduceSum(engine.TensorMultiply(engine.Tanh(p), z), null);
                a = engine.ReduceSum(engine.TensorMultiply(engine.TensorLogSoftmax(p, 1), y), null);
            }
            var loss = engine.TensorAdd(a, b);
            plan = scope.CompileTraining(new[] { p }, loss);
        }
        using (plan)
        {
            plan.Step();
            AssertClose(Reference(p, y, z, rows, cols), plan.Gradients[0], "dP");
        }
    }
}