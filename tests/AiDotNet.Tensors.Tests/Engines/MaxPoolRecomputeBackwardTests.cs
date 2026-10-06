using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A graph-mode MaxPool node saves only its geometry, so a compiled CPU training plan recovers each window's winner
/// by re-scanning the forward input (<see cref="CpuEngine.MaxPool2DBackwardRecomputeInto{T}"/>). The winner rule must
/// be exactly the saved-index one (first strict maximum, NaN never wins, an all-NaN/-inf window routes to index 0),
/// and the additions must land in the same order, so these compare bit for bit against the indexed path.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class MaxPoolRecomputeBackwardTests
{
    private static Tensor<float> Grid(int[] shape, int seed, int levels)
    {
        // A coarse value grid makes ties common, so "first maximum wins" is actually exercised.
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.Next(levels) - levels / 2;
        return t;
    }

    private static Tensor<float> Rnd(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertBitEqual(Tensor<float> expected, Tensor<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}: element {i} expected {expected[i]:R} got {actual[i]:R}");
    }

    [Theory]
    [InlineData(2, 3, 8, 8, 2, 2)]    // tiling windows
    [InlineData(2, 2, 7, 9, 2, 2)]    // tiling with an uncovered last row and column
    [InlineData(1, 3, 9, 9, 3, 2)]    // overlapping windows: a cell can win twice
    [InlineData(2, 2, 6, 10, 2, 3)]   // stride wider than the window: gaps between windows
    [InlineData(1, 1, 3, 3, 3, 1)]    // one window covering the whole plane
    [InlineData(2, 3, 6, 34, 2, 2)]   // 17 windows per row: two 8-window vector blocks plus a scalar tail
    [InlineData(1, 2, 5, 35, 2, 2)]   // vector blocks with an uncovered last row and column
    public void MatchesSavedIndexBackward(int n, int c, int h, int w, int pool, int stride)
    {
        var engine = new CpuEngine();
        var x = Grid(new[] { n, c, h, w }, 11, 5);
        // Windows with no finite winner: all NaN, and all -inf. Both route to plane index 0 under the indexed rule.
        x[0] = float.NaN; x[1] = float.NaN; x[w] = float.NaN; x[w + 1] = float.NaN;
        if (h >= 4 && w >= 4) { x[2] = float.NegativeInfinity; x[3] = float.NegativeInfinity; x[w + 2] = float.NegativeInfinity; x[w + 3] = float.NegativeInfinity; }
        var pools = new[] { pool, pool };
        var strides = new[] { stride, stride };
        engine.MaxPool2DWithTensorIndices(x, pools, strides, out var indices);
        var gradOut = Rnd(indices._shape, 12);
        if (gradOut.Length > 3) gradOut[3] = -0f;   // its winner must receive 0 + (-0) = +0, as the scatter writes it
        var expected = engine.MaxPool2DBackwardWithTensorIndices(gradOut, indices, x._shape, pools, strides);

        var actual = new Tensor<float>(x._shape);
        for (int i = 0; i < actual.Length; i++) actual[i] = 123f;   // stale contents must be fully replaced
        engine.MaxPool2DBackwardRecomputeInto(actual, gradOut, x, pool, pool, stride, stride, accumulate: false);
        AssertBitEqual(expected, actual, "overwrite");

        var prior = Rnd(x._shape, 13);
        var accumulated = new Tensor<float>(x._shape);
        prior.AsSpan().CopyTo(accumulated.AsWritableSpan());
        engine.MaxPool2DBackwardRecomputeInto(accumulated, gradOut, x, pool, pool, stride, stride, accumulate: true);
        var expectedSum = new Tensor<float>(x._shape);
        for (int i = 0; i < expectedSum.Length; i++) expectedSum[i] = prior[i] + expected[i];
        AssertBitEqual(expectedSum, accumulated, "accumulate");
    }

    /// <summary>
    /// The 2x2 stride-2 forward (MaxPool2DInto, the compiled plan's replay) runs eight windows per AVX step. It must
    /// keep the scalar rule exactly: start from the window's first tap and take a later tap only when strictly
    /// greater, so a NaN first tap survives, a later NaN never replaces, and -0/+0 ties keep the first.
    /// </summary>
    [Theory]
    [InlineData(2, 3, 6, 34)]   // 17 windows per row: two vector blocks plus a scalar tail
    [InlineData(1, 2, 5, 35)]   // odd height and width: the last row and column are not covered
    [InlineData(64, 16, 28, 28)] // the parity CNN's pool (parallel path)
    public void ForwardMatchesScalarRule(int n, int c, int h, int w)
    {
        var engine = new CpuEngine();
        var x = Grid(new[] { n, c, h, w }, 41, 5);
        x[0] = float.NaN;                       // NaN first tap: the window's max is NaN
        x[3] = float.NaN;                       // NaN later tap: never replaces
        x[4] = -0f; x[5] = 0f; x[w + 4] = -0f; x[w + 5] = -0f;   // a -0/+0 tie keeps the first (-0)
        int oh = (h - 2) / 2 + 1, ow = (w - 2) / 2 + 1;
        var y = new Tensor<float>(new[] { n, c, oh, ow });
        engine.MaxPool2DInto(y, x, 2, 2, 0);
        for (int p = 0; p < n * c; p++)
        for (int r = 0; r < oh; r++)
        for (int col = 0; col < ow; col++)
        {
            int b0 = p * h * w + 2 * r * w + 2 * col;
            float m = x[b0];
            float v = x[b0 + 1]; if (v > m) m = v;
            v = x[b0 + w]; if (v > m) m = v;
            v = x[b0 + w + 1]; if (v > m) m = v;
            float got = y[(p * oh + r) * ow + col];
            Assert.True(BitConverter.SingleToInt32Bits(m) == BitConverter.SingleToInt32Bits(got),
                $"plane {p} ({r},{col}): expected {m:R} got {got:R}");
        }
    }

    [Fact]
    public void ResultIsIndependentOfThreadCount()
    {
        var engine = new CpuEngine();
        var x = Grid(new[] { 8, 16, 28, 28 }, 21, 7);
        engine.MaxPool2DWithTensorIndices(x, new[] { 2, 2 }, new[] { 2, 2 }, out var indices);
        var gradOut = Rnd(indices._shape, 22);
        int prior = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var serial = new Tensor<float>(x._shape);
            engine.MaxPool2DBackwardRecomputeInto(serial, gradOut, x, 2, 2, 2, 2, accumulate: false);
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var parallel = new Tensor<float>(x._shape);
            engine.MaxPool2DBackwardRecomputeInto(parallel, gradOut, x, 2, 2, 2, 2, accumulate: false);
            AssertBitEqual(serial, parallel, "threads");
        }
        finally
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }

    /// <summary>
    /// End to end through a compiled training plan: the graph-mode pool node's gradient (now the recompute
    /// specialization) equals the eager tape's saved-index gradient bit for bit, and a second step does not add onto
    /// the first step's gradient.
    /// </summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(3, 2)]
    public void CompiledPlanGradientMatchesTape(int pool, int stride)
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = new CpuEngine();
            var p = Grid(new[] { 2, 3, 9, 11 }, 31, 6);
            int outH = (9 - pool) / stride + 1, outW = (11 - pool) / stride + 1;
            var coef = Rnd(new[] { 2, 3, outH, outW }, 32);

            float[] tapeGrad;
            using (var tape = new GradientTape<float>())
            {
                var y = engine.MaxPool2DWithTensorIndices(p, new[] { pool, pool }, new[] { stride, stride }, out _);
                var loss = engine.ReduceSum(engine.TensorMultiply(y, coef), null);
                tapeGrad = (float[])tape.ComputeGradients(loss, new[] { p })[p].GetFlattenedData().Clone();
            }

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var y = engine.MaxPool2DWithIndices(p, new[] { pool, pool }, new[] { stride, stride }, out _);
                engine.ReduceSum(engine.TensorMultiply(y, coef), null);
                plan = scope.CompileTraining(new[] { p });
            }
            try
            {
                for (int step = 0; step < 2; step++)
                {
                    plan.Step();
                    var g = plan.Gradients[0].AsSpan();
                    Assert.Equal(tapeGrad.Length, g.Length);
                    for (int i = 0; i < g.Length; i++)
                        Assert.True(BitConverter.SingleToInt32Bits(tapeGrad[i]) == BitConverter.SingleToInt32Bits(g[i]),
                            $"step {step} element {i}: tape {tapeGrad[i]:R} plan {g[i]:R}");
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
