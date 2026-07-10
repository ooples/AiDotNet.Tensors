using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression tests for the compiled/GraphMode <c>ReduceMax</c> path.
///
/// The GraphMode-recorded ReduceMax execute delegate used to compute a single
/// GLOBAL max over the whole input and write it to <c>output[0]</c> only —
/// correct solely when the output is scalar. For a genuine axis reduction (the
/// max-subtraction step of a numerically-stable softmax over a class/feature
/// axis) the output has many elements, so the delegate left <c>output[1..]</c>
/// as whatever the rented pool buffer last held. On a fresh zero-init buffer
/// that produced merely-wrong values; on a pooled buffer carrying a prior op's
/// NaN/Inf tail it leaked straight into softmax → loss/grad NaN, which surfaced
/// as intermittent, net-core-only training NaNs (e.g. SwinUNETR segmentation).
///
/// These tests pin the compiled ReduceMax output to the eager result across
/// every axis/keepDims combination so the delegate must fill the ENTIRE output.
/// </summary>
public class CompiledReduceMaxAxisTests
{
    [Theory]
    [InlineData(1, true)]
    [InlineData(1, false)]
    [InlineData(2, true)]
    [InlineData(0, true)]
    public void CompiledReduceMax_OverAxis_MatchesEager_FillsWholeOutput(int axis, bool keepDims)
    {
        var engine = new CpuEngine();

        // [batch=1, rows=3, cols=4] with distinct per-position values so that
        // each reduced slot's max is DIFFERENT from the global max — a delegate
        // that writes only output[0] (the global max) fails every other slot.
        var input = new Tensor<double>(new[]
        {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 1.0, 1.5, 2.5,
        }, new[] { 1, 3, 4 });

        var expected = engine.ReduceMax(input, new[] { axis }, keepDims, out _);

        using var cache = new CompiledModelCache<double>();
        var plan = cache.GetOrCompileInference(input._shape,
            () => engine.ReduceMax(input, new[] { axis }, keepDims, out _));
        var compiled = plan.Execute();

        Assert.Equal(expected.Shape.ToArray(), compiled.Shape.ToArray());
        Assert.Equal(expected.Length, compiled.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(double.IsNaN(compiled[i]) || double.IsInfinity(compiled[i]),
                $"compiled ReduceMax(axis={axis},keepDims={keepDims}) produced non-finite at [{i}] " +
                "— an unwritten output slot leaked pool garbage.");
            Assert.Equal(expected[i], compiled[i], 10);
        }
    }
}
