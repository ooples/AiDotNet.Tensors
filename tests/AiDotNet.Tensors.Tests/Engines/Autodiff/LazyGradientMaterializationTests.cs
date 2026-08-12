using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression coverage for AccumulateGrad's lazy in-place copy.
/// </summary>
/// <remarks>
/// <para>
/// The copy used to be computed eagerly, before the branch that decides whether it is needed.
/// The createGraph branch deliberately keeps the incoming <c>grad</c> so the recorded tape entry
/// chains back through the original <c>GradFn</c>, so that copy was allocated, never read, and
/// immediately garbage. Making it lazy must not disturb either contract, and the two contracts
/// pull in opposite directions -- which is why both are asserted here rather than only checking
/// that the call succeeds.
/// </para>
/// <para>
/// The graphs below route every parameter through <c>Permute</c>/<c>Reshape</c>, because those
/// are the ops whose VJP hands back a NON-contiguous gradient. On a contiguous gradient the lazy
/// and eager forms are trivially identical, so a contiguous-only test would pass no matter what
/// the code did.
/// </para>
/// </remarks>
public class LazyGradientMaterializationTests
{
    private static Tensor<float> Ramp(int[] shape, float scale = 0.5f)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ((i * 37) % 19 - 9) * scale;
        return t;
    }

    /// <summary>
    /// Normal backward must hand the optimizer a CONTIGUOUS gradient even though every
    /// contribution arrives through a permute. This is the branch that consumes the copy.
    /// </summary>
    [Fact]
    public void NormalBackward_NonContiguousContribution_StoresContiguousGradient()
    {
        var engine = new CpuEngine();
        var w = Ramp([4, 4]);

        using var tape = new GradientTape<float>();
        var permuted = engine.TensorPermute<float>(w, [1, 0]);
        var flat = engine.Reshape<float>(permuted, [16]);
        var loss = engine.ReduceSum<float>(engine.TensorMultiply<float>(flat, flat), null);

        var grads = tape.ComputeGradients(loss, new[] { w });

        Assert.True(grads.ContainsKey(w));
        Tensor<float> g = grads[w];
        Assert.True(g.IsContiguous,
            "normal backward must materialize a contiguous gradient; the optimizer writes into it in place");
        Assert.Equal(w.Length, g.Length);
    }

    /// <summary>
    /// Repeated accumulation into the same parameter: the destination is written more than
    /// once, which is the path that in-place-adds into the stored accumulator.
    /// </summary>
    [Fact]
    public void NormalBackward_RepeatedAccumulation_MatchesAnalyticGradient()
    {
        var engine = new CpuEngine();
        var w = Ramp([3, 3], 0.25f);

        using var tape = new GradientTape<float>();
        // w is reached by TWO paths, so its slot is first-written and then accumulated into.
        var viaPermute = engine.Reshape<float>(engine.TensorPermute<float>(w, [1, 0]), [9]);
        var direct = engine.Reshape<float>(w, [9]);
        var loss = engine.ReduceSum<float>(engine.TensorAdd<float>(viaPermute, direct), null);

        var grads = tape.ComputeGradients(loss, new[] { w });
        Tensor<float> g = grads[w];

        // d/dw of sum(wT + w) is 2 everywhere: one from each path.
        Assert.True(g.IsContiguous);
        for (int i = 0; i < g.Length; i++)
            Assert.Equal(2.0f, g[i], 4);
    }

    /// <summary>
    /// createGraph must keep the gradient differentiable. This is the branch that never reads
    /// the copy, and the one the eager form was wasting a full parameter-sized allocation on.
    /// </summary>
    [Fact]
    public void CreateGraphBackward_ProducesDifferentiableGradient()
    {
        var engine = new CpuEngine();
        var w = Ramp([4, 4], 0.125f);

        using var tape = new GradientTape<float>();
        var flat = engine.Reshape<float>(engine.TensorPermute<float>(w, [1, 0]), [16]);
        var loss = engine.ReduceSum<float>(engine.TensorMultiply<float>(flat, flat), null);

        var grads = tape.ComputeGradients(loss, new[] { w }, createGraph: true);
        Tensor<float> g = grads[w];

        Assert.Equal(w.Length, g.Length);
        // The point of createGraph: the gradient is itself a tape value, so it can be
        // differentiated again. A detached copy here would silently break double-backward.
        Assert.NotNull(g.GradFn);
    }

    /// <summary>
    /// The numbers, not just the plumbing: d/dw of sum(w*w) is 2w, whichever branch ran.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Backward_NumericResultIsIndependentOfCreateGraph(bool createGraph)
    {
        var engine = new CpuEngine();
        var w = Ramp([2, 3], 0.5f);

        using var tape = new GradientTape<float>();
        var flat = engine.Reshape<float>(engine.TensorPermute<float>(w, [1, 0]), [6]);
        var loss = engine.ReduceSum<float>(engine.TensorMultiply<float>(flat, flat), null);

        var grads = tape.ComputeGradients(loss, new[] { w }, createGraph);
        Tensor<float> g = grads[w];

        for (int i = 0; i < w.Length; i++)
            Assert.Equal(2.0f * w[i], g[i], 3);
    }
}
