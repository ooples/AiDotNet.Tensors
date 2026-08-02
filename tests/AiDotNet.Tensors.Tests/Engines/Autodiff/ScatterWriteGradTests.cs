using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Gradients for the overwrite-scatter family: <c>TensorSelectScatter</c>, <c>TensorSliceScatter</c>
/// and <c>TensorPut</c>.
/// </summary>
/// <remarks>
/// <para>
/// All three write <c>source</c> over part of <c>tensor</c> and recorded nothing, so neither operand
/// received a gradient. They were invisible to the gradcheck sweep until it could construct valid
/// arguments for them (index tensors in range, matching source ranks) — adding those table entries is
/// what exposed all three.
/// </para>
/// <para>
/// An overwrite splits the gradient cleanly: overwritten positions came entirely from <c>source</c>, so
/// the destination must receive the incoming gradient with exactly those positions ZEROED. Asserting
/// only "a gradient exists" would miss a backward that double-counts the overwritten region, so these
/// tests check the zeros explicitly.
/// </para>
/// </remarks>
public class ScatterWriteGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public ScatterWriteGradTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Seq(int[] shape, double start = 1.0)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + i;
        return t;
    }

    /// <summary>Row 1 of a [3,2] tensor is replaced, so only row 1's gradient goes to source.</summary>
    [Fact]
    public void SelectScatter_DestinationLosesGradientExactlyWhereOverwritten()
    {
        var tensor = Seq([3, 2]);
        var source = Seq([2], 100);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorSelectScatter(tensor, source, dim: 0, index: 1);
        // Distinct weight per output cell so misrouted gradient is detectable.
        var w = Seq([3, 2], 1);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        var gt = grads[tensor];
        var gs = grads[source];
        // Row 0 and row 2 keep their weights; row 1 (indices 2,3) is zeroed.
        Assert.Equal(1.0, gt![0]);
        Assert.Equal(2.0, gt[1]);
        Assert.Equal(0.0, gt[2]);
        Assert.Equal(0.0, gt[3]);
        Assert.Equal(5.0, gt[4]);
        Assert.Equal(6.0, gt[5]);
        // source received exactly row 1's weights.
        Assert.Equal(3.0, gs![0]);
        Assert.Equal(4.0, gs[1]);
    }

    /// <summary>Rows [1,3) of a [4,2] tensor are replaced.</summary>
    [Fact]
    public void SliceScatter_DestinationLosesGradientExactlyWhereOverwritten()
    {
        var tensor = Seq([4, 2]);
        var source = Seq([2, 2], 100);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorSliceScatter(tensor, source, dim: 0, start: 1, length: 2);
        var w = Seq([4, 2], 1);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        var gt = grads[tensor];
        var gs = grads[source];
        Assert.Equal(1.0, gt![0]);
        Assert.Equal(2.0, gt[1]);
        for (int i = 2; i <= 5; i++) Assert.Equal(0.0, gt[i]);   // rows 1-2 overwritten
        Assert.Equal(7.0, gt[6]);
        Assert.Equal(8.0, gt[7]);
        for (int i = 0; i < 4; i++) Assert.Equal(3.0 + i, gs![i]);
    }

    [Fact]
    public void Put_DestinationLosesGradientExactlyWhereOverwritten()
    {
        var tensor = Seq([2, 3]);
        var source = Seq([2], 100);
        var indices = new Tensor<int>([2]);
        indices[0] = 1; indices[1] = 4;

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorPut(tensor, indices, source);
        var w = Seq([2, 3], 1);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        var gt = grads[tensor];
        Assert.Equal(1.0, gt![0]);
        Assert.Equal(0.0, gt[1]);   // overwritten
        Assert.Equal(3.0, gt[2]);
        Assert.Equal(4.0, gt[3]);
        Assert.Equal(0.0, gt[4]);   // overwritten
        Assert.Equal(6.0, gt[5]);
        Assert.Equal(2.0, grads[source]![0]);
        Assert.Equal(5.0, grads[source]![1]);
    }

    /// <summary>
    /// DUPLICATE indices: the forward writes in order, so the LAST write to a position wins and the
    /// earlier source element was discarded — it must receive 0. Giving every duplicate the same
    /// gradient would invent gradient for a value that never reached the output.
    /// </summary>
    [Fact]
    public void Put_DuplicateIndices_OnlyTheLastWriterGetsGradient()
    {
        var tensor = Seq([4]);
        var source = Seq([3], 100);          // source[0], source[1] both target index 2
        var indices = new Tensor<int>([3]);
        indices[0] = 2; indices[1] = 2; indices[2] = 0;

        // Forward: index 2 ends up holding source[1], not source[0].
        var fwd = _engine.TensorPut(tensor, indices, source);
        Assert.Equal(101.0, fwd[2]);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var w = Seq([4], 1);                 // weights 1,2,3,4
        var loss = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorPut(tensor, indices, source), w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        var gs = grads[source];
        _out.WriteLine($"gradSource = [{gs![0]}, {gs[1]}, {gs[2]}]");
        Assert.Equal(0.0, gs[0]);            // overwritten by source[1] -> no influence
        Assert.Equal(3.0, gs[1]);            // won index 2, weight 3
        Assert.Equal(1.0, gs[2]);            // wrote index 0, weight 1

        var gt = grads[tensor];
        Assert.Equal(0.0, gt![0]);           // overwritten
        Assert.Equal(2.0, gt[1]);
        Assert.Equal(0.0, gt[2]);            // overwritten
        Assert.Equal(4.0, gt[3]);
    }
}
