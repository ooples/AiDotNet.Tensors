using System;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Gradients for <c>TensorIndexPut</c> and for <c>TensorMaskedFill</c>'s <c>Tensor&lt;Bit&gt;</c>
/// overload — both exposed once the gradcheck sweep could construct valid arguments.
/// </summary>
/// <remarks>
/// <c>TensorMaskedFill(Tensor&lt;Bit&gt;)</c> recorded for GraphMode and AutoTracer but never called
/// <c>DifferentiableOps.Record*</c>, so a Bit-masked fill silently produced no gradient while the
/// <c>Tensor&lt;bool&gt;</c> overload produced one. <c>TensorIndexPut</c> recorded nothing at all.
/// </remarks>
public class IndexPutAndBitMaskGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public IndexPutAndBitMaskGradTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Seq(int[] shape, double start = 1.0)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + i;
        return t;
    }

    private static Tensor<int> Idx(params int[] v)
    {
        var t = new Tensor<int>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    /// <summary>
    /// A filled position takes a constant, so its gradient is 0; unfilled positions pass through.
    /// </summary>
    [Fact]
    public void MaskedFill_BitMask_BlocksGradientAtFilledPositions()
    {
        var tensor = Seq([2, 3]);
        var mask = new Tensor<Bit>([2, 3]);
        for (int i = 0; i < 6; i++) mask[i] = i % 2 == 0;   // fill even positions

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorMaskedFill(tensor, mask, 0.25);
        var w = Seq([2, 3], 1.0);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor]);

        Assert.True(grads.TryGetValue(tensor, out var g) && g is not null, "no gradient recorded");
        for (int i = 0; i < 6; i++)
        {
            double expected = i % 2 == 0 ? 0.0 : w[i];
            _out.WriteLine($"[{i}] masked={i % 2 == 0} grad={g![i]} expected={expected}");
            Assert.Equal(expected, g![i]);
        }
    }

    /// <summary>Overwrite mode: written positions lose the destination's gradient entirely.</summary>
    [Fact]
    public void IndexPut_Overwrite_MovesGradientToSource()
    {
        var tensor = Seq([3, 2]);
        var source = Seq([2], 100);
        // Two index tensors (one per axis) addressing (0,0) and (1,1) -> flat 0 and 3.
        var indices = new[] { Idx(0, 1), Idx(0, 1) };

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorIndexPut(tensor, indices, source, accumulate: false);
        var w = Seq([3, 2], 1.0);        // weights 1..6
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        var gt = grads[tensor];
        Assert.Equal(0.0, gt![0]);       // flat 0 overwritten
        Assert.Equal(2.0, gt[1]);
        Assert.Equal(3.0, gt[2]);
        Assert.Equal(0.0, gt[3]);        // flat 3 overwritten
        Assert.Equal(5.0, gt[4]);
        Assert.Equal(6.0, gt[5]);
        Assert.Equal(1.0, grads[source]![0]);
        Assert.Equal(4.0, grads[source]![1]);
    }

    /// <summary>
    /// ACCUMULATE mode adds instead of replacing, so the destination keeps its FULL gradient
    /// everywhere — including the written positions. This is the case an overwrite-only backward
    /// gets wrong.
    /// </summary>
    [Fact]
    public void IndexPut_Accumulate_DestinationKeepsFullGradient()
    {
        var tensor = Seq([3, 2]);
        var source = Seq([2], 100);
        var indices = new[] { Idx(0, 1), Idx(0, 1) };

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorIndexPut(tensor, indices, source, accumulate: true);
        var w = Seq([3, 2], 1.0);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        var gt = grads[tensor];
        for (int i = 0; i < 6; i++) Assert.Equal(w[i], gt![i]);   // nothing zeroed
        Assert.Equal(1.0, grads[source]![0]);
        Assert.Equal(4.0, grads[source]![1]);
    }

    /// <summary>Duplicate targets under overwrite: only the last writer influenced the output.</summary>
    [Fact]
    public void IndexPut_DuplicateTargets_OnlyLastWriterGetsGradient()
    {
        var tensor = Seq([2, 2]);
        var source = Seq([2], 100);
        var indices = new[] { Idx(0, 0), Idx(1, 1) };   // both address (0,1) -> flat 1

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorIndexPut(tensor, indices, source, accumulate: false);
        Assert.Equal(101.0, outT[1]);                  // source[1] won
        var w = Seq([2, 2], 1.0);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        Assert.Equal(0.0, grads[source]![0]);          // overwritten, no influence
        Assert.Equal(2.0, grads[source]![1]);          // winner takes weight of flat 1
        Assert.Equal(0.0, grads[tensor]![1]);
    }
}
