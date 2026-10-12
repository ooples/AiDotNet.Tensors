using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Review follow-ups not covered elsewhere: storage adoption keeps copy-on-write aliasing; a strided fill is one write;
/// an in-place op on a history whose tape has been disposed proceeds; empty optimizer parameters are skipped.
/// </summary>
public class InPlaceReviewFollowUpTests
{
    private readonly IEngine _engine = new CpuEngine();

    private static Tensor<double> X() => new Tensor<double>(new[] { -1.5, -0.25, 0.5, 2.0 }, new[] { 4 });

    [Fact]
    public void SetInPlace_FromACopyOnWriteClone_KeepsSeeingTheSourcesWrites()
    {
        var source = X();
        var peer = source.Clone();          // copy-on-write: source and peer share storage until one writes
        var target = new Tensor<double>(new[] { 2 });
        _engine.TensorSetInPlace(target, source);
        source[0] = 99;                     // detaches source's family from the peer; the target joined that family
        Assert.Equal(99, target[0]);
        Assert.Equal(-1.5, peer[0]);
    }

    [Fact]
    public void StridedFill_IsOneWrite()
    {
        var m = new Tensor<double>(new double[16], new[] { 4, 4 });
        var column = _engine.TensorSwapAxes(m, 0, 1);   // a non-contiguous view of m
        Assert.False(column.IsContiguous);
        int before = column.Version;
        column.Fill(5);
        Assert.Equal(before + 1, column.Version);
        Assert.All(m.ToArray(), v => Assert.Equal(5, v));
    }

    [Fact]
    public void InPlaceOnADisposedTapesHistory_Proceeds()
    {
        var x = X();
        Tensor<double> y;
        using (new GradientTape<double>()) y = _engine.TensorMultiplyScalar(x, 2.0);
        // The first tape is gone, so no backward can read y's old value; a new tape may overwrite it.
        using var tape = new GradientTape<double>();
        _engine.ReLUInPlace(y);
        Assert.Equal(0.0, y[0]);
        Assert.Equal(4.0, y[3]);
    }

    [Fact]
    public void EmptyBoundParameters_DoNotDivideByZero()
    {
        // AddShapedParameter refuses a zero dimension; a zero-element tensor bound the normal way still reaches Step,
        // where the RMS and the factored statistics would divide by zero.
        foreach (ShapedOptimizerBase optimizer in new ShapedOptimizerBase[] { new AdafactorOptimizer(), new MuonOptimizer() })
        {
            var p = new Tensor<float>(new[] { 0, 4 });
            optimizer.AddParamGroup().AddParameter(p);
            optimizer.Step(new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> { [p] = new Tensor<float>(new[] { 0, 4 }) });
            Assert.Equal(0, p.Length);
        }
    }
}
