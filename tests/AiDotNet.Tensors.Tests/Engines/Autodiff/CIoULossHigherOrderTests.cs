using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// CIoU's designed gradient, to second order. CIoU = DIoU + α·v with the aspect-ratio weight α = v / (1 - IoU + v)
/// held constant during backpropagation (the CIoU paper; torchvision's complete_box_iou_loss computes it under
/// no_grad), so finite differences of the loss cannot check it: they differentiate α. What can be checked exactly is
/// that the backward is the derivative of DIoU(p, t) + α₀·v(p, t) with α₀ frozen at the evaluation point — at first
/// AND second order, in BOTH boxes (v depends on the target's aspect ratio too, which the backward used to drop).
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CIoULossHigherOrderTests : IDisposable
{
    private readonly IEngine _previous;
    private readonly CpuEngine _e = new();

    public CIoULossHigherOrderTests()
    {
        _previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _e;
    }

    public void Dispose() => AiDotNetEngine.Current = _previous;

    private static Tensor<double> Boxes(params double[] values) => new(values, new[] { values.Length / 4, 4 });

    // v = (4/π²)·(atan(w_t/h_t) - atan(w_p/h_p))², widths relu(x2 - x1) + 1e-7, as the loss defines it.
    private Tensor<double> Aspect(Tensor<double> p, Tensor<double> t)
    {
        int n = p.Shape[0];
        Tensor<double> Extent(Tensor<double> b, int lo, int hi) => _e.TensorAddScalar(_e.ReLU(_e.TensorSubtract(
            _e.TensorSlice(b, new[] { 0, hi }, new[] { n, 1 }), _e.TensorSlice(b, new[] { 0, lo }, new[] { n, 1 }))), 1e-7);
        var diff = _e.TensorSubtract(_e.TensorAtan(_e.TensorDivide(Extent(t, 0, 2), Extent(t, 1, 3))),
            _e.TensorAtan(_e.TensorDivide(Extent(p, 0, 2), Extent(p, 1, 3))));
        return _e.Reshape(_e.TensorMultiplyScalar(_e.TensorMultiply(diff, diff), 4.0 / (Math.PI * Math.PI)), new[] { n });
    }

    // DIoU(p, t) + α₀·v(p, t), α₀ = v / (1 - IoU + v + 1e-7) evaluated at the given boxes and then held constant.
    private Func<Tensor<double>, Tensor<double>, Tensor<double>> FrozenAlphaReference(Tensor<double> p0, Tensor<double> t0)
    {
        Tensor<double> alpha;
        using (new NoGradScope<double>())
        {
            var v = Aspect(p0, t0);
            var iou = _e.TensorNegate(_e.TensorAddScalar(_e.TensorIoULoss(p0, t0), -1.0));
            alpha = _e.TensorDivide(v, _e.TensorAddScalar(_e.TensorAdd(_e.TensorNegate(iou), v), 1.0 + 1e-7));
        }
        return (p, t) => _e.TensorAdd(_e.TensorDIoULoss(p, t), _e.TensorMultiply(alpha, Aspect(p, t)));
    }

    // (∇L, ∇⟨∇L, w⟩) for both boxes, with every backward on one tape and the first under create-graph.
    private (Dictionary<Tensor<double>, Tensor<double>> First, Dictionary<Tensor<double>, Tensor<double>> Second) Derivatives(
        Func<Tensor<double>, Tensor<double>, Tensor<double>> loss, Tensor<double> p, Tensor<double> t, Tensor<double> u,
        Tensor<double> wp, Tensor<double> wt)
    {
        using var tape = new GradientTape<double>();
        var l = _e.ReduceSum(_e.TensorMultiply(loss(p, t), u), null);
        var first = tape.ComputeGradients(l, new[] { p, t }, createGraph: true);
        var h = _e.TensorAdd(_e.ReduceSum(_e.TensorMultiply(first[p], wp), null), _e.ReduceSum(_e.TensorMultiply(first[t], wt), null));
        var second = tape.ComputeGradients(h, new[] { p, t });
        return (first, second);
    }

    [Fact]
    public void CIoULoss_FirstAndSecondDerivatives_AreThoseOfTheFrozenAlphaLoss_InBothBoxes()
    {
        var p = Boxes(0.1, 0.2, 1.4, 0.9, 0.5, 0.3, 1.1, 2.0, -0.4, 0.0, 0.7, 0.6);
        var t = Boxes(0.3, 0.1, 1.2, 1.5, 0.2, 0.6, 1.6, 1.4, -0.2, -0.3, 0.9, 0.4);
        var u = new Tensor<double>(new[] { 0.7, -1.1, 0.9 }, new[] { 3 });
        var wp = new Tensor<double>(new[] { 0.3, -0.5, 0.8, 0.2, -0.7, 0.4, 0.1, -0.9, 0.6, 0.5, -0.2, 0.3 }, new[] { 3, 4 });
        var wt = new Tensor<double>(new[] { -0.4, 0.6, 0.2, -0.8, 0.5, 0.1, -0.3, 0.7, -0.6, 0.2, 0.9, -0.1 }, new[] { 3, 4 });

        var ours = Derivatives((a, b) => _e.TensorCIoULoss(a, b), p, t, u, wp, wt);
        var reference = Derivatives(FrozenAlphaReference(p, t), p, t, u, wp, wt);

        foreach (var (name, box) in new[] { ("predicted", p), ("target", t) })
        {
            for (int i = 0; i < box.Length; i++)
            {
                Assert.True(Math.Abs(ours.First[box][i] - reference.First[box][i]) < 1e-10,
                    $"{name}[{i}] first order: {ours.First[box][i]:G12} vs {reference.First[box][i]:G12}");
                Assert.True(Math.Abs(ours.Second[box][i] - reference.Second[box][i]) < 1e-9,
                    $"{name}[{i}] second order: {ours.Second[box][i]:G12} vs {reference.Second[box][i]:G12}");
            }
        }
    }
}
