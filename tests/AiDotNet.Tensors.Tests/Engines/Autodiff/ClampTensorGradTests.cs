using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <c>TensorClampTensor</c> must record a gradient, and must route it to whichever operand supplied
/// each output element.
/// </summary>
/// <remarks>
/// Found by the gradcheck sweep ("TensorClampTensor: no gradient for ANY of its 3 tensor input(s)").
/// The op returned its result with no <c>DifferentiableOps.Record*</c> call while its
/// <c>TensorClampMin</c> / <c>TensorClampMax</c> siblings both record. Gradient routing follows
/// PyTorch's clamp: an unclamped element passes its gradient to the tensor, a low-clamped element to
/// <c>min</c>, a high-clamped element to <c>max</c>.
/// </remarks>
public class ClampTensorGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public ClampTensorGradTests(ITestOutputHelper o) => _out = o;

    /// <summary>
    /// Central finite differences on every supplied operand. Perturbations are 1e-6 while operands sit
    /// well away from the clamp boundaries, so no probe crosses a kink.
    /// </summary>
    private void CheckAll(
        Tensor<double> tensor, Tensor<double>? min, Tensor<double>? max, string label)
    {
        Func<Tensor<double>> fwd = () => _engine.TensorClampTensor(tensor, min, max);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = fwd();
        var loss = _engine.ReduceSum(outT, null);

        var wrt = new List<Tensor<double>> { tensor };
        if (min is not null) wrt.Add(min);
        if (max is not null) wrt.Add(max);
        var grads = tape.ComputeGradients(loss, wrt.ToArray());

        const double eps = 1e-6;
        foreach (var t in wrt)
        {
            string name = ReferenceEquals(t, tensor) ? "tensor" : ReferenceEquals(t, min) ? "min" : "max";
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"{label}: no gradient for {name}");
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + eps; double lp = _engine.TensorSum(fwd());
                t[i] = orig - eps; double lm = _engine.TensorSum(fwd());
                t[i] = orig;
                double numerical = (lp - lm) / (2 * eps);
                _out.WriteLine($"{label} d/d{name}[{i}] analytical={g![i]:G10} numerical={numerical:G10}");
                Assert.True(Math.Abs(g[i] - numerical) < 1e-6,
                    $"{label}: d/d{name}[{i}] analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    private static Tensor<double> Of(params double[] vals)
    {
        var t = new Tensor<double>([vals.Length]);
        for (int i = 0; i < vals.Length; i++) t[i] = vals[i];
        return t;
    }

    [Fact]
    public void MinAndMax_GradientMatchesFiniteDifferences()
    {
        // below min | inside | above max | inside
        var tensor = Of(-1.0, 0.20, 5.0, 0.60);
        var min = Of(0.0, 0.0, 0.0, 0.0);
        var max = Of(1.0, 1.0, 1.0, 1.0);
        CheckAll(tensor, min, max, "min+max");
    }

    [Fact]
    public void MinOnly_GradientMatchesFiniteDifferences()
    {
        var tensor = Of(-2.0, 0.30, 0.70, -0.50);
        var min = Of(0.0, 0.0, 0.0, 0.0);
        CheckAll(tensor, min, null, "min-only");
    }

    [Fact]
    public void MaxOnly_GradientMatchesFiniteDifferences()
    {
        var tensor = Of(2.0, 0.30, 0.70, 3.50);
        var max = Of(1.0, 1.0, 1.0, 1.0);
        CheckAll(tensor, null, max, "max-only");
    }

    /// <summary>Broadcast bounds: a scalar bound accumulates gradient from every element it clamped.</summary>
    [Fact]
    public void BroadcastBounds_GradientIsSumReducedOverBroadcastPositions()
    {
        var tensor = new Tensor<double>([2, 3]);
        double[] vals = { -1.0, 0.25, 5.0, -2.0, 0.75, 6.0 };
        for (int i = 0; i < 6; i++) tensor[i] = vals[i];
        var min = new Tensor<double>([1]); min[0] = 0.0;
        var max = new Tensor<double>([1]); max[0] = 1.0;

        CheckAll(tensor, min, max, "broadcast");

        // Two elements were below min and two above max, so each scalar bound must have
        // collected exactly 2.0 of gradient from a plain sum loss.
        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(_engine.TensorClampTensor(tensor, min, max), null);
        var grads = tape.ComputeGradients(loss, [tensor, min, max]);
        Assert.Equal(2.0, grads[min]![0]);
        Assert.Equal(2.0, grads[max]![0]);
    }

    /// <summary>
    /// The forward applies min FIRST and then max, so when max lies below min the max bound wins and
    /// must receive the gradient. Pins that the backward mirrors the forward's comparison order.
    /// </summary>
    [Fact]
    public void MaxBelowMin_MaxWinsAndReceivesTheGradient()
    {
        var tensor = Of(0.5);
        var min = Of(2.0);   // pushes 0.5 up to 2.0 ...
        var max = Of(1.0);   // ... then max pulls it down to 1.0

        var outT = _engine.TensorClampTensor(tensor, min, max);
        Assert.Equal(1.0, outT[0]);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(_engine.TensorClampTensor(tensor, min, max), null);
        var grads = tape.ComputeGradients(loss, [tensor, min, max]);

        Assert.Equal(0.0, grads[tensor]![0]);
        Assert.Equal(0.0, grads[min]![0]);
        Assert.Equal(1.0, grads[max]![0]);
    }
}
