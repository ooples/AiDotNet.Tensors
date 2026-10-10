using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Pins the vectorized double ELU (FastExpDouble256 instead of a per-lane Math.Exp) and the dense single-axis
/// ReduceMax kernel (which also produces the max indices ReduceMaxBackward routes the gradient through).
/// </summary>
public class EluAndReduceMaxFastPathTests
{
    [Theory]
    [InlineData(1.0, 19)]     // SIMD body + scalar tail
    [InlineData(0.37, 4096)]  // alpha != 1, long run
    [InlineData(2.5, 3)]      // tail only
    public void Elu_MatchesScalarReference_Double(double alpha, int n)
    {
        var data = new double[n];
        var rng = new Random(n);
        for (int i = 0; i < n; i++) data[i] = (rng.NextDouble() * 2 - 1) * (i % 5 == 0 ? 800 : 4);   // includes < -708
        if (n > 7) data[7] = double.NaN;
        var y = new CpuEngine().ELU(new Tensor<double>(data, new[] { n }), alpha);
        for (int i = 0; i < n; i++)
        {
            double x = data[i], expected = x > 0 ? x : alpha * (Math.Exp(x) - 1.0);
            if (double.IsNaN(expected)) { Assert.True(double.IsNaN(y[i]), $"i={i}: NaN expected, got {y[i]}"); continue; }
            Assert.True(Math.Abs(expected - y[i]) <= 1e-14 * Math.Max(1, Math.Abs(expected)), $"i={i} x={x}: {y[i]:R} vs {expected:R}");
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void ReduceMax_SingleAxis_ValuesAndGradient_Double(int axis)
    {
        var engine = new CpuEngine();
        int[] shape = { 5, 4, 6 };
        var data = new double[5 * 4 * 6];
        for (int i = 0; i < data.Length; i++) data[i] = Math.Round(Math.Sin(i * 1.7) * 3);   // rounding makes ties
        var x = new Tensor<double>(data, shape);
        var outShape = new List<int>(shape); outShape.RemoveAt(axis);
        var w = new Tensor<double>(outShape.ToArray());
        for (int i = 0; i < w.Length; i++) w[i] = 1 + i * 0.25;                               // distinct weights

        double[] grad;
        Tensor<double> y;
        using (var tape = new GradientTape<double>())
        {
            y = engine.ReduceMax(x, new[] { axis }, keepDims: false);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null);
            var g = tape.ComputeGradients(loss, new[] { x })[x];
            grad = new double[g.Length];
            for (int i = 0; i < g.Length; i++) grad[i] = g.GetFlat(i);
        }

        // Reference: first (lowest axis index) max gets the output's weight; everything else gets zero.
        var expectedGrad = new double[data.Length];
        int[] strides = { shape[1] * shape[2], shape[2], 1 };
        for (int flatOut = 0; flatOut < w.Length; flatOut++)
        {
            int rem = flatOut; var coord = new int[3]; int k = outShape.Count - 1;
            for (int d = 2; d >= 0; d--) { if (d == axis) continue; coord[d] = rem % shape[d]; rem /= shape[d]; }
            double best = double.MinValue; int bestFlat = -1;
            for (int a = 0; a < shape[axis]; a++)
            {
                coord[axis] = a;
                int f = coord[0] * strides[0] + coord[1] * strides[1] + coord[2] * strides[2];
                if (data[f] > best) { best = data[f]; bestFlat = f; }
            }
            Assert.Equal(best, y.GetFlat(flatOut));
            expectedGrad[bestFlat] += w[flatOut];
        }
        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(expectedGrad[i] - grad[i]) < 1e-12, $"axis {axis} grad[{i}] {grad[i]} vs {expectedGrad[i]}");
    }
}
