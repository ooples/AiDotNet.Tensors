using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Finite-difference gradient checks for FAN-OUT graphs — one tensor consumed by several downstream
/// branches, where the backward must SUM the gradient contributions of every path.
/// </summary>
/// <remarks>
/// <para>
/// These exist BEFORE the graph-driven training executor that will need them, deliberately. Fan-out
/// accumulation is the one part of branched execution whose failure mode is SILENT: if a shared node
/// receives only one branch's gradient instead of the sum, nothing throws, no shape is wrong, and no
/// existing invariant notices — training simply converges to something worse. Writing the executor first
/// and the checks afterwards would mean developing the risky part against no test that can catch it.
/// </para>
/// <para>
/// A linear chain never exercises this: each node has exactly one consumer, so "accumulate" and "assign"
/// are indistinguishable. Only a branch tells them apart, which is why every case here forks and rejoins.
/// </para>
/// <para>
/// Finite differences are the reference because they are independent of the autodiff implementation. A
/// test that compared the tape against itself, or against a hand-derived formula transcribed from the same
/// reasoning that wrote the backward, would agree with a wrong backward.
/// </para>
/// </remarks>
[Collection("EngineCurrentGlobalState")]
public class FanOutGradientAccumulationTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static Tensor<double> Vec(params double[] values)
    {
        var t = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++) t[i] = values[i];
        return t;
    }

    /// <summary>Sum of all elements — a scalar loss so the seed is unambiguous.</summary>
    private static Tensor<double> SumAll(Tensor<double> t)
    {
        var acc = new Tensor<double>(new[] { 1 });
        var one = new Tensor<double>(new[] { 1 });
        one[0] = 1.0;
        acc = Engine.TensorMultiplyScalar(one, 0.0);
        for (int i = 0; i < t.Length; i++)
        {
            var slice = Engine.TensorNarrow(t, 0, i, 1);
            acc = Engine.TensorAdd(acc, slice);
        }
        return acc;
    }

    /// <summary>
    /// Central-difference gradient of <paramref name="loss"/> with respect to each element of the input.
    /// </summary>
    /// <remarks>
    /// Central rather than forward difference: the forward difference's error is O(h) and would be the
    /// same order as the discrepancies being hunted, so it cannot distinguish a slightly wrong gradient
    /// from its own truncation error.
    /// </remarks>
    private static double[] NumericGradient(double[] x, Func<double[], double> loss, double h = 1e-6)
    {
        var g = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[i] += h;
            minus[i] -= h;
            g[i] = (loss(plus) - loss(minus)) / (2.0 * h);
        }
        return g;
    }

    private static void AssertClose(double[] expected, double[] actual, string what, double tol = 1e-5)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double denom = Math.Max(1.0, Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i])));
            double rel = Math.Abs(expected[i] - actual[i]) / denom;
            if (rel > tol)
            {
                Assert.Fail(
                    $"{what}: element {i} analytic {actual[i]:R} vs finite-difference {expected[i]:R} "
                    + $"(relative {rel:E3}). For a fan-out graph the usual cause is that the shared node "
                    + "received ONE branch's gradient instead of the sum of all branches.");
            }
        }
    }

    [Fact]
    public async Task FirstWriteGradientSlots_OwnIndependentStorage()
    {
        await Task.Yield();

        using var tape = new GradientTape<double>();
        var x = Vec(0.5, -1.25, 2.0);
        var y = Vec(-0.75, 1.5, 0.25);
        var loss = SumAll(Engine.TensorAdd(x, y));

        var grads = tape.ComputeGradients(loss, new List<Tensor<double>> { x, y });

        Assert.NotSame(grads[x], grads[y]);
        double originalY = grads[y][0];
        grads[x][0] = 17.0;
        Assert.Equal(originalY, grads[y][0]);
    }

    [Fact]
    public async Task NestedIdentityFanOut_AccumulatesEveryPathExactlyOnce()
    {
        await Task.Yield();

        using var tape = new GradientTape<double>();
        var x = Vec(0.5, -1.25, 2.0);
        var left = Engine.TensorAdd(x, x);
        var right = Engine.TensorAdd(x, x);
        var loss = SumAll(Engine.TensorAdd(left, right));

        var grads = tape.ComputeGradients(loss, new List<Tensor<double>> { x });

        AssertClose(new[] { 4.0, 4.0, 4.0 }, grads[x].ToArray(), "nested identity fan-out", 1e-12);
    }

    [Fact]
    public void TwoBranchesRejoining_AccumulatesBothGradients()
    {
        // THE core case. x feeds two branches, both rejoin: loss = sum(2x) + sum(3x).
        // dL/dx must be 2 + 3 = 5 per element. A backward that overwrites instead of accumulating gives
        // either 2 or 3 — a plausible-looking gradient that is silently wrong by 40-60%.
        var start = new[] { 0.5, -1.25, 2.0 };

        double Loss(double[] v)
        {
            double s = 0.0;
            foreach (var e in v) s += (2.0 * e) + (3.0 * e);
            return s;
        }

        using var tape = new GradientTape<double>();
        var x = Vec(start);

        var branchA = Engine.TensorMultiplyScalar(x, 2.0);
        var branchB = Engine.TensorMultiplyScalar(x, 3.0);
        var loss = Engine.TensorAdd(SumAll(branchA), SumAll(branchB));

        var grads = tape.ComputeGradients(loss, new List<Tensor<double>> { x });
        Assert.True(grads.ContainsKey(x), "No gradient was produced for the fan-out source at all.");

        var analytic = new double[start.Length];
        for (int i = 0; i < start.Length; i++) analytic[i] = Convert.ToDouble(grads[x][i]);

        AssertClose(NumericGradient(start, Loss), analytic, "two branches rejoining");
    }

    [Fact]
    public void ThreeBranchesRejoining_AccumulatesAllThree()
    {
        // Three paths, because a backward that accumulates only the LAST two (a subtly different bug from
        // overwriting) still passes the two-branch case.
        var start = new[] { 1.5, -0.75 };

        double Loss(double[] v)
        {
            double s = 0.0;
            foreach (var e in v) s += e + (4.0 * e) + (0.5 * e);
            return s;
        }

        using var tape = new GradientTape<double>();
        var x = Vec(start);

        var a = Engine.TensorMultiplyScalar(x, 1.0);
        var b = Engine.TensorMultiplyScalar(x, 4.0);
        var c = Engine.TensorMultiplyScalar(x, 0.5);
        var loss = Engine.TensorAdd(Engine.TensorAdd(SumAll(a), SumAll(b)), SumAll(c));

        var grads = tape.ComputeGradients(loss, new List<Tensor<double>> { x });
        var analytic = new double[start.Length];
        for (int i = 0; i < start.Length; i++) analytic[i] = Convert.ToDouble(grads[x][i]);

        AssertClose(NumericGradient(start, Loss), analytic, "three branches rejoining");
    }

    [Fact]
    public void AsymmetricBranchDepths_StillAccumulate()
    {
        // ABCNet's actual shape: a shared trunk feeding a SHALLOW head and a DEEPER one. If accumulation
        // depends on both paths finishing at the same depth, this is where it breaks.
        var start = new[] { 0.3, 1.1, -2.2 };

        double Loss(double[] v)
        {
            double s = 0.0;
            foreach (var e in v)
            {
                double shallow = 2.0 * e;
                double deep = ((e * 3.0) + 1.0) * 2.0;   // two ops before rejoining
                s += shallow + deep;
            }
            return s;
        }

        using var tape = new GradientTape<double>();
        var x = Vec(start);

        var shallowHead = Engine.TensorMultiplyScalar(x, 2.0);
        var deepHead = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, 3.0), 1.0), 2.0);
        var loss = Engine.TensorAdd(SumAll(shallowHead), SumAll(deepHead));

        var grads = tape.ComputeGradients(loss, new List<Tensor<double>> { x });
        var analytic = new double[start.Length];
        for (int i = 0; i < start.Length; i++) analytic[i] = Convert.ToDouble(grads[x][i]);

        AssertClose(NumericGradient(start, Loss), analytic, "asymmetric branch depths");
    }

    [Fact]
    public void OneBranchDiscarded_ContributesNoGradient()
    {
        // The opposite error: over-accumulation. A branch computed but NOT used by the loss must add
        // nothing. A backward that walks every recorded op rather than only the loss's ancestors would
        // fold the dead branch in and inflate the gradient, which finite differences catch immediately
        // because the dead branch does not affect the loss at all.
        var start = new[] { 0.9, -1.4 };

        double Loss(double[] v)
        {
            double s = 0.0;
            foreach (var e in v) s += 2.0 * e;   // the 7x branch below is deliberately unused
            return s;
        }

        using var tape = new GradientTape<double>();
        var x = Vec(start);

        var used = Engine.TensorMultiplyScalar(x, 2.0);
        _ = Engine.TensorMultiplyScalar(x, 7.0);   // recorded, never reaches the loss
        var loss = SumAll(used);

        var grads = tape.ComputeGradients(loss, new List<Tensor<double>> { x });
        var analytic = new double[start.Length];
        for (int i = 0; i < start.Length; i++) analytic[i] = Convert.ToDouble(grads[x][i]);

        AssertClose(NumericGradient(start, Loss), analytic, "discarded branch");
    }
}
