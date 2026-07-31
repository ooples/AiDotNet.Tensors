using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <c>TensorWhere(Tensor&lt;T&gt; condition, x, y)</c> must record a gradient, and must weight it by the
/// SELECTION the forward made rather than by the raw mask values.
/// </summary>
/// <remarks>
/// <para>
/// Found by the gradcheck sweep: "TensorWhere: no gradient for ANY of its 3 tensor input(s)". This
/// overload computed its result and returned it with no <c>DifferentiableOps.Record*</c> call at all,
/// even though <c>OpRegistry</c> classifies <c>TensorWhere</c> as differentiable and
/// <c>WhereBackward</c> already contained a dedicated <c>Tensor&lt;T&gt;</c>-mask branch waiting for it.
/// The <c>Tensor&lt;bool&gt;</c> / <c>Tensor&lt;Bit&gt;</c> overloads and the GPU override all recorded;
/// only this one did not.
/// </para>
/// <para>
/// Separately, that mask branch multiplied the incoming gradient by the raw condition values while the
/// forward selects on <c>!= 0</c>. A mask entry of 2.0 therefore scaled gradX by 2 and produced
/// gradY = (1 - 2) = -1 — gradient invented where the forward performed a plain selection.
/// </para>
/// </remarks>
public class TensorWhereGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public TensorWhereGradTests(ITestOutputHelper o) => _out = o;

    /// <summary>
    /// d/dx of where(c, x, y) is 1 where c selects x and 0 elsewhere; symmetrically for y.
    /// Verified against central finite differences.
    /// </summary>
    [Fact]
    public void BinaryMask_GradientMatchesFiniteDifferences()
    {
        const int n = 6;
        var cond = new Tensor<double>([n]);
        var x = new Tensor<double>([n]);
        var y = new Tensor<double>([n]);
        var rng = new Random(17);
        for (int i = 0; i < n; i++)
        {
            cond[i] = i % 2 == 0 ? 1.0 : 0.0;
            x[i] = 0.3 + rng.NextDouble();
            y[i] = 0.3 + rng.NextDouble();
        }

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorWhere(cond, x, y);
        var loss = _engine.ReduceSum(outT, null);
        var grads = tape.ComputeGradients(loss, [x, y]);

        Assert.True(grads.TryGetValue(x, out var gx) && gx is not null, "no gradient recorded for x");
        Assert.True(grads.TryGetValue(y, out var gy) && gy is not null, "no gradient recorded for y");

        const double eps = 1e-6;
        foreach (var (name, t, g) in new[] { ("x", x, gx!), ("y", y, gy!) })
        {
            for (int i = 0; i < n; i++)
            {
                double orig = t[i];
                t[i] = orig + eps; double lp = _engine.TensorSum(_engine.TensorWhere(cond, x, y));
                t[i] = orig - eps; double lm = _engine.TensorSum(_engine.TensorWhere(cond, x, y));
                t[i] = orig;
                double numerical = (lp - lm) / (2 * eps);
                _out.WriteLine($"d/d{name}[{i}] analytical={g[i]:G10} numerical={numerical:G10} cond={cond[i]}");
                Assert.True(Math.Abs(g[i] - numerical) < 1e-6,
                    $"where gradient d/d{name}[{i}]: analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    /// <summary>
    /// The forward treats ANY non-zero condition as true, so a mask of 2.0 must yield exactly the same
    /// gradient as a mask of 1.0 — a pure selection, never a scaling.
    /// </summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(2.0)]
    [InlineData(-3.5)]
    [InlineData(0.25)]
    public void NonBinaryTruthyMask_IsASelectionNotAScaling(double truthy)
    {
        const int n = 4;
        var cond = new Tensor<double>([n]);
        var x = new Tensor<double>([n]);
        var y = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
        {
            cond[i] = i % 2 == 0 ? truthy : 0.0;
            x[i] = 0.5 + i * 0.1;
            y[i] = 0.9 - i * 0.1;
        }

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorWhere(cond, x, y);
        var loss = _engine.ReduceSum(outT, null);
        var grads = tape.ComputeGradients(loss, [x, y]);
        var gx = grads[x];
        var gy = grads[y];

        for (int i = 0; i < n; i++)
        {
            bool selectsX = i % 2 == 0;
            // The forward selected one operand verbatim, so d/dselected == 1 and d/dother == 0.
            Assert.Equal(selectsX ? 1.0 : 0.0, gx![i]);
            Assert.Equal(selectsX ? 0.0 : 1.0, gy![i]);
        }
    }

    /// <summary>The forward's selection itself must honour non-zero-means-true.</summary>
    [Fact]
    public void Forward_TreatsAnyNonZeroAsTrue()
    {
        var cond = new Tensor<double>([3]);
        cond[0] = 2.0; cond[1] = 0.0; cond[2] = -1.0;
        var x = new Tensor<double>([3]); x[0] = 10; x[1] = 11; x[2] = 12;
        var y = new Tensor<double>([3]); y[0] = 20; y[1] = 21; y[2] = 22;

        var r = _engine.TensorWhere(cond, x, y);
        Assert.Equal(10.0, r[0]);
        Assert.Equal(21.0, r[1]);
        Assert.Equal(12.0, r[2]);
    }
}
