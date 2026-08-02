using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Control experiment for <see cref="DifferentiableOpsGradCheckSweep"/>.
/// </summary>
/// <remarks>
/// TensorLog is the ideal control: d/dx Σ log(x) = 1/x, so the expected gradient is known in
/// closed form and neither the analytical nor the numerical value needs to be trusted. If the
/// sweep reports a mismatch here, the harness is at fault, not the op.
/// </remarks>
public class GradCheckHarnessSanity
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public GradCheckHarnessSanity(ITestOutputHelper output) => _out = output;

    [Fact]
    public void TensorLog_AnalyticalNumericalAndClosedForm_Agree()
    {
        var x = new Tensor<double>([6]);
        var rng = new Random(1234);
        for (int i = 0; i < x.Length; i++) x[i] = 0.35 + rng.NextDouble() * 0.6;

        using var tape = new GradientTape<double>();
        var y = _engine.TensorLog(x);
        var loss = _engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, [x]);
        var analytical = grads[x];

        // Sweep eps: if TensorSum accumulates at reduced precision, tiny eps is swamped by
        // summation error and the finite difference is unreliable at 1e-6 while improving as eps
        // grows. That signature distinguishes "harness precision" from "wrong gradient".
        foreach (var e in new[] { 1e-6, 1e-5, 1e-4, 1e-3 })
        {
            double o = x[0];
            x[0] = o + e;
            double p = _engine.TensorSum(_engine.TensorLog(x));
            x[0] = o - e;
            double mm = _engine.TensorSum(_engine.TensorLog(x));
            x[0] = o;
            _out.WriteLine($"eps={e:G1}: numerical={(p - mm) / (2 * e):G10}  (closed form {1.0 / o:G10})");
        }

        const double eps = 1e-3;
        for (int k = 0; k < 4; k++)
        {
            double orig = x[k];

            x[k] = orig + eps;
            double lp = _engine.TensorSum(_engine.TensorLog(x));
            x[k] = orig - eps;
            double lm = _engine.TensorSum(_engine.TensorLog(x));
            x[k] = orig;

            double numerical = (lp - lm) / (2 * eps);
            double closedForm = 1.0 / orig;

            _out.WriteLine($"[{k}] x={orig:G17}  closed-form 1/x={closedForm:G10}  " +
                           $"analytical={analytical[k]:G10}  numerical={numerical:G10}");

            // The closed form is the arbiter.
            Assert.True(Math.Abs(analytical[k] - closedForm) / closedForm < 1e-8,
                $"[{k}] analytical {analytical[k]:G10} != 1/x {closedForm:G10} — the OP is wrong.");
            Assert.True(Math.Abs(numerical - closedForm) / closedForm < 1e-3,
                $"[{k}] numerical {numerical:G10} != 1/x {closedForm:G10} — the HARNESS is wrong.");
        }
    }

    /// <summary>
    /// Second control: does re-invoking an op after mutating its input in place actually observe
    /// the new value? If any caching keyed on tensor identity is in play, finite differences
    /// computed by the sweep would be silently stale.
    /// </summary>
    [Fact]
    public void InPlaceMutation_IsObservedByReinvocation()
    {
        var x = new Tensor<double>([4]);
        for (int i = 0; i < x.Length; i++) x[i] = 0.5;

        double before = _engine.TensorSum(_engine.TensorLog(x));
        x[0] = 0.9;
        double after = _engine.TensorSum(_engine.TensorLog(x));

        _out.WriteLine($"sum(log) before={before:G17} after={after:G17}");
        Assert.True(Math.Abs(after - before) > 1e-9,
            "Mutating the input in place did not change the recomputed forward — the sweep's finite " +
            "differences would be measuring a stale cached result.");
    }
}
