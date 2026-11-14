using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Why an op whose output moves smoothly with its input still records no gradient.</summary>
public enum NoGradientReason
{
    /// <summary>A random draw: torch's samplers (bernoulli, uniform_, ...) are not differentiable either.</summary>
    RandomDraw,

    /// <summary>An iterative reconstruction algorithm, not an op (GriffinLim runs 60 fixed-point iterations).</summary>
    IterativeAlgorithm,

    /// <summary>Intermediate quantities of a loss whose own backward consumes them (the CTC lattice).</summary>
    ConsumedByOwnLossBackward,

    /// <summary>A scaled low-precision kernel, not differentiable in torch either (torch._scaled_mm).</summary>
    LowPrecisionKernel,
}

/// <summary>
/// Audits <see cref="OpRegistry.NonDifferentiableOps"/> against the ops themselves.
/// </summary>
/// <remarks>
/// <para>
/// The classification was never checked. STFT and the FFT family sat in the set on the stated ground that their out
/// parameters cannot flow through the Record* helpers, which is false (STFT now records an edge per out result), and
/// because every sweep selected ops by a tensor return type, nothing ever finite-differenced them: STFT trained nothing
/// upstream of it. A loss through a classified op silently gets no gradient, so a wrong entry is a silent defect.
/// </para>
/// <para>
/// This drives every op in the set the sweep can invoke and finite-differences its output (the weighted sum of every
/// out result for an op with no return) at a few elements of each tensor input. A derivative that is nonzero and stable
/// under halving the step is a smooth dependence: the op is differentiable and must record a backward, unless
/// <see cref="NoGradientBy"/> says why not. Comparisons, indicators, rounding, integer and index ops have a zero
/// derivative at generic points and pass; an op whose output varies between identical calls is reported, not audited.
/// </para>
/// </remarks>
public partial class DifferentiableOpsGradCheckSweep
{
    /// <summary>The ops that may depend smoothly on their input and still record no gradient, each with its reason.</summary>
    internal static readonly Dictionary<string, NoGradientReason> NoGradientBy = new(StringComparer.Ordinal)
    {
        ["GriffinLim"] = NoGradientReason.IterativeAlgorithm,
        ["TensorCTCForwardBackward"] = NoGradientReason.ConsumedByOwnLossBackward,
        ["TensorCTCLabelPosteriors"] = NoGradientReason.ConsumedByOwnLossBackward,
        ["TensorScaledMatMul"] = NoGradientReason.LowPrecisionKernel,
        ["TensorScaledGroupedMatMul"] = NoGradientReason.LowPrecisionKernel,
        ["TensorBernoulli"] = NoGradientReason.RandomDraw,
        ["TensorBinomial"] = NoGradientReason.RandomDraw,
        ["TensorPoisson"] = NoGradientReason.RandomDraw,
        ["TensorUniform"] = NoGradientReason.RandomDraw,
        ["TensorCauchy"] = NoGradientReason.RandomDraw,
        ["TensorExponential"] = NoGradientReason.RandomDraw,
        ["TensorGeometric"] = NoGradientReason.RandomDraw,
        ["TensorLogNormal"] = NoGradientReason.RandomDraw,
        ["TensorMultinomial"] = NoGradientReason.RandomDraw,
        ["TensorRandLike"] = NoGradientReason.RandomDraw,
        ["TensorRandnLike"] = NoGradientReason.RandomDraw,
        ["TensorRandintLike"] = NoGradientReason.RandomDraw,
    };

    [Fact]
    public void EveryNonDifferentiableOp_HasNoSmoothDerivative()
    {
        var smooth = new List<string>();
        var exemptedAndSmooth = new HashSet<string>(StringComparer.Ordinal);
        var exemptedAndAudited = new HashSet<string>(StringComparer.Ordinal);
        var notDriven = new List<string>();
        var nondeterministic = new List<string>();
        int audited = 0;

        foreach (var def in typeof(IEngine).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                     .Where(m => !m.IsSpecialName && m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1)
                     .Where(HasTensorOutput))
        {
            string name = def.Name;
            if (!OpRegistry.NonDifferentiableOps.Contains(name)) continue;
            MethodInfo m;
            try { m = def.MakeGenericMethod(typeof(double)); }
            catch (ArgumentException) { notDriven.Add($"{name}: cannot bind <double>"); continue; }

            var args = DriveArgs(m, name);
            if (args is null) { notDriven.Add($"{name}|{ParamFingerprint(m)}: no valid invocation"); continue; }

            var inputs = InputTensors<double>(m, args);
            if (inputs.Length == 0) continue; // nothing to depend on (a creation op given only sizes)
            if (Evaluate(m, args) is not { } f0 || !(Evaluate(m, args) is { } f1 && f0.Equals(f1)))
            {
                nondeterministic.Add($"{name}|{ParamFingerprint(m)}");
                continue;
            }

            audited++;
            if (NoGradientBy.ContainsKey(name)) exemptedAndAudited.Add(name);
            if (SmoothDependence(m, args, inputs) is not { } finding) continue;
            if (NoGradientBy.ContainsKey(name)) exemptedAndSmooth.Add(name);
            else smooth.Add($"{name}|{ParamFingerprint(m)}: {finding}");
        }

        _out.WriteLine($"audited          : {audited}");
        _out.WriteLine($"smooth, no grad  : {smooth.Count}");
        _out.WriteLine($"exempted         : {string.Join(", ", exemptedAndSmooth.OrderBy(x => x))}");
        foreach (var s in smooth.OrderBy(x => x)) _out.WriteLine("SMOOTH  " + s);
        foreach (var s in nondeterministic.OrderBy(x => x)) _out.WriteLine("varies  " + s);
        foreach (var s in notDriven.OrderBy(x => x)) _out.WriteLine("skip    " + s);

        // An exemption for an op the audit drove and found constant is stale: the reason no longer describes the op.
        var stale = exemptedAndAudited.Except(exemptedAndSmooth).OrderBy(x => x).ToList();
        Assert.True(smooth.Count == 0,
            $"{smooth.Count} op(s) in OpRegistry.NonDifferentiableOps depend smoothly on their input, so a loss through "
            + "them silently gets no gradient. Record a backward and move each to DifferentiableOps, or add it to "
            + "NoGradientBy with its reason:\n" + string.Join("\n", smooth.OrderBy(x => x)));
        Assert.True(stale.Count == 0, "NoGradientBy entries with no smooth dependence (remove them): " + string.Join(", ", stale));
    }

    private object[]? DriveArgs(MethodInfo m, string name)
    {
        string fingerprint = $"{name}|{ParamFingerprint(m)}";
        bool hasTable = OpCases.TryGetValue(fingerprint, out var caseFactory)
                     || OpCases.TryGetValue($"{name}/{m.GetParameters().Length}", out caseFactory)
                     || OpCases.TryGetValue(name, out caseFactory);
        foreach (var shape in hasTable ? new[] { Array.Empty<int>() } : new[] { new[] { 6 }, new[] { 2, 3 } })
        {
            var rng = new Random(1234);
            object[]? args;
            try { args = hasTable && caseFactory is not null ? caseFactory(rng) : BuildArgs(m, shape, rng, out _); }
            catch (Exception ex) when (ex is ArgumentException or InvalidOperationException) { continue; }
            if (args is null) continue;
            if (Evaluate(m, args) is null) continue;
            return args;
        }
        return null;
    }

    // The op's output reduced to one number (Σ over its outputs as the sweeps weigh them), or null when it throws.
    private double? Evaluate(MethodInfo m, object[] args)
    {
        try
        {
            using var _ = new NoGradScope<double>();
            var output = InvokeOutput<double>(_engine, m, args);
            double sum = 0;
            foreach (var v in output.GetFlattenedData()) sum += v;
            return sum;
        }
        catch (Exception ex) when (ex is TargetInvocationException or ArgumentException or InvalidOperationException
                                       or NotSupportedException or IndexOutOfRangeException)
        {
            return null;
        }
    }

    // The first element of an input along which the output's derivative is nonzero and stable (the same at step h
    // and h/2): a smooth dependence. A jump or a kink is not stable under halving, a plateau is zero.
    private string? SmoothDependence(MethodInfo m, object[] args, Tensor<double>[] inputs)
    {
        const double h = 1e-4;
        for (int t = 0; t < inputs.Length; t++)
        {
            var x = inputs[t];
            int n = x.Length;
            foreach (int i in new[] { 0, n / 3, (2 * n) / 3, n - 1 }.Where(i => i >= 0 && i < n).Distinct())
            {
                double original = x[i];
                double? Derivative(double step)
                {
                    x[i] = original + step;
                    var plus = Evaluate(m, args);
                    x[i] = original - step;
                    var minus = Evaluate(m, args);
                    x[i] = original;
                    return plus is { } p && minus is { } q ? (p - q) / (2 * step) : null;
                }

                if (Derivative(h) is not { } d1 || Derivative(h / 2) is not { } d2) continue;
                if (double.IsNaN(d1) || double.IsInfinity(d1) || Math.Abs(d1) <= 1e-6) continue;
                if (Math.Abs(d1 - d2) <= 1e-4 + 1e-3 * Math.Abs(d1))
                    return $"d/d(input {t})[{i}] = {d1:G6} (step {h}), {d2:G6} (step {h / 2})";
            }
        }
        return null;
    }
}
