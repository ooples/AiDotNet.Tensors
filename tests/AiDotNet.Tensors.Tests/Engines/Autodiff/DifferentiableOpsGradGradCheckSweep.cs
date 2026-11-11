using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Second-order counterpart of <see cref="DifferentiableOpsGradCheckSweep"/>: every op classified
/// <c>DifferentiableOps</c> must survive <c>torch.autograd.gradgradcheck</c> — the gradient of
/// Σ_i ⟨∂⟨f(x), u⟩/∂x_i, v_i⟩ taken by a second backward through a <c>createGraph</c> first backward must match
/// the central differences of the first-order gradient.
/// </summary>
/// <remarks>
/// <para>
/// A backward that computes its gradient with a raw kernel or a host loop records nothing on the tape, so under
/// <c>createGraph</c> its gradient is a constant: the second backward silently returns zero (or nothing) for every
/// input behind it. Physics-informed networks (a Laplacian through a tanh MLP), gradient penalties and Hessian-vector
/// products all depend on this, and the first-order sweep cannot see it.
/// </para>
/// <para>
/// The ops the first-order sweep exempts are still run here: a kink or a designed (surrogate / straight-through)
/// gradient breaks the FIRST-order finite-difference comparison, but the second-order one compares the backward's own
/// derivative with the central differences of that same backward, which still has to hold. Only their second-order
/// mismatches are asserted. The stochastic ops are the exception: their forward differs between evaluations.
/// </para>
/// </remarks>
public partial class DifferentiableOpsGradGradCheckSweep
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public DifferentiableOpsGradGradCheckSweep(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Ops whose second derivative cannot be checked numerically, with the reason. Anything here is a deliberate,
    /// documented exclusion, not a silent gap.
    /// </summary>
    internal static readonly Dictionary<string, string> NotSecondOrderCheckable = new(StringComparer.Ordinal)
    {
        // The forward draws fresh randomness on every call, so neither derivative has a finite-difference reference.
        ["TensorDropout"] = "stochastic forward",
        ["TensorRandomLike"] = "stochastic forward",
        // An iterative reconstruction, exempted from the first-order sweep for the same reason.
        ["GriffinLim"] = "iterative phase reconstruction, not a single op",
        // Straight-through estimator: the forward is piecewise constant, so finite differences of its (designed,
        // mask-valued) gradient are zero while the recorded second backward is not. Its backward is still recorded
        // (gradient ⊙ mask), so it stays connected under create-graph.
        ["TensorFakeQuantize"] = "straight-through estimator: piecewise-constant forward, designed gradient",
    };

    /// <summary>
    /// Inputs that are non-differentiable by design (torch gives them no gradient), per op, with the reason: their
    /// first-order gradient is a constant zero, so the finite differences of the OTHER inputs' gradients with respect
    /// to them have no analytical counterpart. The op's remaining inputs are still checked.
    /// </summary>
    internal static readonly Dictionary<string, (int[] Inputs, string Reason)> NonDifferentiableInputs = new(StringComparer.Ordinal)
    {
        ["RoIAlign"] = (new[] { 1 }, "boxes: torchvision's roi_align gives rois no gradient"),
        ["PsRoIAlign"] = (new[] { 1 }, "boxes: torchvision's ps_roi_align gives rois no gradient"),
        ["RoIPool"] = (new[] { 1 }, "boxes: torchvision's roi_pool gives rois no gradient (and the bins are discontinuous in them)"),
    };

    /// <summary>
    /// Parameter variants (see <see cref="ParameterVariants"/>) whose numerical second-order check is invalid, with the
    /// reason, keyed <c>Op|parameter=Value</c>.
    /// </summary>
    internal static readonly Dictionary<string, string> NotSecondOrderCheckableVariants = new(StringComparer.Ordinal)
    {
    };

    /// <summary>
    /// Variants whose FIRST-order finite differences are invalid (a kink or a designed gradient the probe can hit),
    /// so only their second-order mismatches are asserted, as for the base ops in the first-order sweep's Exempt table.
    /// </summary>
    internal static readonly Dictionary<string, string> FirstOrderExemptVariants = new(StringComparer.Ordinal)
    {
    };

    /// <summary>
    /// The base arguments plus, one parameter at a time, every other value of each mode-like parameter: each value of
    /// an enum (an activation kind, a reduction kind, a padding or sampling mode), the other value of a bool (keepDims,
    /// includeSelf, alignCorners...), each axis of an <c>axis</c>/<c>dim</c> int, and each single axis plus all axes of
    /// an <c>axes</c>/<c>dims</c> array. A backward that is correct for the one value the table happens to pass and
    /// wrong for another (FusedLinear's Tanh epilogue was swept only as None) cannot hide this way. A variant the op
    /// rejects as an invalid argument is not one it supports, so it is dropped.
    /// </summary>
    // The per-area partial files (DifferentiableOpsGradGradCheckSweep.Exemptions*.cs) add their documented exclusions
    // here, after the initializers above, so areas never edit one shared table.
    static DifferentiableOpsGradGradCheckSweep()
    {
        AddExemptionsFusedActivations();
        AddExemptionsReductionsAndIndexing();
        AddExemptionsSpatialAndAttention();
        AddExemptionsLinearAlgebraAndLosses();
    }

    static partial void AddExemptionsFusedActivations();
    static partial void AddExemptionsReductionsAndIndexing();
    static partial void AddExemptionsSpatialAndAttention();
    static partial void AddExemptionsLinearAlgebraAndLosses();

    private readonly List<string> _rejectedVariants = new();

    private IEnumerable<(string Variant, object[] Args)> ParameterVariants(System.Reflection.MethodInfo m, object[] baseArgs)
    {
        yield return (string.Empty, baseArgs);
        var ps = m.GetParameters();
        int rank = baseArgs.OfType<Tensor<double>>().FirstOrDefault()?.Rank ?? 1;
        for (int i = 0; i < ps.Length; i++)
        {
            var type = ps[i].ParameterType;
            var pname = ps[i].Name ?? $"arg{i}";
            var lower = pname.ToLowerInvariant();
            var values = new List<(string, object?)>();
            if (type.IsEnum && !type.IsDefined(typeof(FlagsAttribute), false))
            {
                foreach (var v in Enum.GetValues(type))
                    if (!Equals(v, baseArgs[i])) values.Add(($"{pname}={v}", v));
            }
            else if (type == typeof(bool) && baseArgs[i] is bool b)
            {
                values.Add(($"{pname}={!b}", !b));
            }
            else if (type == typeof(int) && lower is "axis" or "dim" or "dimension" && baseArgs[i] is int axis)
            {
                for (int a = -rank; a < rank; a++)
                    if (a != axis && a != axis - rank && a != axis + rank) values.Add(($"{pname}={a}", a));
            }
            else if (type == typeof(int[]) && lower is "axes" or "dims" or "dimensions" && baseArgs[i] is int[] axes)
            {
                for (int a = 0; a < rank; a++)
                    if (!(axes.Length == 1 && axes[0] == a)) values.Add(($"{pname}=[{a}]", new[] { a }));
                if (axes.Length != rank) values.Add(($"{pname}=all", Enumerable.Range(0, rank).ToArray()));
            }

            foreach (var (label, value) in values)
            {
                var args = DifferentiableOpsGradCheckSweep.CopyArgs(baseArgs);
                args[i] = value!;
                bool rejected = false;
                try
                {
                    using (new NoGradScope<double>()) _ = m.Invoke(_engine, DifferentiableOpsGradCheckSweep.CopyArgs(args));
                }
                catch (Exception ex) when (DifferentiableOpsGradCheckSweep.Inner(ex) is ArgumentException or NotSupportedException
                                           or IndexOutOfRangeException or InvalidOperationException)
                {
                    rejected = true;
                    _rejectedVariants.Add($"{m.Name}|{label}: {DifferentiableOpsGradCheckSweep.Inner(ex).Message.Split('\n')[0]}");
                }
                if (!rejected) yield return (label, args);
            }
        }
    }

    [Fact]
    public void EveryDifferentiableOp_SecondDerivativeMatchesFiniteDifferences()
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _engine;
        var skipped = new List<string>();
        var secondOrderFailures = new List<string>();
        var firstOrderFailures = new List<string>();
        var threw = new List<string>();
        var exempted = new List<string>();
        var noGradient = new List<string>();
        var recordedTwice = new List<string>();
        int ok = 0;
        try
        {
            var harness = new DifferentiableOpsGradCheckSweep(_out);
            // Local iteration aid: GRADGRAD_OPS=Tanh,Softmax restricts the sweep to those ops (the assertion still holds
            // for whatever runs).
            var only = Environment.GetEnvironmentVariable("GRADGRAD_OPS") is { Length: > 0 } filter
                ? new HashSet<string>(filter.Split(',').Select(s => s.Trim()), StringComparer.Ordinal)
                : null;
            foreach (var (name, m, baseArgs) in harness.DrivableOps(skipped, includeDelegators: true))
            foreach (var (variant, args) in ParameterVariants(m, baseArgs))
            {
                if (only is not null && !only.Contains(name)) continue;
                if (NotSecondOrderCheckable.TryGetValue(name, out var why)) { exempted.Add($"{name} ({why})"); continue; }
                if (variant.Length > 0 && NotSecondOrderCheckableVariants.TryGetValue($"{name}|{variant}", out var whyVariant))
                {
                    exempted.Add($"{name}|{variant} ({whyVariant})");
                    continue;
                }
                bool firstOrderExempt = DifferentiableOpsGradCheckSweep.Exempt.ContainsKey(name)
                    || (variant.Length > 0 && FirstOrderExemptVariants.ContainsKey($"{name}|{variant}"));
                string label = $"{name}|{DifferentiableOpsGradCheckSweep.ParamFingerprint(m)}" + (variant.Length > 0 ? $" {{{variant}}}" : "");

                var inputs = args.SelectMany(a => a switch
                {
                    Tensor<double> t => new[] { t },
                    Tensor<double>[] arr => arr,
                    IEnumerable<Tensor<double>> seq => seq.ToArray(),
                    _ => Array.Empty<Tensor<double>>(),
                }).Distinct().ToArray();

                // Only the inputs the backward differentiates: an input a first-order backward gives no gradient
                // (an index tensor, a target) is the first-order sweep's business.
                int[] differentiated;
                try
                {
                    using var tape = new GradientTape<double>();
                    var outT = (Tensor<double>)m.Invoke(_engine, DifferentiableOpsGradCheckSweep.CopyArgs(args))!;
                    if (outT.GradFn?.Backward?.Method is { } bm) label += $" [{bm.DeclaringType?.Name}.{bm.Name}]";
                    // One producer per tensor: an op that records itself AND the engine ops it is built from leaves its
                    // output with two tape entries, and a create-graph backward (a tape walk) runs both, doubling the
                    // gradient, while the first-order graph walk (GradFn) hides it.
                    int producers = 0;
                    for (int i = 0; i < tape.Entries.Count; i++)
                        if (ReferenceEquals(tape.Entries[i].Output, outT)) producers++;
                    if (producers > 1) recordedTwice.Add($"{label}: {producers} tape entries produce the output");
                    var grads = tape.ComputeGradients(_engine.ReduceSum(outT, null), inputs);
                    var excluded = NonDifferentiableInputs.TryGetValue(name, out var nd) ? nd.Inputs : Array.Empty<int>();
                    differentiated = Enumerable.Range(0, inputs.Length)
                        .Where(i => !excluded.Contains(i) && grads.TryGetValue(inputs[i], out var g) && g is not null).ToArray();
                }
                catch (Exception ex)
                {
                    threw.Add($"{label}: first-order backward threw {DifferentiableOpsGradCheckSweep.Inner(ex).GetType().Name}");
                    continue;
                }
                if (differentiated.Length == 0) { noGradient.Add(label); continue; }

                GradCheckResult result;
                try
                {
                    // The output is squared before it is contracted, so the op's upstream gradient depends on its
                    // inputs. Contracting the bare output with a constant weight cannot see a backward that drops the
                    // upstream gradient's graph: a linear op's (Narrow, ReduceSum, a transpose) gradient is then a
                    // constant whatever its backward records, and its second derivative is zero either way.
                    result = GradCheck.GradGradCheckMany<double>(
                        _ =>
                        {
                            var y = (Tensor<double>)m.Invoke(_engine, DifferentiableOpsGradCheckSweep.CopyArgs(args))!;
                            return _engine.TensorMultiply(y, y);
                        }, inputs,
                        new GradCheckOptions
                        {
                            Engine = _engine,
                            Epsilon = 1e-6,
                            AbsoluteTolerance = 1e-4,
                            RelativeTolerance = 1e-3,
                            InputsToCheck = differentiated,
                            MaxElementsPerInput = 4,
                            MaxReportedMismatches = 3,
                        });
                }
                catch (Exception ex)
                {
                    var inner = DifferentiableOpsGradCheckSweep.Inner(ex);
                    var frames = (inner.StackTrace ?? string.Empty).Split('\n').Select(l => l.Trim()).Take(4);
                    threw.Add($"{label}: createGraph backward threw {inner.GetType().Name}: {inner.Message} @ {string.Join(" <- ", frames)}");
                    continue;
                }

                var second = result.WorstMismatches.Where(x => x.Order == 2).ToList();
                var first = result.WorstMismatches.Where(x => x.Order == 1).ToList();
                if (second.Count > 0)
                    secondOrderFailures.Add($"{label}: " + string.Join("; ", second.Select(x =>
                        $"input {x.InputIndex}[{x.ElementIndex}] analytical {x.Analytical:G6} vs numerical {x.Numerical:G6}")));
                else if (first.Count > 0 && !firstOrderExempt)
                    firstOrderFailures.Add($"{label}: " + string.Join("; ", first.Select(x =>
                        $"input {x.InputIndex}[{x.ElementIndex}] analytical {x.Analytical:G6} vs numerical {x.Numerical:G6}")));
                else
                    ok++;
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }

        _out.WriteLine($"second-order OK            : {ok}");
        _out.WriteLine($"SECOND-ORDER MISMATCH      : {secondOrderFailures.Count}");
        _out.WriteLine($"first-order (random u) bad : {firstOrderFailures.Count}");
        _out.WriteLine($"THREW                      : {threw.Count}");
        _out.WriteLine($"no first-order gradient    : {noGradient.Count} (the first-order sweep owns these)");
        _out.WriteLine($"exempt (documented)        : {exempted.Count}");
        _out.WriteLine($"skipped (harness)          : {skipped.Count}");
        foreach (var s in secondOrderFailures.OrderBy(x => x)) _out.WriteLine("GRADGRAD  " + s);
        foreach (var s in firstOrderFailures.OrderBy(x => x)) _out.WriteLine("GRAD      " + s);
        foreach (var s in threw.OrderBy(x => x)) _out.WriteLine("THREW     " + s);
        foreach (var s in noGradient.OrderBy(x => x)) _out.WriteLine("NO-GRAD   " + s);
        foreach (var s in recordedTwice.OrderBy(x => x)) _out.WriteLine("TWICE     " + s);
        foreach (var s in exempted.OrderBy(x => x)) _out.WriteLine("exempt    " + s);
        foreach (var s in skipped.OrderBy(x => x)) _out.WriteLine("skip      " + s);
        _out.WriteLine($"variants the ops reject as invalid arguments: {_rejectedVariants.Count}");
        foreach (var s in _rejectedVariants.OrderBy(x => x)) _out.WriteLine("rejected  " + s);

        var all = secondOrderFailures.Concat(firstOrderFailures).Concat(threw).Concat(recordedTwice).Concat(skipped.Select(s => "SKIPPED " + s)).ToList();
        Assert.True(all.Count == 0,
            $"{secondOrderFailures.Count} op(s) fail gradgradcheck, {firstOrderFailures.Count} fail the random-weight first-order " +
            $"check, {threw.Count} threw, {skipped.Count} could not be driven:\n" + string.Join("\n", all.OrderBy(x => x).Take(80)));
    }

    /// <summary>
    /// Third derivatives: with every backward run under <c>createGraph</c> on ONE tape (the way a PINN differentiates a
    /// Laplacian residual with respect to its weights), the gradient of K(x) = Σ_i ⟨∂h/∂x_i, w_i⟩, h = Σ_i ⟨∂L/∂x_i, v_i⟩,
    /// L = ⟨op(x)², u⟩, must match the central differences of K. This is what catches a backward whose create-graph
    /// ops are recorded somewhere the next create-graph backward does not walk (a nested tape).
    /// </summary>
    [Fact]
    public void EveryDifferentiableOp_ThirdDerivativeMatchesFiniteDifferences()
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _engine;
        var skipped = new List<string>();
        var failures = new List<string>();
        var threw = new List<string>();
        int ok = 0, exempt = 0;
        try
        {
            var harness = new DifferentiableOpsGradCheckSweep(_out);
            var only = Environment.GetEnvironmentVariable("GRADGRAD_OPS") is { Length: > 0 } filter
                ? new HashSet<string>(filter.Split(',').Select(s => s.Trim()), StringComparer.Ordinal)
                : null;
            foreach (var (name, m, args) in harness.DrivableOps(skipped, includeDelegators: true))
            {
                if (only is not null && !only.Contains(name)) continue;
                // The first-order exemptions (kinks, stochastic forwards, designed gradients) cannot be finite-differenced
                // at third order either; the second-order sweep already covers their graph connectivity.
                if (NotSecondOrderCheckable.ContainsKey(name) || DifferentiableOpsGradCheckSweep.Exempt.ContainsKey(name)) { exempt++; continue; }
                string label = $"{name}|{DifferentiableOpsGradCheckSweep.ParamFingerprint(m)}";
                var inputs = args.SelectMany(a => a switch
                {
                    Tensor<double> t => new[] { t },
                    Tensor<double>[] arr => arr,
                    IEnumerable<Tensor<double>> seq => seq.ToArray(),
                    _ => Array.Empty<Tensor<double>>(),
                }).Distinct().ToArray();
                var excluded = NonDifferentiableInputs.TryGetValue(name, out var nd) ? nd.Inputs : Array.Empty<int>();

                try
                {
                    Tensor<double>[] sources;
                    int[] outputShape;
                    using (var tape = new GradientTape<double>())
                    {
                        var y = (Tensor<double>)m.Invoke(_engine, DifferentiableOpsGradCheckSweep.CopyArgs(args))!;
                        outputShape = (int[])y._shape.Clone();
                        var g = tape.ComputeGradients(_engine.ReduceSum(y, null), inputs);
                        sources = Enumerable.Range(0, inputs.Length)
                            .Where(i => !excluded.Contains(i) && g.TryGetValue(inputs[i], out var gi) && gi is not null)
                            .Select(i => inputs[i]).ToArray();
                    }
                    if (sources.Length == 0) continue;

                    var u = Weights(outputShape, 1);
                    var v = sources.Select((s, i) => Weights(s._shape, 100 + i)).ToArray();
                    var w = sources.Select((s, i) => Weights(s._shape, 200 + i)).ToArray();

                    // K(x) and, when asked, its gradient by a third create-graph backward on the same tape.
                    double K(bool differentiate, out Dictionary<Tensor<double>, Tensor<double>>? third)
                    {
                        third = null;
                        using var tape = new GradientTape<double>();
                        var y = (Tensor<double>)m.Invoke(_engine, DifferentiableOpsGradCheckSweep.CopyArgs(args))!;
                        var loss = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorMultiply(y, y), u), null);
                        var g1 = tape.ComputeGradients(loss, sources, createGraph: true);
                        var h = Contract(g1, sources, v);
                        if (h is null) return 0;
                        var g2 = tape.ComputeGradients(h, sources, createGraph: true);
                        var k = Contract(g2, sources, w);
                        if (k is null) return 0;
                        if (differentiate) third = tape.ComputeGradients(k, sources, createGraph: true);
                        return k[0];
                    }

                    K(differentiate: true, out var analytic);
                    var bad = new List<string>();
                    for (int s = 0; s < sources.Length && bad.Count < 3; s++)
                    {
                        var src = sources[s];
                        var a = analytic is not null && analytic.TryGetValue(src, out var ga) && ga is not null ? ga : null;
                        for (int e = 0; e < Math.Min(4, src.Length) && bad.Count < 3; e++)
                        {
                            double original = src[e];
                            const double eps = 1e-5;
                            src[e] = original + eps;
                            double hi = K(false, out _);
                            src[e] = original - eps;
                            double lo = K(false, out _);
                            src[e] = original;
                            double numerical = (hi - lo) / (2 * eps);
                            double got = a is null ? 0 : a[e];
                            if (!(Math.Abs(got - numerical) <= 1e-4 + 1e-3 * Math.Abs(numerical)))
                                bad.Add($"input {Array.IndexOf(inputs, src)}[{e}] analytical {got:G6} vs numerical {numerical:G6}");
                        }
                    }
                    if (bad.Count > 0) failures.Add($"{label}: {string.Join("; ", bad)}");
                    else ok++;
                }
                catch (Exception ex)
                {
                    var inner = DifferentiableOpsGradCheckSweep.Inner(ex);
                    var frames = (inner.StackTrace ?? string.Empty).Split('\n').Select(l => l.Trim()).Take(3);
                    threw.Add($"{label}: threw {inner.GetType().Name}: {inner.Message} @ {string.Join(" <- ", frames)}");
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }

        _out.WriteLine($"third-order OK      : {ok}");
        _out.WriteLine($"THIRD-ORDER MISMATCH: {failures.Count}");
        _out.WriteLine($"THREW               : {threw.Count}");
        _out.WriteLine($"exempt              : {exempt}");
        foreach (var s in failures.OrderBy(x => x)) _out.WriteLine("GRAD3  " + s);
        foreach (var s in threw.OrderBy(x => x)) _out.WriteLine("THREW3 " + s);
        var all = failures.Concat(threw).Concat(skipped.Select(s => "SKIPPED " + s)).ToList();
        Assert.True(all.Count == 0, $"{failures.Count} op(s) fail the third-order check, {threw.Count} threw:\n"
            + string.Join("\n", all.OrderBy(x => x).Take(80)));
    }

    private static Tensor<double> Weights(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>((int[])shape.Clone());
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    // Σ_i ⟨grads[sources_i], weights_i⟩ as recorded ops; null when no source has a gradient.
    private Tensor<double>? Contract(Dictionary<Tensor<double>, Tensor<double>> grads, Tensor<double>[] sources, Tensor<double>[] weights)
    {
        Tensor<double>? total = null;
        for (int i = 0; i < sources.Length; i++)
        {
            if (!grads.TryGetValue(sources[i], out var g) || g is null) continue;
            var term = _engine.ReduceSum(_engine.TensorMultiply(g, weights[i]), null);
            total = total is null ? term : _engine.TensorAdd(total, term);
        }
        return total;
    }
}
