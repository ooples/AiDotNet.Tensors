using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Sweeps every op classified <c>DifferentiableOps</c> and checks its recorded gradient against
/// central finite differences.
/// </summary>
/// <remarks>
/// <para>
/// The existing coverage checks that ops are CLASSIFIED (TapeCompletenessTests) and that GPU
/// matches CPU (GpuCpuAutoDifferentialTests). Neither checks that a gradient claimed to exist is
/// actually CORRECT. That gap is not hypothetical: Spectrogram recorded a tape node whose backward
/// delegated to ISTFT, a synthesis operator, and returned gradients ~1/nFft off with varying sign;
/// MelSpectrogram was classified non-differentiable and returned none at all; three GPU audio
/// overrides returned results without recording. Every one of those passed the existing suites.
/// </para>
/// <para>
/// This sweep is deliberately reported rather than asserted per-op: it invokes ops reflectively,
/// so it cannot construct valid arguments for all of them. Ops it cannot drive are listed as
/// SKIPPED with the reason, so the coverage gap is visible instead of silent. Only genuine
/// disagreements fail the test.
/// </para>
/// </remarks>
public class DifferentiableOpsGradCheckSweep
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public DifferentiableOpsGradCheckSweep(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Ops excluded from the numeric check, with the reason. Anything here is a deliberate,
    /// documented exclusion — not a silent gap.
    /// </summary>
    private static readonly Dictionary<string, string> Exempt = new(StringComparer.Ordinal)
    {
        // Non-smooth at points a random probe can land on. Finite differences straddle the kink
        // and disagree with any one-sided subgradient choice.
        ["TensorAbs"] = "non-smooth at 0",
        ["TensorSign"] = "piecewise constant",
        ["TensorReLU"] = "kink at 0",
        ["TensorLeakyReLU"] = "kink at 0",
        ["TensorMaximum"] = "kink where operands tie",
        ["TensorMinimum"] = "kink where operands tie",
        ["TensorClamp"] = "kink at the clamp bounds",
        ["TensorFloor"] = "piecewise constant",
        ["TensorCeiling"] = "piecewise constant",
        ["TensorRound"] = "piecewise constant",
        ["TensorTruncate"] = "piecewise constant",
        ["TensorHardTanh"] = "kink at the saturation bounds",
        ["TensorHardSigmoid"] = "kink at the saturation bounds",
        ["TensorReLU6"] = "kink at 0 and 6",

        // Stochastic — the forward differs between the taped call and each probe call.
        ["TensorDropout"] = "stochastic forward",
        ["TensorRandomLike"] = "stochastic forward",

        // Reductions over indices: the gradient is a selection, and a probe can move which
        // index wins.
        ["TensorMax"] = "argmax can switch between probes",
        ["TensorMin"] = "argmin can switch between probes",

        // Iterative algorithms rather than single ops.
        ["GriffinLim"] = "iterative phase reconstruction, not a single op",
    };

    /// <summary>
    /// Per-op argument tables: the semantically valid shapes and couplings each op actually
    /// requires.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the real fix for the coverage gap. Reflective synthesis cannot infer that
    /// <c>TensorMatMul</c> needs the last dim of <c>a</c> to equal the first of <c>b</c>, that
    /// convolutions want NCHW plus a matching kernel, or that a spectrum's bin count is tied to a
    /// transform length — so 79 ops were skipped with "threw ArgumentException" and another 31 for
    /// parameter types no heuristic can invent. Name-based heuristics helped with scalars but
    /// cannot express shape relationships.
    /// </para>
    /// <para>
    /// Entries win over the heuristic path. Ops with neither a working entry nor a working
    /// heuristic are reported individually as NEEDS TABLE ENTRY, so the remaining gap is an
    /// explicit, shrinkable list rather than a silent skip count.
    /// </para>
    /// </remarks>
    private static readonly Dictionary<string, Func<Random, object[]>> OpCases = new(StringComparer.Ordinal)
    {
        // --- matmul family: inner dimensions must agree ---
        ["TensorMatMul"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 2], r)],
        ["TensorMatMulTransposed"] = r => [SafeTensor([2, 3], r), SafeTensor([2, 3], r)],
        ["BatchMatMul"] = r => [SafeTensor([2, 2, 3], r), SafeTensor([2, 3, 2], r)],
        ["TensorOuterProduct"] = r => [SafeTensor([3], r), SafeTensor([4], r)],
        ["TensorOuter"] = r => [SafeTensor([3], r), SafeTensor([4], r)],
        ["TensorVecDot"] = r => [SafeTensor([4], r), SafeTensor([4], r)],
        ["TensorInner"] = r => [SafeTensor([4], r), SafeTensor([4], r)],
        ["TensorKron"] = r => [SafeTensor([2, 2], r), SafeTensor([2, 2], r)],
        ["TensorTrace"] = r => [SafeTensor([3, 3], r)],
        ["TensorCosineSimilarity"] = r => [SafeTensor([2, 4], r), SafeTensor([2, 4], r), 1, 1e-8],

        // --- elementwise binaries needing matched shapes ---
        ["TensorAddMany"] = r => [new[] { SafeTensor([4], r), SafeTensor([4], r), SafeTensor([4], r) }],
        ["TensorMultiplyMany"] = r => [new[] { SafeTensor([4], r), SafeTensor([4], r) }],
        ["TensorLdexp"] = r => [SafeTensor([4], r), SafeTensor([4], r)],

        // --- normalization: last-dim normalized shapes ---
        ["LayerNorm"] = r => [SafeTensor([2, 4], r), SafeTensor([4], r), SafeTensor([4], r), 1e-5],
        ["RMSNorm"] = r => [SafeTensor([2, 4], r), SafeTensor([4], r), 1e-6],

        // --- shape ops: the target shape must be consistent with the input ---
        ["Reshape"] = r => [SafeTensor([2, 3], r), new[] { 3, 2 }],
        ["TensorSqueeze"] = r => [SafeTensor([1, 4], r), 0],
        ["TensorTile"] = r => [SafeTensor([2, 2], r), new[] { 2, 1 }],
        ["TensorConcatenate"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }, 0],
        ["Concat"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }, 0],
        ["TensorStack"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }, 0],
        ["TensorHStack"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }],
        ["TensorVStack"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }],
        ["TensorColumnStack"] = r => [new[] { SafeTensor([3], r), SafeTensor([3], r) }],
        ["TensorRowStack"] = r => [new[] { SafeTensor([3], r), SafeTensor([3], r) }],
        ["TensorRot90"] = r => [SafeTensor([3, 3], r), 1, new[] { 0, 1 }],
        ["TensorSlice"] = r => [SafeTensor([4, 4], r), new[] { 0, 0 }, new[] { 2, 2 }],
        ["TensorBlockDiag"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }],
        ["TensorCartesianProd"] = r => [new[] { SafeTensor([2], r), SafeTensor([2], r) }],

        // --- spectral: bin count is tied to the transform length ---
        ["IRFFT"] = r => [SafeTensor([2 * (8 / 2 + 1)], r), 8],
        ["Spectrogram"] = r => [SafeTensor([64], r), 16, 4, 16, HannWindowFor(16)],

        // --- reductions with explicit axes ---
        ["ReduceMax"] = r => [SafeTensor([2, 3], r), new[] { 1 }, false],
    };

    /// <summary>Hann window of exactly nFft samples, matching CpuEngine's own definition.</summary>
    private static Tensor<double> HannWindowFor(int nFft)
    {
        var w = new Tensor<double>([nFft]);
        for (int i = 0; i < nFft; i++)
            w[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / Math.Max(1, nFft - 1));
        return w;
    }

    private static Tensor<double> SafeTensor(int[] shape, Random rng)
    {
        // Values in [0.35, 0.95]: away from 0 (log/sqrt/reciprocal domains, |.| kink) and away
        // from 1 (acos/atanh edges), all strictly positive so domain-restricted ops are valid.
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.35 + rng.NextDouble() * 0.6;
        return t;
    }

    private static bool IsTensorDouble(Type t) =>
        t == typeof(Tensor<double>) ||
        (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Tensor<>));

    /// <summary>
    /// Best-effort argument construction. Returns null when a parameter type is not something
    /// this harness can synthesize, which is reported as a skip.
    /// </summary>
    private static object[]? BuildArgs(MethodInfo m, int[] shape, Random rng, out int firstTensorIdx)
    {
        firstTensorIdx = -1;
        var ps = m.GetParameters();
        var args = new object[ps.Length];

        for (int i = 0; i < ps.Length; i++)
        {
            var pt = ps[i].ParameterType;

            if (pt.IsByRef) return null;                     // out/ref ops are covered elsewhere

            if (IsTensorDouble(pt))
            {
                args[i] = SafeTensor(shape, rng);
                if (firstTensorIdx < 0) firstTensorIdx = i;
                continue;
            }

            // Name-aware synthesis. Blanket values are actively harmful for semantically loaded
            // parameters: 0.5 for an `epsilon` is not a stabilizer but a large additive constant
            // (it produced a bogus ReduceLogVariance mismatch and collapsed BinaryCrossEntropy's
            // [eps, 1-eps] clamp to a point, making its forward constant), and 1 for an
            // `outputLength` desynchronizes a transform from its own spectrum.
            var pname = (ps[i].Name ?? string.Empty).ToLowerInvariant();
            bool IsEpsilon() => pname.Contains("epsilon") || pname == "eps" || pname.Contains("tolerance");
            bool IsLength() => pname.Contains("outputlength") || pname == "n" || pname == "nfft" || pname.Contains("signallength");
            bool IsAxis() => pname is "axis" or "dim" or "dimension";

            if (pt == typeof(double)) { args[i] = IsEpsilon() ? 1e-7 : 0.5; continue; }
            if (pt == typeof(float)) { args[i] = IsEpsilon() ? 1e-7f : 0.5f; continue; }
            if (pt == typeof(bool)) { args[i] = ps[i].HasDefaultValue && ps[i].DefaultValue is bool b ? b : false; continue; }
            if (pt == typeof(int))
            {
                if (IsLength()) { args[i] = shape[^1]; continue; }
                if (IsAxis()) { args[i] = shape.Length - 1; continue; }
                args[i] = ps[i].HasDefaultValue && ps[i].DefaultValue is int d ? d : 1;
                continue;
            }
            if (pt == typeof(int?) || pt == typeof(double?) || pt == typeof(float?))
            {
                var inner = Nullable.GetUnderlyingType(pt)!;
                if (IsEpsilon() && inner == typeof(double)) { args[i] = 1e-7; continue; }
                if (IsLength() && inner == typeof(int)) { args[i] = shape[^1]; continue; }
                args[i] = ps[i].HasDefaultValue ? ps[i].DefaultValue! : null!;
                continue;
            }
            if (pt == typeof(int[]))
            {
                // axes/dims default to the last axis, which is valid for any rank.
                args[i] = new[] { shape.Length - 1 };
                continue;
            }

            // A generic-T scalar (e.g. TensorBinaryCrossEntropy's `T epsilon`) — bind by name too.
            if (pt == typeof(double) || pt.IsGenericParameter)
            {
                args[i] = IsEpsilon() ? 1e-7 : 0.5;
                continue;
            }

            if (ps[i].HasDefaultValue) { args[i] = ps[i].DefaultValue!; continue; }

            return null;                                      // unsupported parameter type
        }

        return firstTensorIdx >= 0 ? args : null;
    }

    [Fact]
    public void EveryDifferentiableOp_GradientMatchesFiniteDifferences()
    {
        var engineType = typeof(IEngine);
        var candidates = engineType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => !m.IsSpecialName && m.IsGenericMethodDefinition)
            .Where(m => m.GetGenericArguments().Length == 1)
            .Where(m => IsTensorDouble(m.ReturnType))
            .ToList();

        var shapes = new[] { new[] { 6 }, new[] { 2, 3 } };
        var mismatches = new List<string>();
        var noGradient = new List<string>();
        var skipped = new List<string>();
        var checkedOk = new List<string>();
        var exempted = new List<string>();

        foreach (var def in candidates)
        {
            var name = def.Name;
            if (name.Contains('`')) name = name.Substring(0, name.IndexOf('`'));

            if (!OpRegistry.DifferentiableOps.Contains(name)) continue;
            if (Exempt.TryGetValue(name, out var why)) { exempted.Add($"{name} ({why})"); continue; }

            MethodInfo m;
            try { m = def.MakeGenericMethod(typeof(double)); }
            catch (Exception ex) { skipped.Add($"{name}: cannot bind <double> ({ex.GetType().Name})"); continue; }

            bool handled = false;
            string lastSkip = "no shape produced a valid invocation";

            // The per-op table wins over reflective synthesis: it is the only way to express shape
            // relationships (matmul inner dims, spectrum/transform-length coupling, NCHW layouts).
            bool hasTable = OpCases.TryGetValue(name, out var caseFactory);
            var shapesToTry = hasTable ? new[] { Array.Empty<int>() } : shapes;

            foreach (var shape in shapesToTry)
            {
                var rng = new Random(1234);
                object[]? args;
                if (hasTable)
                {
                    try { args = caseFactory!(rng); }
                    catch (Exception ex) { lastSkip = $"table entry threw {ex.GetType().Name}"; continue; }
                }
                else
                {
                    args = BuildArgs(m, shape, rng, out _);
                }
                if (args is null) { lastSkip = "unsupported parameter types (NEEDS TABLE ENTRY)"; continue; }

                // Sanity: does it even run untaped on this shape?
                try { _ = m.Invoke(_engine, CopyArgs(args)); }
                catch (Exception ex) { lastSkip = $"{shape.Length}D threw {Inner(ex).GetType().Name}"; continue; }

                // Check EVERY tensor parameter, not just the first. Taking the first is wrong for
                // ops whose leading tensor is not a differentiable input — TensorWhere's leading
                // argument is the condition mask, which correctly receives no gradient. Only flag
                // an op when NO tensor input receives one.
                // Flatten array-typed tensor arguments too. The variadic ops (TensorAddMany,
                // TensorConcatenate, TensorStack, TensorBlockDiag and friends) pass their tensors
                // wrapped in a single Tensor<double>[], which OfType<Tensor<double>>() does not
                // match -- leaving tensorInputs empty, so `got` is always null and the sweep
                // reports "no gradient" for them the moment any is classified differentiable.
                // That is a false accusation against a working op, not a caught regression.
                var tensorInputs = args
                    .SelectMany(arg => arg switch
                    {
                        Tensor<double> t => new[] { t },
                        Tensor<double>[] arr => arr,
                        _ => Array.Empty<Tensor<double>>()
                    })
                    .ToArray();
                Tensor<double> input;
                Tensor<double> analytical;
                try
                {
                    using var tape = new GradientTape<double>();
                    var outT = (Tensor<double>)m.Invoke(_engine, CopyArgs(args))!;
                    var loss = _engine.ReduceSum(outT, null);
                    var grads = tape.ComputeGradients(loss, tensorInputs);

                    var got = tensorInputs.FirstOrDefault(t => grads.TryGetValue(t, out var gg) && gg is not null);
                    if (got is null)
                    {
                        noGradient.Add($"{name}: no gradient for ANY of its {tensorInputs.Length} tensor input(s)");
                        handled = true;
                        break;
                    }
                    input = got;
                    analytical = grads[got];
                }
                catch (Exception ex)
                {
                    lastSkip = $"backward threw {Inner(ex).GetType().Name}: {Inner(ex).Message}";
                    continue;
                }

                if (analytical.Length != input.Length)
                {
                    mismatches.Add($"{name}: gradient shape [{string.Join(",", analytical.Shape.ToArray())}] " +
                                   $"does not match input [{string.Join(",", input.Shape.ToArray())}] " +
                                   $"| args: {DescribeArgs(m, args)}");
                    handled = true;
                    break;
                }

                // Central finite differences on a few elements.
                const double eps = 1e-6;
                var bad = new List<string>();
                int probes = Math.Min(4, input.Length);
                for (int k = 0; k < probes; k++)
                {
                    double orig = input[k];
                    double lp, lm;
                    try
                    {
                        input[k] = orig + eps;
                        lp = _engine.TensorSum((Tensor<double>)m.Invoke(_engine, CopyArgs(args))!);
                        input[k] = orig - eps;
                        lm = _engine.TensorSum((Tensor<double>)m.Invoke(_engine, CopyArgs(args))!);
                    }
                    finally { input[k] = orig; }

                    double numerical = (lp - lm) / (2 * eps);
                    double a = analytical[k];
                    double denom = Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(numerical)));
                    if (Math.Abs(a - numerical) / denom > 1e-4)
                        bad.Add($"[{k}] analytical {a:G6} vs numerical {numerical:G6}");
                }

                if (bad.Count > 0)
                    mismatches.Add($"{name} on {shape.Length}D: " + string.Join("; ", bad.Take(3)));
                else
                    checkedOk.Add(name);

                handled = true;
                break;
            }

            if (!handled) skipped.Add($"{name}: {lastSkip}");
        }

        _out.WriteLine($"gradient-checked OK : {checkedOk.Count}");
        _out.WriteLine($"MISMATCH            : {mismatches.Count}");
        _out.WriteLine($"NO GRADIENT         : {noGradient.Count}");
        _out.WriteLine($"exempt (documented) : {exempted.Count}");
        _out.WriteLine($"skipped (harness)   : {skipped.Count}");
        _out.WriteLine("");
        foreach (var s in mismatches) _out.WriteLine("MISMATCH   " + s);
        foreach (var s in noGradient) _out.WriteLine("NO-GRAD    " + s);
        _out.WriteLine("");
        foreach (var s in skipped.OrderBy(x => x)) _out.WriteLine("skip  " + s);

        if (mismatches.Count > 0 || noGradient.Count > 0)
        {
            Assert.Fail(
                $"{mismatches.Count} op(s) disagree with finite differences and {noGradient.Count} record no " +
                "gradient despite being classified differentiable.\n" +
                string.Join("\n", mismatches.Concat(noGradient).Take(40)));
        }
    }

    private static object[] CopyArgs(object[] args) => (object[])args.Clone();

    /// <summary>
    /// Renders the synthesized arguments so a reported finding is reproducible by hand. Without
    /// this, a mismatch says nothing about WHICH configuration produced it, and this sweep has
    /// already generated several findings that were artifacts of its own argument choices.
    /// </summary>
    private static string DescribeArgs(MethodInfo m, object[] args)
    {
        var ps = m.GetParameters();
        var parts = new List<string>(args.Length);
        for (int i = 0; i < args.Length; i++)
        {
            string v = args[i] switch
            {
                null => "null",
                Tensor<double> t => $"Tensor[{string.Join(",", t.Shape.ToArray())}]",
                int[] a => $"[{string.Join(",", a)}]",
                _ => Convert.ToString(args[i], System.Globalization.CultureInfo.InvariantCulture) ?? "?",
            };
            parts.Add($"{ps[i].Name}={v}");
        }
        return string.Join(", ", parts);
    }

    private static Exception Inner(Exception ex) => ex is TargetInvocationException { InnerException: { } inner } ? inner : ex;
}
