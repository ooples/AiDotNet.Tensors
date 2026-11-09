using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Rank conformance gate for every engine op that takes an axis. It finds the ops by reflection over
/// <see cref="IEngine"/>, so a new axis op is covered as soon as it is declared, with no test to write.
/// <para>
/// The rule: an op that runs on a rank-2 tensor along an axis must also run on ranks 1, 3 and 4 along every axis,
/// and give the same answer as the rank-2 op applied to the tensor with that axis moved last and the other axes
/// flattened into rows (the "lifted" reference). TensorIndexSelect and TensorScatter both failed this rule (their CPU
/// kernels accepted only 2-D input), and a GPU engine falls back to the CPU kernel whenever a gradient tape is
/// active, so such gaps break training on every device.
/// </para>
/// <para>
/// An op that is rank-specific by definition goes in <see cref="Exempt"/> with the reason. The test fails when an op
/// is in neither list and the probe cannot build arguments it accepts at rank 2, so nothing drops out silently.
/// </para>
/// </summary>
public sealed class AxisOpsRankConformanceTests
{
    private readonly ITestOutputHelper _out;
    public AxisOpsRankConformanceTests(ITestOutputHelper output) => _out = output;

    private static readonly string[] AxisNames = { "axis", "dim" };
    private static readonly string[] GradNames = { "gradOutput", "outputGradient", "gradient", "grad" };
    private static readonly string[] OutputNames = { "output", "softmaxOutput" };

    /// <summary>Ops outside the rule by definition, with the reason.</summary>
    private static readonly Dictionary<string, string> Exempt = new()
    {
        ["TensorCross"] = "the cross product is defined only along an axis of length 3",
        ["TensorExpandDims"] = "inserts an axis: its valid axes run to rank, not rank - 1",
        ["TensorSqueeze"] = "removes a length-1 axis; the probe's axes are longer than 1",
        ["TensorSqueezeInPlace"] = "removes a length-1 axis; the probe's axes are longer than 1",
        ["TensorUnflatten"] = "rewrites the shape; not an op along the axis",
        ["TensorUniqueDim"] = "the output length depends on the data",
    };

    /// <summary>Ops that run on every rank but cannot be compared with the lifted reference, with the reason.</summary>
    private static readonly Dictionary<string, string> RunOnly = new()
    {
        ["GumbelSoftmax"] = "draws noise per element; the lifted call draws it in a different order",
        ["TensorCategoricalSample"] = "draws one sample per fibre; the lifted call draws them in a different order",
        ["TensorNormExceptDim"] = "reduces every axis but one; the lifted reshape changes which axes those are",
        ["TensorUnsqueezeInPlace"] = "inserts an axis, so the result's rank follows the input's and has no rank-2 form",
    };

    private static readonly int[][] Shapes =
    {
        new[] { 6 },
        new[] { 4, 6 },
        new[] { 2, 4, 6 },
        new[] { 2, 4, 2, 6 },
    };

    private static List<MethodInfo> AllEngineMethods() =>
        typeof(IEngine).GetMethods()
            .Concat(typeof(IEngine).GetInterfaces().SelectMany(i => i.GetMethods()))
            .Where(m => m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1)
            .Distinct().ToList();

    private static bool HasAxis(MethodInfo m) =>
        m.GetParameters().Any(p => p.ParameterType == typeof(int) && AxisNames.Contains(p.Name));

    private static bool IsTensorOfT(Type t, MethodInfo m)
    {
        var g = m.GetGenericArguments()[0];
        if (t.IsByRef) t = t.GetElementType()!;
        return t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Tensor<>) && t.GetGenericArguments()[0] == g;
    }

    /// <summary>How the probe shapes the secondary tensors: like the input, like the indices, the input with the
    /// axis cut to 2 or 1, or the input with the axis removed.</summary>
    private enum Sec { X, Indices, Cut2, Cut1, Removed }

    /// <summary>Index tensor: <c>Flat</c> = 1-D, two entries; <c>Full</c> = the input's shape;
    /// <c>Cut</c> = the input's shape with the axis cut to 2.</summary>
    private enum Idx { None, Flat, Full, Cut }

    private sealed class Plan
    {
        public Plan(Idx index, Sec secondary)
        {
            Index = index;
            Secondary = secondary;
        }

        public Idx Index { get; }
        public Sec Secondary { get; }

        public override string ToString() => $"Plan {{ Index = {Index}, Secondary = {Secondary} }}";
    }

    [Fact]
    public void Every_axis_op_that_runs_on_rank_2_runs_on_every_rank_and_matches_the_lifted_reference()
    {
        var engine = new CpuEngine();
        var all = AllEngineMethods();
        var failures = new List<string>();
        var unprobed = new List<string>();
        int probed = 0;
        foreach (var def in all.Where(m => HasAxis(m) && m.GetParameters().Any(p => IsTensorOfT(p.ParameterType, m)))
                     .OrderBy(m => m.Name))
        {
            string sig = Signature(def);
            if (Exempt.ContainsKey(def.Name)) continue;
            var m = def.MakeGenericMethod(typeof(float));
            var fwd = Forward(all, def);
            var plan = FindRank2Plan(engine, m, fwd, out string why);
            if (plan is null)
            {
                unprobed.Add($"{sig}: {why}");
                continue;
            }
            probed++;
            foreach (var shape in Shapes)
            {
                if (shape.Length == 2) continue;
                for (int ax = 0; ax < shape.Length; ax++)
                {
                    string where = $"{sig} rank {shape.Length} axis {ax} [{string.Join(",", shape)}]";
                    object?[] args;
                    object?[] pristine;
                    object? actual;
                    try
                    {
                        args = BuildArgs(engine, m, fwd, shape, ax, plan);
                        // An in-place op mutates args, so the lifted reference starts from a copy taken before the call.
                        pristine = args.Select(a => a switch
                        {
                            Tensor<float> f => f.Clone(),
                            Tensor<int> i => (object)i.Clone(),
                            _ => a,
                        }).ToArray();
                        actual = Invoke(engine, m, args);
                    }
                    catch (Exception e)
                    {
                        failures.Add($"{where}: {Inner(e)}");
                        continue;
                    }
                    if (RunOnly.ContainsKey(def.Name) || !CanLift(m)) continue;
                    try
                    {
                        var lifted = Lift(pristine, m, shape, ax, plan);
                        var expected = Unlift(Invoke(engine, m, lifted), shape, ax);
                        if (Compare(actual, expected) is { } diff)
                            failures.Add($"{where}: differs from the lifted rank-2 reference: {diff}");
                    }
                    catch (Exception e) { failures.Add($"{where}: lifted rank-2 call failed: {Inner(e)}"); }
                }
            }
        }

        _out.WriteLine($"probed {probed} ops; {failures.Count} failures; {unprobed.Count} unprobed");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        foreach (var u in unprobed) _out.WriteLine("UNPROBED " + u);
        Assert.True(failures.Count == 0 && unprobed.Count == 0,
            $"{failures.Count} rank failures; {unprobed.Count} axis ops the probe cannot run at rank 2 "
            + "(make them run, or add them to Exempt with the reason):\n"
            + string.Join("\n", failures.Concat(unprobed.Select(u => "UNPROBED " + u))));
    }

    private static string Inner(Exception e)
    {
        while (e is TargetInvocationException { InnerException: not null } t) e = t.InnerException!;
        return $"{e.GetType().Name}: {e.Message.Split('\n')[0]}";
    }

    private static string Signature(MethodInfo m) =>
        $"{m.Name}({string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name.Split('`')[0].TrimEnd('&') + " " + p.Name))})";

    /// <summary>The forward op of a <c>*Backward</c> op: same name without the suffix, taking an axis.</summary>
    private static MethodInfo? Forward(List<MethodInfo> all, MethodInfo m)
    {
        if (!m.Name.EndsWith("Backward", StringComparison.Ordinal)) return null;
        string name = m.Name.Substring(0, m.Name.Length - "Backward".Length);
        var names = new HashSet<string>(m.GetParameters().Select(p => p.Name!));
        return all.Where(f => f.Name == name && HasAxis(f))
            .OrderByDescending(f => f.GetParameters().Count(p => names.Contains(p.Name!)))
            .FirstOrDefault()?.MakeGenericMethod(typeof(float));
    }

    private static IEnumerable<Plan> Candidates(MethodInfo m)
    {
        var ps = m.GetParameters();
        bool hasIdx = ps.Any(p => p.ParameterType == typeof(Tensor<int>));
        bool hasSec = ps.Count(p => p.ParameterType == typeof(Tensor<float>)) > 1;
        var secs = hasSec ? new[] { Sec.X, Sec.Removed, Sec.Cut1 } : new[] { Sec.X };
        if (!hasIdx)
        {
            foreach (var s in secs) yield return new Plan(Idx.None, s);
            yield break;
        }
        // torch order: index_select / index_add (1-D index), then gather / scatter (index of the input's rank).
        yield return new Plan(Idx.Flat, Sec.X);
        if (hasSec) yield return new Plan(Idx.Flat, Sec.Cut2);
        yield return new Plan(Idx.Full, Sec.X);
        if (hasSec) yield return new Plan(Idx.Cut, Sec.Indices);
        yield return new Plan(Idx.Cut, Sec.X);
        if (hasSec)
            foreach (var i in new[] { Idx.Flat, Idx.Full })
                foreach (var s in new[] { Sec.Removed, Sec.Cut1 })
                    yield return new Plan(i, s);
    }

    private static Plan? FindRank2Plan(IEngine engine, MethodInfo m, MethodInfo? fwd, out string why)
    {
        why = "";
        foreach (var plan in Candidates(m))
        {
            bool ok = true;
            for (int ax = 0; ax < 2 && ok; ax++)
            {
                try { Invoke(engine, m, BuildArgs(engine, m, fwd, Shapes[1], ax, plan)); }
                catch (Exception e)
                {
                    ok = false;
                    if (why.Length == 0) why = $"{plan} axis {ax}: {Inner(e)}";
                }
            }
            if (ok) return plan;
        }
        return null;
    }

    // ---- argument construction --------------------------------------------------------------------------------

    private static float[] Data(int n, int salt) =>
        Enumerable.Range(0, n).Select(i => 0.1f + ((i * 37 + salt * 11) % 101) / 101f).ToArray();

    private static int Prod(IEnumerable<int> s) => s.Aggregate(1, (a, b) => a * b);

    private static Tensor<float> Random(int[] shape, int salt) =>
        new(shape, new Vector<float>(Data(Prod(shape), salt)));

    private static int[] WithAxis(int[] shape, int ax, int len)
    {
        var s = (int[])shape.Clone();
        s[ax] = len;
        return s;
    }

    private static int[] Removed(int[] shape, int ax) => shape.Where((_, i) => i != ax).ToArray();

    /// <summary>Indices along <paramref name="ax"/>, distinct within each fibre so scatters are order-free.</summary>
    private static Tensor<int>? Indices(int[] shape, int ax, Idx mode)
    {
        int n = shape[ax];
        if (mode == Idx.None) return null;
        if (mode == Idx.Flat) return new Tensor<int>(new[] { n - 1, 0 }, new[] { 2 });
        var s = mode == Idx.Cut ? WithAxis(shape, ax, 2) : shape;
        int outer = Prod(s.Take(ax)), k = s[ax], inner = Prod(s.Skip(ax + 1));
        var d = new int[outer * k * inner];
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < k; j++)
                for (int i = 0; i < inner; i++)
                    d[(o * k + j) * inner + i] = (j + 1 + o + i) % n;
        return new Tensor<int>(d, s);
    }

    private static object? Invoke(IEngine engine, MethodInfo m, object?[] args)
    {
        var r = m.Invoke(engine, args);
        // An in-place op (TensorCumSumInPlace) returns nothing: its result is the tensor it mutated, the first one.
        if (m.ReturnType == typeof(void))
            r = args.OfType<Tensor<float>>().FirstOrDefault();
        var outs = m.GetParameters().Where(p => p.IsOut).Select(p => args[p.Position]).ToList();
        return outs.Count == 0 ? r : new object?[] { r }.Concat(outs).ToArray();
    }

    private static Tensor<float>? FirstTensor(object? r) => r switch
    {
        Tensor<float> t => t,
        object?[] a => a.Select(FirstTensor).FirstOrDefault(t => t is not null),
        null => null,
        _ when r.GetType().FullName!.StartsWith("System.ValueTuple", StringComparison.Ordinal)
            => r.GetType().GetFields().Select(f => FirstTensor(f.GetValue(r))).FirstOrDefault(t => t is not null),
        _ => null,
    };

    private static object?[] BuildArgs(IEngine engine, MethodInfo m, MethodInfo? fwd, int[] shape, int ax, Plan plan)
    {
        var ps = m.GetParameters();
        var x = Random(shape, 1);
        var idx = Indices(shape, ax, plan.Index);
        int[] secShape = plan.Secondary switch
        {
            Sec.Indices => idx!.Shape.ToArray(),
            Sec.Cut2 => WithAxis(shape, ax, 2),
            Sec.Cut1 => WithAxis(shape, ax, 1),
            Sec.Removed => Removed(shape, ax),
            _ => shape,
        };
        var sec = Random(secShape, 2);

        // A backward op gets its forward's input, the forward's output, and a gradient of the output's shape.
        Tensor<float>? fOut = null;
        if (fwd is not null)
        {
            var fps = fwd.GetParameters();
            var fargs = BuildArgs(engine, fwd, null, shape, ax, new Plan(plan.Index, Sec.X));
            fOut = FirstTensor(Invoke(engine, fwd, fargs));
            var fx = fps.Where(p => p.ParameterType == typeof(Tensor<float>)).Select(p => fargs[p.Position]).FirstOrDefault();
            if (fx is Tensor<float> t) x = t;
            if (fps.FirstOrDefault(p => p.ParameterType == typeof(Tensor<int>)) is { } fi) idx = (Tensor<int>?)fargs[fi.Position];
        }

        bool primaryTaken = false;
        bool hasList = ps.Any(p => p.ParameterType == typeof(Tensor<float>[]) || p.ParameterType == typeof(IReadOnlyList<Tensor<float>>));
        var args = new object?[ps.Length];
        foreach (var p in ps)
        {
            var t = p.ParameterType;
            object? v;
            if (p.IsOut) v = null;
            else if (t == typeof(Tensor<float>) && fOut is not null && GradNames.Contains(p.Name)) v = Random(fOut.Shape.ToArray(), 3);
            else if (t == typeof(Tensor<float>) && fOut is not null && OutputNames.Contains(p.Name)) v = fOut;
            else if (t == typeof(Tensor<float>) && hasList && p.Name == "destination") v = Random(WithAxis(shape, ax, 2 * shape[ax]), 4);
            else if (t == typeof(Tensor<float>)) { v = primaryTaken ? sec : x; primaryTaken = true; }
            else if (t == typeof(Tensor<int>)) v = idx;
            else if (t == typeof(Tensor<float>[])) v = new[] { x, x };
            else if (t == typeof(IReadOnlyList<Tensor<float>>)) v = new List<Tensor<float>> { x, x };
            else if (t == typeof(int) && AxisNames.Contains(p.Name)) v = ax;
            else if (t == typeof(int)) v = IntArg(p.Name!);
            else if (t == typeof(int?)) v = null;
            else if (t == typeof(int[]) && p.Name is "sourceShape") v = (int[])x.Shape.ToArray();
            else if (t == typeof(int[])) v = new[] { shape[ax] };
            else if (p.HasDefaultValue) v = p.DefaultValue;
            else if (t == typeof(float)) v = 1e-5f;
            else if (t == typeof(double)) v = 1.0;
            else if (t == typeof(bool)) v = false;
            else if (t.IsEnum) v = Enum.GetValues(t).GetValue(0);
            else throw new NotSupportedException($"the probe cannot build {t.Name} {p.Name}");
            args[p.Position] = v;
        }
        return args;
    }

    private static int IntArg(string name) => name switch
    {
        "start" or "index" => 0,
        "chunks" or "numSplits" or "sections" or "repeats" => 2,
        _ => 1, // k, n, size, step, length, order
    };

    // ---- lifted reference ---------------------------------------------------------------------------------------

    private static bool CanLift(MethodInfo m) =>
        !m.GetParameters().Any(p => p.ParameterType == typeof(int?) && !p.HasDefaultValue)
        && m.Name is not ("TensorSelect" or "TensorSliceAxis" or "TensorUnstack" or "TensorUnfold");

    private static T[] MoveAxisLast<T>(T[] d, int[] shape, int ax)
    {
        int outer = Prod(shape.Take(ax)), n = shape[ax], inner = Prod(shape.Skip(ax + 1));
        var r = new T[d.Length];
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < n; j++)
                for (int i = 0; i < inner; i++)
                    r[(o * inner + i) * n + j] = d[(o * n + j) * inner + i];
        return r;
    }

    private static T[] MoveLastToAxis<T>(T[] d, int[] others, int n, int ax)
    {
        int outer = Prod(others.Take(ax)), inner = Prod(others.Skip(ax));
        var r = new T[d.Length];
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < n; j++)
                for (int i = 0; i < inner; i++)
                    r[(o * n + j) * inner + i] = d[(o * inner + i) * n + j];
        return r;
    }

    /// <summary>A tensor of the probe's rank goes to [rows, axis length]; one with the axis removed goes to [rows].</summary>
    private static object? LiftTensor<T>(Tensor<T> t, int rank, int ax, bool keep)
    {
        if (keep) return t;
        var s = t.Shape.ToArray();
        if (s.Length == rank)
            return new Tensor<T>(MoveAxisLast(t.ToArray(), s, ax), new[] { t.Length / s[ax], s[ax] });
        if (s.Length == rank - 1)
            return new Tensor<T>(t.ToArray(), new[] { t.Length });
        return t;
    }

    private static object?[] Lift(object?[] args, MethodInfo m, int[] shape, int ax, Plan plan)
    {
        var ps = m.GetParameters();
        int rank = shape.Length;
        var r = new object?[args.Length];
        foreach (var p in ps)
        {
            var a = args[p.Position];
            bool flatIndex = p.ParameterType == typeof(Tensor<int>) && plan.Index == Idx.Flat;
            r[p.Position] = a switch
            {
                _ when p.IsOut => null,
                Tensor<float> f => LiftTensor(f, rank, ax, false),
                Tensor<int> i => LiftTensor(i, rank, ax, flatIndex),
                Tensor<float>[] fs => fs.Select(f => (Tensor<float>)LiftTensor(f, rank, ax, false)!).ToArray(),
                List<Tensor<float>> fl => fl.Select(f => (Tensor<float>)LiftTensor(f, rank, ax, false)!).ToList(),
                int when p.ParameterType == typeof(int) && AxisNames.Contains(p.Name) => 1,
                int[] s when p.Name is "sourceShape" => new[] { Prod(s) / s[ax], s[ax] },
                _ => a,
            };
        }
        return r;
    }

    private static object? Unlift(object? r, int[] shape, int ax)
    {
        switch (r)
        {
            case null: return null;
            case object?[] arr: return arr.Select(o => Unlift(o, shape, ax)).ToArray();
            case Tensor<float> f: return UnliftTensor(f, shape, ax);
            case Tensor<int> i: return UnliftTensor(i, shape, ax);
            default:
                var tt = r.GetType();
                if (tt.FullName!.StartsWith("System.ValueTuple", StringComparison.Ordinal))
                    return tt.GetFields().Select(fi => Unlift(fi.GetValue(r), shape, ax)).ToArray();
                return r;
        }
    }

    private static object UnliftTensor<T>(Tensor<T> t, int[] shape, int ax)
    {
        var s = t.Shape.ToArray();
        var others = Removed(shape, ax);
        int rows = Prod(others);
        if (s.Length == 1 && s[0] == rows) // the axis was reduced away
            return new Tensor<T>(t.ToArray(), others.Length == 0 ? new[] { 1 } : others);
        if (s.Length != 2 || s[0] != rows) return t;
        var d = MoveLastToAxis(t.ToArray(), others, s[1], ax);
        return new Tensor<T>(d, WithAxis(shape, ax, s[1]));
    }

    private static string? Compare(object? a, object? e)
    {
        switch (a)
        {
            case null: return e is null ? null : "null result";
            case object?[] aa when e is object?[] ea:
                for (int i = 0; i < Math.Min(aa.Length, ea.Length); i++)
                    if (Compare(aa[i], ea[i]) is { } d) return $"item {i}: {d}";
                return aa.Length == ea.Length ? null : $"{aa.Length} items vs {ea.Length}";
            case Tensor<float> af when e is Tensor<float> ef:
                return CompareData(af.Shape.ToArray(), ef.Shape.ToArray(), af.ToArray().Select(v => (double)v).ToArray(), ef.ToArray().Select(v => (double)v).ToArray());
            case Tensor<int> ai when e is Tensor<int> ei:
                return CompareData(ai.Shape.ToArray(), ei.Shape.ToArray(), ai.ToArray().Select(v => (double)v).ToArray(), ei.ToArray().Select(v => (double)v).ToArray());
            default:
                var tt = a.GetType();
                if (tt.FullName!.StartsWith("System.ValueTuple", StringComparison.Ordinal))
                    return Compare(tt.GetFields().Select(fi => fi.GetValue(a)).ToArray(), e);
                return null;
        }
    }

    private static string? CompareData(int[] sa, int[] se, double[] a, double[] e)
    {
        // A rank-1 input lifts to [1, n], so its per-row result comes back as [n] or [1, n]: compare the data.
        if (!sa.SequenceEqual(se) && !(Prod(sa) == Prod(se) && (sa.Length <= 1 || se.Length <= 1 || sa[0] == 1 || se[0] == 1)))
            return $"shape [{string.Join(",", sa)}] vs lifted [{string.Join(",", se)}]";
        for (int i = 0; i < a.Length; i++)
        {
            // A NaN or an infinity fails every tolerance comparison, so it must match exactly: NaN with NaN, and an
            // infinity with the same infinity.
            if (double.IsNaN(a[i]) || double.IsNaN(e[i]) || double.IsInfinity(a[i]) || double.IsInfinity(e[i]))
            {
                if (!(double.IsNaN(a[i]) && double.IsNaN(e[i])) && !a[i].Equals(e[i]))
                    return $"element {i}: {a[i]} vs lifted {e[i]}";
                continue;
            }
            if (Math.Abs(a[i] - e[i]) > 1e-4 * (1 + Math.Abs(e[i])))
                return $"element {i}: {a[i]} vs lifted {e[i]}";
        }
        return null;
    }
}
