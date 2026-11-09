// Copyright (c) CrowdTrainer LLC. All rights reserved.

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
/// A tensor op is a function of its inputs' logical values, not of their memory layout. For every engine op whose
/// arguments are only same-shape tensors (plus defaulted params), found by reflection over <see cref="IEngine"/>, the
/// result on non-contiguous (permuted) views must equal the result on contiguous copies of the same values, for
/// float and double. Kernels that read the storage array directly instead of the logical layout fail here.
/// </summary>
public sealed class StridedViewConformanceTests
{
    private readonly ITestOutputHelper _out;

    public StridedViewConformanceTests(ITestOutputHelper output) => _out = output;

    private static readonly string[] NonDeterministic =
    {
        "Dropout", "Random", "Rand", "Noise", "Bernoulli", "Multinomial", "Sample", "Gumbel", "Uniform", "Normal",
        "Shuffle", "Perturb", "Stochastic", "Poisson",
    };

    private static readonly int[][] Shapes = { new[] { 3, 5 }, new[] { 2, 3, 4 }, new[] { 2, 3, 2, 3 } };

    /// <summary>An int axis/dim parameter without a default: probed at the first and the last axis.</summary>
    private static bool IsAxis(ParameterInfo p) =>
        p.ParameterType == typeof(int) && !p.HasDefaultValue && (p.Name == "axis" || p.Name == "dim");

    private static IEnumerable<MethodInfo> SameShapeOps() =>
        typeof(IEngine).GetMethods()
            .Concat(typeof(IEngine).GetInterfaces().SelectMany(i => i.GetMethods()))
            .Where(m => m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1)
            .Where(m =>
            {
                var tensorT = typeof(Tensor<>).MakeGenericType(m.GetGenericArguments()[0]);
                var ps = m.GetParameters();
                int tensors = ps.Count(p => p.ParameterType == tensorT);
                return m.ReturnType == tensorT && tensors >= 1 && tensors <= 3
                    && ps.All(p => p.ParameterType == tensorT || IsAxis(p) || (p.HasDefaultValue && !p.ParameterType.IsByRef));
            })
            .Where(m => !NonDeterministic.Any(n => m.Name.Contains(n)))
            .Distinct()
            .OrderBy(m => m.Name);

    private static T[] Data<T>(int n, int salt) =>
        Enumerable.Range(0, n).Select(i => (T)Convert.ChangeType(0.1 + ((i * 37 + salt * 11) % 101) / 101.0, typeof(T))).ToArray();

    /// <summary>A permuted view (last two dims swapped back) and a contiguous tensor with the same logical values.</summary>
    private static (Tensor<T> View, Tensor<T> Dense) Pair<T>(int[] shape, int salt, bool offset)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        int r = shape.Length;
        var swapped = (int[])shape.Clone();
        (swapped[r - 1], swapped[r - 2]) = (swapped[r - 2], swapped[r - 1]);
        var perm = Enumerable.Range(0, r).ToArray();
        (perm[r - 1], perm[r - 2]) = (perm[r - 2], perm[r - 1]);
        Tensor<T> view;
        if (offset)
        {
            // A slice past the first row of a larger tensor: contiguous, but at a non-zero storage offset.
            var bigger = (int[])shape.Clone();
            bigger[0] += 1;
            view = new Tensor<T>(Data<T>(n + n / shape[0], salt), bigger).Slice(0, 1);
        }
        else view = new Tensor<T>(Data<T>(n, salt), swapped).Transpose(perm);
        var dense = new Tensor<T>(view.ToArray(), (int[])shape.Clone());
        return (view, dense);
    }

    private static string Inner(Exception e)
    {
        while (e is TargetInvocationException { InnerException: not null } t) e = t.InnerException!;
        return $"{e.GetType().Name}: {e.Message.Split('\n')[0]}";
    }

    private List<string> Sweep<T>(out int probed)
    {
        IEngine cpu = new CpuEngine();
        var failures = new List<string>();
        probed = 0;
        foreach (var def in SameShapeOps())
        {
            var m = def.MakeGenericMethod(typeof(T));
            bool ran = false;
            foreach (var shape in Shapes)
            foreach (int axis in new[] { 0, -1 })
            foreach (bool offset in new[] { false, true })
            {
                if (axis == 0 && !m.GetParameters().Any(IsAxis)) continue;
                string where = $"{def.Name}<{typeof(T).Name}> [{string.Join(",", shape)}]" + (m.GetParameters().Any(IsAxis) ? $" axis {axis}" : "") + (offset ? " offset" : " permuted");
                var ps = m.GetParameters();
                var views = new object?[ps.Length];
                var dense = new object?[ps.Length];
                int salt = 1;
                for (int i = 0; i < ps.Length; i++)
                {
                    if (ps[i].ParameterType == typeof(Tensor<T>))
                    {
                        var (v, d) = Pair<T>(shape, salt++, offset);
                        views[i] = v;
                        dense[i] = d;
                    }
                    else views[i] = dense[i] = IsAxis(ps[i]) ? axis : ps[i].DefaultValue;
                }
                Tensor<T> expected;
                try { expected = (Tensor<T>)m.Invoke(cpu, dense)!; }
                catch { continue; } // the op does not take this shape
                ran = true;
                Tensor<T> actual;
                try { actual = (Tensor<T>)m.Invoke(cpu, views)!; }
                catch (Exception e) { failures.Add($"{where}: threw on a strided view but not on its contiguous copy: {Inner(e)}"); continue; }
                if (!expected.Shape.ToArray().SequenceEqual(actual.Shape.ToArray()))
                {
                    failures.Add($"{where}: shape [{string.Join(",", actual.Shape.ToArray())}] vs contiguous [{string.Join(",", expected.Shape.ToArray())}]");
                    continue;
                }
                var e1 = expected.ToArray();
                var a1 = actual.ToArray();
                for (int i = 0; i < e1.Length; i++)
                {
                    double x = Convert.ToDouble(a1[i]), y = Convert.ToDouble(e1[i]);
                    bool same = double.IsNaN(y) ? double.IsNaN(x)
                        : double.IsInfinity(y) ? x.Equals(y)
                        : !double.IsNaN(x) && !double.IsInfinity(x) && Math.Abs(x - y) <= 1e-5 * (1 + Math.Abs(y));
                    if (!same) { failures.Add($"{where}: element {i}: strided {x} vs contiguous {y}"); break; }
                }
            }
            if (ran) probed++;
        }
        return failures;
    }

    [Fact]
    public void Float_ops_give_the_same_result_on_strided_views()
    {
        var failures = Sweep<float>(out int probed);
        _out.WriteLine($"float: probed {probed} ops; {failures.Count} failures");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        Assert.True(failures.Count == 0, $"{failures.Count} layout-dependent results:\n" + string.Join("\n", failures));
    }

    [Fact]
    public void Double_ops_give_the_same_result_on_strided_views()
    {
        var failures = Sweep<double>(out int probed);
        _out.WriteLine($"double: probed {probed} ops; {failures.Count} failures");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        Assert.True(failures.Count == 0, $"{failures.Count} layout-dependent results:\n" + string.Join("\n", failures));
    }
}
