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
/// An elementwise backward <c>XBackward(gradOutput, saved)</c> pairs each gradient element with the saved forward
/// tensor's element, so the two must have one shape. Found by reflection over <see cref="IEngine"/>: every op named
/// <c>*Backward</c> whose arguments are two same-type tensors (plus defaulted params) must throw on a mismatch rather
/// than read the shorter one past its end.
/// </summary>
public sealed class BackwardShapeContractTests
{
    private readonly ITestOutputHelper _out;

    public BackwardShapeContractTests(ITestOutputHelper output) => _out = output;

    private static readonly (int[] Grad, int[] Saved)[] Mismatches =
    {
        (new[] { 2, 3, 4 }, new[] { 4 }),
        (new[] { 4 }, new[] { 2, 3, 4 }),
        (new[] { 2, 3, 4 }, new[] { 2, 1, 4 }),
        (new[] { 3, 4 }, new[] { 4, 3 }),
    };

    /// <summary>
    /// Backwards that read their operands as rows along an axis (or as [batch, classes]), so two equal-count tensors of
    /// different shapes do not pair up either. The rest are flat elementwise: equal counts are enough.
    /// </summary>
    private static readonly HashSet<string> RowStructured = new HashSet<string>
    {
        "SoftmaxBackward", "SparsemaxBackward", "CrossEntropyBackward",
    };

    internal static IEnumerable<MethodInfo> TwoTensorBackwards() =>
        typeof(IEngine).GetMethods()
            .Concat(typeof(IEngine).GetInterfaces().SelectMany(i => i.GetMethods()))
            .Where(m => m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1 && m.Name.EndsWith("Backward"))
            .Where(m =>
            {
                var tensorT = typeof(Tensor<>).MakeGenericType(m.GetGenericArguments()[0]);
                var ps = m.GetParameters();
                return m.ReturnType == tensorT && ps.Count(p => p.ParameterType == tensorT) == 2
                    && ps.All(p => p.ParameterType == tensorT || (p.HasDefaultValue && !p.ParameterType.IsByRef));
            })
            .Distinct()
            .OrderBy(m => m.Name);

    private static Tensor<float> Make(int[] shape, int salt)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<float>(Enumerable.Range(0, n).Select(i => 0.1f + ((i * 37 + salt * 11) % 101) / 101f).ToArray(), shape);
    }

    /// <summary>Checks every mismatch on <paramref name="engine"/>; returns the ops that accepted one.</summary>
    internal static List<string> Violations(IEngine engine, out int probed)
    {
        var bad = new List<string>();
        probed = 0;
        foreach (var def in TwoTensorBackwards())
        {
            probed++;
            var m = def.MakeGenericMethod(typeof(float));
            foreach (var (g, s) in Mismatches)
            {
                bool sameCount = g.Aggregate(1, (a, b) => a * b) == s.Aggregate(1, (a, b) => a * b);
                if (sameCount && !RowStructured.Contains(def.Name)) continue;
                int t = 0;
                var args = m.GetParameters().Select(p => p.ParameterType == typeof(Tensor<float>)
                    ? (object)Make(t++ == 0 ? g : s, t)
                    : p.DefaultValue).ToArray();
                try
                {
                    var r = (Tensor<float>)m.Invoke(engine, args)!;
                    _ = r.ToArray();
                    bad.Add($"{def.Name} grad [{string.Join(",", g)}] saved [{string.Join(",", s)}] -> accepted, result [{string.Join(",", r.Shape.ToArray())}]");
                }
                catch (TargetInvocationException e) when (e.InnerException is ArgumentException) { }
                catch (TargetInvocationException e)
                {
                    bad.Add($"{def.Name} grad [{string.Join(",", g)}] saved [{string.Join(",", s)}] -> {e.InnerException!.GetType().Name}: {e.InnerException.Message.Split('\n')[0]}");
                }
            }
        }
        return bad;
    }

    [Fact]
    public void Cpu_elementwise_backwards_reject_a_gradient_of_another_shape()
    {
        var bad = Violations(new CpuEngine(), out int probed);
        _out.WriteLine($"probed {probed} backward ops; {bad.Count} violations");
        // An empty sweep (a filter that matched nothing) would pass with no violations.
        Assert.True(probed > 0, "The sweep probed no backward op.");
        foreach (var b in bad) _out.WriteLine("FAIL " + b);
        Assert.True(bad.Count == 0, $"{bad.Count} backward shape-contract violations:\n" + string.Join("\n", bad));
    }
}
