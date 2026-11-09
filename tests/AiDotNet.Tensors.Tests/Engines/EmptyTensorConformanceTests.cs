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
/// Zero-size tensors are ordinary values in torch: an empty batch, a sequence of length 0. For every engine op whose
/// arguments are only same-shape tensors (plus defaulted params) and whose result has its input's shape (an elementwise
/// op or backward), found by reflection over <see cref="IEngine"/>, a tensor with a zero dimension must give an empty
/// result of that shape rather than throw.
/// </summary>
public sealed class EmptyTensorConformanceTests
{
    private readonly ITestOutputHelper _out;

    public EmptyTensorConformanceTests(ITestOutputHelper output) => _out = output;

    private static readonly string[] NonDeterministic =
    {
        "Dropout", "Random", "Rand", "Noise", "Bernoulli", "Multinomial", "Sample", "Gumbel", "Uniform", "Normal",
        "Shuffle", "Perturb", "Stochastic", "Poisson",
    };

    /// <summary>Each keeps an even last dim, which interleaved-complex and RoPE ops require of any input.</summary>
    private static readonly int[][] EmptyShapes = { new[] { 0 }, new[] { 3, 0 }, new[] { 0, 4 }, new[] { 2, 0, 4 } };

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
                    && ps.All(p => p.ParameterType == tensorT || (p.HasDefaultValue && !p.ParameterType.IsByRef));
            })
            .Where(m => !NonDeterministic.Any(n => m.Name.Contains(n)))
            .Distinct()
            .OrderBy(m => m.Name);

    private static object?[] Args(MethodInfo m, int[] shape)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        int salt = 1;
        return m.GetParameters().Select(p => p.ParameterType == typeof(Tensor<float>)
            ? (object)new Tensor<float>(Enumerable.Range(0, n).Select(i => 0.1f + ((i * 37 + salt++ * 11) % 101) / 101f).ToArray(), (int[])shape.Clone())
            : p.DefaultValue).ToArray();
    }

    private static string Inner(Exception e)
    {
        while (e is TargetInvocationException { InnerException: not null } t) e = t.InnerException!;
        return $"{e.GetType().Name}: {e.Message.Split('\n')[0]}";
    }

    [Fact]
    public void Shape_preserving_ops_accept_zero_size_tensors()
    {
        IEngine cpu = new CpuEngine();
        var failures = new List<string>();
        int probed = 0;
        foreach (var def in SameShapeOps())
        {
            var m = def.MakeGenericMethod(typeof(float));
            // Ranks at which the op maps a non-empty input to a result of the same shape; an empty shape is probed only
            // at those ranks (an op may legitimately require rank 2, say).
            var ranks = new HashSet<int>();
            foreach (var probe in new[] { new[] { 5 }, new[] { 3, 5 }, new[] { 2, 3, 4 } })
            {
                try
                {
                    var r = (Tensor<float>)m.Invoke(cpu, Args(m, probe))!;
                    if (r.Shape.ToArray().SequenceEqual(probe)) ranks.Add(probe.Length);
                }
                catch { }
            }
            if (ranks.Count == 0) continue;
            probed++;
            foreach (var shape in EmptyShapes.Where(e => ranks.Contains(e.Length)))
            {
                string where = $"{def.Name} [{string.Join(",", shape)}]";
                try
                {
                    var r = (Tensor<float>)m.Invoke(cpu, Args(m, shape))!;
                    if (!r.Shape.ToArray().SequenceEqual(shape))
                        failures.Add($"{where}: result shape [{string.Join(",", r.Shape.ToArray())}]");
                    else if (r.Length != 0)
                        failures.Add($"{where}: result has {r.Length} elements");
                }
                catch (Exception e) { failures.Add($"{where}: {Inner(e)}"); }
            }
        }
        _out.WriteLine($"probed {probed} shape-preserving ops; {failures.Count} failures");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        Assert.True(failures.Count == 0, $"{failures.Count} empty-tensor failures:\n" + string.Join("\n", failures));
    }
}
