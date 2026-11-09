// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// GPU/CPU parity for every engine op whose arguments are only same-shape float tensors (unary, binary, ternary
/// elementwise ops and their backwards), found by reflection over <see cref="IEngine"/>, so a new op is covered when
/// it is declared. Each runs on ranks 1 to 4, on contiguous inputs and on non-contiguous (permuted) views, and must
/// give the CPU engine's result.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class TensorOpsGpuCpuParitySweepTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _out;

    public TensorOpsGpuCpuParitySweepTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _out = output;
    }

    /// <summary>Ops whose result is random by definition, or which are not a function of their inputs alone.</summary>
    private static readonly string[] NonDeterministic =
    {
        "Dropout", "Random", "Rand", "Noise", "Bernoulli", "Multinomial", "Sample", "Gumbel", "Uniform", "Normal",
        "Shuffle", "Perturb", "Stochastic", "Poisson",
    };

    private static readonly int[][] Shapes =
    {
        new[] { 7 },
        new[] { 3, 5 },
        new[] { 2, 3, 4 },
        new[] { 2, 3, 2, 3 },
    };

    private static IEnumerable<MethodInfo> SameShapeOps() =>
        typeof(IEngine).GetMethods()
            .Concat(typeof(IEngine).GetInterfaces().SelectMany(i => i.GetMethods()))
            .Where(m => m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1)
            .Where(m =>
            {
                var g = m.GetGenericArguments()[0];
                var tensorT = typeof(Tensor<>).MakeGenericType(g);
                var ps = m.GetParameters();
                int tensors = ps.Count(p => p.ParameterType == tensorT);
                return m.ReturnType == tensorT && tensors >= 1 && tensors <= 3
                    && ps.All(p => p.ParameterType == tensorT || (p.HasDefaultValue && !p.ParameterType.IsByRef));
            })
            .Where(m => !NonDeterministic.Any(n => m.Name.Contains(n)))
            .Distinct()
            .OrderBy(m => m.Name);

    private static float[] Data(int n, int salt) =>
        Enumerable.Range(0, n).Select(i => 0.1f + ((i * 37 + salt * 11) % 101) / 101f).ToArray();

    private static Tensor<float> Make(int[] shape, int salt, bool strided)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        if (!strided || shape.Length < 2) return new Tensor<float>(Data(n, salt), (int[])shape.Clone());
        // A permuted view of a tensor with the last two dims swapped: same logical shape, non-contiguous strides.
        var swapped = (int[])shape.Clone();
        int r = shape.Length;
        (swapped[r - 1], swapped[r - 2]) = (swapped[r - 2], swapped[r - 1]);
        var baseT = new Tensor<float>(Data(n, salt), swapped);
        var perm = Enumerable.Range(0, shape.Length).ToArray();
        (perm[r - 1], perm[r - 2]) = (perm[r - 2], perm[r - 1]);
        return baseT.Transpose(perm);
    }

    private static object?[] Args(MethodInfo m, int[] shape, bool strided)
    {
        int salt = 1;
        return m.GetParameters().Select(p => p.ParameterType == typeof(Tensor<float>)
            ? (object)Make(shape, salt++, strided)
            : p.DefaultValue).ToArray();
    }

    private static string Inner(Exception e)
    {
        while (e is TargetInvocationException { InnerException: not null } t) e = t.InnerException!;
        return $"{e.GetType().Name}: {e.Message.Split('\n')[0]}";
    }

    [SkippableFact]
    public void Same_shape_ops_match_the_cpu_on_every_rank_and_on_strided_views()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        IEngine cpu = new CpuEngine();
        var failures = new List<string>();
        int probed = 0;
        foreach (var def in SameShapeOps())
        {
            var m = def.MakeGenericMethod(typeof(float));
            bool ran = false;
            foreach (var shape in Shapes)
                foreach (var strided in new[] { false, true })
                {
                    if (strided && shape.Length < 2) continue;
                    string where = $"{def.Name} [{string.Join(",", shape)}]{(strided ? " strided" : "")}";
                    Tensor<float> expected;
                    try { expected = (Tensor<float>)m.Invoke(cpu, Args(m, shape, strided))!; }
                    catch { continue; } // the op does not take this shape on the CPU either
                    ran = true;
                    Tensor<float> actual;
                    try { actual = (Tensor<float>)m.Invoke(gpu, Args(m, shape, strided))!; }
                    catch (Exception e) { failures.Add($"{where}: GPU threw where the CPU did not: {Inner(e)}"); continue; }
                    var e1 = expected.ToArray();
                    var a1 = actual.ToArray();
                    if (!expected.Shape.ToArray().SequenceEqual(actual.Shape.ToArray()))
                    {
                        failures.Add($"{where}: shape [{string.Join(",", actual.Shape.ToArray())}] vs cpu [{string.Join(",", expected.Shape.ToArray())}]");
                        continue;
                    }
                    for (int i = 0; i < e1.Length; i++)
                    {
                        float x = a1[i], y = e1[i];
                        bool same = float.IsNaN(y) ? float.IsNaN(x)
                            : float.IsInfinity(y) ? x.Equals(y)
                            : !float.IsNaN(x) && !float.IsInfinity(x) && Math.Abs(x - y) <= 1e-3f * (1 + Math.Abs(y));
                        if (!same)
                        {
                            failures.Add($"{where}: element {i}: gpu {x} vs cpu {y}");
                            break;
                        }
                    }
                }
            if (ran) probed++;
        }
        _out.WriteLine($"probed {probed} ops; {failures.Count} failures");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        Assert.True(failures.Count == 0, $"{failures.Count} GPU/CPU mismatches:\n" + string.Join("\n", failures));
    }
}
