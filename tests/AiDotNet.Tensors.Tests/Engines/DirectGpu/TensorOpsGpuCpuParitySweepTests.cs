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

    /// <summary>
    /// Ops whose result is random by definition, by exact name: a substring match also dropped deterministic ops
    /// (NativeNormalizeRows, PixelShuffle, GridSample and the backwards of random ops).
    /// </summary>
    internal static readonly HashSet<string> NonDeterministic = new()
    {
        "Dropout", "GumbelSoftmax", "TensorAlphaDropout", "TensorBernoulli", "TensorCategoricalSample",
        "TensorChannelDropout", "TensorDropoutMask", "TensorFeatureAlphaDropout", "TensorLogNormal", "TensorMultinomial",
        "TensorNormal", "TensorPoisson", "TensorRandLike", "TensorRandint", "TensorRandintLike", "TensorRandnLike",
        "TensorRandomNormal", "TensorRandomUniform", "TensorRandomUniformRange", "TensorRandperm", "TensorUniform",
        "TensorBinomial",
    };

    /// <summary>
    /// Ops the CPU engine rejects for every same-shape float input the sweep builds, with the reason. Any other op
    /// that never runs fails the test, so a CPU regression cannot silently drop an op from the comparison.
    /// </summary>
    private static readonly Dictionary<string, string> NoSameShapeForm = new()
    {
        ["BatchMatMul"] = "contracts operands of different shapes",
        ["TensorBatchMatMul"] = "contracts operands of different shapes",
        ["TensorMatMul"] = "contracts operands of different shapes",
        ["TensorAddMM"] = "contracts operands of different shapes",
        // Compared GPU vs CPU (forward, gradients, residency) in TorchModelOpsGpuTests.
        ["TensorHyperbolicLinear"] = "contracts operands of different shapes",
        ["TensorAddmv"] = "contracts operands of different shapes",
        ["TensorAddbmm"] = "contracts operands of different shapes",
        ["TensorBaddbmm"] = "contracts operands of different shapes",
        ["TensorGroupedMatMul"] = "contracts operands of different shapes",
        ["OctonionMatMulTensor"] = "contracts operands of different shapes",
        ["FusedLinearGELU"] = "contracts operands of different shapes",
        ["FusedLinearReLU"] = "contracts operands of different shapes",
        ["FusedLinearSigmoid"] = "contracts operands of different shapes",
        ["FusedLinearSwish"] = "contracts operands of different shapes",
        ["FusedLinearTanh"] = "contracts operands of different shapes",
        ["RBFKernel"] = "contracts operands of different shapes",
        ["BoxIou"] = "takes [n, 4] boxes",
        ["CompleteBoxIou"] = "takes [n, 4] boxes",
        ["DistanceBoxIou"] = "takes [n, 4] boxes",
        ["GeneralizedBoxIou"] = "takes [n, 4] boxes",
        ["TensorCIoULoss"] = "takes [n, 4] boxes",
        ["TensorDIoULoss"] = "takes [n, 4] boxes",
        ["TensorGIoULoss"] = "takes [n, 4] boxes",
        ["TensorIoULoss"] = "takes [n, 4] boxes",
        ["Conv3D"] = "its kernel's shape differs from the input's",
        ["DepthwiseConv1D"] = "its kernel's shape differs from the input's",
        ["TensorConvTbc"] = "its kernel's shape differs from the input's",
        ["TensorConvTranspose1D"] = "its kernel's shape differs from the input's",
        ["ForwardSplat"] = "its flow field has its own [N, 2, H, W] shape",
        ["ForwardSplatBackwardInput"] = "its flow field has its own [N, 2, H, W] shape",
        ["GLUBackward"] = "its gradient has half the input's split dimension",
        ["GeGLUBackward"] = "its gradient has half the input's split dimension",
        ["ReGLUBackward"] = "its gradient has half the input's split dimension",
        ["SwiGLUBackward"] = "its gradient has half the input's split dimension",
        ["GridSample"] = "its sampling grid has its own shape",
        ["GridSampleBackwardGrid"] = "its sampling grid has its own shape",
        ["TensorGridSample3D"] = "its sampling grid has its own shape",
        ["TensorTrilinearInterpolate"] = "interpolates a rank-5 volume; the sweep's ranks stop at 4",
        ["TensorTrilinearInterpolateBackward"] = "interpolates a rank-5 volume; the sweep's ranks stop at 4",
        ["TensorBitwiseAnd"] = "an integer-only op; the CPU rejects float",
        ["TensorBitwiseLeftShift"] = "an integer-only op; the CPU rejects float",
        ["TensorBitwiseNot"] = "an integer-only op; the CPU rejects float",
        ["TensorBitwiseOr"] = "an integer-only op; the CPU rejects float",
        ["TensorBitwiseRightShift"] = "an integer-only op; the CPU rejects float",
        ["TensorBitwiseXor"] = "an integer-only op; the CPU rejects float",
        ["TensorGcd"] = "an integer-only op; the CPU rejects float",
        ["TensorLcm"] = "an integer-only op; the CPU rejects float",
        ["TensorDiagonalScatter"] = "its source is a diagonal, not the input's shape",
        ["TensorGatherClassValues"] = "gathers with class indices of another shape",
        // Compared GPU vs the scalar recurrence in NumericRecurrenceGpuTests.FritschCarlsonLimit_MatchesScalarRecurrence.
        ["TensorFritschCarlsonLimit"] = "its secants are [..., n - 1] against slopes [..., n]",
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
            .Where(m => !NonDeterministic.Contains(m.Name))
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

    /// <summary>The CPU engine's rejection of an input it does not take: an argument or shape error, not a fault.</summary>
    internal static bool RejectsArguments(Exception e)
    {
        while (e is TargetInvocationException { InnerException: not null } t) e = t.InnerException;
        return e is ArgumentException or NotSupportedException or InvalidOperationException or RankException;
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
        var unprobed = new List<string>();
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
                    catch (TargetInvocationException e) when (RejectsArguments(e))
                    {
                        continue; // the op does not take this shape on the CPU either
                    }
                    catch (Exception e)
                    {
                        failures.Add($"{where}: the CPU engine failed: {Inner(e)}");
                        continue;
                    }
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
            else if (!NoSameShapeForm.ContainsKey(def.Name)) unprobed.Add(def.Name);
        }
        _out.WriteLine($"probed {probed} ops; {failures.Count} failures");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        Assert.True(probed > 0, "The sweep probed no op.");
        Assert.True(unprobed.Count == 0,
            "The CPU engine ran these ops on no same-shape input, so they were never compared: " + string.Join(", ", unprobed));
        Assert.True(failures.Count == 0, $"{failures.Count} GPU/CPU mismatches:\n" + string.Join("\n", failures));
    }

    /// <summary>Broadcastable (left, right) shape pairs: trailing vector, middle singleton, outer product, scalar-like.</summary>
    private static readonly (int[] A, int[] B)[] BroadcastPairs =
    {
        (new[] { 2, 3, 4 }, new[] { 4 }),
        (new[] { 4 }, new[] { 2, 3, 4 }),
        (new[] { 2, 1, 4 }, new[] { 3, 1 }),
        (new[] { 3, 1 }, new[] { 1, 5 }),
        (new[] { 2, 3, 4 }, new[] { 1 }),
        (new[] { 2, 3, 4 }, new[] { 2, 1, 4 }),
    };

    /// <summary>
    /// GPU/CPU parity for the same two-tensor ops on broadcastable shapes: wherever the CPU engine broadcasts, the GPU
    /// must give the same shape and values (and must not throw).
    /// </summary>
    [SkippableFact]
    public void Two_tensor_ops_match_the_cpu_on_broadcast_shapes()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        IEngine cpu = new CpuEngine();
        var failures = new List<string>();
        int probed = 0;
        foreach (var def in SameShapeOps().Where(d => d.GetParameters().Count(p => p.ParameterType.IsGenericType
            && p.ParameterType.GetGenericTypeDefinition() == typeof(Tensor<>)) == 2))
        {
            var m = def.MakeGenericMethod(typeof(float));
            bool ran = false;
            foreach (var (sa, sb) in BroadcastPairs)
            {
                object?[] MakeArgs()
                {
                    int t = 0;
                    return m.GetParameters().Select(p => p.ParameterType == typeof(Tensor<float>)
                        ? (object)Make(t++ == 0 ? sa : sb, t, false)
                        : p.DefaultValue).ToArray();
                }
                string where = $"{def.Name} [{string.Join(",", sa)}]x[{string.Join(",", sb)}]";
                Tensor<float> expected;
                try { expected = (Tensor<float>)m.Invoke(cpu, MakeArgs())!; }
                catch (Exception e) when (RejectsArguments(e)) { continue; }   // the op does not broadcast these shapes
                catch (Exception e) { failures.Add($"{where}: the CPU failed: {Inner(e)}"); continue; }
                ran = true;
                Tensor<float> actual;
                try { actual = (Tensor<float>)m.Invoke(gpu, MakeArgs())!; }
                catch (Exception e) { failures.Add($"{where}: GPU threw where the CPU did not: {Inner(e)}"); continue; }
                if (!expected.Shape.ToArray().SequenceEqual(actual.Shape.ToArray()))
                {
                    failures.Add($"{where}: shape [{string.Join(",", actual.Shape.ToArray())}] vs cpu [{string.Join(",", expected.Shape.ToArray())}]");
                    continue;
                }
                var e1 = expected.ToArray();
                var a1 = actual.ToArray();
                for (int i = 0; i < e1.Length; i++)
                {
                    float x = a1[i], y = e1[i];
                    bool same = float.IsNaN(y) ? float.IsNaN(x)
                        : float.IsInfinity(y) ? x.Equals(y)
                        : !float.IsNaN(x) && !float.IsInfinity(x) && Math.Abs(x - y) <= 1e-3f * (1 + Math.Abs(y));
                    if (!same) { failures.Add($"{where}: element {i}: gpu {x} vs cpu {y}"); break; }
                }
            }
            if (ran) probed++;
        }
        _out.WriteLine($"broadcast: probed {probed} ops; {failures.Count} failures");
        foreach (var f in failures) _out.WriteLine("FAIL " + f);
        Assert.True(probed > 0, "The broadcast sweep compared no op.");
        Assert.True(failures.Count == 0, $"{failures.Count} GPU/CPU broadcast mismatches:\n" + string.Join("\n", failures));
    }

    /// <summary>The GPU overrides must enforce the same backward shape contract as the CPU engine.</summary>
    [SkippableFact]
    public void Gpu_elementwise_backwards_reject_a_gradient_of_another_shape()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var bad = BackwardShapeContractTests.Violations(_fixture.Engine!, out int probed);
        _out.WriteLine($"gpu: probed {probed} backward ops; {bad.Count} violations");
        // An empty sweep (a filter that matched nothing) would pass with no violations.
        Assert.True(probed > 0, "The sweep probed no backward op.");
        foreach (var b in bad) _out.WriteLine("FAIL " + b);
        Assert.True(bad.Count == 0, $"{bad.Count} GPU backward shape-contract violations:\n" + string.Join("\n", bad));
    }
}
