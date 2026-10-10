using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1112: the PyTorch-parity elementwise ops that computed on the host now run device kernels on CUDA. Each op is fed
/// the edge values the CPU lambdas special-case -- negatives, +-0, magnitudes past expf's range, +-inf and NaN -- plus a
/// few thousand random values, and must match the CPU element by element (NaN only where the CPU is NaN), launch a
/// kernel, and record no host fallback.
/// </summary>
[Collection("VulkanGlobalState")]
public class TorchParityElementwiseDeviceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public TorchParityElementwiseDeviceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static readonly float[] Edges =
    {
        float.NegativeInfinity, -1e30f, -800f, -88f, -5f, -1.5f, -1f, -0.25f, -0f, 0f, 1e-7f, 0.25f, 0.5f, 1f, 1.5f,
        5f, 88f, 800f, 1e30f, float.PositiveInfinity, float.NaN,
    };

    private static Tensor<float> Input(int salt, Func<float, float>? map = null)
    {
        var rng = new Random(salt);
        int n = Edges.Length + 3000;
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++)
        {
            float v = i < Edges.Length ? Edges[i] : (float)((rng.NextDouble() - 0.5) * 12);
            t[i] = map is null ? v : map(v);
        }
        return t;
    }

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "These device kernels are CUDA-only.");
        return gpu;
    }

    private static void AssertMatches(string op, Tensor<float> cpu, Tensor<float> gpu)
    {
        Assert.Equal(cpu._shape, gpu._shape);
        for (int i = 0; i < cpu.Length; i++)
        {
            float c = cpu[i], g = gpu[i];
            if (float.IsNaN(c) || float.IsNaN(g)) { Assert.True(float.IsNaN(c) && float.IsNaN(g), $"{op}[{i}]: cpu {c} gpu {g}"); continue; }
            if (float.IsInfinity(c) || float.IsInfinity(g)) { Assert.Equal(c, g); continue; }
            Assert.True(Math.Abs(c - g) <= 2e-6 * Math.Max(1.0, Math.Abs(c)), $"{op}[{i}]: cpu {c:R} gpu {g:R}");
        }
    }

    private static Tensor<float> OnDevice(string op, Func<Tensor<float>> run)
    {
        long launches = GpuLaunchProbe.Count;
        int fallbacks = GpuLaunchProbe.Fallbacks.Count(f => f.StartsWith(op, StringComparison.Ordinal));
        var r = run();
        _ = r[0];   // materialize the deferred device result
        Assert.True(GpuLaunchProbe.Count > launches, $"{op}: no kernel launched");
        Assert.Equal(fallbacks, GpuLaunchProbe.Fallbacks.Count(f => f.StartsWith(op, StringComparison.Ordinal)));
        return r;
    }

    public static TheoryData<string> UnaryOps() => new() { "celu", "celu2", "hardtanh", "logsigmoid", "softsign", "logit", "logitEps", "sinc", "erf" };

    [SkippableTheory]
    [MemberData(nameof(UnaryOps))]
    public void Unary_MatchesCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var x = op.StartsWith("logit") ? Input(11, v => float.IsNaN(v) ? v : 1f / (1f + MathF.Exp(-v))) : Input(7);
        Func<IEngine, Tensor<float>> f = op switch
        {
            "celu" => e => e.TensorCelu(x),
            "celu2" => e => e.TensorCelu(x, 2.5),
            "hardtanh" => e => e.TensorHardtanh(x, -0.75, 2.0),
            "logsigmoid" => e => e.TensorLogSigmoid(x),
            "softsign" => e => e.TensorSoftsign(x),
            "logit" => e => e.TensorLogit(x),
            "logitEps" => e => e.TensorLogit(x, 1e-3),
            "sinc" => e => e.TensorSinc(x),
            _ => e => e.TensorErf(x),
        };
        string probeName = "Tensor" + char.ToUpperInvariant(op[0]) + op.Substring(1);
        AssertMatches(op, f(cpu), OnDevice(probeName, () => f(gpu)));
    }

    public static TheoryData<string> BinaryOps() => new() { "fmax", "fmin", "heaviside", "greaterEqual", "lessEqual" };

    [SkippableTheory]
    [MemberData(nameof(BinaryOps))]
    public void Binary_MatchesCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var a = Input(3);
        var b = Input(5);
        for (int i = 0; i < 40; i++) b[i] = i % 3 == 0 ? float.NaN : (i % 3 == 1 ? a[i] : -a[i]);   // NaN, ties, mirrors
        Func<IEngine, Tensor<float>> f = op switch
        {
            "fmax" => e => e.TensorFmax(a, b),
            "fmin" => e => e.TensorFmin(a, b),
            "heaviside" => e => e.TensorHeaviside(a, b),
            "greaterEqual" => e => e.TensorGreaterEqual(a, b),
            _ => e => e.TensorLessEqual(a, b),
        };
        string probeName = "Tensor" + char.ToUpperInvariant(op[0]) + op.Substring(1);
        AssertMatches(op, f(cpu), OnDevice(probeName, () => f(gpu)));
    }
}
