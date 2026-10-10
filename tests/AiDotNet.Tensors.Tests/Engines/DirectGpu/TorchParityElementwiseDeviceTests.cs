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

    private static void AssertMatches(string op, Tensor<float> cpu, Tensor<float> gpu, double tolerance = 2e-6)
    {
        Assert.Equal(cpu._shape, gpu._shape);
        for (int i = 0; i < cpu.Length; i++)
        {
            float c = cpu[i], g = gpu[i];
            if (float.IsNaN(c) || float.IsNaN(g)) { Assert.True(float.IsNaN(c) && float.IsNaN(g), $"{op}[{i}]: cpu {c} gpu {g}"); continue; }
            if (float.IsInfinity(c) || float.IsInfinity(g)) { Assert.Equal(c, g); continue; }
            Assert.True(Math.Abs(c - g) <= tolerance * Math.Max(1.0, Math.Abs(c)), $"{op}[{i}]: cpu {c:R} gpu {g:R}");
        }
    }

    // GpuLaunchProbe.Fallbacks entries read "<count>x <op>: <detail>"; sum the counts recorded for this op.
    internal static long FallbackCount(string op) => GpuLaunchProbe.Fallbacks.Sum(entry =>
    {
        int x = entry.IndexOf("x ", StringComparison.Ordinal);
        if (x <= 0 || !long.TryParse(entry.Substring(0, x), out long count)) return 0L;
        string key = entry.Substring(x + 2);
        return key == op || key.StartsWith(op + ":", StringComparison.Ordinal) ? count : 0L;
    });

    private static Tensor<float> OnDevice(string op, Func<Tensor<float>> run)
    {
        long launches = GpuLaunchProbe.Count;
        long fallbacks = FallbackCount(op);
        var r = run();
        _ = r[0];   // materialize the deferred device result
        Assert.True(GpuLaunchProbe.Count > launches, $"{op}: no kernel launched");
        Assert.Equal(fallbacks, FallbackCount(op));
        return r;
    }

    public static TheoryData<string> UnaryOps() => new()
    {
        "celu", "celu2", "hardtanh", "hardtanhNaNBound", "logsigmoid", "softsign", "logit", "logitEps", "logitNaNEps", "sinc", "erf",
        "besselJ0", "besselJ1", "besselY0", "besselY1", "modifiedBesselI0", "modifiedBesselI1", "erfcx", "ndtr", "ndtri",
        "entr", "signbit", "isPosInf", "isNegInf",
    };

    [SkippableTheory]
    [MemberData(nameof(UnaryOps))]
    public void Unary_MatchesCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        // logit / ndtri take probabilities: the edges plus a sigmoid of the random values (and, for ndtri, the out-of-range
        // edges that must give NaN).
        var x = op.StartsWith("logit") || op == "ndtri"
            ? Input(11, v => float.IsNaN(v) || (op == "ndtri" && (v == 1e30f || v == -1e30f)) ? (v > 0 ? 1.25f : v < 0 ? -0.25f : v)
                : v == 0f ? 0f : v == 1f ? 1f : 1f / (1f + MathF.Exp(-v)))
            : Input(7);
        Func<IEngine, Tensor<float>> f = op switch
        {
            "celu" => e => e.TensorCelu(x),
            "celu2" => e => e.TensorCelu(x, 2.5),
            "hardtanh" => e => e.TensorHardtanh(x, -0.75, 2.0),
            "hardtanhNaNBound" => e => e.TensorHardtanh(x, double.NaN, 2.0),   // NaN bound: every output NaN, as on the CPU
            "logsigmoid" => e => e.TensorLogSigmoid(x),
            "softsign" => e => e.TensorSoftsign(x),
            "logit" => e => e.TensorLogit(x),
            "logitEps" => e => e.TensorLogit(x, 1e-3),
            "logitNaNEps" => e => e.TensorLogit(x, double.NaN),
            "sinc" => e => e.TensorSinc(x),
            "erf" => e => e.TensorErf(x),
            "besselJ0" => e => e.TensorBesselJ0(x),
            "besselJ1" => e => e.TensorBesselJ1(x),
            "besselY0" => e => e.TensorBesselY0(x),
            "besselY1" => e => e.TensorBesselY1(x),
            "modifiedBesselI0" => e => e.TensorModifiedBesselI0(x),
            "modifiedBesselI1" => e => e.TensorModifiedBesselI1(x),
            "erfcx" => e => e.TensorErfcx(x),
            "ndtr" => e => e.TensorNdtr(x),
            "ndtri" => e => e.TensorNdtri(x),
            "entr" => e => e.TensorEntr(x),
            "signbit" => e => e.TensorSignbit(x),
            "isPosInf" => e => e.TensorIsPosInf(x),
            _ => e => e.TensorIsNegInf(x),
        };
        // CUDA's Bessel / erfcx / ndtri carry a few 1e-6 of error against the CPU's double-precision quadratures.
        double tolerance = op.Contains("essel") || op == "erfcx" || op == "ndtri" ? 1e-5 : 2e-6;
        // The fallback record is keyed by the engine op, not by this case's variant label.
        string baseOp = op == "celu2" ? "celu" : op.StartsWith("hardtanh") ? "hardtanh" : op.StartsWith("logit") ? "logit" : op;
        string probeName = "Tensor" + char.ToUpperInvariant(baseOp[0]) + baseOp.Substring(1);
        AssertMatches(op, f(cpu), OnDevice(probeName, () => f(gpu)), tolerance);
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
