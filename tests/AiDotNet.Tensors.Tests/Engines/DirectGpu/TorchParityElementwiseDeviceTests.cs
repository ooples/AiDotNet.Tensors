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
public abstract class TorchParityElementwiseDeviceTestsBase
{
    private readonly IDirectGpuTensorEngineFixture _fixture;
    protected TorchParityElementwiseDeviceTestsBase(IDirectGpuTensorEngineFixture fixture) => _fixture = fixture;

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
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "This backend has no PyTorch-parity device kernels.");
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
        "entr", "signbit", "isPosInf", "isNegInf", "logNdtr", "modifiedBesselK0", "modifiedBesselK1",
        "scaledModifiedBesselK0", "scaledModifiedBesselK1", "sphericalBesselJ0", "airyAi", "mvlgamma", "mvlgamma4",
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
            // Airy: x over [-30, 30] reaches the Maclaurin, K_{1/3}, Schlafli-quadrature and asymptotic branches.
            : op == "airyAi" ? Input(7, v => Math.Abs(v) > 6 || float.IsNaN(v) ? v : v * 5f)
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
            "logNdtr" => e => e.TensorLogNdtr(x),
            "modifiedBesselK0" => e => e.TensorModifiedBesselK0(x),
            "modifiedBesselK1" => e => e.TensorModifiedBesselK1(x),
            "scaledModifiedBesselK0" => e => e.TensorScaledModifiedBesselK0(x),
            "scaledModifiedBesselK1" => e => e.TensorScaledModifiedBesselK1(x),
            "sphericalBesselJ0" => e => e.TensorSphericalBesselJ0(x),
            "airyAi" => e => e.TensorAiryAi(x),
            "mvlgamma" => e => e.TensorMvlgamma(x, 1),
            "mvlgamma4" => e => e.TensorMvlgamma(x, 4),
            "isNegInf" => e => e.TensorIsNegInf(x),
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, "no case for this op"),
        };
        // CUDA's Bessel / erfcx / ndtri carry a few 1e-6 of error against the CPU's double-precision quadratures.
        double tolerance = op.Contains("essel") || op == "erfcx" || op == "ndtri" || op == "airyAi" || op.StartsWith("mvlgamma") ? 1e-5 : 2e-6;
        // The fallback record is keyed by the engine op, not by this case's variant label.
        string baseOp = op == "celu2" ? "celu" : op.StartsWith("hardtanh") ? "hardtanh" : op.StartsWith("logit") ? "logit"
            : op.StartsWith("mvlgamma") ? "mvlgamma" : op;
        string probeName = "Tensor" + char.ToUpperInvariant(baseOp[0]) + baseOp.Substring(1);
        AssertMatches(op, f(cpu), OnDevice(probeName, () => f(gpu)), tolerance);
    }

    public static TheoryData<string> BinaryOps() => new() { "fmax", "fmin", "heaviside", "greaterEqual", "lessEqual", "igamma", "igammac" };

    [SkippableTheory]
    [MemberData(nameof(BinaryOps))]
    public void Binary_MatchesCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var a = Input(3);
        var b = Input(5);
        for (int i = 0; i < 40; i++) b[i] = i % 3 == 0 ? float.NaN : (i % 3 == 1 ? a[i] : -a[i]);   // NaN, ties, mirrors
        if (op.StartsWith("igamma"))
        {
            // P / Q(a, x) need a > 0, x >= 0: the edge rows keep their negatives, zeros and infinities (the NaN, 0 and 1
            // branches); the rest are positive, straddling x = a + 1 (series vs continued fraction).
            for (int i = Edges.Length; i < a.Length; i++) { a[i] = Math.Abs(a[i]) * 2; b[i] = Math.Abs(b[i]) * 2; }
            int e = Edges.Length;
            b[e] = 0f; b[e + 1] = float.PositiveInfinity; b[e + 2] = 0f;   // x = 0 and x = inf at a positive a
            a[e] = 1.5f; a[e + 1] = 2.5f; a[e + 2] = float.PositiveInfinity;
        }
        Func<IEngine, Tensor<float>> f = op switch
        {
            "fmax" => e => e.TensorFmax(a, b),
            "fmin" => e => e.TensorFmin(a, b),
            "heaviside" => e => e.TensorHeaviside(a, b),
            "greaterEqual" => e => e.TensorGreaterEqual(a, b),
            "igamma" => e => e.TensorIgamma(a, b),
            "igammac" => e => e.TensorIgammac(a, b),
            "lessEqual" => e => e.TensorLessEqual(a, b),
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, "no case for this op"),
        };
        string probeName = "Tensor" + char.ToUpperInvariant(op[0]) + op.Substring(1);
        AssertMatches(op, f(cpu), OnDevice(probeName, () => f(gpu)), op.StartsWith("igamma") ? 1e-5 : 2e-6);
    }

    // The launchers check every buffer's capacity before a kernel can index past it (a sticky CUDA fault otherwise).
    [SkippableFact]
    public void Launchers_RejectUndersizedBuffers()
    {
        var gpu = GpuOrSkip();
        var backend = gpu.GetBackend();
        var kernels = (ITorchParityElementwiseBackend)backend!;
        using var small = backend!.AllocateBuffer(8);
        using var large = backend.AllocateBuffer(64);
        Assert.Throws<ArgumentException>(() => kernels.TorchParityUnary(TorchParityUnaryOp.Erf, large, small, 64, 0f, 0f));
        Assert.Throws<ArgumentException>(() => kernels.TorchParityBinary(TorchParityBinaryOp.Fmax, small, large, large, 64));
        Assert.Throws<ArgumentException>(() => kernels.TorchParityDiagflat(large, small, 8, 8, 0));
        Assert.Throws<ArgumentException>(() => kernels.TorchParitySample(small, 64, 0, 0, 1, 1));
    }

    public static TheoryData<string> PolynomialOps() => new()
    {
        "ChebyshevPolynomialT", "ChebyshevPolynomialU", "ChebyshevPolynomialV", "ChebyshevPolynomialW",
        "ShiftedChebyshevPolynomialT", "ShiftedChebyshevPolynomialU", "ShiftedChebyshevPolynomialV", "ShiftedChebyshevPolynomialW",
        "HermitePolynomialH", "HermitePolynomialHe", "LaguerrePolynomialL", "LegendrePolynomialP",
    };

    private static Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>> Polynomial(string op) => op switch
    {
        "ChebyshevPolynomialT" => (e, x, n) => e.TensorChebyshevPolynomialT(x, n),
        "ChebyshevPolynomialU" => (e, x, n) => e.TensorChebyshevPolynomialU(x, n),
        "ChebyshevPolynomialV" => (e, x, n) => e.TensorChebyshevPolynomialV(x, n),
        "ChebyshevPolynomialW" => (e, x, n) => e.TensorChebyshevPolynomialW(x, n),
        "ShiftedChebyshevPolynomialT" => (e, x, n) => e.TensorShiftedChebyshevPolynomialT(x, n),
        "ShiftedChebyshevPolynomialU" => (e, x, n) => e.TensorShiftedChebyshevPolynomialU(x, n),
        "ShiftedChebyshevPolynomialV" => (e, x, n) => e.TensorShiftedChebyshevPolynomialV(x, n),
        "ShiftedChebyshevPolynomialW" => (e, x, n) => e.TensorShiftedChebyshevPolynomialW(x, n),
        "HermitePolynomialH" => (e, x, n) => e.TensorHermitePolynomialH(x, n),
        "HermitePolynomialHe" => (e, x, n) => e.TensorHermitePolynomialHe(x, n),
        "LaguerrePolynomialL" => (e, x, n) => e.TensorLaguerrePolynomialL(x, n),
        "LegendrePolynomialP" => (e, x, n) => e.TensorLegendrePolynomialP(x, n),
        _ => throw new ArgumentOutOfRangeException(nameof(op), op, "no case for this op"),
    };

    // Per-element degrees cover the CPU's degree rules -- NaN gives NaN, a fraction truncates toward zero (2.7 -> 2,
    // -0.7 -> 0), a negative degree gives 0, degree 0 gives 1 even at a NaN x -- and run up to degree 40; x spans the
    // edges and [-1.2, 1.2] (where the Chebyshev / Legendre values stay bounded and the comparison is meaningful).
    [SkippableTheory]
    [MemberData(nameof(PolynomialOps))]
    public void Polynomial_PerElementDegree_MatchesCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var x = Input(13, v => Math.Abs(v) > 6 || float.IsNaN(v) ? v : v / 5f);
        var rng = new Random(17);
        float[] special = { float.NaN, -3f, -0.7f, 0f, 0.5f, 1f, 2.7f, 2f, 3f, 7f };
        var n = new Tensor<float>((int[])x._shape.Clone());
        for (int i = 0; i < n.Length; i++) n[i] = i < 3 * special.Length ? special[i % special.Length] : rng.Next(0, 41);
        var f = Polynomial(op);
        // The recurrences grow like (2x)^n (Hermite: n!-ish), so compare relative to magnitude in float.
        AssertMatches(op, f(cpu, x, n), OnDevice("Tensor" + op, () => f(gpu, x, n)), 2e-5);
    }

    // An infinite or over-limit degree is the CPU's ArgumentOutOfRangeException, and a degree shaped unlike x its shape
    // error; the device path must leave both to the CPU rather than compute a result.
    [SkippableFact]
    public void Polynomial_InvalidDegree_ThrowsLikeCpu()
    {
        var gpu = GpuOrSkip();
        var x = Input(23);
        var scalar = new Tensor<float>(new[] { 1 });
        scalar[0] = 3f;
        Assert.Throws<ArgumentException>(() => { var r = gpu.TensorLegendrePolynomialP(x, scalar); _ = r[0]; });
        var n = new Tensor<float>((int[])x._shape.Clone());
        n[5] = float.PositiveInfinity;
        Assert.Throws<ArgumentOutOfRangeException>(() => { var r = gpu.TensorChebyshevPolynomialT(x, n); _ = r[0]; });
        n[5] = 2e6f;
        Assert.Throws<ArgumentOutOfRangeException>(() => { var r = gpu.TensorHermitePolynomialH(x, n); _ = r[0]; });
    }
}

/// <summary>The cases above on the machine's default DirectGpu backend.</summary>
[Collection("VulkanGlobalState")]
public class TorchParityElementwiseDeviceTests : TorchParityElementwiseDeviceTestsBase, IClassFixture<DirectGpuTensorEngineTestFixture>
{
    public TorchParityElementwiseDeviceTests(DirectGpuTensorEngineTestFixture fixture) : base(fixture) { }
}

/// <summary>The cases above on the Vulkan backend (#1127).</summary>
[Collection("VulkanGlobalState")]
public class TorchParityElementwiseVulkanDeviceTests : TorchParityElementwiseDeviceTestsBase, IClassFixture<VulkanTensorEngineTestFixture>
{
    public TorchParityElementwiseVulkanDeviceTests(VulkanTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("DirectGpuSerial")]
public class TorchParityElementwiseHipDeviceTests : TorchParityElementwiseDeviceTestsBase, IClassFixture<HipTensorEngineTestFixture>
{
    public TorchParityElementwiseHipDeviceTests(HipTensorEngineTestFixture fixture) : base(fixture) { }
}
