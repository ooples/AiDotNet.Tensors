using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1127: the double library the Vulkan parity kernels compute in (GLSL has exp / log / sin / cos for float only).
/// Each function runs on the device over a spread of arguments and is compared with .NET in units in the last place;
/// the special functions are compared with the CPU reference they port, after the one rounding to float the kernels
/// perform.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanTorchParityMathTests
{
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.SupportsTorchParityDoubles) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with shaderFloat64 and the runtime GLSL compiler.");
        return backend!;
    }

    private static double[] Evaluate(VulkanBackend backend, string expression, double[] inputs, bool special)
    {
        var bits = new float[2 * inputs.Length];
        Buffer.BlockCopy(inputs, 0, bits, 0, inputs.Length * sizeof(double));
        using var input = backend.AllocateBuffer(bits);
        using var output = backend.AllocateBuffer(2 * inputs.Length);
        backend.EvaluateTorchParityDoubleExpression(VulkanTorchParityMath.Doubles(expression), input, output, inputs.Length, special);
        var result = new double[inputs.Length];
        Buffer.BlockCopy(backend.DownloadBuffer(output), 0, result, 0, inputs.Length * sizeof(double));
        return result;
    }

    private static double Ulps(double want, double got)
    {
        if (double.IsNaN(want) || double.IsNaN(got)) return double.IsNaN(want) && double.IsNaN(got) ? 0 : double.PositiveInfinity;
        if (want == got) return 0;
        if (double.IsInfinity(want) || double.IsInfinity(got)) return double.PositiveInfinity;
        double a = Math.Abs(want);
        long next = BitConverter.DoubleToInt64Bits(a) + 1;
        return Math.Abs(want - got) / (BitConverter.Int64BitsToDouble(next) - a);
    }

    private static readonly double[] Edges =
    {
        0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 88, -88, 800, -800, 1e30, -1e30, 1647098.9, 1647099.1, 4e6,
        double.PositiveInfinity, double.NegativeInfinity, double.NaN,
    };

    private static double[] Inputs(string kind)
    {
        var rng = new Random(kind.Length * 7919 + kind[0]);
        double LogUniform(double lo, double hi) => Math.Exp(Math.Log(lo) + rng.NextDouble() * (Math.Log(hi) - Math.Log(lo)));
        double Signed(double v) => rng.Next(2) == 0 ? v : -v;
        double[] Many(int n, Func<double> f) => Enumerable.Range(0, n).Select(_ => f()).ToArray();
        return kind switch
        {
            "exp" => Edges.Concat(Many(8000, () => -750 + 1465 * rng.NextDouble())).Concat(Many(2000, () => Signed(LogUniform(1e-20, 1)))).ToArray(),
            "positive" => Edges.Concat(Many(8000, () => LogUniform(1e-300, 1e308))).Concat(Many(2000, () => 0.5 + rng.NextDouble())).ToArray(),
            "small" => Edges.Concat(Many(8000, () => Signed(LogUniform(1e-12, 100)))).ToArray(),
            "medium" => Many(8000, () => Signed(LogUniform(100, 1.6e6))),
            // Arguments next to multiples of pi / 2, where the reduction cancels almost every bit.
            "nearPio2" => Many(8000, () => Signed(Math.Round(LogUniform(1, 1e15)) * Math.PI / 2)),
            "large" => Many(12000, () => Signed(LogUniform(1.6e6, 1e308))),
            "hyperbolic" => Edges.Concat(Many(8000, () => Signed(LogUniform(1e-8, 700)))).ToArray(),
            "integers" => Edges.Concat(Many(3000, () => Signed(Math.Round(LogUniform(1, 1e6)) + 0.5))).Concat(Many(3000, () => Signed(LogUniform(1e-3, 1e17)))).ToArray(),
            "wide" => Edges.Concat(Many(8000, () => Signed(LogUniform(1e-3, 1e300)))).ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
    }

    public static TheoryData<string, string, string, double> ElementaryCases() => new()
    {
        { "exp", "d_exp(x)", "exp", 1 }, { "log", "d_log(x)", "positive", 1 },
        { "sin", "d_sin(x)", "small", 1 }, { "cos", "d_cos(x)", "small", 1 },
        { "sin", "d_sin(x)", "medium", 1 }, { "cos", "d_cos(x)", "medium", 1 },
        { "sin", "d_sin(x)", "nearPio2", 1 }, { "cos", "d_cos(x)", "nearPio2", 1 },
        { "sin", "d_sin(x)", "large", 1 }, { "cos", "d_cos(x)", "large", 1 },
        { "sinh", "d_sinh(x)", "hyperbolic", 4 }, { "cosh", "d_cosh(x)", "hyperbolic", 4 },
        { "rint", "d_rint(x)", "integers", 0 }, { "trunc", "d_trunc(x)", "integers", 0 },
        { "fmod", "d_fmod(x, 7.3)", "wide", 0 }, { "fmod2", "d_fmod(x, 2.0)", "integers", 0 },
    };

    [SkippableTheory]
    [MemberData(nameof(ElementaryCases))]
    public void ElementaryFunction_AgreesWithDotNet(string name, string expression, string inputKind, double maxUlps)
    {
        var backend = BackendOrSkip();
        Func<double, double> reference = name switch
        {
            "exp" => Math.Exp, "log" => Math.Log, "sin" => Math.Sin, "cos" => Math.Cos, "sinh" => Math.Sinh, "cosh" => Math.Cosh,
            "rint" => v => Math.Round(v, MidpointRounding.ToEven), "trunc" => Math.Truncate,
            "fmod" => v => v % 7.3, _ => v => v % 2.0,
        };
        double[] inputs = Inputs(inputKind);
        double[] actual = Evaluate(backend, expression, inputs, special: false);
        for (int i = 0; i < inputs.Length; i++)
        {
            double want = reference(inputs[i]);
            double ulps = Ulps(want, actual[i]);
            Assert.True(ulps <= maxUlps, $"{expression} at x = {inputs[i]:R}: .NET {want:R}, device {actual[i]:R} ({ulps:G3} ulp)");
        }
    }

    // The kernels move values between float buffers and doubles by bit pattern (a GPU's own conversions flush
    // subnormals to zero): widening must be exact and narrowing must round to nearest even, subnormals included.
    [SkippableFact]
    public void FloatConversions_AreExact_IncludingSubnormals()
    {
        var backend = BackendOrSkip();
        var rng = new Random(99);
        double[] doubles = Edges
            .Concat(new[] { 1e-39, -6e-39, 1.401298464324817e-45, 7e-46, 7.1e-46, 1.1754943508222875e-38, 1.17549428e-38, 3.4028234663852886e38, 3.4028235677973366e38, 3.5e38, 1e-50, 16777217.0, 1.0000000596046448 })
            .Concat(Enumerable.Range(0, 6000).Select(_ => (rng.Next(2) == 0 ? 1 : -1) * Math.Exp(-110 + 200 * rng.NextDouble()))).ToArray();
        double[] narrowed = Evaluate(backend, "double(d_to_float_bits(x))", doubles, special: false);
        for (int i = 0; i < doubles.Length; i++)
        {
            float want = (float)doubles[i];
            uint wantBits = unchecked((uint)TestHelpers.MathCompat.SingleToInt32Bits(want));
            if (float.IsNaN(want)) { Assert.True((((uint)narrowed[i]) & 0x7FFFFFFFu) > 0x7F800000u, $"NaN input {i}"); continue; }
            Assert.True(wantBits == (uint)narrowed[i], $"d_to_float_bits({doubles[i]:R}): want {wantBits:X8} ({want:R}), device {(uint)narrowed[i]:X8}");
        }

        // Widening: feed float bit patterns as integers held in doubles.
        float[] floats = { 0f, -0f, 1f, -1.5f, float.Epsilon, -float.Epsilon, 1e-39f, 6e-39f, 1.17549435e-38f, float.MaxValue, float.PositiveInfinity, float.NegativeInfinity, 3.1415927f, 1e-30f };
        double[] asBits = floats.Select(f => (double)unchecked((uint)TestHelpers.MathCompat.SingleToInt32Bits(f))).ToArray();
        double[] widened = Evaluate(backend, "d_from_float_bits(uint(x))", asBits, special: false);
        for (int i = 0; i < floats.Length; i++)
            Assert.True(BitConverter.DoubleToInt64Bits((double)floats[i]) == BitConverter.DoubleToInt64Bits(widened[i]),
                $"d_from_float_bits({floats[i]:R}): want {(double)floats[i]:R}, device {widened[i]:R}");
    }

    public static TheoryData<string, string> SpecialCases() => new()
    {
        { "Erf", "sf_erf(x)" }, { "Erfc", "sf_erfc(x)" }, { "Erfcx", "sf_erfcx(x)" }, { "Ndtr", "sf_ndtr(x)" }, { "LogNdtr", "sf_log_ndtr(x)" },
        { "Ndtri", "sf_ndtri(x)" }, { "LogGamma", "sf_lgamma(x)" }, { "SphericalBesselJ0", "sf_spherical_bessel_j0(x)" },
        { "AiryAi", "sf_airy_ai(x)" }, { "SinPi", "sf_sin_pi(x)" },
        { "BesselJ0", "sf_bessel_j(0, x)" }, { "BesselJ1", "sf_bessel_j(1, x)" }, { "BesselY0", "sf_bessel_y(0, x)" }, { "BesselY1", "sf_bessel_y(1, x)" },
        { "BesselI0", "sf_bessel_i(0, x)" }, { "BesselI1", "sf_bessel_i(1, x)" },
        { "BesselK0", "sf_bessel_k(0.0, x)" }, { "BesselK1", "sf_bessel_k(1.0, x)" },
        { "BesselK0e", "sf_bessel_k_scaled(0.0, x)" }, { "BesselK1e", "sf_bessel_k_scaled(1.0, x)" },
    };

    [SkippableTheory]
    [MemberData(nameof(SpecialCases))]
    public void SpecialFunction_MatchesTheCpuReference_AfterRoundingToFloat(string name, string expression)
    {
        var backend = BackendOrSkip();
        Func<double, double> reference = name switch
        {
            "Erf" => SpecialFunctions.Erf, "Erfc" => SpecialFunctions.Erfc, "Erfcx" => SpecialFunctions.Erfcx, "Ndtr" => SpecialFunctions.Ndtr,
            "LogNdtr" => SpecialFunctions.LogNdtr, "Ndtri" => SpecialFunctions.Ndtri, "LogGamma" => SpecialFunctions.LogGamma,
            "SphericalBesselJ0" => SpecialFunctions.SphericalBesselJ0, "AiryAi" => SpecialFunctions.AiryAi, "SinPi" => SpecialFunctions.SinPi,
            "BesselJ0" => v => SpecialFunctions.BesselJ(0, v), "BesselJ1" => v => SpecialFunctions.BesselJ(1, v),
            "BesselY0" => v => SpecialFunctions.BesselY(0, v), "BesselY1" => v => SpecialFunctions.BesselY(1, v),
            "BesselI0" => v => SpecialFunctions.BesselI(0, v), "BesselI1" => v => SpecialFunctions.BesselI(1, v),
            "BesselK0" => v => SpecialFunctions.BesselK(0, v), "BesselK1" => v => SpecialFunctions.BesselK(1, v),
            "BesselK0e" => v => SpecialFunctions.BesselKScaled(0, v), _ => v => SpecialFunctions.BesselKScaled(1, v),
        };
        // Arguments a float tensor can hold (the kernels only ever see those), across each function's branches.
        var rng = new Random(name.Length * 31 + name[0]);
        double F(double v) => (float)v;
        double[] inputs = new double[] { 0, -0.0, 1, -1, 0.5, 2.5, 88, -88, 800, -800, 1e30, -1e30, double.PositiveInfinity, double.NegativeInfinity, double.NaN }
            .Concat(Enumerable.Range(0, 2500).Select(_ => F(name == "Ndtri" ? rng.NextDouble() : name == "AiryAi" ? (rng.NextDouble() - 0.5) * 60 : (rng.NextDouble() - 0.5) * 12)))
            .Concat(Enumerable.Range(0, 1200).Select(_ => F((rng.Next(2) == 0 ? 1 : -1) * Math.Exp(Math.Log(1e-6) + rng.NextDouble() * Math.Log(1e10)))))
            .ToArray();
        double[] actual = Evaluate(backend, expression, inputs, special: true);
        for (int i = 0; i < inputs.Length; i++)
        {
            float want = (float)reference(inputs[i]), got = (float)actual[i];
            if (float.IsNaN(want) || float.IsNaN(got)) { Assert.True(float.IsNaN(want) && float.IsNaN(got), $"{expression} at {inputs[i]:R}: cpu {want:R}, device {got:R}"); continue; }
            if (float.IsInfinity(want) || float.IsInfinity(got)) { Assert.True(want == got, $"{expression} at {inputs[i]:R}: cpu {want:R}, device {got:R}"); continue; }
            Assert.True(Math.Abs((double)want - got) <= 1e-6 * Math.Max(1.0, Math.Abs(want)), $"{expression} at {inputs[i]:R}: cpu {want:R}, device {got:R}");
        }
    }

    // No device needed: libshaderc compiles on the CPU, so this runs wherever the native library loads (CI included)
    // and proves every kernel source is valid GLSL for Vulkan.
    [SkippableFact]
    public void EveryParityShader_CompilesToSpirv()
    {
        VulkanGlslCompiler compiler;
        try { compiler = new VulkanGlslCompiler(); }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or BadImageFormatException) { compiler = null!; }
        Skip.If(compiler is null || !compiler.IsAvailable, "libshaderc is not available.");
        using (compiler)
        {
            int count = 0;
            foreach (var (name, source) in VulkanTorchParityElementwiseKernels.All())
            {
                uint[]? spirv = compiler!.CompileToSpirv(source);
                Assert.True(spirv is { Length: > 5 }, $"{name} did not compile: {compiler.LastError}");
                Assert.Equal(0x07230203u, spirv![0]);   // SPIR-V magic number
                count++;
            }
            // One per unary and binary op, plus the 31 other kernels.
            Assert.Equal(Enum.GetValues(typeof(TorchParityUnaryOp)).Length + Enum.GetValues(typeof(TorchParityBinaryOp)).Length + 31, count);
            Assert.NotNull(compiler.CompileToSpirv(VulkanTorchParityElementwiseKernels.DoubleProbe("d_exp(x)", special: false)));
        }
    }

    [Fact]
    public void KernelsThatNeedDoubles_AreExactlyThoseThatComputeInDouble()
    {
        foreach (TorchParityUnaryOp op in Enum.GetValues(typeof(TorchParityUnaryOp)))
            Assert.Equal(VulkanTorchParityElementwiseKernels.NeedsDoubles(op), VulkanTorchParityElementwiseKernels.Unary(op).Contains("double "));
        foreach (TorchParityBinaryOp op in Enum.GetValues(typeof(TorchParityBinaryOp)))
            Assert.Equal(VulkanTorchParityElementwiseKernels.NeedsDoubles(op), VulkanTorchParityElementwiseKernels.Binary(op).Contains("double "));
    }

    [Fact]
    public void TwoOverPi_HasTheKnownLeadingWords()
    {
        uint[] words = VulkanTorchParityMath.TwoOverPiWords(VulkanTorchParityMath.TwoOverPiWordCount);
        Assert.Equal(VulkanTorchParityMath.TwoOverPiWordCount, words.Length);
        // 2 / pi = 0x0.A2F9836E 4E441529 FC2757D1 F534DDC0 DB629599 3C439041 ...
        Assert.Equal(new uint[] { 0xA2F9836Eu, 0x4E441529u, 0xFC2757D1u, 0xF534DDC0u, 0xDB629599u, 0x3C439041u }, words.Take(6).ToArray());
        // Asking for fewer words gives a prefix: the guard bits make the leading words independent of the count.
        Assert.Equal(words.Take(8).ToArray(), VulkanTorchParityMath.TwoOverPiWords(8));
    }

    [Fact]
    public void Doubles_SuffixesFloatingLiterals_AndNothingElse()
    {
        Assert.Equal("x * 0.5lf + 1e-17lf - 3.lf", VulkanTorchParityMath.Doubles("x * 0.5 + 1e-17 - 3."));
        Assert.Equal("a[2u * i] = 0x7FF00000u; int k = 10; v1 = pio2_1;", VulkanTorchParityMath.Doubles("a[2u * i] = 0x7FF00000u; int k = 10; v1 = pio2_1;"));
        Assert.Equal("o[i] = 0.0f; y = 1.0lf;", VulkanTorchParityMath.Doubles("o[i] = 0.0f; y = 1.0;"));
        Assert.Equal("d = 2.5e+3lf * 0x1E5u;", VulkanTorchParityMath.Doubles("d = 2.5e+3 * 0x1E5u;"));
        // Every literal in the shipped library is suffixed.
        Assert.DoesNotMatch(@"(?<![\w.])\d+\.\d*(?:[eE][-+]?\d+)?(?![\w.])", VulkanTorchParityMath.ElementaryFunctions);
        Assert.Contains("const uint TWO_OVER_PI[40]", VulkanTorchParityMath.Tables);
        Assert.Contains("const double SF_GL96_X[96]", VulkanTorchParityMath.Tables);
    }
}
