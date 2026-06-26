// The JIT machine-code GEMM generator is net5+ only (it needs System.Runtime.Intrinsics),
// so the BlasManaged.Jit namespace does not exist on net471. Guard the whole file to match
// the src guard on JitGemmGenerator (#if NET5_0_OR_GREATER); otherwise net471 fails CS0234.
#if NET5_0_OR_GREATER
using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged.Jit;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #475 Phase 1/2/4: locks in the libxsmm-style JIT GEMM generator — the specialized FP32/FP64
/// kernels (and the fused bias + ReLU epilogue) must match a double-precision scalar reference and
/// must actually fire. This is the regression gate that protects the "we exceed OpenBLAS" win.
/// </summary>
public class JitGemmGeneratorTests
{
    // The ISA/OS contract JitGemmGenerator.IsSupported must honor (everything EXCEPT the Enabled
    // master switch). On a runner that meets this, IsSupported MUST be true — otherwise a silent
    // support-gate regression has disabled the JIT path; the tests assert that rather than skipping.
    // On a runner that does NOT meet it (no AVX2/FMA, non-x64), the JIT path must report unsupported
    // and the TryRun* entrypoints must return false so callers fall back to the managed kernels.
    private static bool HardwareSupportsJit =>
        Fma.IsSupported && Avx2.IsSupported &&
        RuntimeInformation.ProcessArchitecture == Architecture.X64 &&
        (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || RuntimeInformation.IsOSPlatform(OSPlatform.Linux));
    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1);
        return a;
    }

    private static double MaxAbsErr(float[] got, double[] reference)
    {
        double e = 0;
        for (int i = 0; i < got.Length; i++) e = Math.Max(e, Math.Abs((double)got[i] - reference[i]));
        return e;
    }

    // C := relu?(A·B + bias?), row-major, double accumulation reference.
    private static double[] ScalarRef(float[] a, float[] b, int m, int n, int k, float[] bias, bool relu)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += (double)a[i * k + p] * b[p * n + j];
                if (bias != null) s += bias[j];
                if (relu && s < 0) s = 0;
                c[i * n + j] = s;
            }
        return c;
    }

    public static TheoryData<int, int, int> Shapes() => new()
    {
        { 6, 16, 64 }, { 6, 8, 32 }, { 12, 32, 128 }, { 24, 48, 96 },
        { 48, 64, 256 }, { 64, 64, 128 }, { 96, 96, 200 }, { 128, 128, 64 },
        { 7, 8, 50 }, { 13, 24, 17 }, { 1, 16, 33 }, { 5, 40, 1 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public unsafe void Fp32_MatchesScalarReference(int m, int n, int k)
    {
        var a = Rand(m * k, 1); var b = Rand(k * n, 2); var c = new float[m * n];
        bool ok;
        fixed (float* pa = a, pb = b, pc = c)
            ok = JitGemmGenerator.TryRunFp32(pa, k, pb, n, pc, n, m, n, k);
        if (!HardwareSupportsJit)
        {
            // No AVX2/FMA or non-x64: the JIT path must report unsupported and decline so the caller
            // takes the managed kernel. (Asserts the fallback contract instead of silently passing.)
            Assert.False(JitGemmGenerator.IsSupported);
            Assert.False(ok);
            return;
        }
        Assert.True(JitGemmGenerator.IsSupported, "JIT must report supported on this AVX2/FMA x64 host");
        Assert.True(ok, $"JIT FP32 must fire for {m}x{n}x{k} (n%8==0)");
        Assert.True(MaxAbsErr(c, ScalarRef(a, b, m, n, k, null, false)) < 1e-3, "FP32 result must match scalar ref");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public unsafe void Fp32_FusedBiasRelu_MatchesScalarReference(int m, int n, int k)
    {
        var a = Rand(m * k, 3); var b = Rand(k * n, 4); var bias = Rand(n, 5); var c = new float[m * n];
        bool ok;
        fixed (float* pa = a, pb = b, pc = c, pbias = bias)
            ok = JitGemmGenerator.TryRunFp32(pa, k, pb, n, pc, n, m, n, k, pbias, relu: true);
        if (!HardwareSupportsJit)
        {
            Assert.False(JitGemmGenerator.IsSupported);
            Assert.False(ok);
            return;
        }
        Assert.True(JitGemmGenerator.IsSupported, "JIT must report supported on this AVX2/FMA x64 host");
        Assert.True(ok, $"fused JIT must fire for {m}x{n}x{k}");
        Assert.True(MaxAbsErr(c, ScalarRef(a, b, m, n, k, bias, relu: true)) < 1e-3, "fused relu(A·B+bias) must match scalar ref");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public unsafe void Fp64_MatchesScalarReference(int m, int n, int k)
    {
        var af = Rand(m * k, 6); var bf = Rand(k * n, 7);
        var a = Array.ConvertAll(af, x => (double)x);
        var b = Array.ConvertAll(bf, x => (double)x);
        var c = new double[m * n];
        bool ok;
        fixed (double* pa = a, pb = b, pc = c)
            ok = JitGemmGenerator.TryRunFp64(pa, k, pb, n, pc, n, m, n, k);
        if (!HardwareSupportsJit)
        {
            Assert.False(JitGemmGenerator.IsSupported);
            Assert.False(ok);
            return;
        }
        Assert.True(JitGemmGenerator.IsSupported, "JIT must report supported on this AVX2/FMA x64 host");
        if ((n & 3) != 0)
        {
            // FP64 lane is 4: partial-4 columns are out of the JIT envelope, so it must DECLINE
            // (return false) and let the managed path handle it — verify the decline, don't skip.
            Assert.False(ok, $"JIT FP64 must decline n%4!=0 ({m}x{n}x{k})");
            return;
        }
        Assert.True(ok, $"JIT FP64 must fire for {m}x{n}x{k} (n%4==0)");
        double e = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += a[i * k + p] * b[p * n + j];
                e = Math.Max(e, Math.Abs(c[i * n + j] - s));
            }
        Assert.True(e < 1e-9, "FP64 result must match scalar ref");
    }

    // ── #475 Phase 2: native BF16 path (vdpbf16ps) — encoding-verified here; correctness/perf on
    //    AVX-512-BF16 hardware or under Intel SDE (--verify-bf16gemm). ──────────────────────────

    [Fact]
    public void Bf16Microkernel_EmitsValidEncoding()
    {
        if (!HardwareSupportsJit)
        {
            // Non-x64 / no AVX2: the JIT path is unsupported; the x64 EVEX emitter is not exercised.
            Assert.False(JitGemmGenerator.IsSupported);
            return;
        }
        byte[] code = MachineCodeBf16Kernel.EmitGemmMicrokernelWindows();
        Assert.True(code.Length > 16, "BF16 microkernel must emit a non-trivial body");
        Assert.Equal(0xC3, code[^1]);          // ret
        Assert.Contains((byte)0x62, code);     // EVEX prefix — vdpbf16ps is raw EVEX (no .NET intrinsic)
    }

    [Fact]
    public void Bf16_StaysOffUntilExplicitlyEnabled()
    {
        // vdpbf16ps faults (#UD) without AVX-512-BF16, so the JIT BF16 path must never auto-run.
        Assert.False(JitGemmGenerator.EnableBf16);
        var a = new ushort[16]; var b = new ushort[16]; var c = new float[4];
        Assert.False(JitGemmGenerator.TryRunBf16(a, b, c, 2, 8, 2));
    }
}
#endif
