using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #475 Phase 1/2/4: locks in the libxsmm-style JIT GEMM generator — the specialized FP32/FP64
/// kernels (and the fused bias + ReLU epilogue) must match a double-precision scalar reference and
/// must actually fire. This is the regression gate that protects the "we exceed OpenBLAS" win.
/// </summary>
public class JitGemmGeneratorTests
{
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
        if (!JitGemmGenerator.IsSupported) return; // net471 / non-x64 — managed path covers it
        var a = Rand(m * k, 1); var b = Rand(k * n, 2); var c = new float[m * n];
        bool ok;
        fixed (float* pa = a, pb = b, pc = c)
            ok = JitGemmGenerator.TryRunFp32(pa, k, pb, n, pc, n, m, n, k);
        Assert.True(ok, $"JIT FP32 must fire for {m}x{n}x{k} (n%8==0)");
        Assert.True(MaxAbsErr(c, ScalarRef(a, b, m, n, k, null, false)) < 1e-3, "FP32 result must match scalar ref");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public unsafe void Fp32_FusedBiasRelu_MatchesScalarReference(int m, int n, int k)
    {
        if (!JitGemmGenerator.IsSupported) return;
        var a = Rand(m * k, 3); var b = Rand(k * n, 4); var bias = Rand(n, 5); var c = new float[m * n];
        bool ok;
        fixed (float* pa = a, pb = b, pc = c, pbias = bias)
            ok = JitGemmGenerator.TryRunFp32(pa, k, pb, n, pc, n, m, n, k, pbias, relu: true);
        Assert.True(ok, $"fused JIT must fire for {m}x{n}x{k}");
        Assert.True(MaxAbsErr(c, ScalarRef(a, b, m, n, k, bias, relu: true)) < 1e-3, "fused relu(A·B+bias) must match scalar ref");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public unsafe void Fp64_MatchesScalarReference(int m, int n, int k)
    {
        if (!JitGemmGenerator.IsSupported) return;
        if ((n & 3) != 0) return; // FP64 lane is 4
        var af = Rand(m * k, 6); var bf = Rand(k * n, 7);
        var a = Array.ConvertAll(af, x => (double)x);
        var b = Array.ConvertAll(bf, x => (double)x);
        var c = new double[m * n];
        bool ok;
        fixed (double* pa = a, pb = b, pc = c)
            ok = JitGemmGenerator.TryRunFp64(pa, k, pb, n, pc, n, m, n, k);
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
}
