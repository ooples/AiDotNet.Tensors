using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

#if !NET471 // The direct-parallel kernel + the #368 carve-out are net5+/x86 only.
/// <summary>
/// #368 thin-M fast path. Verifies (a) the no-pack direct-parallel kernel
/// <see cref="SimdGemm.SgemmDirectParallelMInto"/> matches a naive reference across
/// the routed regime including the M %% 6 != 0 tail, and (b) the full
/// <c>SimdGemm.Sgemm</c> → <c>BlasManaged.Gemm</c> carve-out path is numerically
/// correct (a launch/tail/dispatch bug shows as O(1) error, not FMA rounding). A
/// perf probe (env-gated) confirms thin-M now runs on the fast kernel instead of the
/// ~55 GF/s machine-code path it previously fell to.
/// </summary>
public class BlasManagedThinMDirectTests
{
    private readonly ITestOutputHelper _out;
    public BlasManagedThinMDirectTests(ITestOutputHelper o) => _out = o;

    public static System.Collections.Generic.IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 128, 784, 512 },  // AIsEval MLP L0
        new object[] { 256, 512, 512 },
        new object[] { 512, 784, 512 },
        new object[] { 130, 520, 512 },  // M % 6 != 0 -> tail rows
        new object[] { 200, 640, 256 },  // N = 256
        new object[] { 64, 1024, 512 },  // edges of the routed K/M range
        new object[] { 1024, 784, 512 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void DirectParallelMInto_MatchesReference(int m, int k, int n)
    {
        var rng = RandomHelper.CreateSeededRandom(368);
        var a = RandF(m * k, rng);
        var b = RandF(k * n, rng);
        var got = new float[m * n];
        var want = new float[m * n];

        Reference(a, b, want, m, k, n);
        SimdGemm.SgemmDirectParallelMInto(a, b, got, m, k, n);

        Assert.True(RelErr(got, want) < 1e-3,
            $"SgemmDirectParallelMInto rel error at {m}x{k}x{n} — tail/launch bug.");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Gemm_ThinMCarveout_MatchesReference(int m, int k, int n)
    {
        // SimdGemm.Sgemm forwards to BlasManaged.Gemm<float>, which (#368) routes
        // this thin-M regime through the direct-parallel kernel. Verify end-to-end.
        var rng = RandomHelper.CreateSeededRandom(369);
        var a = RandF(m * k, rng);
        var b = RandF(k * n, rng);
        var got = new float[m * n];
        var want = new float[m * n];

        Reference(a, b, want, m, k, n);
#pragma warning disable CS0618 // Sgemm shim forwards to BlasManaged.Gemm — exactly what we want to exercise.
        SimdGemm.Sgemm(a, b, got, m, k, n);
#pragma warning restore CS0618

        Assert.True(RelErr(got, want) < 1e-3,
            $"BlasManaged.Gemm thin-M carve-out rel error at {m}x{k}x{n}.");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void DgemmDirectParallelMInto_MatchesReference(int m, int k, int n)
    {
        var rng = RandomHelper.CreateSeededRandom(640);
        var a = RandD(m * k, rng);
        var b = RandD(k * n, rng);
        var got = new double[m * n];
        var want = new double[m * n];

        ReferenceD(a, b, want, m, k, n);
        SimdGemm.DgemmDirectParallelMInto(a, b, got, m, k, n);

        Assert.True(RelErrD(got, want) < 1e-10,
            $"DgemmDirectParallelMInto rel error at {m}x{k}x{n} — tail/launch bug.");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void GemmDouble_ThinMCarveout_MatchesReference(int m, int k, int n)
    {
        // BlasManaged.Gemm<double> routes this thin-M regime through the FP64 direct
        // kernel (#368). Verify end-to-end vs a naive reference.
        var rng = RandomHelper.CreateSeededRandom(641);
        var a = RandD(m * k, rng);
        var b = RandD(k * n, rng);
        var got = new double[m * n];
        var want = new double[m * n];

        ReferenceD(a, b, want, m, k, n);
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
            a, k, false, b, n, false, got, n, m, n, k, default);

        Assert.True(RelErrD(got, want) < 1e-10,
            $"BlasManaged.Gemm<double> thin-M carve-out rel error at {m}x{k}x{n}.");
    }

    // Transposed thin-M (#368 strided kernels — no transpose materialised).
    // transA: C = Aᵀ·B, A stored [k,m]. transB: C = A·Bᵀ, B stored [n,k].
    [Theory]
    [MemberData(nameof(Shapes))]
    public void GemmDouble_ThinM_TransA_MatchesReference(int m, int k, int n)
    {
        var rng = RandomHelper.CreateSeededRandom(820);
        var A = RandD(m * k, rng); var B = RandD(k * n, rng); // logical A[m,k], B[k,n]
        var aStored = TransposeD(A, m, k);                    // [k,m] = Aᵀ (lda=m)
        var got = new double[m * n]; var want = new double[m * n];
        ReferenceD(A, B, want, m, k, n);
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
            aStored, m, true, B, n, false, got, n, m, n, k, default);
        Assert.True(RelErrD(got, want) < 1e-10, $"double transA at {m}x{k}x{n}.");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void GemmDouble_ThinM_TransB_MatchesReference(int m, int k, int n)
    {
        var rng = RandomHelper.CreateSeededRandom(821);
        var A = RandD(m * k, rng); var B = RandD(k * n, rng); // logical A[m,k], B[k,n]
        var bStored = TransposeD(B, k, n);                    // [n,k] = Bᵀ (ldb=k)
        var got = new double[m * n]; var want = new double[m * n];
        ReferenceD(A, B, want, m, k, n);
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
            A, k, false, bStored, k, true, got, n, m, n, k, default);
        Assert.True(RelErrD(got, want) < 1e-10, $"double transB at {m}x{k}x{n}.");
    }

    private static double[] TransposeD(double[] src, int rows, int cols)
    {
        var d = new double[rows * cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) d[j * rows + i] = src[i * cols + j];
        return d;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void GemmFloat_ThinM_TransA_MatchesReference(int m, int k, int n)
    {
        var rng = RandomHelper.CreateSeededRandom(822);
        var A = RandF(m * k, rng); var B = RandF(k * n, rng);
        var aStored = TransposeF(A, m, k);
        var got = new float[m * n]; var want = new float[m * n];
        Reference(A, B, want, m, k, n);
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            aStored, m, true, B, n, false, got, n, m, n, k, default);
        Assert.True(RelErr(got, want) < 1e-3, $"float transA at {m}x{k}x{n}.");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void GemmFloat_ThinM_TransB_MatchesReference(int m, int k, int n)
    {
        var rng = RandomHelper.CreateSeededRandom(823);
        var A = RandF(m * k, rng); var B = RandF(k * n, rng);
        var bStored = TransposeF(B, k, n);
        var got = new float[m * n]; var want = new float[m * n];
        Reference(A, B, want, m, k, n);
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            A, k, false, bStored, k, true, got, n, m, n, k, default);
        Assert.True(RelErr(got, want) < 1e-3, $"float transB at {m}x{k}x{n}.");
    }

    private static float[] TransposeF(float[] src, int rows, int cols)
    {
        var d = new float[rows * cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) d[j * rows + i] = src[i * cols + j];
        return d;
    }

    // Fused bias+activation epilogue at thin-M: #368 applies it after the direct
    // kernel so FusedLinear hits the fast path. Verify out = ReLU(A·B + bias).
    [Theory]
    [InlineData(128, 784, 512)]
    [InlineData(256, 512, 256)]
    public void Gemm_ThinMEpilogue_BiasReLU_MatchesReference(int m, int k, int n)
    {
        var rng = RandomHelper.CreateSeededRandom(910);
        var a = RandF(m * k, rng);
        var b = RandF(k * n, rng);
        var bias = RandF(n, rng);
        var got = new float[m * n];
        var want = new float[m * n];

        Reference(a, b, want, m, k, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                want[i * n + j] = Math.Max(0f, want[i * n + j] + bias[j]);

        var opts = new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float>
        {
            Epilogue = new AiDotNet.Tensors.Engines.BlasManaged.Epilogue<float>
            {
                BiasN = bias,
                Activation = AiDotNet.Tensors.Engines.FusedActivationType.ReLU,
            },
        };
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            a, k, false, b, n, false, got, n, m, n, k, opts);

        Assert.True(RelErr(got, want) < 1e-3, $"thin-M bias+ReLU epilogue at {m}x{k}x{n}.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public unsafe void Gemm_ThinM_NowBeatsMachineCodePath()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") != "1")
            return; // perf gate — dedicated hardware only

        var rng = RandomHelper.CreateSeededRandom(368);
        const int M = 128, K = 784, N = 512;  // AIsEval MLP L0
        var a = RandF(M * K, rng);
        var b = RandF(K * N, rng);
        var c = new float[M * N];
        double flop = 2.0 * M * N * K;

        double Min(Action f) { for (int i = 0; i < 50; i++) f(); double mn = double.MaxValue; for (int i = 0; i < 1000; i++) { var sw = Stopwatch.StartNew(); f(); sw.Stop(); mn = Math.Min(mn, sw.Elapsed.TotalMilliseconds); } return mn; }

#pragma warning disable CS0618
        double gemmMs = Min(() => SimdGemm.Sgemm(a, b, c, M, K, N));
#pragma warning restore CS0618
        double gemmGf = flop / (gemmMs * 1e6);
        _out.WriteLine($"BlasManaged.Gemm<float> thin-M L0 [128x784x512]: {gemmGf:F0} GF/s (pre-#368 ~55).");

        // FP64 thin-M at the same shape — is double also slow (no direct-parallel
        // double kernel exists) or does the machine-code FP64 6x8 path handle it?
        var ad = new double[M * K]; var bd = new double[K * N]; var cd = new double[M * N];
        for (int i = 0; i < ad.Length; i++) ad[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < bd.Length; i++) bd[i] = rng.NextDouble() * 2 - 1;
        double dMs = Min(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
            ad, K, false, bd, N, false, cd, N, M, N, K, default));
        double dGf = flop / (dMs * 1e6);
        _out.WriteLine($"BlasManaged.Gemm<double> thin-M L0 [128x784x512]: {dGf:F0} GF/s (pre-#368 ~60).");

        // transposed (strided kernels — pre-#368 the strategy gave ~57).
        var bT = new double[N * K]; for (int i = 0; i < bT.Length; i++) bT[i] = rng.NextDouble() * 2 - 1;
        var cTb = new double[M * N];
        double tbMs = Min(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
            ad, K, false, bT, K, true, cTb, N, M, N, K, default));
        _out.WriteLine($"BlasManaged.Gemm<double> thin-M L0 transB: {flop / (tbMs * 1e6):F0} GF/s (pre-#368 ~57).");
        var aT = new double[K * M]; for (int i = 0; i < aT.Length; i++) aT[i] = rng.NextDouble() * 2 - 1;
        var cTa = new double[M * N];
        double taMs = Min(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<double>(
            aT, M, true, bd, N, false, cTa, N, M, N, K, default));
        _out.WriteLine($"BlasManaged.Gemm<double> thin-M L0 transA: {flop / (taMs * 1e6):F0} GF/s (pre-#368 ~53).");

        var bTf = RandF(N * K, rng); var cTbf = new float[M * N];
        double tbfMs = Min(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            a, K, false, bTf, K, true, cTbf, N, M, N, K, default));
        _out.WriteLine($"BlasManaged.Gemm<float>  thin-M L0 transB: {flop / (tbfMs * 1e6):F0} GF/s");
        var aTf = RandF(K * M, rng); var cTaf = new float[M * N];
        double tafMs = Min(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            aTf, M, true, b, N, false, cTaf, N, M, N, K, default));
        _out.WriteLine($"BlasManaged.Gemm<float>  thin-M L0 transA: {flop / (tafMs * 1e6):F0} GF/s");
        // Pre-#368 double routed to the ~60 GF/s machine-code/strategy path; the new
        // 4x8 direct kernel clears it comfortably. 100 GF/s floor catches a regression.
        Assert.True(dGf > 100, $"BlasManaged.Gemm<double> thin-M L0 = {dGf:F0} GF/s — below the 100 GF/s floor (regressed off the #368 FP64 direct path?).");
        // Pre-#368 this routed to the ~55 GF/s machine-code path; the direct kernel
        // is ~200-464 GF/s here. A generous 100 GF/s floor catches a regression to
        // the slow path without being flaky on shared hardware.
        Assert.True(gemmGf > 100, $"BlasManaged.Gemm thin-M L0 = {gemmGf:F0} GF/s — below the 100 GF/s floor (regressed off the #368 direct path?).");
    }

    private static void Reference(float[] a, float[] b, float[] c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                c[i * n + j] = acc;
            }
    }

    private static double RelErr(float[] got, float[] want)
    {
        double maxAbs = 0, maxMag = 0;
        for (int i = 0; i < want.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i]));
            maxMag = Math.Max(maxMag, Math.Abs(want[i]));
        }
        return maxAbs / Math.Max(1e-6, maxMag);
    }

    private static float[] RandF(int len, Random rng)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static void ReferenceD(double[] a, double[] b, double[] c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0.0;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                c[i * n + j] = acc;
            }
    }

    private static double RelErrD(double[] got, double[] want)
    {
        double maxAbs = 0, maxMag = 0;
        for (int i = 0; i < want.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i]));
            maxMag = Math.Max(maxMag, Math.Abs(want[i]));
        }
        return maxAbs / Math.Max(1e-9, maxMag);
    }

    private static double[] RandD(int len, Random rng)
    {
        var a = new double[len];
        for (int i = 0; i < len; i++) a[i] = rng.NextDouble() * 2 - 1;
        return a;
    }
}
#endif
