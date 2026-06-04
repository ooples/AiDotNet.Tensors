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
        _out.WriteLine($"BlasManaged.Gemm thin-M L0 [128x784x512]: {gemmGf:F0} GF/s (pre-#368 ~55 on the strategy/machine-code path).");
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
}
#endif
