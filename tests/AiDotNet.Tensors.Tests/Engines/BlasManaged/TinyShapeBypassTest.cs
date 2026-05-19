using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue C (#371) task C.1: verifies the tiny-shape bypass in
/// <see cref="BlasManagedLib.Gemm{T}"/> produces bit-exact output vs the regular
/// path and dispatches faster (skips strategy / autotune / options-switch).
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class TinyShapeBypassTest
{
    private readonly ITestOutputHelper _output;

    public TinyShapeBypassTest(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Tiny shape (M*N*K = 32K, below the 100K bypass threshold) must produce
    /// the same result whether routed via bypass or via the full dispatcher path.
    /// Use PackingMode.ForcePackBoth on one side to force the non-bypass path.
    /// </summary>
    [Fact]
    public void Bypass_BitExact_Vs_ForcedFullPath_FP32()
    {
        const int M = 32, N = 32, K = 32;  // 32K work, under bypass threshold
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cBypass = new float[M * N];
        var cForcedPath = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Default options → bypass fires (M*N*K=32K below threshold).
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBypass, N, M, N, K);

        // ForcePackBoth forces the full dispatcher path even for tiny shapes.
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cForcedPath, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });

        for (int i = 0; i < cBypass.Length; i++)
            Assert.True(cBypass[i] == cForcedPath[i],
                $"Tiny-shape bypass mismatch at [{i / N}, {i % N}]: " +
                $"bypass={cBypass[i]:G9} forced={cForcedPath[i]:G9}");
    }

    [Fact]
    public void Bypass_BitExact_Vs_ForcedFullPath_FP64()
    {
        const int M = 32, N = 32, K = 32;
        var rng = new Random(42);
        var a = new double[M * K];
        var b = new double[K * N];
        var cBypass = new double[M * N];
        var cForcedPath = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cBypass, N, M, N, K);
        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cForcedPath, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth });

        for (int i = 0; i < cBypass.Length; i++)
            Assert.Equal(cBypass[i], cForcedPath[i]);
    }

    [Fact]
    public void Bypass_Faster_Than_Full_Path_On_Tiny_Shape()
    {
        // Tiny shape where dispatcher overhead dominates real compute. The
        // bypass should beat the full path by a clear margin.
        const int M = 16, N = 16, K = 16;  // 4K work, deep below threshold
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        const int Iters = 1000;

        // Warmup both paths.
        for (int i = 0; i < 50; i++)
        {
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K);
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K);
        sw.Stop();
        double bypassUs = sw.Elapsed.TotalMicroseconds / Iters;

        sw.Restart();
        for (int i = 0; i < Iters; i++)
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });
        sw.Stop();
        double forcedUs = sw.Elapsed.TotalMicroseconds / Iters;

        double speedup = forcedUs / bypassUs;
        _output.WriteLine($"Tiny bypass: bypass={bypassUs:F2}us forced={forcedUs:F2}us speedup={speedup:F2}x");
        Assert.True(speedup >= 2.0,
            $"Expected bypass >=2x faster than forced full path, got {speedup:F2}x " +
            $"(bypass={bypassUs:F2}us, forced={forcedUs:F2}us)");
    }

    [Fact]
    public void Bypass_Skipped_When_Above_Threshold()
    {
        // Bigger shape — bypass must NOT fire (M*N*K > threshold). Verify by
        // ensuring the full path's output is still correct (this is the path
        // most tests already exercise; mostly a smoke check).
        const int M = 128, N = 128, K = 128;  // 2M work, well above threshold
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K);

        // Sanity: result should be non-zero (would be all-zero if c.Clear hit but
        // the kernel never ran).
        bool anyNonZero = false;
        for (int i = 0; i < c.Length; i++) if (c[i] != 0) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Result must be non-zero — kernel didn't run?");
    }
}
