using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue B (#370) task B.3: verifies <see cref="PackAOnlyStrategy"/>'s N-axis
/// parallel split is bit-exact vs serial and delivers a speedup on Wide-N shapes.
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class PackAOnlyNAxisParallelTest
{
    private readonly ITestOutputHelper _output;

    public PackAOnlyNAxisParallelTest(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void PackAOnly_NAxisParallel_BitExact_Vs_Serial_FP32()
    {
        // Wide-N shape where PackAOnly is selected (K small enough to skip B-pack
        // but big enough that streaming would be slower). PackAOnly currently uses
        // Mr=Nr=4 in BlasManaged.Gemm so M and N must be multiples of 4.
        const int M = 32, N = 512, K = 64;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cSerial = new float[M * N];
        var cParallel = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(
            a, K, false, b, N, false, cSerial, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 1 });

        BlasManagedLib.Gemm<float>(
            a, K, false, b, N, false, cParallel, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 8 });

        for (int i = 0; i < cSerial.Length; i++)
            Assert.True(
                cSerial[i] == cParallel[i],
                $"Mismatch at [{i / N}, {i % N}]: serial={cSerial[i]:G9} parallel={cParallel[i]:G9}");
    }

    [Fact]
    public void PackAOnly_NAxisParallel_BitExact_Vs_Serial_FP64()
    {
        const int M = 32, N = 512, K = 64;
        var rng = new Random(42);
        var a = new double[M * K];
        var b = new double[K * N];
        var cSerial = new double[M * N];
        var cParallel = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        BlasManagedLib.Gemm<double>(
            a, K, false, b, N, false, cSerial, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 1 });

        BlasManagedLib.Gemm<double>(
            a, K, false, b, N, false, cParallel, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 8 });

        for (int i = 0; i < cSerial.Length; i++)
            Assert.Equal(cSerial[i], cParallel[i]);
    }

    [SkippableFact]
    public void PackAOnly_NAxisParallel_Delivers_Speedup_On_Wide_N()
    {
        // Per-thread overhead dominates on low-core hosts: the parallel
        // configuration spawns NumThreads=8 even when the box only has 2-4
        // physical cores, so the parallel path loses to the serial baseline
        // (CI run 26304260634 measured 0.30x on a 4-vCPU runner). The
        // contract this test pins is "parallel pack-A wins on adequately
        // multi-core hardware"; gate to >=8 cores so we don't false-positive
        // on undersized CI runners. Bit-exactness vs serial is covered by
        // the unconditional [Fact] siblings above, so undersized-runner
        // skip doesn't lose any correctness coverage.
        Skip.IfNot(Environment.ProcessorCount >= 8,
            $"Requires >=8 logical processors for NumThreads=8 to deliver speedup; have {Environment.ProcessorCount}.");

        // Need M < procs*mr*2 so AxisSelector picks N over M. With procs=8 and
        // PackAOnly's Mr=4, the M-axis threshold is 64; using M=32 forces N.
        const int M = 32, N = 4096, K = 64;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int i = 0; i < 3; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 1 });

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 1 });
        sw.Stop();
        double serialMs = sw.Elapsed.TotalMilliseconds / 10;

        for (int i = 0; i < 3; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 8 });

        sw.Restart();
        for (int i = 0; i < 10; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackAOnly, NumThreads = 8 });
        sw.Stop();
        double parallelMs = sw.Elapsed.TotalMilliseconds / 10;

        double speedup = serialMs / parallelMs;
        _output.WriteLine($"PackAOnly N-axis: serial={serialMs:F2}ms parallel={parallelMs:F2}ms speedup={speedup:F2}x");
        Assert.True(speedup >= 2.0, $"Expected >=2x speedup, got {speedup:F2}x");
    }
}
