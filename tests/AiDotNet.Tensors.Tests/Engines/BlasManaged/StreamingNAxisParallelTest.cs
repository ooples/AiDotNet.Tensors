using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue B (#370) task B.2: verifies <see cref="StreamingStrategy"/>'s N-axis
/// parallel split is correct (bit-exact vs serial) and delivers a speedup on
/// Wide-N shapes.
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class StreamingNAxisParallelTest
{
    private readonly ITestOutputHelper _output;

    public StreamingNAxisParallelTest(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void Streaming_NAxisParallel_BitExact_Vs_Serial_FP32()
    {
        // Wide-N shape that triggers N-axis split: M small, N very large, K small.
        const int M = 16, N = 512, K = 8;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cSerial = new float[M * N];
        var cParallel = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Force PackingMode.ForceStreaming so we exercise StreamingStrategy directly.
        // Both calls use the same code path; parallelism is decided by the strategy
        // based on shape × procs. With M=16, N=512, K=8, work = 65536 — above the
        // 32K parallel threshold and N=512 has enough Nr=8 blocks for >1 procs.
        BlasManagedLib.Gemm<float>(
            a, K, transA: false,
            b, N, transB: false,
            cSerial, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 1 });

        BlasManagedLib.Gemm<float>(
            a, K, transA: false,
            b, N, transB: false,
            cParallel, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 8 });

        // Deterministic mode (the default) MUST give bit-exact output regardless of thread count.
        for (int i = 0; i < cSerial.Length; i++)
        {
            Assert.True(
                cSerial[i] == cParallel[i],
                $"Mismatch at [{i / N}, {i % N}]: serial={cSerial[i]:G9} parallel={cParallel[i]:G9}");
        }
    }

    [Fact]
    public void Streaming_NAxisParallel_BitExact_Vs_Serial_FP64()
    {
        const int M = 16, N = 512, K = 8;
        var rng = new Random(42);
        var a = new double[M * K];
        var b = new double[K * N];
        var cSerial = new double[M * N];
        var cParallel = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        BlasManagedLib.Gemm<double>(
            a, K, false, b, N, false, cSerial, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForceStreaming, NumThreads = 1 });

        BlasManagedLib.Gemm<double>(
            a, K, false, b, N, false, cParallel, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForceStreaming, NumThreads = 8 });

        for (int i = 0; i < cSerial.Length; i++)
            Assert.Equal(cSerial[i], cParallel[i]);
    }

    [Fact]
    public void Streaming_NAxisParallel_BitExact_TransB_FP32()
    {
        // transB=true exercises the [N, K] stored layout. Column slice in the
        // logical sense becomes row slice in the stored sense (contiguous).
        const int M = 16, N = 512, K = 8;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[N * K];  // [N, K] when transB
        var cSerial = new float[M * N];
        var cParallel = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(
            a, K, false,
            b, K, transB: true,
            cSerial, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 1 });

        BlasManagedLib.Gemm<float>(
            a, K, false,
            b, K, transB: true,
            cParallel, N, M, N, K,
            new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 8 });

        for (int i = 0; i < cSerial.Length; i++)
            Assert.Equal(cSerial[i], cParallel[i]);
    }

    [SkippableFact]
    public void Streaming_NAxisParallel_Delivers_Speedup_On_Wide_N()
    {
        // CI runs on 4-vCPU boxes: NumThreads=8 oversubscribes and parallel
        // loses to serial. Same fix landed on PR #402 commit 5354a2a6.
        // Bit-exactness vs serial is asserted by the unconditional sibling
        // tests, so undersized-runner skip costs no correctness coverage.
        Skip.IfNot(Environment.ProcessorCount >= 8,
            $"Requires >=8 logical processors for NumThreads=8 to deliver speedup; have {Environment.ProcessorCount}.");

        // Big enough work that parallel overhead is amortized.
        const int M = 32, N = 4096, K = 256;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Warmup
        for (int i = 0; i < 3; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 1 });

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 1 });
        sw.Stop();
        double serialMs = sw.Elapsed.TotalMilliseconds / 10;

        for (int i = 0; i < 3; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 8 });

        sw.Restart();
        for (int i = 0; i < 10; i++)
            BlasManagedLib.Gemm<float>(
                a, K, false, b, N, false, c, N, M, N, K,
                new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming, NumThreads = 8 });
        sw.Stop();
        double parallelMs = sw.Elapsed.TotalMilliseconds / 10;

        double speedup = serialMs / parallelMs;
        _output.WriteLine($"Streaming N-axis: serial={serialMs:F2}ms parallel={parallelMs:F2}ms speedup={speedup:F2}x");
        Assert.True(speedup >= 2.0, $"Expected >=2x speedup, got {speedup:F2}x (serial={serialMs:F2}ms, parallel={parallelMs:F2}ms)");
    }
}
