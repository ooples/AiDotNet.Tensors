// Verifies the singleton CUDA backend survives concurrent host-side submission
// from many threads without a sticky CUDA-700 (illegal address) that trips the
// process-wide GPU circuit breaker. Before the PushContext serialization lock,
// many threads interleaving cuMemAllocAsync / cuLaunchKernel / cuMemcpy*Async +
// cuStreamSynchronize on the ONE shared stream raced and faulted the context.
// Runs only when a GPU is present; CPU-only hosts early-return (no-op pass).

#if !NETFRAMEWORK
#nullable disable

using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GpuConcurrentSubmissionTests : IDisposable
{
    private readonly CpuEngine _cpu = new CpuEngine();
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _ready;

    public GpuConcurrentSubmissionTests()
    {
        try { _gpu = new DirectGpuTensorEngine(); _ready = _gpu.IsGpuAvailable; }
        catch { _ready = false; }
    }

    public void Dispose() => _gpu?.Dispose();

    [Fact]
    public async System.Threading.Tasks.Task ConcurrentGpuMatMul_DoesNotTripCircuitBreakerOrCorrupt()
    {
        await System.Threading.Tasks.Task.Yield();
        if (!_ready) return;
        // A prior GPU test in this process may already have tripped the (once-only,
        // process-wide) breaker; only meaningful to assert from a clean state.
        if (AiDotNetEngine.GpuCircuitBroken) return;

        const int M = 48, K = 48, N = 48;
        var a = Rand(1, M, K);
        var b = Rand(2, K, N);
        var reference = _cpu.TensorMatMul(a, b);
        int refLen = reference.Length;

        int threads = Math.Max(8, Environment.ProcessorCount);
        const int itersPerThread = 60;
        Exception failure = null;

        Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, _ =>
        {
            try
            {
                for (int i = 0; i < itersPerThread; i++)
                {
                    var c = _gpu.TensorMatMul(a, b);
                    // Force the download + validate: corruption from a stream race shows
                    // up as NaN/Inf or values off the CPU reference.
                    for (int j = 0; j < refLen; j++)
                    {
                        float v = c[j];
                        if (double.IsNaN(v) || double.IsInfinity(v) ||
                            Math.Abs(v - reference[j]) > 1e-2f)
                            throw new InvalidOperationException(
                                $"corrupt GPU result[{j}]={v} vs CPU ref {reference[j]}");
                    }
                }
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref failure, ex, null);
            }
        });

        Assert.False(AiDotNetEngine.GpuCircuitBroken,
            "GPU circuit breaker tripped — a CUDA fault (sticky 700) occurred under concurrent submission.");
        Assert.True(failure == null,
            $"Concurrent GPU submission faulted/corrupted: {failure?.GetType().Name}: {failure?.Message}");
    }

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }
}
#endif
