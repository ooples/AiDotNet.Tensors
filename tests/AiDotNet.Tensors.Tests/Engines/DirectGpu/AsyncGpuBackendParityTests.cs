#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IAsyncGpuBackend"/> on every backend (#1142): stream-ordered upload, GEMM, fused GEMM + bias + ReLU and
/// download produce the CPU result; recorded events complete, time forward, and a sync point reports completion.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class AsyncGpuBackendParityTests
{
    private const int M = 5, N = 7, K = 3;

    private static float[] Values(int n, int salt)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((i * 7919 + salt * 31) % 61 - 30) * 0.0625f;
        return v;
    }

    private static void Check(IDirectGpuBackend backend)
    {
        var async = Assert.IsAssignableFrom<IAsyncGpuBackend>(backend);
        var a = Values(M * K, 1);
        var b = Values(K * N, 2);
        var bias = Values(N, 3);
        var expectedGemm = new double[M * N];
        var expectedFused = new double[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double s = 0;
                for (int k = 0; k < K; k++) s += a[i * K + k] * b[k * N + j];
                expectedGemm[i * N + j] = s;
                expectedFused[i * N + j] = Math.Max(s + bias[j], 0);
            }

        var stream = async.CreateStream(GpuStreamType.Compute);
        using var bufA = backend.AllocateBuffer(M * K);
        using var bufB = async.AllocateBufferAsync(b, stream);
        using var bufBias = backend.AllocateBuffer(bias);
        using var gemm = backend.AllocateBuffer(M * N);
        using var fused = backend.AllocateBuffer(M * N);
        var start = async.CreateEvent(enableTiming: true);
        async.RecordEvent(start, stream);
        async.UploadBufferAsync(a, bufA, stream);
        async.GemmAsync(bufA, bufB, gemm, M, N, K, 1f, 0f, stream);
        async.FusedGemmBiasActivationAsync(bufA, bufB, bufBias, fused, M, N, K, FusedActivationType.ReLU, stream);
        var end = async.CreateEvent(enableTiming: true);
        async.RecordEvent(end, stream);
        var syncPoint = async.CreateSyncPoint(stream);
        async.SynchronizeStream(stream);

        Assert.True(async.QueryEventComplete(end));
        Assert.True(async.QueryStreamComplete(stream));
        Assert.True(async.GetEventElapsedTime(start, end) >= 0f);
        syncPoint.Wait();
        Assert.True(syncPoint.IsComplete);

        var gemmHost = new float[M * N];
        var fusedHost = new float[M * N];
        async.DownloadBufferAsync(gemm, gemmHost, stream);
        async.DownloadBufferAsync(fused, fusedHost, stream);
        async.SynchronizeStream(stream);
        for (int i = 0; i < M * N; i++)
        {
            Assert.True(Math.Abs(gemmHost[i] - expectedGemm[i]) <= 1e-4, $"GemmAsync[{i}]: {gemmHost[i]} vs {expectedGemm[i]}");
            Assert.True(Math.Abs(fusedHost[i] - expectedFused[i]) <= 1e-4, $"FusedGemmBiasActivationAsync[{i}]: {fusedHost[i]} vs {expectedFused[i]}");
        }
    }

    [SkippableFact]
    public void OpenCl_AsyncGpu_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_AsyncGpu_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_AsyncGpu_MatchesCpu()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_AsyncGpu_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
