using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1027: the Vulkan backend is a process-wide singleton, and the test collections that use it run in
/// parallel, so two threads dispatch on it at once. Each thread here checks its own results against a
/// CPU oracle while the other runs; any cross-thread interference shows up as a mismatch.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class VulkanConcurrentDispatchTests
{
    private const int M = 8, K = 128, N = 64;
    private const int Iterations = 60;

    private static bool Ready
    {
        get
        {
            try { return VulkanBackend.Instance.Initialize() && VulkanBackend.Instance.IsGlslCompilerAvailable; }
            catch (InvalidOperationException) { return false; }
            catch (DllNotFoundException) { return false; }
        }
    }

    [SkippableFact]
    public void DequantGemm_And_Gemm_OnTwoThreads_EachMatchTheirOracle()
    {
        Skip.IfNot(Ready, $"Vulkan with a GLSL compiler is not available: {VulkanBackend.Instance.InitializationFailure}");
        var backend = VulkanBackend.Instance;
        using var start = new Barrier(2);
        int quantRuns = 0, gemmRuns = 0;
        string? failure = null;

        void Fail(string message) => Interlocked.CompareExchange(ref failure, message, null);

        var quant = Task.Run(() =>
        {
            var rng = RandomHelper.CreateSeededRandom(1027);
            var act = new float[M * K];
            var w = new int[K * N];
            for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < w.Length; i++) w[i] = rng.Next(-128, 128);
            var scales = new[] { 0.03f };
            var expected = new float[M * N];
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    float acc = 0f;
                    for (int k = 0; k < K; k++) acc += act[i * K + k] * w[k * N + j];
                    expected[i * N + j] = acc * scales[0];
                }
            start.SignalAndWait();
            for (int run = 0; run < Iterations && Volatile.Read(ref failure) is null; run++)
            {
                IGpuBuffer? a = null, s = null, wb = null, o = null;
                try
                {
                    a = backend.AllocateBuffer(act); s = backend.AllocateBuffer(scales); wb = backend.AllocateIntBuffer(w);
                    o = backend.DequantGemmInt(a, wb, s, M, K, N, K * N, 1);
                    var actual = backend.DownloadBuffer(o);
                    for (int idx = 0; idx < expected.Length; idx++)
                    {
                        float tol = 1e-2f + 1e-3f * Math.Abs(expected[idx]);
                        if (Math.Abs(expected[idx] - actual[idx]) > tol)
                        {
                            Fail($"DequantGemmInt run {run}: [{idx}] expected {expected[idx]}, got {actual[idx]}");
                            return;
                        }
                    }
                    quantRuns++;
                }
                catch (Exception ex)
                {
                    Fail($"DequantGemmInt run {run} threw {ex}");
                    return;
                }
                finally { a?.Dispose(); s?.Dispose(); wb?.Dispose(); o?.Dispose(); }
            }
        });

        var gemm = Task.Run(() =>
        {
            const int gm = 64, gn = 96, gk = 80;
            var rng = RandomHelper.CreateSeededRandom(1028);
            var ga = new float[gm * gk];
            var gb = new float[gk * gn];
            for (int i = 0; i < ga.Length; i++) ga[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < gb.Length; i++) gb[i] = (float)(rng.NextDouble() - 0.5);
            var expected = new float[gm * gn];
            for (int i = 0; i < gm; i++)
                for (int p = 0; p < gk; p++)
                    for (int j = 0; j < gn; j++) expected[i * gn + j] += ga[i * gk + p] * gb[p * gn + j];
            start.SignalAndWait();
            for (int run = 0; run < Iterations && Volatile.Read(ref failure) is null; run++)
            {
                IGpuBuffer? a = null, b = null, c = null;
                try
                {
                    a = backend.AllocateBuffer(ga); b = backend.AllocateBuffer(gb); c = backend.AllocateBuffer(gm * gn);
                    backend.Gemm(a, b, c, gm, gn, gk);
                    var actual = backend.DownloadBuffer(c);
                    for (int idx = 0; idx < expected.Length; idx++)
                    {
                        if (Math.Abs(expected[idx] - actual[idx]) > 1e-3f + 1e-3f * Math.Abs(expected[idx]))
                        {
                            Fail($"Gemm run {run}: [{idx}] expected {expected[idx]}, got {actual[idx]}");
                            return;
                        }
                    }
                    gemmRuns++;
                }
                catch (Exception ex)
                {
                    Fail($"Gemm run {run} threw {ex}");
                    return;
                }
                finally { a?.Dispose(); b?.Dispose(); c?.Dispose(); }
            }
        });

        Task.WaitAll(quant, gemm);
        Assert.True(failure is null, failure);
        Assert.Equal(Iterations, quantRuns);
        Assert.Equal(Iterations, gemmRuns);
    }
}