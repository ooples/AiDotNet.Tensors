#if NETCOREAPP
using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class Fp16ReviewRegressionTests
{
    private static CudaBackend RequireAmpere()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || backend.GetDirectPtxRuntime().ComputeCapabilityMajor < 8)
        {
            backend.Dispose();
            Skip.If(true, "Requires CUDA compute capability 8.0+.");
        }
        return backend;
    }

    private static byte[] HalfBytes(int count, float value) =>
        MemoryMarshal.AsBytes(Enumerable.Repeat((Half)value, count).ToArray().AsSpan()).ToArray();

    [SkippableFact]
    public void ReplacingRoute_DoesNotInvalidateAKernelAlreadyReadByADispatch()
    {
        using var backend = RequireAmpere();
        const int m = 128, n = 128, k = 32;
        var config = new PtxTensorCoreGemmConfiguration(128, 128, 16, 64, 64, 2, 8);
        backend.ForceFp16AccumulateRouteForTests(m, n, k, config);
        var routes = (System.Collections.IDictionary)typeof(CudaBackend)
            .GetField("_fp16AccumulateRoutes", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(backend)!;
        object oldRoute = routes.Values.Cast<object>().Single();
        var oldKernel = (PtxTensorCoreGemmKernel)oldRoute.GetType()
            .GetProperty("Kernel", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(oldRoute)!;
        // Precisely reproduce the race ordering: read old route, replace it, then launch the old kernel.
        backend.ForceFp16AccumulateRouteForTests(m, n, k, config);
        using var a = backend.AllocateByteBuffer(m * k * 2);
        using var b = backend.AllocateByteBuffer(k * n * 2);
        using var c = backend.AllocateBuffer(m * n);
        backend.UploadBytes(a, HalfBytes(m * k, 0.5f));
        backend.UploadBytes(b, HalfBytes(k * n, 0.25f));
        oldKernel.Launch(a.Handle, b.Handle, c.Handle, m, n, k, k, n, n, 1f, 0f);
        backend.GetDirectPtxRuntime().Synchronize();
        Assert.All(backend.DownloadBuffer(c), x => Assert.Equal(4f, x));
    }

    [SkippableFact]
    public void CachedBuildFailure_PreservesTheOriginalErrorAcrossOtherFailures()
    {
        using var backend = RequireAmpere();
        backend.ForceMmaRateForTests(60, 120);
        var failures = (System.Collections.Generic.Dictionary<(int, bool, bool), string>)typeof(CudaBackend)
            .GetField("_directPtxFlashAttentionFailed", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(backend)!;
        const string original = "flash-attention-specialization-failed-to-load: original JIT diagnostic";
        failures.Add((64, false, true), original);
        failures.Add((64, false, false), original);
        using var qkv = backend.AllocateByteBuffer(64 * 2);
        using var output = backend.AllocateBuffer(64);
        using var stats = backend.AllocateBuffer(1);
        long before = backend.DirectPtxFlashAttentionDispatchCount;
        foreach (float scale in new[] { 0.125f, 0f, 0.125f })
        {
            Assert.False(backend.TryDirectPtxFlashAttention(qkv, qkv, qkv, output, stats,
                1, 1, 1, 1, 1, 64, scale, false, reducedPrecisionRequested: true));
            Assert.Equal(scale == 0 ? "flash-attention-scale-not-positive-finite" : original, backend.DirectPtxLastError);
        }
        Assert.Equal(before, backend.DirectPtxFlashAttentionDispatchCount);
    }

    [SkippableTheory]
    [InlineData(false, 1)]
    [InlineData(true, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 2)]
    public void CachedRoute_RechecksDeterminismOnEveryDispatch(bool recorded, int batch)
    {
        using var backend = RequireAmpere();
        const int m = 1024, n = 512, k = 1024;
        backend.ForceFp16AccumulateRouteForTests(m, n, k, new(128, 128, 16, 64, 64, 2, 8),
            batch: batch, recorded: recorded);
        using var a = backend.AllocateByteBuffer(m * k * 2);
        using var b = backend.AllocateByteBuffer(batch * k * n * 2);
        using var c = backend.AllocateBuffer(batch * m * n);
        backend.UploadBytes(a, HalfBytes(m * k, 0.5f));
        backend.UploadBytes(b, HalfBytes(batch * k * n, 0.25f));
        try
        {
            // Exercise both transitions using the same cached route, without timing-dependent winners.
            foreach (bool deterministic in new[] { false, true, false, true })
            {
                BlasProvider.SetThreadLocalDeterministicMode(deterministic);
                long before = backend.Fp16AccumulateDispatchCount;
                bool routed = batch == 1
                    ? backend.TryFp16AccumulateGemm(a, b, c, m, n, k)
                    : backend.TryFp16AccumulateConvBatched(a, b, c, m, n, k, batch, false);
                bool expected = recorded || !deterministic;
                Assert.Equal(expected, routed);
                Assert.Equal(before + (expected ? 1 : 0), backend.Fp16AccumulateDispatchCount);
                var verdict = backend.Fp16AccumulateVerdict(m, n, k, batch: batch)!.Value;
                Assert.Equal(expected, verdict.Routed);
                if (!expected) Assert.Contains("deterministic mode", verdict.Reason);
                if (routed) Assert.All(backend.DownloadBuffer(c), x => Assert.Equal(128f, x));
            }
        }
        finally
        {
            BlasProvider.SetThreadLocalDeterministicMode(null);
        }
    }

    [Theory]
    [InlineData(8, 0, "sm_80")]
    [InlineData(8, 6, "sm_86")]
    [InlineData(8, 7, "sm_80")]
    [InlineData(8, 9, "sm_80")]
    [InlineData(9, 0, "sm_80")]
    [InlineData(10, 0, "sm_80")]
    [InlineData(12, 0, "sm_80")]
    public void FlashAttention_EmitsCompatiblePtxTarget(int major, int minor, string target)
    {
        string ptx = PtxFlashAttentionGenerator.Emit(major, minor, 64, false, true, true);
        Assert.Contains(".version 7.1", ptx);
        Assert.Contains(".target " + target, ptx);
        Assert.DoesNotContain("RESCALED", ptx);
    }

    [SkippableFact]
    public void InvalidScales_AreRejectedBeforeKernelDispatch()
    {
        using var backend = RequireAmpere();
        using var kernel = new PtxFlashAttentionKernel(backend.GetDirectPtxRuntime(), 64, false, true, true);
        using var qkv = backend.AllocateByteBuffer(64 * 2);
        using var output = backend.AllocateBuffer(64);
        using var stats = backend.AllocateBuffer(1);
        long before = backend.DirectPtxFlashAttentionDispatchCount;
        foreach (float scale in new[] { 0f, -1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 3e38f, float.MaxValue })
        {
            Assert.False(backend.TryDirectPtxFlashAttention(qkv, qkv, qkv, output, stats,
                1, 1, 1, 1, 1, 64, scale, false, reducedPrecisionRequested: true));
            Assert.Equal("flash-attention-scale-not-positive-finite", backend.DirectPtxLastError);
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => kernel.Launch(qkv.Handle, qkv.Handle,
                qkv.Handle, output.Handle, stats.Handle, 1, 1, 1, 1, 1, scale));
            Assert.Equal("scale", error.ParamName);
        }
        Assert.Equal(before, backend.DirectPtxFlashAttentionDispatchCount);
    }

    [SkippableTheory]
    [InlineData(0.125f)]
    [InlineData(1e38f)]
    public void SupportedScales_KeepZeroScoresUniform(float scale)
    {
        using var backend = RequireAmpere();
        const int sequence = 16, d = 64;
        using var qk = backend.AllocateByteBuffer(sequence * d * 2);
        using var v = backend.AllocateByteBuffer(sequence * d * 2);
        using var output = backend.AllocateBuffer(sequence * d);
        using var stats = backend.AllocateBuffer(sequence);
        backend.UploadBytes(qk, HalfBytes(sequence * d, 0));
        backend.UploadBytes(v, HalfBytes(sequence * d, 1));
        Assert.True(backend.TryDirectPtxFlashAttention(qk, qk, v, output, stats,
            1, 1, 1, sequence, sequence, d, scale, false, reducedPrecisionRequested: true), backend.DirectPtxLastError);
        Assert.All(backend.DownloadBuffer(output), x => Assert.InRange(Math.Abs(x - 1f), 0f, 1e-5f));
        Assert.All(backend.DownloadBuffer(stats), x => Assert.InRange(Math.Abs(x - Math.Log(sequence)), 0, 1e-5));
    }

    [SkippableFact]
    public void OverflowRedo_AlsoProducesCorrectLogSumExp()
    {
        using var backend = RequireAmpere();
        var runtime = backend.GetDirectPtxRuntime();
        using var kernel = new PtxFlashAttentionKernel(runtime, 64, false, true, true);
        const int sequence = 16, d = 64;
        using var qk = backend.AllocateByteBuffer(sequence * d * 2);
        using var v = backend.AllocateByteBuffer(sequence * d * 2);
        using var output = backend.AllocateBuffer(sequence * d);
        using var stats = backend.AllocateBuffer(sequence);
        backend.UploadBytes(qk, HalfBytes(sequence * d, 32));
        backend.UploadBytes(v, HalfBytes(sequence * d, 1));
        // Every raw score is 64 * 32 * 32 = 65536: beyond FP16, but finite in the FP32 redo.
        kernel.Launch(qk.Handle, qk.Handle, v.Handle, output.Handle, stats.Handle,
            1, 1, 1, sequence, sequence, 0.125f);
        runtime.Synchronize();
        Assert.All(backend.DownloadBuffer(output), x => Assert.InRange(Math.Abs(x - 1f), 0f, 1e-5f));
        double expected = 8192 + Math.Log(sequence);
        Assert.All(backend.DownloadBuffer(stats), x => Assert.InRange(Math.Abs(x - expected), 0, 0.002));
    }
}
#endif
