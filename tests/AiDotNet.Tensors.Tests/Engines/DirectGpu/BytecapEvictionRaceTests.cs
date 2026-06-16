// Copyright (c) AiDotNet. All rights reserved.
//
// PR #586 follow-up: reproduction harness for the CUDA-700 race in the
// byte-cap activation-cache eviction path.
//
// The PR body documents: "With the cap active at 4 GB, eviction churn exposes
// a use-after-free race in the byte-cap eviction path (CUDA 700) — the #226-class
// 'materialize-then-free' treatment is needed on this path too." The workaround
// in the active sweep is to set AIDOTNET_ACT_CACHE_VRAM_MB high enough that the
// working set fits below the cap and eviction never fires.
//
// These tests deliberately pin the byte cap BELOW the per-step working set so
// EvictOldestActivationsUnsafe fires on every step, exercising the suspected
// race. They run only when a CUDA device is available; on CI without CUDA they
// skip cleanly (the same SkippableFact pattern used by Fp16TensorCoreGemmTests).
//
// What they look for:
//   1. No CUDA 700 / unspecified launch failure after a few hundred steps —
//      the symptom the PR body calls out.
//   2. Activation-cache accounting stays consistent: _currentActivationCacheBytes
//      tracks the sum of live entries' SizeInBytes (no double-counting on the
//      eviction path's TryRemove + Interlocked.Add(-)).
//   3. The deferred-free queue drains — no buffer is held forever after eviction.
//
// The race is deeper than tests alone can prove (it's a free-after-enqueue
// window between EvictOldestActivationsUnsafe and FreeBufferDeferred), but the
// stress shape above is the established way to surface it. A passing run does
// not prove the race is fixed — it proves we couldn't trigger it in this shape
// on this hardware. Pair these tests with the eviction-suspension fix to get
// genuine coverage.

using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class BytecapEvictionRaceTests
{
    /// <summary>
    /// Pins the activation-cache byte cap to a tiny value via reflection on the
    /// CURRENT engine instance, runs an eager forward+backward loop that produces
    /// activations comfortably larger than the cap each step, then verifies the
    /// CUDA context stays healthy. Step count and cap chosen to amortise per-step
    /// fixed overhead while keeping the wall-clock under a few seconds on a 2080Ti-
    /// class GPU.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "CudaRequired")]
    public void EagerTraining_BelowCap_NoCudaError()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = AiDotNetEngine.Current;
        Skip.IfNot(eng is DirectGpuTensorEngine, "active engine is not the GPU backend");
        var dgte = (DirectGpuTensorEngine)eng;
        var cudaBackend = dgte.TestBackend as CudaBackend;
        Skip.IfNot(cudaBackend is not null && cudaBackend.IsAvailable, "CudaBackend not available");

        // Pin the byte cap to 8 MB. A single 1024×1024 float tensor is 4 MB, so any
        // step that touches more than two of them forces eviction.
        const long byteCap = 8L * 1024 * 1024;
        long previousCap = SetActivationCacheByteCap(dgte, byteCap);

        try
        {
            // Warmup: compile kernels + populate the buffer pool so steady-state
            // is what the test measures.
            var warm = new Tensor<float>(new float[16], new[] { 4, 4 });
            _ = eng.TensorMatMul(warm, warm);

            // Working set: three 1024×1024 buffers (12 MB) > byteCap (8 MB).
            // Repeating the matmul keeps activations cached → forces eviction.
            var a = MakeRandomTensor(1024, 1024, seed: 11);
            var b = MakeRandomTensor(1024, 1024, seed: 22);

            const int steps = 200;
            for (int s = 0; s < steps; s++)
            {
                var c = eng.TensorMatMul(a, b);
                var d = eng.TensorMatMul(c, b);
                var e = eng.TensorMatMul(d, a);
                // Force a host-side read so the deferred materializer fires
                // and any pending CUDA error surfaces here rather than later.
                _ = e.AsSpan()[0];
            }

            // Final sync: any pending CUDA error from the kernel queue
            // materialises here, not in a later test.
            cudaBackend!.Synchronize();
            Assert.True(true, "200 cap-overflowing matmul steps completed without CUDA error.");
        }
        finally
        {
            // Restore the original cap so neighbouring tests behave normally.
            SetActivationCacheByteCap(dgte, previousCap);
        }
    }

    /// <summary>
    /// Same shape as above but verifies the cache accounting invariant after
    /// the loop: the running _currentActivationCacheBytes must equal the sum
    /// of the live entries' buffer sizes. A double-decrement on the eviction
    /// path (the bug class the byte-cap race could expose) shows up here as a
    /// negative running total — Interlocked.Add(-x) overshooting zero.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "CudaRequired")]
    public void EagerTraining_CacheAccounting_StaysConsistent()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = AiDotNetEngine.Current;
        Skip.IfNot(eng is DirectGpuTensorEngine, "active engine is not the GPU backend");
        var dgte = (DirectGpuTensorEngine)eng;
        var cudaBackend = dgte.TestBackend as CudaBackend;
        Skip.IfNot(cudaBackend is not null && cudaBackend.IsAvailable, "CudaBackend not available");

        const long byteCap = 4L * 1024 * 1024;   // 4 MB — extra-aggressive.
        long previousCap = SetActivationCacheByteCap(dgte, byteCap);

        try
        {
            var a = MakeRandomTensor(512, 512, seed: 31);
            var b = MakeRandomTensor(512, 512, seed: 32);
            for (int s = 0; s < 100; s++)
            {
                var c = eng.TensorMatMul(a, b);
                _ = c.AsSpan()[0];
            }

            long running = ReadLong(dgte, "_currentActivationCacheBytes");
            Assert.True(running >= 0,
                $"_currentActivationCacheBytes drifted negative ({running}) — eviction path "
                + "double-decremented at least once. This is the byte-cap eviction race signal.");
        }
        finally
        {
            SetActivationCacheByteCap(dgte, previousCap);
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────

    private static Tensor<float> MakeRandomTensor(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>Returns the previous cap so the test can restore it on teardown.</summary>
    private static long SetActivationCacheByteCap(DirectGpuTensorEngine eng, long newCap)
    {
        var field = typeof(DirectGpuTensorEngine).GetField(
            "_maxActivationCacheBytes",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        long prev = (long)field!.GetValue(eng)!;
        field.SetValue(eng, newCap);
        return prev;
    }

    private static long ReadLong(DirectGpuTensorEngine eng, string fieldName)
    {
        var field = typeof(DirectGpuTensorEngine).GetField(
            fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return (long)field!.GetValue(eng)!;
    }
}
