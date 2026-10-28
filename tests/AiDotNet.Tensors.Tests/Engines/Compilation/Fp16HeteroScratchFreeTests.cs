// FP16-in-capture structural peak fix (#633): the hetero Execute-replay backward routes every op's SCRATCH
// (cross-entropy Clamp/Log/Sign/Abs/Divide chain, TensorMultiply-backward temps) through the activation cache,
// where the FP32 specialized path reuses preallocated buffers. MixedPrecisionCompiledPlan now releases each
// node's backward scratch right after that node's backward, protecting the live gradient accumulators
// (AIDOTNET_FP16_NO_SCRATCH_FREE opts out). These tests prove on the GPU engine that (1) the gradients are
// unchanged by the scratch release (correctness) and (2) the post-backward resident activation-cache bytes are
// lower with the release on (the peak win). Runs on any DirectGpuTensorEngine (CUDA/OpenCL/…); skips on CPU-only.

#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection(MixedPrecisionTestCollection.Name)] // serializes MixedPrecisionEmit.TestOverrideEnabled + the env toggles
public class Fp16HeteroScratchFreeTests
{
    private static float[] RandData(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    private static Tensor<float> Rand(int[] shape, int seed) => new(RandData(shape, seed), shape);

    // Build a captured hetero order + loss whose BACKWARD creates per-op scratch: a matmul chain with an
    // element-wise square (TensorMultiply — its backward allocates temporaries) before the reduction.
    private static (ILazyNode[] order, Tensor<float> loss) CaptureHetero(DirectGpuTensorEngine gpu)
    {
        var input = Rand(new[] { 64, 128 }, 1);
        var w1 = Rand(new[] { 128, 128 }, 2);
        var w2 = Rand(new[] { 128, 128 }, 3);

        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        CompiledTrainingPlan<float> plan;
        try
        {
            using (var scope = GraphMode.Enable())
            using (new AutocastScope(PrecisionMode.Float16))
            {
                var h = gpu.TensorMatMul(input, w1);
                var sq = gpu.TensorMultiply(h, h);     // backward scratch generator
                var y = gpu.TensorMatMul(sq, w2);
                gpu.ReduceSum(y, null);
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }

        Assert.True(plan.HasFp16HeteroForward, "graph must capture FP16 hetero nodes.");
        return (plan.Fp16HeteroOrderForTest, plan.LossOutput);
    }

    // Entries die with their storage, so a garbage collection inside the measured window changes the bytes it reads.
    // The window runs in a no-GC region when one can start: on net471 the small gen-0 budget let background allocation
    // from earlier tests trigger a collection in every one of 8 attempts. If the region cannot start, or the window
    // outgrows it, the collected flag still reports it and the caller retries.
    private static bool BeginNoGc()
    {
        try { return GC.TryStartNoGCRegion(32L << 20); }
        catch (InvalidOperationException) { return false; }   // already in a region
    }

    private static void EndNoGc(bool started)
    {
        if (started && System.Runtime.GCSettings.LatencyMode == System.Runtime.GCLatencyMode.NoGCRegion)
            GC.EndNoGCRegion();
    }

    // Run one ISOLATED forward+backward with scratch-free on/off, returning the param-grad L2 norm (correctness
    // fingerprint) and the resident activation-cache bytes right after the backward. Each run captures its OWN
    // fresh graph (distinct tensors) and clears the activation cache first, so the two runs don't share cache
    // state (a shared snapshot would treat the other run's entries as pre-existing and confound the A/B).
    private sealed record ScratchRun(double GradNorm, long ForwardBytes, long PeakBackwardBytes, long AfterBackwardBytes,
        long GradBytes, bool Collected);

    private static ScratchRun RunOnce(DirectGpuTensorEngine gpu, bool scratchFree)
    {
        var prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_FP16_NO_SCRATCH_FREE");
        Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_SCRATCH_FREE", scratchFree ? null : "1");
        try
        {
            var (order, loss) = CaptureHetero(gpu);
            gpu.ClearActivationCache(); // isolate this run's cache accounting from the capture + the other run
            using var plan = MixedPrecisionCompiledPlan.FromCapturedOrder(gpu, order, loss, paging: false);
            // Replicate the real Step condition: eviction suspended for the whole forward+backward (#226), so
            // scratch that nothing frees genuinely accumulates.
            gpu.SuspendActivationEviction();
            // Resident bytes are the activation cache PLUS the results that own their device buffers: since #1566 a
            // backward node's scratch is an owned result, not a cache entry, so the cache alone cannot see it.
            DirectGpuTensorEngine.TrackOwnedResultBytes = true;
            gpu.ResetOwnedResultTracking();
            try
            {
                bool quiet = BeginNoGc();
                int gen0 = GC.CollectionCount(0);
                MixedPrecisionGraphBackward.Result grads;
                long forwardBytes, afterBytes;
                bool collected;
                try
                {
                    plan.Forward();
                    forwardBytes = gpu.CurrentActivationCacheBytes + gpu.LiveOwnedResultBytes;
                    grads = plan.Backward();
                    afterBytes = gpu.CurrentActivationCacheBytes + gpu.LiveOwnedResultBytes;
                    collected = GC.CollectionCount(0) != gen0;
                }
                finally { EndNoGc(quiet); }
                double sum = 0;
                // Every gradient the backward holds, in both maps. The device stores FP16 gradients up-converted to FP32,
                // so each element is four device bytes either way.
                long gradBytes = 0;
                foreach (var kv in grads.Fp16) gradBytes += (long)kv.Value.Length * sizeof(float);
                foreach (var kv in grads.Fp32)
                {
                    var a = kv.Value.ToArray();
                    gradBytes += (long)a.Length * sizeof(float);
                    for (int i = 0; i < a.Length; i++) sum += (double)a[i] * a[i];
                }
                return new ScratchRun(Math.Sqrt(sum), forwardBytes, plan.PeakBackwardResidentBytesForTest, afterBytes,
                    gradBytes, collected);
            }
            finally { gpu.ResumeActivationEviction(); DirectGpuTensorEngine.TrackOwnedResultBytes = false; }
        }
        finally { Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_SCRATCH_FREE", prevEnv); }
    }

    // Run one ISOLATED forward (+ backward for the grad fingerprint) with the forward Half-resident store on/off,
    // returning the param-grad L2 norm and the resident activation-cache bytes right AFTER the forward (before
    // the backward adds grads/scratch) — so the measurement isolates the activation STORAGE dtype.
    private static (double gradNorm, long cacheBytesAfterForward, bool collected) RunForwardStoreOnce(DirectGpuTensorEngine gpu, bool store)
    {
        var prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_FP16_NO_FWD_STORE");
        // store on = default (env unset); store off = opt out via env (engages the FP32 up-cast store).
        Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_FWD_STORE", store ? null : "1");
        try
        {
            var (order, loss) = CaptureHetero(gpu);
            gpu.ClearActivationCache();
            using var plan = MixedPrecisionCompiledPlan.FromCapturedOrder(gpu, order, loss, paging: false);
            gpu.SuspendActivationEviction();
            DirectGpuTensorEngine.TrackOwnedResultBytes = true;
            gpu.ResetOwnedResultTracking();
            try
            {
                bool quiet = BeginNoGc();
                int gen0 = GC.CollectionCount(0);
                long bytes;
                bool collected;
                try
                {
                    plan.Forward();
                    // Resident activation storage right after the forward: the activation cache PLUS the results that
                    // own their device buffers (FP32 results are no longer cache entries, so the cache alone undercounts).
                    bytes = gpu.CurrentActivationCacheBytes + gpu.LiveOwnedResultBytes;
                    collected = GC.CollectionCount(0) != gen0;
                }
                finally { EndNoGc(quiet); }
                var grads = plan.Backward();
                double sum = 0;
                foreach (var kv in grads.Fp32)
                {
                    var a = kv.Value.ToArray();
                    for (int i = 0; i < a.Length; i++) sum += (double)a[i] * a[i];
                }
                return (Math.Sqrt(sum), bytes, collected);
            }
            finally { gpu.ResumeActivationEviction(); DirectGpuTensorEngine.TrackOwnedResultBytes = false; }
        }
        finally { Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_FWD_STORE", prevEnv); }
    }

    [SkippableFact]
    public void ForwardHalfStore_PreservesGradients_AndLowersResidentActivationBytes()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/…).");

        // The Half-resident store needs a half-output GEMM. Skip on the CAPABILITY, not on "the bytes came out
        // equal": inferring it from the measurement let a store that silently stopped engaging pass as a skip.
        Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend { SupportsHgemm: true },
            "backend has no half-output GEMM, so the forward Half-store cannot engage.");
        var off = RunForwardStoreOnce(gpu, store: false);
        var on = RunForwardStoreOnce(gpu, store: true);

        // (1) Correctness: keeping activations Half-resident must not change the gradients (the captured graph
        // already intends Half activations; the fused Half backward reads them directly).
        Assert.True(Math.Abs(on.gradNorm - off.gradNorm) <= 1e-3 * (1 + Math.Abs(off.gradNorm)),
            $"forward Half-store changed the gradient norm: on={on.gradNorm} off={off.gradNorm}.");

        // (2) The win: the Half-resident matmul activations occupy 2 bytes/elem vs 4 (FP32 up-cast), so the
        // resident activation-cache bytes after the forward are strictly lower with the store on.
        // As in the scratch-free test below: entries die with their storage, so a collection during either forward frees
        // entries by itself (net471: on=65540 off=32768 in about one run in six). Compared on the first attempt neither arm
        // collected in; the test fails if no attempt is clean, so it never passes without making the comparison.
        const int Attempts = 8;
        for (int attempt = 0; ; attempt++)
        {
            if (!on.collected && !off.collected)
            {
                Assert.True(on.cacheBytesAfterForward < off.cacheBytesAfterForward,
                    $"forward Half-store should lower resident activation bytes: on={on.cacheBytesAfterForward} off={off.cacheBytesAfterForward}.");
                return;
            }
            Assert.True(attempt + 1 < Attempts,
                $"a garbage collection ran during every one of {Attempts} attempts, so the resident-bytes comparison was never made.");
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            off = RunForwardStoreOnce(gpu, store: false);
            on = RunForwardStoreOnce(gpu, store: true);
        }
    }

    [SkippableFact]
    public void ScratchFree_PreservesGradients_AndAccumulatesNoBackwardScratch()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/…).");

        var off = RunOnce(gpu, scratchFree: false);
        var on = RunOnce(gpu, scratchFree: true);

        // (1) Correctness: the scratch release must not change the gradients. Over-eviction would degrade to a
        // re-upload (still correct); a wrong protect set that freed a live grad would corrupt this norm.
        Assert.True(Math.Abs(on.GradNorm - off.GradNorm) <= 1e-3 * (1 + Math.Abs(off.GradNorm)),
            $"scratch-free changed the gradient norm: on={on.GradNorm} off={off.GradNorm}.");

        // (2) No backward scratch accumulates: after every backward node, the resident bytes (activation cache plus
        // owned results) are at most the forward's resident bytes plus the gradients the backward holds. Before #1566
        // the per-op scratch were cache entries that only the scratch-free release freed, so this held with it on and
        // failed with it off. Since #1566 the scratch are owned results freed with their tensors, so both arms must
        // hold it (#1570). A garbage collection inside the window frees dead buffers on its own and could hide a
        // build-up, so the check is made on an attempt neither arm collected in, and the test fails if none is clean.
        const int Attempts = 8;
        for (int attempt = 0; ; attempt++)
        {
            if (!on.Collected && !off.Collected)
            {
                foreach (var (arm, run) in new[] { ("on", on), ("off", off) })
                {
                    long bound = run.ForwardBytes + run.GradBytes;
                    Assert.True(run.PeakBackwardBytes > 0, $"scratch-free {arm}: the backward peak was not sampled.");
                    Assert.True(run.PeakBackwardBytes <= bound,
                        $"scratch-free {arm}: backward scratch accumulated. Peak resident {run.PeakBackwardBytes} bytes > " +
                        $"forward {run.ForwardBytes} + gradients {run.GradBytes}.");
                    Assert.True(run.AfterBackwardBytes <= bound,
                        $"scratch-free {arm}: backward scratch outlived the backward. Resident {run.AfterBackwardBytes} " +
                        $"bytes > forward {run.ForwardBytes} + gradients {run.GradBytes}.");
                }
                return;
            }
            Assert.True(attempt + 1 < Attempts,
                $"a garbage collection ran during every one of {Attempts} attempts, so the resident-bytes check was never made.");
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            off = RunOnce(gpu, scratchFree: false);
            on = RunOnce(gpu, scratchFree: true);
        }
    }
}
