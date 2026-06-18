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

    // Run one ISOLATED forward+backward with scratch-free on/off, returning the param-grad L2 norm (correctness
    // fingerprint) and the resident activation-cache bytes right after the backward. Each run captures its OWN
    // fresh graph (distinct tensors) and clears the activation cache first, so the two runs don't share cache
    // state (a shared snapshot would treat the other run's entries as pre-existing and confound the A/B).
    private static (double gradNorm, long cacheBytesAfter) RunOnce(DirectGpuTensorEngine gpu, bool scratchFree)
    {
        var prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_FP16_NO_SCRATCH_FREE");
        Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_SCRATCH_FREE", scratchFree ? null : "1");
        try
        {
            var (order, loss) = CaptureHetero(gpu);
            gpu.ClearActivationCache(); // isolate this run's cache accounting from the capture + the other run
            var plan = MixedPrecisionCompiledPlan.FromCapturedOrder(gpu, order, loss, paging: false);
            // Replicate the real Step condition: eviction suspended for the whole forward+backward (#226), so
            // without the scratch-free the per-op backward scratch genuinely accumulates in the cache.
            gpu.SuspendActivationEviction();
            try
            {
                plan.Forward();
                var grads = plan.Backward();
                double sum = 0;
                foreach (var kv in grads.Fp32)
                {
                    var a = kv.Value.ToArray();
                    for (int i = 0; i < a.Length; i++) sum += (double)a[i] * a[i];
                }
                long bytes = gpu.CurrentActivationCacheBytes;
                return (Math.Sqrt(sum), bytes);
            }
            finally { gpu.ResumeActivationEviction(); }
        }
        finally { Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_SCRATCH_FREE", prevEnv); }
    }

    // Run one ISOLATED forward (+ backward for the grad fingerprint) with the forward Half-resident store on/off,
    // returning the param-grad L2 norm and the resident activation-cache bytes right AFTER the forward (before
    // the backward adds grads/scratch) — so the measurement isolates the activation STORAGE dtype.
    private static (double gradNorm, long cacheBytesAfterForward) RunForwardStoreOnce(DirectGpuTensorEngine gpu, bool store)
    {
        var prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_FP16_NO_FWD_STORE");
        // store on = default (env unset); store off = opt out via env (engages the FP32 up-cast store).
        Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_FWD_STORE", store ? null : "1");
        try
        {
            var (order, loss) = CaptureHetero(gpu);
            gpu.ClearActivationCache();
            var plan = MixedPrecisionCompiledPlan.FromCapturedOrder(gpu, order, loss, paging: false);
            gpu.SuspendActivationEviction();
            try
            {
                plan.Forward();
                long bytes = gpu.CurrentActivationCacheBytes;  // resident activation storage right after forward
                var grads = plan.Backward();
                double sum = 0;
                foreach (var kv in grads.Fp32)
                {
                    var a = kv.Value.ToArray();
                    for (int i = 0; i < a.Length; i++) sum += (double)a[i] * a[i];
                }
                return (Math.Sqrt(sum), bytes);
            }
            finally { gpu.ResumeActivationEviction(); }
        }
        finally { Environment.SetEnvironmentVariable("AIDOTNET_FP16_NO_FWD_STORE", prevEnv); }
    }

    [SkippableFact]
    public void ForwardHalfStore_PreservesGradients_AndLowersResidentActivationBytes()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/…).");

        var off = RunForwardStoreOnce(gpu, store: false);
        var on = RunForwardStoreOnce(gpu, store: true);

        // The Half-resident store only engages on a backend with a half-output GEMM; if it didn't (bytes equal),
        // this backend lacks SupportsHgemm — skip rather than fail (the store no-ops to the FP32 path there).
        Skip.If(on.cacheBytesAfterForward == off.cacheBytesAfterForward,
            "forward Half-store did not engage (backend has no half-output GEMM) — nothing to measure.");

        // (1) Correctness: keeping activations Half-resident must not change the gradients (the captured graph
        // already intends Half activations; the fused Half backward reads them directly).
        Assert.True(Math.Abs(on.gradNorm - off.gradNorm) <= 1e-3 * (1 + Math.Abs(off.gradNorm)),
            $"forward Half-store changed the gradient norm: on={on.gradNorm} off={off.gradNorm}.");

        // (2) The win: the Half-resident matmul activations occupy 2 bytes/elem vs 4 (FP32 up-cast), so the
        // resident activation-cache bytes after the forward are strictly lower with the store on.
        Assert.True(on.cacheBytesAfterForward < off.cacheBytesAfterForward,
            $"forward Half-store should lower resident activation bytes: on={on.cacheBytesAfterForward} off={off.cacheBytesAfterForward}.");
    }

    [SkippableFact]
    public void ScratchFree_PreservesGradients_AndLowersResidentCacheBytes()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/…).");

        var off = RunOnce(gpu, scratchFree: false);
        var on = RunOnce(gpu, scratchFree: true);

        // (1) Correctness: the scratch release must not change the gradients. Over-eviction would degrade to a
        // re-upload (still correct); a wrong protect set that freed a live grad would corrupt this norm.
        Assert.True(Math.Abs(on.gradNorm - off.gradNorm) <= 1e-3 * (1 + Math.Abs(off.gradNorm)),
            $"scratch-free changed the gradient norm: on={on.gradNorm} off={off.gradNorm}.");

        // (2) Peak win: with the release on, the per-op backward scratch is gone after each node, so the
        // resident activation-cache bytes after the backward are strictly lower (the held grads remain).
        Assert.True(on.cacheBytesAfter < off.cacheBytesAfter,
            $"scratch-free should lower resident cache bytes: on={on.cacheBytesAfter} off={off.cacheBytesAfter}.");
    }
}
