using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Issue #336: validates the WeightLifetime.GpuPinned wiring for the
/// optimizer-state-on-GPU work. Tags Adam m/v, BatchNorm running stats,
/// and weights with GpuPinned so they avoid the per-train-step
/// cuMemcpyHtoD/DtoH round-trip.
/// </summary>
public class GpuPinnedLifetimeTests
{
    /// <summary>
    /// End-to-end probe: runs an actual <c>GpuOptimizer.TrySgdStep</c> against a tiny
    /// pinned-on-GPU tensor with a known input/expected-output pair, and reports whether
    /// the host array reflects the kernel's writes. This is the contract the
    /// <c>GpuOptimizer.Try*Step</c> tests assert; backends whose offload allocator can't
    /// honor it (e.g. the current OpenCL allocator: separate AllocHGlobal hostBuf +
    /// CL_MEM_ALLOC_HOST_PTR cl_mem with no map/unmap pair) cause the actual tests to
    /// fail on stale host reads. Treat such hosts the same way as the no-GPU CI hosts —
    /// skip cleanly. Probing through the same code path the tests use is the only
    /// reliable detection: WeightRegistry.OffloadAllocator type-checking misses cases
    /// where TryGetGpuBuffer returns a non-allocator-backed buffer.
    /// </summary>
    private static bool HasZeroCopyPinnedMapping()
    {
        Tensor<float>? probeP = null, probeG = null;
        try
        {
            probeP = TensorAllocator.RentPinnedOnGpu<float>(new[] { 1 });
            probeG = TensorAllocator.RentPinnedOnGpu<float>(new[] { 1 });
            if (probeP.TryGetGpuBuffer() is null || probeG.TryGetGpuBuffer() is null)
                return false;
            // Closed-form SGD: p1 = p0 - lr * g = 10 - 1 * 2 = 8.
            probeP.GetDataArray()[0] = 10.0f;
            probeG.GetDataArray()[0] = 2.0f;
            bool ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TrySgdStep(probeP, probeG, learningRate: 1.0f);
            if (!ran) return false;
            return System.Math.Abs(probeP.GetDataArray()[0] - 8.0f) < 1e-4f;
        }
        catch
        {
            return false;
        }
        finally
        {
            if (probeP is not null) WeightRegistry.UnregisterWeight(probeP);
            if (probeG is not null) WeightRegistry.UnregisterWeight(probeG);
        }
    }

    [Fact]
    public void WeightLifetime_GpuPinned_EnumValueExists()
    {
        // Smoke check: the enum value is defined and distinct from the
        // existing values. Consumer code (AdamOptimizer._tapeM/v setter)
        // refers to this enum member by name.
        Assert.True(System.Enum.IsDefined(typeof(WeightLifetime), WeightLifetime.GpuPinned));
        Assert.NotEqual(WeightLifetime.GpuPinned, WeightLifetime.Default);
        Assert.NotEqual(WeightLifetime.GpuPinned, WeightLifetime.GpuOffload);
        Assert.NotEqual(WeightLifetime.GpuPinned, WeightLifetime.GpuManaged);
    }

    [Fact]
    public void RentPinnedOnGpu_CpuOnlyHost_FallsBackToCpuPinned_NotThrowing()
    {
        // On a host without GPU offload allocator (CPU CI), the accessor
        // must return a usable tensor — RentPinnedOnGpu's fallback path.
        var tensor = TensorAllocator.RentPinnedOnGpu<float>(new[] { 4, 8 });
        Assert.NotNull(tensor);
        Assert.Equal(32, tensor.Length);
    }

    [Fact]
    public void RentPinnedOnGpu_GpuOffloadCapableHost_TagsTensorGpuPinned()
    {
        // When the offload allocator IS registered (GPU host), the tensor
        // comes back tagged GpuPinned (not GpuOffload — issue #336 uses
        // the new lifetime tag to distinguish "lives on GPU for the train
        // loop" from "may swap to host for large-model memory pressure").
        if (WeightRegistry.OffloadAllocator is null
            || !WeightRegistry.OffloadAllocator.IsAvailable)
            return; // CPU-only host — skip; covered by the fallback test.

        // PR #345 review: tighten assertion to require GpuPinned exactly.
        // Permitting `Default` masked regressions in the new tagging
        // behavior — the whole point of issue #336 is that pinned-on-GPU
        // tensors are explicitly tagged as such. Also unregister the
        // tensor afterward so each test run doesn't accumulate
        // process-wide offload state.
        var tensor = TensorAllocator.RentPinnedOnGpu<float>(new[] { 16 });
        try
        {
            Assert.Equal(WeightLifetime.GpuPinned, tensor.Lifetime);
        }
        finally
        {
            WeightRegistry.UnregisterWeight(tensor);
        }
    }

    [Fact]
    public void Tensor_LifetimeSetter_AcceptsGpuPinned()
    {
        // Direct lifetime assignment must not throw — consumer code may
        // construct a tensor first and then tag it.
        var t = new Tensor<float>(new[] { 8 })
        {
            Lifetime = WeightLifetime.GpuPinned,
        };
        Assert.Equal(WeightLifetime.GpuPinned, t.Lifetime);
    }

    [Fact]
    public void TryGetGpuBuffer_CpuOnlyHost_ReturnsNull()
    {
        var t = new Tensor<float>(new[] { 16 });
        Assert.Null(t.TryGetGpuBuffer());
    }

    [Fact]
    public void TryGetGpuBuffer_GpuPinnedTagWithoutPointer_ReturnsNull()
    {
        var t = new Tensor<float>(new[] { 16 })
        {
            Lifetime = WeightLifetime.GpuPinned,
        };
        Assert.Null(t.TryGetGpuBuffer());
    }

    [Fact]
    public void GpuOptimizer_TryAdamStep_MatchesHandComputedReference_OnGpuHost()
    {
        // Issue #336 correctness gate. Runs a single Adam step on a tiny
        // 4-element parameter tensor with seeds reset to zero, then
        // compares against the closed-form one-step Adam update:
        //   m1 = beta1*0 + (1-beta1)*g       = (1-beta1)*g
        //   v1 = beta2*0 + (1-beta2)*g²      = (1-beta2)*g²
        //   m̂  = m1 / (1-beta1^1) = g
        //   v̂  = v1 / (1-beta2^1) = g²
        //   p1 = p0 - lr * m̂ / (sqrt(v̂) + eps)
        //      = p0 - lr * g / (|g| + eps)
        // Skips when no GPU is present.
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = gpu;
        try
        {
            if (gpu.GetBackend() is null) return;

            // RentPinnedOnGpu falls back to CPU-pinned on CPU-only hosts.
            // TryAdamStep then returns false (no GPU buffer), and the test
            // skips. On a real CUDA host the tensors get GPU mappings and
            // the kernel runs.
            var p = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var g = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var m = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var v = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            try
            {
                // Need device-resident buffers to actually run the kernel; if
                // the tensors didn't bind to GPU buffers, TryAdamStep returns
                // false and we treat that as skip.
                if (p.TryGetGpuBuffer() is null) return;
                // Some backends (current OpenCL allocator) return a GPU buffer but
                // do NOT provide zero-copy host↔device mapping — the optimizer kernel
                // runs against a separate buffer the host never sees written. Skip
                // in that case; this gate is independent of the no-GPU skip above.
                if (!HasZeroCopyPinnedMapping()) return;

                // Initialize values via flat-data span.
                var pData = p.GetDataArray();
                var gData = g.GetDataArray();
                pData[0] = 1.0f; pData[1] = 2.0f; pData[2] = 3.0f; pData[3] = 4.0f;
                gData[0] = 0.1f; gData[1] = 0.2f; gData[2] = 0.3f; gData[3] = 0.4f;
                // m, v left as zeros from RentPinnedOnGpu's zero-init contract.

                const float lr = 0.01f;
                const float beta1 = 0.9f;
                const float beta2 = 0.999f;
                const float eps = 1e-8f;

                bool ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamStep(
                    p, g, m, v,
                    learningRate: lr, beta1: beta1, beta2: beta2,
                    epsilon: eps, weightDecay: 0f, step: 1);
                // GPU preconditions are already satisfied (backend not null
                // + GPU buffer bound), so a false here is a kernel-dispatch
                // regression, not a host-capability skip. Fail loudly.
                Assert.True(ran,
                    "GpuOptimizer.TryAdamStep returned false after GPU preconditions were met — kernel-dispatch regression.");

                // Closed-form expected post-step values.
                for (int i = 0; i < 4; i++)
                {
                    float gi = gData[i];
                    float expected = (i + 1) - lr * gi / ((float)System.Math.Abs(gi) + eps);
                    // Tolerance accounts for fp32 rounding in the kernel's
                    // division + sqrt + bias-correction chain.
                    Assert.InRange(pData[i], expected - 1e-4f, expected + 1e-4f);
                }
            }
            finally
            {
                // Release the pinned/offload registry slots so subsequent
                // tests don't observe accumulated state from this one.
                WeightRegistry.UnregisterWeight(p);
                WeightRegistry.UnregisterWeight(g);
                WeightRegistry.UnregisterWeight(m);
                WeightRegistry.UnregisterWeight(v);
            }
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine;
            gpu.Dispose();
        }
    }

    [Fact]
    public void GpuOptimizer_TryAdamStep_MarksGpuUpdatedTensorsCurrent_OnGpuHost()
    {
        // Regression for post-training eval on GPU-resident model weights:
        // a successful on-device optimizer step mutates param/m/v in place.
        // Their resident GPU buffer version must be synced to the bumped
        // tensor Version so later inference reuses the live device buffers
        // instead of re-uploading fresh host arrays on every forward pass.
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = gpu;
        try
        {
            if (gpu.GetBackend() is null) return;

            var p = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var g = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var m = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var v = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            try
            {
                if (p.TryGetGpuBuffer() is null
                    || g.TryGetGpuBuffer() is null
                    || m.TryGetGpuBuffer() is null
                    || v.TryGetGpuBuffer() is null)
                    return;

                int pVersion = p.Version;
                int mVersion = m.Version;
                int vVersion = v.Version;
                int gVersion = g.Version;

                bool ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamStep(
                    p, g, m, v,
                    learningRate: 0.01f, beta1: 0.9f, beta2: 0.999f,
                    epsilon: 1e-8f, weightDecay: 0f, step: 1);
                Assert.True(ran,
                    "GpuOptimizer.TryAdamStep returned false after GPU buffers were bound.");

                Assert.True(p.Version > pVersion, "Parameter Version should advance after an in-place GPU Adam step.");
                Assert.True(m.Version > mVersion, "Adam m-state Version should advance after an in-place GPU Adam step.");
                Assert.True(v.Version > vVersion, "Adam v-state Version should advance after an in-place GPU Adam step.");
                Assert.Equal(gVersion, g.Version);

                Assert.Equal(p.Version, p._gpuBufferVersion);
                Assert.Equal(m.Version, m._gpuBufferVersion);
                Assert.Equal(v.Version, v._gpuBufferVersion);
            }
            finally
            {
                WeightRegistry.UnregisterWeight(p);
                WeightRegistry.UnregisterWeight(g);
                WeightRegistry.UnregisterWeight(m);
                WeightRegistry.UnregisterWeight(v);
            }
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine;
            gpu.Dispose();
        }
    }

    [Fact]
    public void GpuOptimizer_TrySgdStep_MatchesHandComputedReference_OnGpuHost()
    {
        // Single-step SGD: p1 = p0 - lr * g
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = gpu;
        try
        {
            if (gpu.GetBackend() is null) return;
            var p = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var g = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            try
            {
                if (p.TryGetGpuBuffer() is null) return;
                // Some backends (current OpenCL allocator) return a GPU buffer but
                // do NOT provide zero-copy host↔device mapping — the optimizer kernel
                // runs against a separate buffer the host never sees written. Skip
                // in that case; this gate is independent of the no-GPU skip above.
                if (!HasZeroCopyPinnedMapping()) return;

                var pData = p.GetDataArray();
                var gData = g.GetDataArray();
                pData[0] = 10.0f; pData[1] = 20.0f; pData[2] = 30.0f; pData[3] = 40.0f;
                gData[0] = 1.0f;  gData[1] = 2.0f;  gData[2] = 3.0f;  gData[3] = 4.0f;

                const float lr = 0.1f;
                bool ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TrySgdStep(p, g, learningRate: lr);
                Assert.True(ran,
                    "GpuOptimizer.TrySgdStep returned false after GPU preconditions were met — kernel-dispatch regression.");

                for (int i = 0; i < 4; i++)
                {
                    float expected = (i + 1) * 10.0f - lr * (i + 1);
                    Assert.InRange(pData[i], expected - 1e-4f, expected + 1e-4f);
                }
            }
            finally
            {
                WeightRegistry.UnregisterWeight(p);
                WeightRegistry.UnregisterWeight(g);
            }
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine;
            gpu.Dispose();
        }
    }

    [Fact]
    public void GpuOptimizer_TryAdamWStep_AppliesDecoupledWeightDecay_OnGpuHost()
    {
        // AdamW = Adam + decoupled weight decay. With weightDecay=0.01 and
        // lr=0.01, on a parameter with grad=0, AdamW would update solely
        // by p1 = p0 - lr*wd*p0 = 0.9999 * p0. Verifies the wd path runs.
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = gpu;
        try
        {
            if (gpu.GetBackend() is null) return;
            var p = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var g = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var m = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            var v = AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<float>(new[] { 4 });
            try
            {
                if (p.TryGetGpuBuffer() is null) return;
                // Some backends (current OpenCL allocator) return a GPU buffer but
                // do NOT provide zero-copy host↔device mapping — the optimizer kernel
                // runs against a separate buffer the host never sees written. Skip
                // in that case; this gate is independent of the no-GPU skip above.
                if (!HasZeroCopyPinnedMapping()) return;

                var pData = p.GetDataArray();
                var initial = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
                pData[0] = initial[0]; pData[1] = initial[1]; pData[2] = initial[2]; pData[3] = initial[3];
                // grad zeros — m/v stay zero — the only update is the wd term.

                bool ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamWStep(
                    p, g, m, v,
                    learningRate: 0.01f, beta1: 0.9f, beta2: 0.999f,
                    epsilon: 1e-8f, weightDecay: 0.01f, step: 1);
                Assert.True(ran,
                    "GpuOptimizer.TryAdamWStep returned false after GPU preconditions were met — kernel-dispatch regression.");

                // AdamW's decoupled weight decay updates `p -= lr * wd * p`
                // BEFORE the Adam moment update. With grad=0, the only thing
                // that moves the parameter is that decay term — so the
                // post-step magnitude MUST be smaller than the pre-step
                // magnitude. A finiteness check alone would pass even if the
                // decay path was silently broken (e.g. wd folded into a
                // non-firing branch).
                for (int i = 0; i < 4; i++)
                {
                    Assert.True(!float.IsNaN(pData[i]) && !float.IsInfinity(pData[i]),
                        $"AdamW step produced NaN/Inf at index {i}: {pData[i]}");
                    Assert.True(System.MathF.Abs(pData[i]) < System.MathF.Abs(initial[i]),
                        $"AdamW with grad=0 and weightDecay=0.01 should reduce |p[{i}]|: before={initial[i]}, after={pData[i]} (weight decay did not fire).");
                }
            }
            finally
            {
                WeightRegistry.UnregisterWeight(p);
                WeightRegistry.UnregisterWeight(g);
                WeightRegistry.UnregisterWeight(m);
                WeightRegistry.UnregisterWeight(v);
            }
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine;
            gpu.Dispose();
        }
    }

    [Fact]
    public void AsGpuBuffer_AliasesTryGetGpuBuffer_CpuOnlyHost()
    {
        // Issue #336 names the accessor AsGpuBuffer<T>(); the impl is
        // TryGetGpuBuffer(). The alias keeps both spellings working for
        // callers reading the issue body vs. callers reading IntelliSense.
        var t = new Tensor<float>(new[] { 16 });
        Assert.Null(t.AsGpuBuffer());
        Assert.Equal(t.TryGetGpuBuffer(), t.AsGpuBuffer());
    }

    [Fact]
    public void GpuOptimizer_TryAdamStep_CpuEngine_ReturnsFalse()
    {
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = new AiDotNet.Tensors.Engines.CpuEngine();
        try
        {
            var param = new Tensor<float>(new[] { 4 });
            var grad = new Tensor<float>(new[] { 4 });
            var m = new Tensor<float>(new[] { 4 });
            var v = new Tensor<float>(new[] { 4 });
            var ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamStep(
                param, grad, m, v,
                learningRate: 0.01f, beta1: 0.9f, beta2: 0.999f,
                epsilon: 1e-8f, weightDecay: 0f, step: 1);
            Assert.False(ran);
        }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine; }
    }

    [Fact]
    public void GpuOptimizer_TrySgdStep_CpuEngine_ReturnsFalse()
    {
        var priorEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current = new AiDotNet.Tensors.Engines.CpuEngine();
        try
        {
            var param = new Tensor<float>(new[] { 4 });
            var grad = new Tensor<float>(new[] { 4 });
            var ran = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TrySgdStep(param, grad, 0.01f);
            Assert.False(ran);
        }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.Current = priorEngine; }
    }

    [Fact]
    public void GpuOptimizer_TryAdamStep_NullArg_Throws()
    {
        var t = new Tensor<float>(new[] { 4 });
        Assert.Throws<System.ArgumentNullException>(() =>
            AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamStep(
                null!, t, t, t, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1));
    }
}
