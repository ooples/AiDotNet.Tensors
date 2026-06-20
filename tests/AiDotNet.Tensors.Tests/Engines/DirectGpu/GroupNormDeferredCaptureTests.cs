using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// On-device (#642 P3) regression coverage for the GPU diffusion ResBlock through the deferred
/// execution graph and CUDA-graph capture. These run only on a real CUDA device (SkippableFact)
/// since CI has no GPU; they lock in three fixes that the mock-only graph tests could not catch:
/// <list type="number">
/// <item>The GroupNorm argument-order bug (numGroups/channels/spatial scrambled) — exercised at
/// channels != numGroups (64 channels, 32 groups), the case that overflowed the save buffers.</item>
/// <item>Deferred-graph correctness for a full ResBlock (GroupNorm-recording + GPU-resident in-place
/// activations) — deferred output must match eager exactly.</item>
/// <item>CUDA-graph capture+replay of the resident graph through the production
/// <see cref="GraphedInferenceStep"/> API replays the captured compute kernels correctly.</item>
/// </list>
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GroupNormDeferredCaptureTests : IDisposable
{
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _gpuAvailable;
    private const float Tolerance = 1e-3f;
    private const int C = 64;       // channels != numGroups (the GroupNorm arg-order overflow case)
    private const int Groups = 32;
    private const int Sp = 16;

    public GroupNormDeferredCaptureTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuAvailable = _gpu.IsGpuAvailable;
        }
        catch (PlatformNotSupportedException) { _gpuAvailable = false; }
        catch (DllNotFoundException) { _gpuAvailable = false; }
    }

    public void Dispose() => (_gpu as IDisposable)?.Dispose();

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // SD-style ResBlock: GroupNorm -> Swish -> Conv -> GroupNorm -> Swish -> Conv -> residual add.
    private Tensor<float> ResBlock(Tensor<float> input, Tensor<float> gamma, Tensor<float> beta,
        Tensor<float> k1, Tensor<float> k2)
    {
        var gpu = _gpu!;
        var h = gpu.GroupNorm(input, Groups, gamma, beta, 1e-5, out _, out _);
        gpu.SwishInPlace(h);
        h = gpu.Conv2D(h, k1, 1, 1, 1);
        h = gpu.GroupNorm(h, Groups, gamma, beta, 1e-5, out _, out _);
        gpu.SwishInPlace(h);
        h = gpu.Conv2D(h, k2, 1, 1, 1);
        return gpu.TensorAdd(input, h);
    }

    // Pure-CPU unit test (no GPU) for the GpuBufferReleaseDeferral depth-counting fix (#642 review,
    // Critical). Runs in CI and locks in the nesting contract: an inner scope's EndAndRelease must
    // NOT flush the outer scope's still-pending releases (the use-after-free the depth counter fixes).
    [Fact]
    public void DeferralGate_NestedScopes_FlushOnlyAtOutermostEnd()
    {
        // (Serialized with the GPU tests via the collection, so no concurrent DeferredScope pollutes
        // the thread-local depth on this thread.)
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        var flushed = new List<string>();

        GpuBufferReleaseDeferral.Begin();                                   // outer
        Assert.True(GpuBufferReleaseDeferral.IsActive);
        Assert.True(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("outer")));

        GpuBufferReleaseDeferral.Begin();                                   // inner
        Assert.True(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("inner")));

        GpuBufferReleaseDeferral.EndAndRelease();                           // inner ends — must NOT flush
        Assert.True(GpuBufferReleaseDeferral.IsActive);
        Assert.Empty(flushed);

        GpuBufferReleaseDeferral.EndAndRelease();                           // outer ends — flush both
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        Assert.Equal(new[] { "outer", "inner" }, flushed);

        // Balanced + idempotent: an extra EndAndRelease is a no-op, and TryDefer returns false.
        GpuBufferReleaseDeferral.EndAndRelease();
        Assert.False(GpuBufferReleaseDeferral.IsActive);
        Assert.False(GpuBufferReleaseDeferral.TryDefer(() => flushed.Add("after")));
        Assert.Equal(2, flushed.Count);
    }

    [SkippableFact]
    public void GroupNorm_ChannelsNeNumGroups_MatchesCpu()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var input = Rand(new[] { 1, C, Sp, Sp }, 1);
        var gamma = Rand(new[] { C }, 2);
        var beta = Rand(new[] { C }, 3);

        var gpuOut = gpu.GroupNorm(input, Groups, gamma, beta, 1e-5, out _, out _);
        var cpuOut = new CpuEngine().GroupNorm(input, Groups, gamma, beta, 1e-5, out _, out _);

        float maxDiff = 0;
        for (int i = 0; i < gpuOut.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(gpuOut[i] - cpuOut[i]));
        Assert.True(maxDiff < Tolerance, $"GPU GroupNorm (channels={C}, groups={Groups}) diverged from CPU: maxDiff={maxDiff}");
    }

    [SkippableFact]
    public void DeferredResBlock_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var input = Rand(new[] { 1, C, Sp, Sp }, 1);
        var gamma = Rand(new[] { C }, 2);
        var beta = Rand(new[] { C }, 3);
        var k1 = Rand(new[] { C, C, 3, 3 }, 4);
        var k2 = Rand(new[] { C, C, 3, 3 }, 5);

        var eager = ResBlock(input, gamma, beta, k1, k2);
        var eagerCopy = new float[eager.Length];
        for (int i = 0; i < eager.Length; i++) eagerCopy[i] = eager[i];

        Tensor<float> deferred;
        using (var scope = gpu.BeginDeferredScope())
        {
            Skip.If(scope is null, "Deferred execution unsupported.");
            deferred = ResBlock(input, gamma, beta, k1, k2);
            scope!.Execute();
        }

        // Shape is part of the contract — Math.Min would mask a wrong-shape regression.
        Assert.Equal(eagerCopy.Length, deferred.Length);
        float maxDiff = 0;
        for (int i = 0; i < eagerCopy.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - deferred[i]));
        Assert.True(maxDiff < Tolerance, $"Deferred ResBlock diverged from eager: maxDiff={maxDiff}");
    }

    [SkippableFact]
    public void ChannelConcat_MatchesCpu_EagerAndDeferred()
    {
        // The UNet decoder skip-connection: TensorConcatenate(new[]{a,b}, axis:1) on NCHW.
        // Channel-axis concat was the #642 gap — the old GPU path only handled the last axis, so
        // this fell to CPU (breaking the device-resident chain). Validate GPU eager + deferred == CPU.
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var a = Rand(new[] { 1, C, Sp, Sp }, 7);
        var b = Rand(new[] { 1, C / 2, Sp, Sp }, 8);

        var cpu = new CpuEngine().TensorConcatenate(new[] { a, b }, axis: 1);
        Assert.Equal(C + C / 2, cpu.Shape[1]);

        var eager = gpu.TensorConcatenate(new[] { a, b }, axis: 1);
        float eagerDiff = 0;
        for (int i = 0; i < cpu.Length; i++) eagerDiff = Math.Max(eagerDiff, Math.Abs(cpu[i] - eager[i]));
        Assert.True(eagerDiff < Tolerance, $"Eager GPU channel-concat diverged from CPU: maxDiff={eagerDiff}");

        Tensor<float> deferred;
        using (var scope = gpu.BeginDeferredScope())
        {
            Skip.If(scope is null, "Deferred execution unsupported.");
            deferred = gpu.TensorConcatenate(new[] { a, b }, axis: 1);
            scope!.Execute();
        }
        float defDiff = 0;
        for (int i = 0; i < cpu.Length; i++) defDiff = Math.Max(defDiff, Math.Abs(cpu[i] - deferred[i]));
        Assert.True(defDiff < Tolerance, $"Deferred GPU channel-concat diverged from CPU: maxDiff={defDiff}");
    }

    // Runs `op` eagerly, then inside a deferred scope; asserts the deferred result matches eager.
    // Proves the op's backend kernel is RECORDED (replays in order) rather than running eagerly
    // mid-record on not-yet-computed buffers (#642 structural-op coverage).
    private void AssertDeferredMatchesEager(string name, Func<Tensor<float>> op)
    {
        var eager = op();
        var eagerCopy = new float[eager.Length];
        for (int i = 0; i < eager.Length; i++) eagerCopy[i] = eager[i];

        Tensor<float> deferred;
        using (var scope = _gpu!.BeginDeferredScope())
        {
            Skip.If(scope is null, "Deferred execution unsupported.");
            deferred = op();
            scope!.Execute();
        }
        // Shape is part of the contract — Math.Min would mask a wrong-shape regression.
        Assert.Equal(eagerCopy.Length, deferred.Length);
        float maxDiff = 0;
        for (int i = 0; i < eagerCopy.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - deferred[i]));
        Assert.True(maxDiff < Tolerance, $"Deferred {name} diverged from eager: maxDiff={maxDiff}");
    }

    [SkippableFact]
    public void DeferredUpsample_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 11);
        AssertDeferredMatchesEager("Upsample", () => _gpu!.Upsample(input, 2, 2));
    }

    [SkippableFact]
    public void DeferredMaxPool2D_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 12);
        AssertDeferredMatchesEager("MaxPool2D", () => _gpu!.MaxPool2D(input, 2, 2, 0));
    }

    [SkippableFact]
    public void DeferredAvgPool2D_MatchesEager()
    {
        // Regression for the CudaBackend.AvgPool2D launch bug (#642): the kernel takes 15 params
        // (… countIncludePad) but the launcher allocated 14 arg slots, so cuLaunchKernel read a
        // garbage 15th pointer → 0xC0000005 on small inputs (surfaced under deferred replay).
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 13);
        AssertDeferredMatchesEager("AvgPool2D", () => _gpu!.AvgPool2D(input, 2, 2, 0));
    }

    [SkippableFact]
    public void DeferredConvTranspose2D_MatchesEager()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var input = Rand(new[] { 1, C, Sp, Sp }, 14);
        var kernel = Rand(new[] { C, C, 2, 2 }, 15);   // [inC, outC, kH, kW]
        AssertDeferredMatchesEager("ConvTranspose2D",
            () => _gpu!.ConvTranspose2D(input, kernel, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 0, 0 }));
    }

    [SkippableFact]
    public void CudaGraphCapture_ReplaysResBlockCorrectly()
    {
        Skip.IfNot(_gpuAvailable, "No CUDA device.");
        var gpu = _gpu!;
        var backend = gpu.GetBackend();
        Skip.If(backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Capture path is CUDA-only.");
        var cudaBackend = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)backend!;

        var input = Rand(new[] { 1, C, Sp, Sp }, 1);
        var gamma = Rand(new[] { C }, 2);
        var beta = Rand(new[] { C }, 3);
        var k1 = Rand(new[] { C, C, 3, 3 }, 4);
        var k2 = Rand(new[] { C, C, 3, 3 }, 5);

        var eager = ResBlock(input, gamma, beta, k1, k2);
        var eagerCopy = new float[eager.Length];
        for (int i = 0; i < eager.Length; i++) eagerCopy[i] = eager[i];

        // Record the resident graph (buffers allocated at record time → Execute is alloc-free).
        var scope = gpu.BeginDeferredScope() as DeferredScope;
        Skip.If(scope is null, "Deferred execution unsupported.");
        ExecutionGraph? graph = null;
        try
        {
            Tensor<float> result = ResBlock(input, gamma, beta, k1, k2);
            graph = scope!.Compile();   // owns the stream pool (GraphCompiler transferred ownership)

            // Warmup: full graph once (H2D populates resident buffers + JIT).
            foreach (var node in graph.TopologicalOrder) node.Execute(backend);
            backend.Synchronize();

            using var step = new GraphedInferenceStep(
                backend, cudaBackend.DefaultStream.Handle,
                () => graph.ExecuteComputeKernelsNoSync(backend),
                new GraphedInferenceStepOptions { ThrowOnUnsupported = false });
            step.Prepare();
            step.Capture();
            Skip.IfNot(step.HasGraph, "CUDA graph capture unsupported on this driver.");

            step.Replay();   // recomputes into the captured output buffer; `result` downloads it

            Assert.Equal(eagerCopy.Length, result.Length);
            float maxDiff = 0;
            for (int i = 0; i < eagerCopy.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - result[i]));
            Assert.True(maxDiff < Tolerance, $"CUDA-graph replay diverged from eager: maxDiff={maxDiff}");
        }
        finally
        {
            // Dispose the scope first (its Dispose may touch the compiled graph), then the graph —
            // which disposes the stream pool the GraphCompiler transferred to it (no leak).
            scope?.Dispose();
            graph?.Dispose();
        }
    }
}
