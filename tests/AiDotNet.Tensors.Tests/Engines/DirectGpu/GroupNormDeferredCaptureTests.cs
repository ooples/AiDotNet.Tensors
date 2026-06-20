using System;
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

        float maxDiff = 0;
        for (int i = 0; i < Math.Min(eagerCopy.Length, deferred.Length); i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - deferred[i]));
        Assert.True(maxDiff < Tolerance, $"Deferred ResBlock diverged from eager: maxDiff={maxDiff}");
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
        var scope = (DeferredScope)gpu.BeginDeferredScope()!;
        try
        {
            Tensor<float> result = ResBlock(input, gamma, beta, k1, k2);
            ExecutionGraph graph = scope.Compile();

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

            float maxDiff = 0;
            for (int i = 0; i < Math.Min(eagerCopy.Length, result.Length); i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(eagerCopy[i] - result[i]));
            Assert.True(maxDiff < Tolerance, $"CUDA-graph replay diverged from eager: maxDiff={maxDiff}");
        }
        finally
        {
            scope.Dispose();
        }
    }
}
