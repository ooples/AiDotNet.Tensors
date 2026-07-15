// Copyright (c) AiDotNet. All rights reserved.
// WebGPU dispatchers for the torch.linalg decomposition kernels (#211 moat #2).
// The *Async methods do the work; the synchronous ILinalgBackend members block on
// them (GetAwaiter().GetResult()) so the engine's sync dispatch runs these on WebGPU
// instead of CPU-falling-back (#775). Caveat: safe under native wgpu (no captured
// SynchronizationContext); a single-threaded UI/JS event-loop host should call the
// *Async methods directly.

#if NET7_0_OR_GREATER
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : ILinalgBackend
{
    void ILinalgBackend.LinalgCholesky(IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
        int batchCount, int n, bool upper)
        => LinalgCholeskyAsync(input, output, info, batchCount, n, upper).GetAwaiter().GetResult();

    void ILinalgBackend.LinalgLuFactor(IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
        int batchCount, int m, int n)
        => LinalgLuFactorAsync(input, output, pivots, batchCount, m, n).GetAwaiter().GetResult();

    void ILinalgBackend.LinalgQrReduced(IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
        => LinalgQrReducedAsync(input, q, r, batchCount, m, n).GetAwaiter().GetResult();

    void ILinalgBackend.LinalgEigh(IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
        => LinalgEighAsync(input, eigenvalues, eigenvectors, batchCount, n).GetAwaiter().GetResult();
    private const string LinalgModuleKey = "Linalg211";

    public async Task LinalgCholeskyAsync(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
        int batchCount, int n, bool upper)
    {
        if (batchCount <= 0 || n <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            LinalgModuleKey + ":Cholesky", WebGpuLinalgKernels.Cholesky, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(batchCount, n, upper ? 1 : 0),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output), AsWgpu(info));
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, batchCount, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task LinalgLuFactorAsync(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            LinalgModuleKey + ":LuFactor", WebGpuLinalgKernels.LuFactor, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(batchCount, m, n),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output), AsWgpu(pivots));
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, batchCount, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task LinalgQrReducedAsync(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
    {
        if (batchCount <= 0 || m <= 0 || n <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            LinalgModuleKey + ":QrReduced", WebGpuLinalgKernels.QrReduced, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(batchCount, m, n),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(q), AsWgpu(r));
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, batchCount, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task LinalgEighAsync(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
    {
        if (batchCount <= 0 || n <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            LinalgModuleKey + ":Eigh", WebGpuLinalgKernels.Eigh, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(batchCount, n),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(eigenvalues), AsWgpu(eigenvectors));
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, batchCount, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
}
#endif
