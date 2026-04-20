// Copyright (c) AiDotNet. All rights reserved.
// WebGPU dispatchers for the torch.linalg decomposition kernels (#211 moat #2).
// Async-only — WebGPU's pipeline creation and dispatch are both async-native.
// The DirectGpuTensorEngine sync ILinalgBackend path doesn't route through
// these today (same pattern as Parity-210's WebGPU variant); callers that
// know they're on WebGPU can call these async methods directly.

#if NET7_0_OR_GREATER
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
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
