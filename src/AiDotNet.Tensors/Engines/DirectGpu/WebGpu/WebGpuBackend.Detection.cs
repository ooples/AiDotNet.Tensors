// Copyright (c) AiDotNet. All rights reserved.
// WebGPU launcher shims for the vision detection kernels (Issue #217). Each
// WGSL source in WebGpuDetectionKernels is a self-contained compute shader
// (one pipeline per source), dispatched 1-D over n*m (IoU) or n (per-box).
//
// Note: WebGPU's compute API is async, so detection ops here are exposed as
// *Async methods. They are NOT wired through DirectGpuTensorEngine's sync
// IDetectionBackend dispatch (same constraint as the Parity-210 surface) —
// the engine falls through to the CpuEngine reference for WebGPU. Direct
// callers who hold a WebGpuBackend can invoke these Async methods.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string DetectionModuleKey = "Detection";

    private async Task DispatchPairwiseIouAsync(string tag, string source,
        IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
    {
        if (n <= 0 || m <= 0) return;
        int total = n * m;
        var pipelineId = await GetOrCreatePipelineAsync(
            DetectionModuleKey + ":" + tag, source, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(n, m),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId,
            AsWgpu(boxesA), AsWgpu(boxesB), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public Task BoxIouAsync(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIouAsync("BoxIou", WebGpuDetectionKernels.BoxIou, boxesA, boxesB, output, n, m);

    public Task GeneralizedBoxIouAsync(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIouAsync("GeneralizedBoxIou", WebGpuDetectionKernels.GeneralizedBoxIou, boxesA, boxesB, output, n, m);

    public Task DistanceBoxIouAsync(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIouAsync("DistanceBoxIou", WebGpuDetectionKernels.DistanceBoxIou, boxesA, boxesB, output, n, m);

    public Task CompleteBoxIouAsync(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIouAsync("CompleteBoxIou", WebGpuDetectionKernels.CompleteBoxIou, boxesA, boxesB, output, n, m);

    public async Task BoxAreaAsync(IGpuBuffer boxes, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            DetectionModuleKey + ":BoxArea", WebGpuDetectionKernels.BoxArea, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(n),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(boxes), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(n);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task BoxConvertAsync(IGpuBuffer boxes, IGpuBuffer output,
        int n, int fromFormat, int toFormat)
    {
        if (n <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            DetectionModuleKey + ":BoxConvert", WebGpuDetectionKernels.BoxConvert, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(n, fromFormat, toFormat),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(boxes), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(n);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task IouFamilyBackwardAsync(
        IGpuBuffer gradOutput, IGpuBuffer boxesA, IGpuBuffer boxesB,
        IGpuBuffer gradA, IGpuBuffer gradB,
        int n, int m, int variant)
    {
        // See CudaBackend.Detection for rationale.
        if (n <= 0 && m <= 0) return;

        if (n > 0)
        {
            var pipeA = await GetOrCreatePipelineAsync(
                DetectionModuleKey + ":IouBackwardA", WebGpuDetectionKernels.IouBackwardA, "main");
            using (var uniforms = new WebGpuBuffer(
                UniformInts(n, m, variant),
                WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst))
            using (var bind = new WebGpuBindGroup(pipeA,
                AsWgpu(gradOutput), AsWgpu(boxesA), AsWgpu(boxesB), AsWgpu(gradA)))
            {
                var (wg, _) = _device.CalculateWorkgroups1D(n);
                await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
                    pipeA, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
                await WebGpuNativeBindings.SubmitAndWaitAsync();
            }
        }

        if (m > 0)
        {
            var pipeB = await GetOrCreatePipelineAsync(
                DetectionModuleKey + ":IouBackwardB", WebGpuDetectionKernels.IouBackwardB, "main");
            using (var uniforms = new WebGpuBuffer(
                UniformInts(n, m, variant),
                WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst))
            using (var bind = new WebGpuBindGroup(pipeB,
                AsWgpu(gradOutput), AsWgpu(boxesA), AsWgpu(boxesB), AsWgpu(gradB)))
            {
                var (wg, _) = _device.CalculateWorkgroups1D(m);
                await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
                    pipeB, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
                await WebGpuNativeBindings.SubmitAndWaitAsync();
            }
        }
    }
}
#endif
