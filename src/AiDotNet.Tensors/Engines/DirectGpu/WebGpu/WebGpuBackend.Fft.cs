// Copyright (c) AiDotNet. All rights reserved.
// WebGPU FFT dispatcher — async launcher for WebGpuFftKernels.Fft.
// Mirrors the Parity210 / Linalg dispatcher style. WebGPU is async-native,
// so the sync IFftBackend signature can't be satisfied here — callers who
// know they're on WebGPU invoke LaunchFftAsync directly.

#if NET7_0_OR_GREATER
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string FftModuleKey = "Fft212";

    public async Task LaunchFftAsync(IGpuBuffer buffer, int batchCount, int n, bool inverse)
    {
        if (batchCount <= 0 || n <= 0) return;
        if ((n & (n - 1)) != 0)
            throw new ArgumentException(
                $"WebGPU LaunchFft requires n to be a power of two (got n = {n}).",
                nameof(n));
        if (buffer is not WebGpuBuffer)
            throw new ArgumentException("Buffer must be WebGpuBuffer.", nameof(buffer));

        var pipelineId = await GetOrCreatePipelineAsync(
            FftModuleKey + ":Fft", WebGpuFftKernels.Fft, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(batchCount, n, inverse ? 1 : 0),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(buffer));
        // One workgroup per batch slice.
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, batchCount, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
}
#endif
