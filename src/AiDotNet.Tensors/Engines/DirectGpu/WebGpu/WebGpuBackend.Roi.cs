// Copyright (c) AiDotNet. All rights reserved.
// WebGPU RoI launcher shims — async-only, same pattern as Detection/Geometry.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string RoiModuleKey = "Roi";

    public async Task RoIAlignAsync(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW,
        float spatialScale, int samplingRatio, bool aligned)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            RoiModuleKey + ":RoIAlign", WebGpuRoiKernels.RoIAlign, "main");
        // Uniform: 7 i32 + 1 f32 + 2 i32 = 10 × 4 bytes = 40.
        var uniform = new float[10];
        uniform[0] = System.BitConverter.Int32BitsToSingle(N);
        uniform[1] = System.BitConverter.Int32BitsToSingle(C);
        uniform[2] = System.BitConverter.Int32BitsToSingle(H);
        uniform[3] = System.BitConverter.Int32BitsToSingle(W);
        uniform[4] = System.BitConverter.Int32BitsToSingle(K);
        uniform[5] = System.BitConverter.Int32BitsToSingle(outH);
        uniform[6] = System.BitConverter.Int32BitsToSingle(outW);
        uniform[7] = spatialScale;
        uniform[8] = System.BitConverter.Int32BitsToSingle(samplingRatio);
        uniform[9] = System.BitConverter.Int32BitsToSingle(aligned ? 1 : 0);
        // Pad to multiple of 16 bytes.
        int padded = ((uniform.Length + 3) / 4) * 4;
        var padBuf = new float[padded];
        Array.Copy(uniform, padBuf, uniform.Length);
        using var u = new WebGpuBuffer(padBuf, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(boxes), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, u.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task RoIPoolAsync(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, float spatialScale)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            RoiModuleKey + ":RoIPool", WebGpuRoiKernels.RoIPool, "main");
        var uniform = new float[8];
        uniform[0] = System.BitConverter.Int32BitsToSingle(N);
        uniform[1] = System.BitConverter.Int32BitsToSingle(C);
        uniform[2] = System.BitConverter.Int32BitsToSingle(H);
        uniform[3] = System.BitConverter.Int32BitsToSingle(W);
        uniform[4] = System.BitConverter.Int32BitsToSingle(K);
        uniform[5] = System.BitConverter.Int32BitsToSingle(outH);
        uniform[6] = System.BitConverter.Int32BitsToSingle(outW);
        uniform[7] = spatialScale;
        using var u = new WebGpuBuffer(uniform, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(boxes), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, u.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
}
#endif
