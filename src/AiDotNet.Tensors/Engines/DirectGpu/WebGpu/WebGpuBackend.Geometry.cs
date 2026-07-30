// Copyright (c) AiDotNet. All rights reserved.
// WebGPU launcher shims for the geometry / sampling kernels (Issue #217).
// WebGPU's compute API is async (*Async methods below); the sync IGeometryBackend
// members block on them via GetAwaiter().GetResult() — the same sync-over-async
// pattern the extended-conv / scatter families use — so the engine's synchronous
// IGeometryBackend dispatch runs these on WebGPU instead of CPU-falling-back (#775).

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IGeometryBackend
{
    private const string GeometryModuleKey = "Geometry";

    void IGeometryBackend.Interpolate2D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win, int Hout, int Wout, int mode, bool alignCorners)
        => Interpolate2DAsync(input, output, N, C, Hin, Win, Hout, Wout, mode, alignCorners)
            .GetAwaiter().GetResult();

    void IGeometryBackend.Pad4D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win, int padN0, int padN1, int padC0, int padC1,
        int padH0, int padH1, int padW0, int padW1, int mode, float padValue)
        => Pad4DAsync(input, output, N, C, Hin, Win, padN0, padN1, padC0, padC1,
            padH0, padH1, padW0, padW1, mode, padValue).GetAwaiter().GetResult();

    void IGeometryBackend.GridSample2D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int N, int H, int W, int C, int outH, int outW, int mode, int padding, bool alignCorners)
        => GridSample2DAsync(input, grid, output, N, H, W, C, outH, outW, mode, padding, alignCorners)
            .GetAwaiter().GetResult();

    void IGeometryBackend.AffineGrid3D(IGpuBuffer theta, IGpuBuffer grid,
        int N, int D, int H, int W, bool alignCorners)
        => AffineGrid3DAsync(theta, grid, N, D, H, W, alignCorners).GetAwaiter().GetResult();

    public async Task Interpolate2DAsync(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win, int Hout, int Wout,
        int mode, bool alignCorners)
    {
        int total = N * C * Hout * Wout;
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            GeometryModuleKey + ":Interpolate2D", WebGpuGeometryKernels.Interpolate2D, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(N, C, Hin, Win, Hout, Wout, mode, alignCorners ? 1 : 0),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task Pad4DAsync(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win,
        int padN0, int padN1, int padC0, int padC1,
        int padH0, int padH1, int padW0, int padW1,
        int mode, float padValue)
    {
        int total = (N + padN0 + padN1) * (C + padC0 + padC1) * (Hin + padH0 + padH1) * (Win + padW0 + padW1);
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            GeometryModuleKey + ":Pad4D", WebGpuGeometryKernels.Pad4D, "main");
        // Uniform: 13 i32 + 1 f32 = 14 × 4 bytes = 56.
        float[] uniformBytes = new float[14];
        uniformBytes[0] = System.BitConverter.Int32BitsToSingle(N);
        uniformBytes[1] = System.BitConverter.Int32BitsToSingle(C);
        uniformBytes[2] = System.BitConverter.Int32BitsToSingle(Hin);
        uniformBytes[3] = System.BitConverter.Int32BitsToSingle(Win);
        uniformBytes[4] = System.BitConverter.Int32BitsToSingle(padN0);
        uniformBytes[5] = System.BitConverter.Int32BitsToSingle(padN1);
        uniformBytes[6] = System.BitConverter.Int32BitsToSingle(padC0);
        uniformBytes[7] = System.BitConverter.Int32BitsToSingle(padC1);
        uniformBytes[8] = System.BitConverter.Int32BitsToSingle(padH0);
        uniformBytes[9] = System.BitConverter.Int32BitsToSingle(padH1);
        uniformBytes[10] = System.BitConverter.Int32BitsToSingle(padW0);
        uniformBytes[11] = System.BitConverter.Int32BitsToSingle(padW1);
        uniformBytes[12] = System.BitConverter.Int32BitsToSingle(mode);
        uniformBytes[13] = padValue;
        // Pad to multiple of 16 bytes for uniform min binding size.
        int padded = ((uniformBytes.Length + 3) / 4) * 4;
        var padBuf = new float[padded];
        Array.Copy(uniformBytes, padBuf, uniformBytes.Length);
        using var uniforms = new WebGpuBuffer(
            padBuf,
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task GridSample2DAsync(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int N, int H, int W, int C, int outH, int outW,
        int mode, int padding, bool alignCorners)
    {
        int total = N * outH * outW;
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            GeometryModuleKey + ":GridSample2D", WebGpuGeometryKernels.GridSample2D, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(N, H, W, C, outH, outW, mode, padding, alignCorners ? 1 : 0),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe,
            AsWgpu(input), AsWgpu(grid), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task AffineGrid3DAsync(IGpuBuffer theta, IGpuBuffer grid,
        int N, int D, int H, int W, bool alignCorners)
    {
        int total = N * D * H * W;
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            GeometryModuleKey + ":AffineGrid3D", WebGpuGeometryKernels.AffineGrid3D, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(N, D, H, W, alignCorners ? 1 : 0),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(theta), AsWgpu(grid));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
}
#endif
