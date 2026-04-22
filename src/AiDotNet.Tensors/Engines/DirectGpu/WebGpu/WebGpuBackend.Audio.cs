// Copyright (c) AiDotNet. All rights reserved.
// WebGPU audio launchers — async-only.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string AudioModuleKey = "Audio";

    public async Task AmplitudeToDBAsync(IGpuBuffer input, IGpuBuffer output, int length,
        float minAmplitude, float topDb, bool clipTopDb)
    {
        if (length <= 0) return;
        if (!(minAmplitude > 0.0f))
            throw new ArgumentException(
                $"minAmplitude must be > 0 (log10 needs a positive floor); got {minAmplitude}.",
                nameof(minAmplitude));
        var pipe = await GetOrCreatePipelineAsync(
            AudioModuleKey + ":AmplitudeToDB", WebGpuAudioKernels.AmplitudeToDB, "main");
        var u = new float[4];
        u[0] = System.BitConverter.Int32BitsToSingle(length);
        u[1] = minAmplitude;
        u[2] = topDb;
        u[3] = System.BitConverter.Int32BitsToSingle(clipTopDb ? 1 : 0);
        using var uniforms = new WebGpuBuffer(u, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(length);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task MuLawEncodingAsync(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        if (qc < 2) throw new ArgumentException("qc must be >= 2.", nameof(qc));
        var pipe = await GetOrCreatePipelineAsync(
            AudioModuleKey + ":MuLawEncoding", WebGpuAudioKernels.MuLawEncoding, "main");
        using var uniforms = new WebGpuBuffer(UniformInts(length, qc),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(length);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task MuLawDecodingAsync(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        if (qc < 2) throw new ArgumentException("qc must be >= 2.", nameof(qc));
        var pipe = await GetOrCreatePipelineAsync(
            AudioModuleKey + ":MuLawDecoding", WebGpuAudioKernels.MuLawDecoding, "main");
        using var uniforms = new WebGpuBuffer(UniformInts(length, qc),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(length);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task ComputeDeltasAsync(IGpuBuffer input, IGpuBuffer output,
        int leading, int timeAxis, int winLength)
    {
        // Widened arithmetic — large feature stacks can wrap in int.
        long totalLong = (long)leading * timeAxis;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"ComputeDeltas total {totalLong} exceeds Int32.MaxValue.");
        int total = (int)totalLong;
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            AudioModuleKey + ":ComputeDeltas", WebGpuAudioKernels.ComputeDeltas, "main");
        using var uniforms = new WebGpuBuffer(UniformInts(leading, timeAxis, winLength),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task ResampleAsync(IGpuBuffer input, IGpuBuffer output,
        int leading, int inLen, int outLen, int up, int down, int halfWidth)
    {
        if (up <= 0 || down <= 0)
            throw new ArgumentException("up and down must both be positive.");
        if (halfWidth < 1)
            throw new ArgumentException("halfWidth must be >= 1 (Hann window needs at least one tap).", nameof(halfWidth));
        long totalLong2 = (long)leading * outLen;
        if (totalLong2 > int.MaxValue)
            throw new OverflowException($"Resample total {totalLong2} exceeds Int32.MaxValue.");
        int total = (int)totalLong2;
        if (total <= 0) return;
        var pipe = await GetOrCreatePipelineAsync(
            AudioModuleKey + ":Resample", WebGpuAudioKernels.Resample, "main");
        using var uniforms = new WebGpuBuffer(UniformInts(leading, inLen, outLen, up, down, halfWidth),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
}
#endif
