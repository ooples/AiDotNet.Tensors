// Copyright (c) AiDotNet. All rights reserved.
// WebGPU launcher shims for the parity-210 kernels. Each WGSL source in
// WebGpuParity210Kernels is a self-contained compute shader (WebGPU
// compiles one pipeline per source), so each method just plumbs the
// right source string + bind group count + push-constant uniform data
// through GetOrCreatePipelineAsync and DispatchComputeWithUniformsAsync.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string Parity210ModuleKey = "Parity210";

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static WebGpuBuffer AsWgpu(IGpuBuffer b) => (WebGpuBuffer)b;

    // Uniform block layout is `std140`-ish — each scalar consumes 16 bytes
    // when alone in a struct, but adjacent scalars pack into a single 16-byte
    // vector. We allocate in 4-element float blocks.
    private static float[] UniformInts(params int[] values)
    {
        // Pad to nearest multiple of 4 for WebGPU uniform buffer min binding size.
        int padded = ((values.Length + 3) / 4) * 4;
        var arr = new float[padded];
        for (int i = 0; i < values.Length; i++)
            arr[i] = System.BitConverter.Int32BitsToSingle(values[i]);
        return arr;
    }

    private static float[] UniformFloats(params float[] values)
    {
        int padded = ((values.Length + 3) / 4) * 4;
        var arr = new float[padded];
        Array.Copy(values, arr, values.Length);
        return arr;
    }

    // -----------------------------------------------------------------------
    // MOVEMENT
    // -----------------------------------------------------------------------

    public async Task Parity210Roll1DAsync(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize, int shift)
    {
        int total = outerSize * axisSize * innerSize;
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":Roll1D", WebGpuParity210Kernels.Roll1D, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(outerSize, axisSize, innerSize, shift),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task Parity210FlipAxisAsync(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        int total = outerSize * axisSize * innerSize;
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":FlipAxis", WebGpuParity210Kernels.FlipAxis, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(outerSize, axisSize, innerSize),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task Parity210TriuAsync(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
        => await DispatchTriuTrilWgpuAsync("Triu", WebGpuParity210Kernels.Triu,
            input, output, batchSize, rows, cols, diagonal);

    public async Task Parity210TrilAsync(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
        => await DispatchTriuTrilWgpuAsync("Tril", WebGpuParity210Kernels.Tril,
            input, output, batchSize, rows, cols, diagonal);

    private async Task DispatchTriuTrilWgpuAsync(string tag, string source,
        IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        int total = batchSize * rows * cols;
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":" + tag, source, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(batchSize, rows, cols, diagonal),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task Parity210DiagEmbedAsync(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int diagLen, int matSize, int offset)
    {
        int total = batchSize * matSize * matSize;
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":DiagEmbed", WebGpuParity210Kernels.DiagEmbed, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(batchSize, diagLen, matSize, offset),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    // -----------------------------------------------------------------------
    // CUMULATIVE
    // -----------------------------------------------------------------------

    public async Task Parity210CumSumAsync(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => await DispatchCumulativeWgpuAsync("CumSum", WebGpuParity210Kernels.CumSumAxis,
            input, output, outerSize, axisSize, innerSize);

    public async Task Parity210CumProdAsync(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => await DispatchCumulativeWgpuAsync("CumProd", WebGpuParity210Kernels.CumProdAxis,
            input, output, outerSize, axisSize, innerSize);

    public async Task Parity210CumMaxAsync(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => await DispatchCumulativeWgpuAsync("CumMax", WebGpuParity210Kernels.CumMaxAxis,
            input, output, outerSize, axisSize, innerSize);

    public async Task Parity210CumMinAsync(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => await DispatchCumulativeWgpuAsync("CumMin", WebGpuParity210Kernels.CumMinAxis,
            input, output, outerSize, axisSize, innerSize);

    public async Task Parity210LogCumSumExpAsync(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => await DispatchCumulativeWgpuAsync("LogCumSumExp", WebGpuParity210Kernels.LogCumSumExpAxis,
            input, output, outerSize, axisSize, innerSize);

    private async Task DispatchCumulativeWgpuAsync(string tag, string source,
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        int total = outerSize * innerSize;
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":" + tag, source, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(outerSize, axisSize, innerSize),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE BINARY
    // -----------------------------------------------------------------------

    public async Task Parity210HypotAsync(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("Hypot", WebGpuParity210Kernels.Hypot, a, b, o, size);

    public async Task Parity210CopysignAsync(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("Copysign", WebGpuParity210Kernels.Copysign, a, b, o, size);

    public async Task Parity210FmodAsync(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("Fmod", WebGpuParity210Kernels.Fmod, a, b, o, size);

    public async Task Parity210RemainderAsync(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("Remainder", WebGpuParity210Kernels.Remainder, a, b, o, size);

    public async Task Parity210FloatPowerAsync(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("FloatPower", WebGpuParity210Kernels.FloatPower, a, b, o, size);

    public async Task Parity210LogAddExpAsync(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("LogAddExp", WebGpuParity210Kernels.LogAddExp, a, b, o, size);

    public async Task Parity210LogAddExp2Async(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("LogAddExp2", WebGpuParity210Kernels.LogAddExp2, a, b, o, size);

    public async Task Parity210XlogyAsync(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("Xlogy", WebGpuParity210Kernels.Xlogy, x, y, o, size);

    public async Task Parity210Xlog1pyAsync(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
        => await DispatchBinaryWgpuAsync("Xlog1py", WebGpuParity210Kernels.Xlog1py, x, y, o, size);

    private async Task DispatchBinaryWgpuAsync(string tag, string source,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
    {
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":" + tag, source, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(size),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(a), AsWgpu(b), AsWgpu(o));
        var (wg, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE UNARY SPECIAL
    // -----------------------------------------------------------------------

    public async Task Parity210ErfcAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("Erfc", WebGpuParity210Kernels.Erfc, input, output, size);

    public async Task Parity210ErfinvAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("Erfinv", WebGpuParity210Kernels.Erfinv, input, output, size);

    public async Task Parity210LgammaAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("Lgamma", WebGpuParity210Kernels.LgammaApprox, input, output, size);

    public async Task Parity210DigammaAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("Digamma", WebGpuParity210Kernels.Digamma, input, output, size);

    public async Task Parity210I0Async(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("I0", WebGpuParity210Kernels.I0, input, output, size);

    public async Task Parity210I1Async(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("I1", WebGpuParity210Kernels.I1, input, output, size);

    public async Task Parity210I0eAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("I0e", WebGpuParity210Kernels.I0e, input, output, size);

    public async Task Parity210I1eAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("I1e", WebGpuParity210Kernels.I1e, input, output, size);

    public async Task Parity210IsFiniteAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("IsFinite", WebGpuParity210Kernels.IsFinite, input, output, size);

    public async Task Parity210IsNanAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("IsNan", WebGpuParity210Kernels.IsNan, input, output, size);

    public async Task Parity210IsInfAsync(IGpuBuffer input, IGpuBuffer output, int size)
        => await DispatchUnaryWgpuAsync("IsInf", WebGpuParity210Kernels.IsInf, input, output, size);

    private async Task DispatchUnaryWgpuAsync(string tag, string source,
        IGpuBuffer input, IGpuBuffer output, int size)
    {
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":" + tag, source, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(size),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task Parity210NanToNumAsync(IGpuBuffer input, IGpuBuffer output, int size,
        float nanVal, float posInfVal, float negInfVal)
    {
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":NanToNum", WebGpuParity210Kernels.NanToNum, "main");
        // Uniform layout: struct { size: i32, nanVal: f32, posInfVal: f32, negInfVal: f32 }
        var bytes = new float[4];
        bytes[0] = System.BitConverter.Int32BitsToSingle(size);
        bytes[1] = nanVal;
        bytes[2] = posInfVal;
        bytes[3] = negInfVal;
        using var uniforms = new WebGpuBuffer(bytes, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(input), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
}
#endif
