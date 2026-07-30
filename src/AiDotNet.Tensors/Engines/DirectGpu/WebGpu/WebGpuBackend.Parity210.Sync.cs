// Copyright (c) AiDotNet. All rights reserved.
// #775: synchronous IParity210Backend for WebGPU. WebGPU's compute submission is
// async (the *Async methods in WebGpuBackend.Parity210.cs); these members block on
// them via GetAwaiter().GetResult() — the same sync-over-async pattern the extended-
// conv / scatter families use — so the engine's sync IParity210Backend dispatch runs
// these on WebGPU instead of CPU-falling-back. CdistL2Async / CosineSimilarityLastAsync
// are added here (their WGSL kernels were registered but had no launcher). Caveat: safe
// under native wgpu (no captured SynchronizationContext); a single-threaded UI/JS
// event-loop host should call the *Async methods directly.
#if NET7_0_OR_GREATER
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IParity210Backend
{
    public async Task Parity210CosineSimilarityLastAsync(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output,
        int n, int d, float eps)
    {
        if (n <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":CosineSimilarityLast", WebGpuParity210Kernels.CosineSimilarityLast, "main");
        // Uniform layout: struct { n: i32, d: i32, eps: f32 }
        var bytes = new float[4];
        bytes[0] = System.BitConverter.Int32BitsToSingle(n);
        bytes[1] = System.BitConverter.Int32BitsToSingle(d);
        bytes[2] = eps;
        using var uniforms = new WebGpuBuffer(bytes, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(a), AsWgpu(b), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(n);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    public async Task Parity210CdistL2Async(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output,
        int n, int m, int d)
    {
        int total = n * m;
        if (total <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            Parity210ModuleKey + ":CdistL2", WebGpuParity210Kernels.CdistL2, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(n, m, d),
            WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipelineId, AsWgpu(x1), AsWgpu(x2), AsWgpu(output));
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    void IParity210Backend.Parity210Roll1D(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int shift)
        => Parity210Roll1DAsync(input, output, outerSize, axisSize, innerSize, shift).GetAwaiter().GetResult();

    void IParity210Backend.Parity210FlipAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize)
        => Parity210FlipAxisAsync(input, output, outerSize, axisSize, innerSize).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Triu(IGpuBuffer input, IGpuBuffer output, int batchSize, int rows, int cols, int diagonal)
        => Parity210TriuAsync(input, output, batchSize, rows, cols, diagonal).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Tril(IGpuBuffer input, IGpuBuffer output, int batchSize, int rows, int cols, int diagonal)
        => Parity210TrilAsync(input, output, batchSize, rows, cols, diagonal).GetAwaiter().GetResult();

    void IParity210Backend.Parity210DiagEmbed(IGpuBuffer input, IGpuBuffer output, int batchSize, int diagLen, int matSize, int offset)
        => Parity210DiagEmbedAsync(input, output, batchSize, diagLen, matSize, offset).GetAwaiter().GetResult();

    void IParity210Backend.Parity210CumSum(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize)
        => Parity210CumSumAsync(input, output, outerSize, axisSize, innerSize).GetAwaiter().GetResult();

    void IParity210Backend.Parity210CumProd(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize)
        => Parity210CumProdAsync(input, output, outerSize, axisSize, innerSize).GetAwaiter().GetResult();

    void IParity210Backend.Parity210CumMax(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize)
        => Parity210CumMaxAsync(input, output, outerSize, axisSize, innerSize).GetAwaiter().GetResult();

    void IParity210Backend.Parity210CumMin(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize)
        => Parity210CumMinAsync(input, output, outerSize, axisSize, innerSize).GetAwaiter().GetResult();

    void IParity210Backend.Parity210LogCumSumExp(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize)
        => Parity210LogCumSumExpAsync(input, output, outerSize, axisSize, innerSize).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Hypot(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Parity210HypotAsync(a, b, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Copysign(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Parity210CopysignAsync(a, b, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Fmod(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Parity210FmodAsync(a, b, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Remainder(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Parity210RemainderAsync(a, b, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210FloatPower(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Parity210FloatPowerAsync(a, b, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210LogAddExp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Parity210LogAddExpAsync(a, b, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210LogAddExp2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Parity210LogAddExp2Async(a, b, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Xlogy(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size)
        => Parity210XlogyAsync(x, y, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Xlog1py(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size)
        => Parity210Xlog1pyAsync(x, y, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Erfc(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210ErfcAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Erfinv(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210ErfinvAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Lgamma(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210LgammaAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210Digamma(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210DigammaAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210I0(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210I0Async(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210I1(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210I1Async(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210I0e(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210I0eAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210I1e(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210I1eAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210IsFinite(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210IsFiniteAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210IsNan(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210IsNanAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210IsInf(IGpuBuffer input, IGpuBuffer output, int size)
        => Parity210IsInfAsync(input, output, size).GetAwaiter().GetResult();

    void IParity210Backend.Parity210NanToNum(IGpuBuffer input, IGpuBuffer output, int size, float nanVal, float posInfVal, float negInfVal)
        => Parity210NanToNumAsync(input, output, size, nanVal, posInfVal, negInfVal).GetAwaiter().GetResult();

    void IParity210Backend.Parity210CosineSimilarityLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int d, float eps)
        => Parity210CosineSimilarityLastAsync(a, b, output, n, d, eps).GetAwaiter().GetResult();

    void IParity210Backend.Parity210CdistL2(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int n, int m, int d)
        => Parity210CdistL2Async(x1, x2, output, n, m, d).GetAwaiter().GetResult();
}
#endif
