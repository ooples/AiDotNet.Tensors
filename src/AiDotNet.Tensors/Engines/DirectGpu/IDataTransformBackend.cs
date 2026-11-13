// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Capability: the device halves of the sweep data ops: the Box-Muller and threshold transforms of uniforms drawn from
/// the caller's System.Random (<see cref="CpuEngine.TensorTruncatedNormalFromRng{T}"/>,
/// <see cref="CpuEngine.TensorBernoulliFromRng{T}(System.Random, LinearAlgebra.Tensor{T}, UniformDrawComparison)"/>,
/// <see cref="CpuEngine.TensorRademacherFromRng{T}"/>), the lfilter recursion
/// (<see cref="CpuEngine.TensorLinearFilter{T}"/>, one thread per signal), the 64-bit-id row gather and in-place scatter,
/// and the integral rounding of <see cref="CpuEngine.TensorCast{TIn, TOut}"/>.
/// </summary>
/// <remarks>
/// Every backend runs <see cref="DataTransformKernelTemplate"/>'s kernels through <see cref="PortableDataTransform"/>:
/// OpenCL, HIP, Vulkan and Metal in their dialects, CUDA the HIP dialect through NVRTC, WebGPU the WGSL restatement. All
/// buffers are float32; row ids ride as exactly-valued floats. A false return launches nothing and the caller runs the
/// host op.
/// </remarks>
internal interface IDataTransformBackend
{
    /// <summary><c>output[i] = mean + std * uv[2i] cos(2 pi uv[2i+1])</c> (sin for <see cref="BoxMullerBranch.Sine"/>),
    /// clamped to [low, high] when <paramref name="clamp"/>; <c>uv[2i]</c> is the radius <c>sqrt(-2 ln u1)</c>, taken on
    /// the host in double.</summary>
    bool TryBoxMuller(IGpuBuffer uniforms, IGpuBuffer output, int count, BoxMullerBranch branch, float mean, float std,
        bool clamp, float low, float high);

    /// <summary>
    /// <c>output[i]</c> = <paramref name="onSuccess"/> when the draw carried as <c>(uv[2i], uv[2i+1])</c> (a float and its
    /// signed residual) is below (or, <paramref name="above"/>, above) its threshold, otherwise
    /// <paramref name="otherwise"/>. The threshold is <paramref name="thresholds"/>[i] when that buffer is given, else
    /// <paramref name="threshold"/>.
    /// </summary>
    bool TryUniformThreshold(IGpuBuffer uniforms, IGpuBuffer? thresholds, IGpuBuffer output, int count, bool above,
        float threshold, float onSuccess, float otherwise);

    /// <summary>
    /// y = lfilter(b, a, x, zi) along the last axis of <paramref name="signals"/> signals of <paramref name="length"/>
    /// samples; <paramref name="workspace"/> holds at least max(1, signals * (K - 1)) floats, K = max(nb, na). With
    /// <paramref name="guard"/> each output is NaN -> 0 and clamped to +-<paramref name="maxAbs"/> before it feeds back.
    /// </summary>
    bool TryLinearFilter(IGpuBuffer x, IGpuBuffer b, IGpuBuffer a, IGpuBuffer? zi, IGpuBuffer y, IGpuBuffer workspace,
        int signals, int length, int nb, int na, bool guard, float maxAbs);

    /// <summary><c>output[k, :] = table[ids[k], :]</c>, zero where <c>ids[k] &lt; 0</c>.</summary>
    bool TryGatherRows(IGpuBuffer table, IGpuBuffer ids, IGpuBuffer output, int rows, int rowSize);

    /// <summary><c>table[ids[k], :] = source[k, :]</c> in place, skipping <c>ids[k] &lt; 0</c> (no two non-negative ids
    /// may be equal).</summary>
    bool TryScatterRows(IGpuBuffer source, IGpuBuffer ids, IGpuBuffer table, int rows, int rowSize);

    /// <summary><c>output[i]</c> = input[i] rounded to an integer under <paramref name="mode"/> and clamped to
    /// [low, high]; NaN becomes 0.</summary>
    bool TryRoundIntegral(IGpuBuffer input, IGpuBuffer output, int count, RoundingMode mode, float low, float high);
}
