using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NN.Losses;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// GPU coverage decision for the PyTorch-parity ops (the <c>CpuEngine.Torch*</c> partials).
/// </summary>
/// <remarks>
/// <para>
/// The composed parity ops (listed in <c>OpRegistry.DelegatorOps</c>: the RNN cells and sequences, the general
/// convolution, bilinear, local response norm, the dropout variants and the rest) are built from engine
/// primitives that this engine already runs on the device; their overrides below pass straight to the composition,
/// whose virtual calls land on this engine's device kernels (they exist so the backend-completeness ratchet sees an
/// explicit decision for every op). Seven ops have device paths in
/// DirectGpuTensorEngine.TorchParityDevice.cs. Degree/radian
/// conversion is routed to the device's scalar multiply below. The reparameterized normal and the dropout variants
/// compose on the device but draw their noise or keep-mask on the host and upload it, so they record a fallback too.
/// </para>
/// <para>
/// The remaining parity ops (special functions, orthogonal polynomials, window functions, samplers, index
/// construction, pooling variants, complex conversions, quantiles, volumetric grid sampling) try a device path first:
/// the <c>Try*OnDevice</c> helpers in the TorchParity* partials, which run on backends implementing
/// <see cref="ITorchParityElementwiseBackend"/>. CUDA (#1112) and Vulkan (#1127) implement it today; HIP, Metal,
/// OpenCL and WebGPU are tracked in #1124, #1125, #1126 and #1128. When a helper declines (another backend, a tape or
/// graph recording it must not bypass, an argument error the CPU reports, or a precision policy), the override falls
/// through to the host implementation EXPLICITLY: the call records a <see cref="GpuLaunchProbe"/> fallback
/// (<c>"&lt;op&gt;: no device kernel"</c> in <see cref="GpuLaunchProbe.Fallbacks"/>), so a GPU workload that reaches
/// the host is visible in the residency diagnostics rather than silently running at host speed.
/// </para>
/// </remarks>
public partial class DirectGpuTensorEngine
{





    // ---- CpuEngine.TorchComplex.cs ----

    /// <inheritdoc/>
    public override Tensor<Complex<T>> TensorComplex<T>(Tensor<T> real, Tensor<T> imag)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (TryComplexOnDevice(real, imag) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorComplex: no device kernel", null);
        return base.TensorComplex<T>(real, imag);
    }

    /// <inheritdoc/>
    public override Tensor<Complex<T>> TensorViewAsComplex<T>(Tensor<T> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (TryViewAsComplexOnDevice(input) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorViewAsComplex: no device kernel", null);
        return base.TensorViewAsComplex<T>(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAngle<T>(Tensor<T> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryParityUnaryOnDevice(input, TorchParityUnaryOp.Angle, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorAngle: no device kernel", null);
        return base.TensorAngle<T>(input);
    }

    // ---- CpuEngine.TorchConvolution.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorConvolution<T>(Tensor<T> input, Tensor<T> weight, Tensor<T>? bias, int[] stride, int[] padding, int[] dilation, bool transposed = false, int[]? outputPadding = null, int groups = 1)
        => base.TensorConvolution<T>(input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups);

    // ---- CpuEngine.TorchCreation.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorAmin<T>(Tensor<T> tensor, int[] axes, bool keepDims = false)
        => base.TensorAmin<T>(tensor, axes, keepDims);

    /// <inheritdoc/>
    /// <remarks>Runs on the device on CUDA (#1112); other backends fall through to the host and record a
    /// <see cref="GpuLaunchProbe"/> fallback (follow-ups #1124-#1128).</remarks>
    public override Tensor<T> TensorAll<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryTruthinessOnDevice("TensorAll", tensor, axes, keepDims, all: true) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorAll: no device kernel", null);
        return base.TensorAll<T>(tensor, axes, keepDims);
    }

    /// <inheritdoc/>
    /// <remarks>Runs on the device on CUDA (#1112); other backends fall through to the host and record a
    /// <see cref="GpuLaunchProbe"/> fallback (follow-ups #1124-#1128).</remarks>
    public override Tensor<T> TensorAny<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryTruthinessOnDevice("TensorAny", tensor, axes, keepDims, all: false) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorAny: no device kernel", null);
        return base.TensorAny<T>(tensor, axes, keepDims);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorArange<T>(double start, double end, double step = 1)
    {
        if (TryRangeOnDevice<T>("TensorArange", start, step, RangeCount(start, end, step, Math.Ceiling((end - start) / step))) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorArange: no device kernel", null);
        return base.TensorArange<T>(start, end, step);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRange<T>(double start, double end, double step = 1)
    {
        if (TryRangeOnDevice<T>("TensorRange", start, step, RangeCount(start, end, step, Math.Floor((end - start) / step + 1))) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorRange: no device kernel", null);
        return base.TensorRange<T>(start, end, step);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogspace<T>(double start, double end, int steps, double logBase = 10)
    {
        if (steps >= 0 && TryGenerateOnDevice<T>("TensorLogspace", new[] { steps }, steps,
                (k, _, output) => k.TorchParityLogspace(output, steps, start, end - start, steps - 1, logBase, steps == 1)) is { } onDevice)
            return onDevice;
        GpuLaunchProbe.OnFallback("TensorLogspace: no device kernel", null);
        return base.TensorLogspace<T>(start, end, steps, logBase);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorZerosLike<T>(Tensor<T> tensor)
    {
        if (TryFillLikeOnDevice("TensorZerosLike", tensor, 0) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorZerosLike: no device kernel", null);
        return base.TensorZerosLike<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorOnesLike<T>(Tensor<T> tensor)
    {
        if (TryFillLikeOnDevice("TensorOnesLike", tensor, 1) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorOnesLike: no device kernel", null);
        return base.TensorOnesLike<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFullLike<T>(Tensor<T> tensor, T value)
    {
        if (TryFillLikeOnDevice("TensorFullLike", tensor, MathHelper.GetNumericOperations<T>().ToDouble(value)) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorFullLike: no device kernel", null);
        return base.TensorFullLike<T>(tensor, value);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorEmptyLike<T>(Tensor<T> tensor)
    {
        if (TryFillLikeOnDevice("TensorEmptyLike", tensor, 0) is { } onDevice) return onDevice;   // the CPU zero-fills too
        GpuLaunchProbe.OnFallback("TensorEmptyLike: no device kernel", null);
        return base.TensorEmptyLike<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandLike<T>(Tensor<T> tensor, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor is not null && TrySampleOnDevice<T>("TensorRandLike", tensor._shape, 0, 0, 1, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorRandLike: no device kernel", null);
        return base.TensorRandLike<T>(tensor!, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandnLike<T>(Tensor<T> tensor, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor is not null && TrySampleOnDevice<T>("TensorRandnLike", tensor._shape, 5, 0, 0, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorRandnLike: no device kernel", null);
        return base.TensorRandnLike<T>(tensor!, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandint<T>(long low, long high, int[] shape, int? seed = null)
    {
        // Draws travel as float values, exact only within +-2^24.
        if (shape is not null && high > low && low >= -(1L << 24) && high <= (1L << 24)
            && TrySampleOnDevice<T>("TensorRandint", shape, 6, low, high - (double)low, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorRandint: no device kernel", null);
        return base.TensorRandint<T>(low, high, shape!, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandintLike<T>(Tensor<T> tensor, long low, long high, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor is not null && high > low && low >= -(1L << 24) && high <= (1L << 24)
            && TrySampleOnDevice<T>("TensorRandintLike", tensor._shape, 6, low, high - (double)low, seed) is { } onDevice)
            return onDevice;
        GpuLaunchProbe.OnFallback("TensorRandintLike: no device kernel", null);
        return base.TensorRandintLike<T>(tensor!, low, high, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandperm<T>(int n, int? seed = null)
    {
        if (TryRandpermOnDevice<T>(n, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorRandperm: no device kernel", null);
        return base.TensorRandperm<T>(n, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHannWindow<T>(int length, bool periodic = true)
    {
        if (TryWindowOnDevice<T>("TensorHannWindow", length, periodic, 0, 0, 0) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorHannWindow: no device kernel", null);
        return base.TensorHannWindow<T>(length, periodic);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHammingWindow<T>(int length, bool periodic = true, double alpha = 0.54, double beta = 0.46)
    {
        if (TryWindowOnDevice<T>("TensorHammingWindow", length, periodic, 1, alpha, beta) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorHammingWindow: no device kernel", null);
        return base.TensorHammingWindow<T>(length, periodic, alpha, beta);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBlackmanWindow<T>(int length, bool periodic = true)
    {
        if (TryWindowOnDevice<T>("TensorBlackmanWindow", length, periodic, 2, 0, 0) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBlackmanWindow: no device kernel", null);
        return base.TensorBlackmanWindow<T>(length, periodic);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBartlettWindow<T>(int length, bool periodic = true)
    {
        if (TryWindowOnDevice<T>("TensorBartlettWindow", length, periodic, 3, 0, 0) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBartlettWindow: no device kernel", null);
        return base.TensorBartlettWindow<T>(length, periodic);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorKaiserWindow<T>(int length, bool periodic = true, double beta = 12)
    {
        if (TryWindowOnDevice<T>("TensorKaiserWindow", length, periodic, 4, beta, 0) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorKaiserWindow: no device kernel", null);
        return base.TensorKaiserWindow<T>(length, periodic, beta);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTrilIndices<T>(int row, int col, int offset = 0)
    {
        if (TryTriIndicesOnDevice<T>("TensorTrilIndices", row, col, offset, lower: true) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorTrilIndices: no device kernel", null);
        return base.TensorTrilIndices<T>(row, col, offset);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTriuIndices<T>(int row, int col, int offset = 0)
    {
        if (TryTriIndicesOnDevice<T>("TensorTriuIndices", row, col, offset, lower: false) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorTriuIndices: no device kernel", null);
        return base.TensorTriuIndices<T>(row, col, offset);
    }

    /// <inheritdoc/>
    public override Tensor<T>[] TensorUnravelIndex<T>(Tensor<T> indices, int[] shape)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryUnravelIndexOnDevice(indices, shape) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorUnravelIndex: no device kernel", null);
        return base.TensorUnravelIndex<T>(indices, shape);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorCombinations<T>(Tensor<T> tensor, int r = 2, bool withReplacement = false)
        => base.TensorCombinations<T>(tensor, r, withReplacement);

    /// <inheritdoc/>
    public override Tensor<T>[] TensorChunk<T>(Tensor<T> tensor, int chunks, int dim = 0)
        => base.TensorChunk<T>(tensor, chunks, dim);

    /// <inheritdoc/>
    public override Tensor<T>[] TensorSplitWithSizes<T>(Tensor<T> tensor, int[] sizes, int dim = 0)
        => base.TensorSplitWithSizes<T>(tensor, sizes, dim);

    /// <inheritdoc/>
    public override Tensor<T> TensorUnflatten<T>(Tensor<T> tensor, int dim, int[] sizes)
        => base.TensorUnflatten<T>(tensor, dim, sizes);

    /// <inheritdoc/>
    public override Tensor<T> TensorSelect<T>(Tensor<T> tensor, int dim, int index)
        => base.TensorSelect<T>(tensor, dim, index);

    /// <inheritdoc/>
    public override Tensor<T> TensorViewAs<T>(Tensor<T> tensor, Tensor<T> other)
        => base.TensorViewAs<T>(tensor, other);

    /// <inheritdoc/>
    public override Tensor<T> TensorSumToSize<T>(Tensor<T> tensor, int[] size)
        => base.TensorSumToSize<T>(tensor, size);

    /// <inheritdoc/>
    /// <remarks>Runs on the device on CUDA (#1112); other backends fall through to the host and record a
    /// <see cref="GpuLaunchProbe"/> fallback (follow-ups #1124-#1128).</remarks>
    public override Tensor<T> TensorQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false, QuantileInterpolation interpolation = QuantileInterpolation.Linear)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryQuantileOnDevice("TensorQuantile", tensor, q, dim, keepDim, interpolation, ignoreNan: false) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorQuantile: no device kernel", null);
        return base.TensorQuantile<T>(tensor, q, dim, keepDim, interpolation);
    }

    /// <inheritdoc/>
    /// <remarks>Runs on the device on CUDA (#1112); other backends fall through to the host and record a
    /// <see cref="GpuLaunchProbe"/> fallback (follow-ups #1124-#1128).</remarks>
    public override Tensor<T> TensorNanQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false, QuantileInterpolation interpolation = QuantileInterpolation.Linear)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryQuantileOnDevice("TensorNanQuantile", tensor, q, dim, keepDim, interpolation, ignoreNan: true) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorNanQuantile: no device kernel", null);
        return base.TensorNanQuantile<T>(tensor, q, dim, keepDim, interpolation);
    }

    /// <inheritdoc/>
    public override (Tensor<T> Std, Tensor<T> Mean) TensorStdMean<T>(Tensor<T> tensor, int[]? axes = null, int correction = 1, bool keepDims = false)
        => base.TensorStdMean<T>(tensor, axes, correction, keepDims);

    /// <inheritdoc/>
    public override (Tensor<T> Var, Tensor<T> Mean) TensorVarMean<T>(Tensor<T> tensor, int[]? axes = null, int correction = 1, bool keepDims = false)
        => base.TensorVarMean<T>(tensor, axes, correction, keepDims);

    /// <inheritdoc/>
    public override Tensor<T> TensorCov<T>(Tensor<T> tensor, int correction = 1)
        => base.TensorCov<T>(tensor, correction);

    /// <inheritdoc/>
    public override Tensor<T> TensorCorrcoef<T>(Tensor<T> tensor)
        => base.TensorCorrcoef<T>(tensor);

    /// <inheritdoc/>
    public override Tensor<T> TensorDiff<T>(Tensor<T> tensor, int n = 1, int dim = -1)
        => base.TensorDiff<T>(tensor, n, dim);

    /// <inheritdoc/>
    public override Tensor<T> TensorTrapezoid<T>(Tensor<T> y, double dx = 1, int dim = -1)
        => base.TensorTrapezoid<T>(y, dx, dim);

    /// <inheritdoc/>
    public override Tensor<T> TensorCumulativeTrapezoid<T>(Tensor<T> y, double dx = 1, int dim = -1)
        => base.TensorCumulativeTrapezoid<T>(y, dx, dim);

    // ---- CpuEngine.TorchGridSample.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorGridSample3D<T>(Tensor<T> input, Tensor<T> grid, GridSampleMode mode = GridSampleMode.Bilinear, GridSamplePadding padding = GridSamplePadding.Zeros, bool alignCorners = false)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryGridSample3DOnDevice(input, grid, mode, padding, alignCorners) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorGridSample3D: no device kernel", null);
        return base.TensorGridSample3D<T>(input, grid, mode, padding, alignCorners);
    }

    // ---- CpuEngine.TorchMisc.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorCelu<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        if (alpha != 0 && TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Celu, (float)alpha, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorCelu: no device kernel", null);
        return base.TensorCelu<T>(tensor, alpha);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHardtanh<T>(Tensor<T> tensor, double minValue = -1.0, double maxValue = 1.0)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Hardtanh, (float)minValue, (float)maxValue) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorHardtanh: no device kernel", null);
        return base.TensorHardtanh<T>(tensor, minValue, maxValue);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogSigmoid<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.LogSigmoid, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLogSigmoid: no device kernel", null);
        return base.TensorLogSigmoid<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSoftsign<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Softsign, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorSoftsign: no device kernel", null);
        return base.TensorSoftsign<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSoftmin<T>(Tensor<T> tensor, int axis = -1)
        => base.TensorSoftmin<T>(tensor, axis);

    /// <inheritdoc/>
    public override Tensor<T> TensorRrelu<T>(Tensor<T> tensor, double lower = 1.0 / 8, double upper = 1.0 / 3, bool training = false, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryRreluOnDevice(tensor, lower, upper, training, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorRrelu: no device kernel", null);
        return base.TensorRrelu<T>(tensor, lower, upper, training, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRsub<T>(Tensor<T> input, Tensor<T> other, double alpha = 1)
        => base.TensorRsub<T>(input, other, alpha);

    /// <inheritdoc/>
    public override Tensor<T> TensorAddcmul<T>(Tensor<T> input, Tensor<T> tensor1, Tensor<T> tensor2, double value = 1)
        => base.TensorAddcmul<T>(input, tensor1, tensor2, value);

    /// <inheritdoc/>
    public override Tensor<T> TensorAddcdiv<T>(Tensor<T> input, Tensor<T> tensor1, Tensor<T> tensor2, double value = 1)
        => base.TensorAddcdiv<T>(input, tensor1, tensor2, value);

    /// <inheritdoc/>
    public override Tensor<T> TensorAddmv<T>(Tensor<T> input, Tensor<T> mat, Tensor<T> vec, double beta = 1, double alpha = 1)
        => base.TensorAddmv<T>(input, mat, vec, beta, alpha);

    /// <inheritdoc/>
    public override Tensor<T> TensorAddr<T>(Tensor<T> input, Tensor<T> vec1, Tensor<T> vec2, double beta = 1, double alpha = 1)
        => base.TensorAddr<T>(input, vec1, vec2, beta, alpha);

    /// <inheritdoc/>
    public override Tensor<T> TensorBaddbmm<T>(Tensor<T> input, Tensor<T> batch1, Tensor<T> batch2, double beta = 1, double alpha = 1)
        => base.TensorBaddbmm<T>(input, batch1, batch2, beta, alpha);

    /// <inheritdoc/>
    public override Tensor<T> TensorAddbmm<T>(Tensor<T> input, Tensor<T> batch1, Tensor<T> batch2, double beta = 1, double alpha = 1)
        => base.TensorAddbmm<T>(input, batch1, batch2, beta, alpha);

    /// <inheritdoc/>
    public override Tensor<T> TensorMsort<T>(Tensor<T> tensor)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryMsortOnDevice(tensor) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorMsort: no device kernel", null);
        return base.TensorMsort<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorDiagflat<T>(Tensor<T> tensor, int offset = 0)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryDiagflatOnDevice(tensor, offset) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorDiagflat: no device kernel", null);
        return base.TensorDiagflat<T>(tensor, offset);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorDiagonalScatter<T>(Tensor<T> input, Tensor<T> src, int offset = 0, int dim1 = 0, int dim2 = 1)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryDiagonalScatterOnDevice(input, src, offset, dim1, dim2) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorDiagonalScatter: no device kernel", null);
        return base.TensorDiagonalScatter<T>(input, src, offset, dim1, dim2);
    }

    // ---- CpuEngine.TorchPool.cs ----

    // ---- CpuEngine.TorchRandom.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorBernoulli<T>(Tensor<T> probabilities, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryValidatedSampleOnDevice(probabilities, null, seed, "TensorBernoulli") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBernoulli: no device kernel", null);
        return base.TensorBernoulli<T>(probabilities, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBinomial<T>(Tensor<T> count, Tensor<T> probability, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryValidatedSampleOnDevice(count, probability, seed, "TensorBinomial") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBinomial: no device kernel", null);
        return base.TensorBinomial<T>(count, probability, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorPoisson<T>(Tensor<T> rates, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryValidatedSampleOnDevice(rates, null, seed, "TensorPoisson") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorPoisson: no device kernel", null);
        return base.TensorPoisson<T>(rates, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNormal<T>(Tensor<T> mean, Tensor<T> std, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        // mean + std * epsilon with a device epsilon: the reparameterized form, so gradients reach mean and std.
        if (mean is not null && std is not null && TrySampleOnDevice<T>("TensorNormal", mean._shape, 5, 0, 0, seed) is { } epsilon)
            return TensorAdd(mean, TensorMultiply(std, epsilon));
        GpuLaunchProbe.OnFallback("TensorNormal: noise drawn on the host", null);
        return base.TensorNormal<T>(mean!, std!, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorUniform<T>(int[] shape, double low = 0, double high = 1, int? seed = null)
    {
        if (shape is not null && TrySampleOnDevice<T>("TensorUniform", shape, 0, low, high, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorUniform: no device kernel", null);
        return base.TensorUniform<T>(shape!, low, high, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorCauchy<T>(int[] shape, double median = 0, double sigma = 1, int? seed = null)
    {
        if (shape is not null && TrySampleOnDevice<T>("TensorCauchy", shape, 1, median, sigma, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorCauchy: no device kernel", null);
        return base.TensorCauchy<T>(shape!, median, sigma, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorExponential<T>(int[] shape, double rate = 1, int? seed = null)
    {
        if (shape is not null && rate > 0 && TrySampleOnDevice<T>("TensorExponential", shape, 2, rate, 0, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorExponential: no device kernel", null);
        return base.TensorExponential<T>(shape!, rate, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGeometric<T>(int[] shape, double p, int? seed = null)
    {
        if (shape is not null && p > 0 && p <= 1 && TrySampleOnDevice<T>("TensorGeometric", shape, 3, p, 0, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorGeometric: no device kernel", null);
        return base.TensorGeometric<T>(shape!, p, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogNormal<T>(int[] shape, double mean = 1, double std = 2, int? seed = null)
    {
        if (shape is not null && TrySampleOnDevice<T>("TensorLogNormal", shape, 4, mean, std, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLogNormal: no device kernel", null);
        return base.TensorLogNormal<T>(shape!, mean, std, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMultinomial<T>(Tensor<T> probabilities, int numSamples, bool replacement = false, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryMultinomialOnDevice(probabilities, numSamples, replacement, seed) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorMultinomial: no device kernel", null);
        return base.TensorMultinomial<T>(probabilities, numSamples, replacement, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (TryDropoutOnDevice(tensor, p, training, seed, alpha: true, spatialDims: -1) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorAlphaDropout: noise drawn on the host", null);
        return base.TensorAlphaDropout<T>(tensor, p, training, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFeatureAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor is not null && TryDropoutOnDevice(tensor, p, training, seed, alpha: true, spatialDims: Math.Max(0, tensor.Rank - 2)) is { } onDevice)
            return onDevice;
        GpuLaunchProbe.OnFallback("TensorFeatureAlphaDropout: noise drawn on the host", null);
        return base.TensorFeatureAlphaDropout<T>(tensor!, p, training, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChannelDropout<T>(Tensor<T> tensor, double p, bool training, int spatialDims, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor is not null && spatialDims >= 0 && spatialDims < tensor.Rank
            && TryDropoutOnDevice(tensor, p, training, seed, alpha: false, spatialDims) is { } onDevice)
            return onDevice;
        GpuLaunchProbe.OnFallback("TensorChannelDropout: noise drawn on the host", null);
        return base.TensorChannelDropout<T>(tensor!, p, training, spatialDims, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBilinear<T>(Tensor<T> input1, Tensor<T> input2, Tensor<T> weight, Tensor<T>? bias = null)
        => base.TensorBilinear<T>(input1, input2, weight, bias);

    /// <inheritdoc/>
    public override Tensor<T> TensorChannelShuffle<T>(Tensor<T> tensor, int groups)
        => base.TensorChannelShuffle<T>(tensor, groups);

    /// <inheritdoc/>
    public override Tensor<T> TensorPixelUnshuffle<T>(Tensor<T> tensor, int downscaleFactor)
        => base.TensorPixelUnshuffle<T>(tensor, downscaleFactor);

    /// <inheritdoc/>
    public override Tensor<T> TensorLocalResponseNorm<T>(Tensor<T> tensor, int size, double alpha = 1e-4, double beta = 0.75, double k = 1)
        => base.TensorLocalResponseNorm<T>(tensor, size, alpha, beta, k);

    /// <inheritdoc/>
    public override Tensor<T> TensorSoftMarginLoss<T>(Tensor<T> input, Tensor<T> target, LossReduction reduction = LossReduction.Mean)
        => base.TensorSoftMarginLoss<T>(input, target, reduction);

    /// <inheritdoc/>
    public override Tensor<T> TensorRenorm<T>(Tensor<T> tensor, double p, int dim, double maxNorm)
        => base.TensorRenorm<T>(tensor, p, dim, maxNorm);

    /// <inheritdoc/>
    public override Tensor<T> TensorNormExceptDim<T>(Tensor<T> tensor, double pow = 2, int dim = 0)
        => base.TensorNormExceptDim<T>(tensor, pow, dim);

    /// <inheritdoc/>
    public override Tensor<T>[] TensorGradient<T>(Tensor<T> tensor, double spacing = 1, int[]? dims = null)
        => base.TensorGradient<T>(tensor, spacing, dims);

    /// <inheritdoc/>
    public override Tensor<T> TensorPadSequence<T>(Tensor<T>[] sequences, bool batchFirst = false, double paddingValue = 0)
        => base.TensorPadSequence<T>(sequences, batchFirst, paddingValue);

    /// <inheritdoc/>
    public override Tensor<int> TensorNonzeroStatic<T>(Tensor<T> tensor, int size, int fillValue = -1)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (TryNonzeroStaticOnDevice(tensor, size, fillValue) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorNonzeroStatic: no device kernel", null);
        return base.TensorNonzeroStatic<T>(tensor, size, fillValue);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorUniqueDim<T>(Tensor<T> tensor, int dim)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.DataDependentOutputShape);
        if (TryUniqueDimOnDevice(tensor, dim) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorUniqueDim: no device kernel", null);
        return base.TensorUniqueDim<T>(tensor, dim);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIndexReduce<T>(Tensor<T> tensor, int dim, Tensor<int> index, Tensor<T> source, ScatterReduceMode reduce, bool includeSelf = true)
        => base.TensorIndexReduce<T>(tensor, dim, index, source, reduce, includeSelf);

    /// <inheritdoc/>
    public override Tensor<T> TensorConvTranspose1D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int outputPadding = 0)
        => base.TensorConvTranspose1D<T>(input, kernel, stride, padding, outputPadding);

    /// <inheritdoc/>
    public override Tensor<T> TensorConvTbc<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias, int pad = 0)
        => base.TensorConvTbc<T>(input, weight, bias, pad);

    // ---- CpuEngine.TorchRnn.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorRnnCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh = null, Tensor<T>? bHh = null, RnnCellType cell = RnnCellType.RnnTanh)
        => base.TensorRnnCell<T>(input, hidden, wIh, wHh, bIh, bHh, cell);

    /// <inheritdoc/>
    public override (Tensor<T> Hidden, Tensor<T> Cell) TensorLstmCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> cell, Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh = null, Tensor<T>? bHh = null)
        => base.TensorLstmCell<T>(input, hidden, cell, wIh, wHh, bIh, bHh);

    /// <inheritdoc/>
    public override Tensor<T> TensorGruCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh = null, Tensor<T>? bHh = null)
        => base.TensorGruCell<T>(input, hidden, wIh, wHh, bIh, bHh);

    /// <inheritdoc/>
    public override (Tensor<T> Output, Tensor<T> Hidden) TensorRecurrent<T>(RnnCellType cell, Tensor<T> input, Tensor<T>? h0, IReadOnlyList<Tensor<T>> weights, bool hasBiases, int numLayers, double dropout = 0, bool training = false, bool bidirectional = false, bool batchFirst = false)
        => base.TensorRecurrent<T>(cell, input, h0, weights, hasBiases, numLayers, dropout, training, bidirectional, batchFirst);

    // ---- CpuEngine.TorchSpecial.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorErf<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Erf, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorErf: no device kernel", null);
        return base.TensorErf<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogit<T>(Tensor<T> tensor, double? eps = null)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Logit, eps.HasValue ? (float)eps.Value : float.NegativeInfinity, eps.HasValue ? (float)(1 - eps.Value) : float.PositiveInfinity) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLogit: no device kernel", null);
        return base.TensorLogit<T>(tensor, eps);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSinc<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Sinc, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorSinc: no device kernel", null);
        return base.TensorSinc<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorDeg2Rad<T>(Tensor<T> tensor)
        => TensorMultiplyScalar(tensor, MathHelper.GetNumericOperations<T>().FromDouble(Math.PI / 180));

    /// <inheritdoc/>
    public override Tensor<T> TensorRad2Deg<T>(Tensor<T> tensor)
        => TensorMultiplyScalar(tensor, MathHelper.GetNumericOperations<T>().FromDouble(180 / Math.PI));

    /// <inheritdoc/>
    public override Tensor<T> TensorPositive<T>(Tensor<T> tensor)
        => base.TensorPositive<T>(tensor);

    /// <inheritdoc/>
    public override Tensor<T> TensorSignbit<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Signbit, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorSignbit: no device kernel", null);
        return base.TensorSignbit<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIsPosInf<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.IsPosInf, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorIsPosInf: no device kernel", null);
        return base.TensorIsPosInf<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIsNegInf<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.IsNegInf, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorIsNegInf: no device kernel", null);
        return base.TensorIsNegInf<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIsReal<T>(Tensor<T> tensor)
    {
        if (TryFillLikeOnDevice("TensorIsReal", tensor, 1) is { } onDevice) return onDevice;   // every real element is real
        GpuLaunchProbe.OnFallback("TensorIsReal: no device kernel", null);
        return base.TensorIsReal<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGreaterEqual<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryParityBinaryOnDevice(a, b, TorchParityBinaryOp.GreaterEqual) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorGreaterEqual: no device kernel", null);
        return base.TensorGreaterEqual<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLessEqual<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryParityBinaryOnDevice(a, b, TorchParityBinaryOp.LessEqual) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLessEqual: no device kernel", null);
        return base.TensorLessEqual<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHeaviside<T>(Tensor<T> tensor, Tensor<T> values)
    {
        if (TryParityBinaryOnDevice(tensor, values, TorchParityBinaryOp.Heaviside) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorHeaviside: no device kernel", null);
        return base.TensorHeaviside<T>(tensor, values);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFmax<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryParityBinaryOnDevice(a, b, TorchParityBinaryOp.Fmax) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorFmax: no device kernel", null);
        return base.TensorFmax<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFmin<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryParityBinaryOnDevice(a, b, TorchParityBinaryOp.Fmin) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorFmin: no device kernel", null);
        return base.TensorFmin<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFloorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorFloorDivide", a, b, 7) is { } onInt32) return onInt32;
        if (TryParityBinaryOnDevice(a, b, TorchParityBinaryOp.FloorDivide) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorFloorDivide: no device kernel", null);
        return base.TensorFloorDivide<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGcd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorGcd", a, b, 5) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorGcd: no device kernel", null);
        return base.TensorGcd<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLcm<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorLcm", a, b, 6) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLcm: no device kernel", null);
        return base.TensorLcm<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseAnd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorBitwiseAnd", a, b, 0) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBitwiseAnd: no device kernel", null);
        return base.TensorBitwiseAnd<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseOr<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorBitwiseOr", a, b, 1) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBitwiseOr: no device kernel", null);
        return base.TensorBitwiseOr<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseXor<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorBitwiseXor", a, b, 2) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBitwiseXor: no device kernel", null);
        return base.TensorBitwiseXor<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseNot<T>(Tensor<T> tensor)
    {
        if (TryInt32OnDevice("TensorBitwiseNot", tensor, null, -1) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBitwiseNot: no device kernel", null);
        return base.TensorBitwiseNot<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseLeftShift<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorBitwiseLeftShift", a, b, 3) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBitwiseLeftShift: no device kernel", null);
        return base.TensorBitwiseLeftShift<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseRightShift<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryInt32OnDevice("TensorBitwiseRightShift", a, b, 4) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBitwiseRightShift: no device kernel", null);
        return base.TensorBitwiseRightShift<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNanSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
        => base.TensorNanSum<T>(tensor, axes, keepDims);

    /// <inheritdoc/>
    public override Tensor<T> TensorNanMean<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
        => base.TensorNanMean<T>(tensor, axes, keepDims);

    /// <inheritdoc/>
    public override Tensor<T> TensorIgamma<T>(Tensor<T> a, Tensor<T> x)
    {
        if (TryParityBinaryOnDevice(a, x, TorchParityBinaryOp.Igamma) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorIgamma: no device kernel", null);
        return base.TensorIgamma<T>(a, x);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIgammac<T>(Tensor<T> a, Tensor<T> x)
    {
        if (TryParityBinaryOnDevice(a, x, TorchParityBinaryOp.Igammac) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorIgammac: no device kernel", null);
        return base.TensorIgammac<T>(a, x);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMvlgamma<T>(Tensor<T> tensor, int p)
    {
        // p < 1 stays on the CPU path, which rejects it; p travels as a float, exact up to 2^24.
        if (p >= 1 && p <= (1 << 24) && TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Mvlgamma, p, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorMvlgamma: no device kernel", null);
        return base.TensorMvlgamma<T>(tensor, p);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorEntr<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Entr, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorEntr: no device kernel", null);
        return base.TensorEntr<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorErfcx<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Erfcx, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorErfcx: no device kernel", null);
        return base.TensorErfcx<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNdtr<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Ndtr, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorNdtr: no device kernel", null);
        return base.TensorNdtr<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogNdtr<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.LogNdtr, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLogNdtr: no device kernel", null);
        return base.TensorLogNdtr<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNdtri<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.Ndtri, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorNdtri: no device kernel", null);
        return base.TensorNdtri<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselJ0<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.BesselJ0, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBesselJ0: no device kernel", null);
        return base.TensorBesselJ0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselJ1<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.BesselJ1, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBesselJ1: no device kernel", null);
        return base.TensorBesselJ1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselY0<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.BesselY0, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBesselY0: no device kernel", null);
        return base.TensorBesselY0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselY1<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.BesselY1, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorBesselY1: no device kernel", null);
        return base.TensorBesselY1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselI0<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.ModifiedBesselI0, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorModifiedBesselI0: no device kernel", null);
        return base.TensorModifiedBesselI0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselI1<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.ModifiedBesselI1, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorModifiedBesselI1: no device kernel", null);
        return base.TensorModifiedBesselI1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselK0<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.ModifiedBesselK0, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorModifiedBesselK0: no device kernel", null);
        return base.TensorModifiedBesselK0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselK1<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.ModifiedBesselK1, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorModifiedBesselK1: no device kernel", null);
        return base.TensorModifiedBesselK1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorScaledModifiedBesselK0<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.ScaledModifiedBesselK0, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorScaledModifiedBesselK0: no device kernel", null);
        return base.TensorScaledModifiedBesselK0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorScaledModifiedBesselK1<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.ScaledModifiedBesselK1, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorScaledModifiedBesselK1: no device kernel", null);
        return base.TensorScaledModifiedBesselK1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSphericalBesselJ0<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.SphericalBesselJ0, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorSphericalBesselJ0: no device kernel", null);
        return base.TensorSphericalBesselJ0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAiryAi<T>(Tensor<T> tensor)
    {
        if (TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.AiryAi, 0f, 0f) is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorAiryAi: no device kernel", null);
        return base.TensorAiryAi<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 0, false, "TensorChebyshevPolynomialT") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialT: no device kernel", null);
        return base.TensorChebyshevPolynomialT<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 1, false, "TensorChebyshevPolynomialU") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialU: no device kernel", null);
        return base.TensorChebyshevPolynomialU<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 2, false, "TensorChebyshevPolynomialV") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialV: no device kernel", null);
        return base.TensorChebyshevPolynomialV<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 3, false, "TensorChebyshevPolynomialW") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialW: no device kernel", null);
        return base.TensorChebyshevPolynomialW<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 0, true, "TensorShiftedChebyshevPolynomialT") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialT: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialT<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 1, true, "TensorShiftedChebyshevPolynomialU") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialU: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialU<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 2, true, "TensorShiftedChebyshevPolynomialV") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialV: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialV<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 3, true, "TensorShiftedChebyshevPolynomialW") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialW: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialW<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHermitePolynomialH<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 4, false, "TensorHermitePolynomialH") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorHermitePolynomialH: no device kernel", null);
        return base.TensorHermitePolynomialH<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHermitePolynomialHe<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 5, false, "TensorHermitePolynomialHe") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorHermitePolynomialHe: no device kernel", null);
        return base.TensorHermitePolynomialHe<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLaguerrePolynomialL<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 6, false, "TensorLaguerrePolynomialL") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLaguerrePolynomialL: no device kernel", null);
        return base.TensorLaguerrePolynomialL<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLegendrePolynomialP<T>(Tensor<T> x, Tensor<T> n)
    {
        if (TryPolynomialOnDevice(x, n, 7, false, "TensorLegendrePolynomialP") is { } onDevice) return onDevice;
        GpuLaunchProbe.OnFallback("TensorLegendrePolynomialP: no device kernel", null);
        return base.TensorLegendrePolynomialP<T>(x, n);
    }
}
