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
/// The remaining parity ops compute on the host. They are long-tail PyTorch surface (special functions,
/// orthogonal polynomials, window functions, samplers, index construction, pooling variants, complex
/// conversions, quantiles, volumetric grid sampling), none of which sits on the training paths this engine keeps
/// device-resident. Each one is overridden here to fall through to the host implementation EXPLICITLY: every
/// call records a <see cref="GpuLaunchProbe"/> fallback (<c>"&lt;op&gt;: guard or route declined"</c> in
/// <see cref="GpuLaunchProbe.Fallbacks"/>), so a GPU workload that reaches one is visible in the residency
/// diagnostics rather than silently running at host speed. Native kernels on the six backends are tracked in
/// https://github.com/ooples/AiDotNet.Tensors/issues/1112; when one lands, its override here is replaced by the
/// device dispatch.
/// </para>
/// </remarks>
public partial class DirectGpuTensorEngine
{





    // ---- CpuEngine.TorchComplex.cs ----

    /// <inheritdoc/>
    public override Tensor<Complex<T>> TensorComplex<T>(Tensor<T> real, Tensor<T> imag)
    {
        GpuLaunchProbe.OnFallback("TensorComplex: no device kernel", null);
        return base.TensorComplex<T>(real, imag);
    }

    /// <inheritdoc/>
    public override Tensor<Complex<T>> TensorViewAsComplex<T>(Tensor<T> input)
    {
        GpuLaunchProbe.OnFallback("TensorViewAsComplex: no device kernel", null);
        return base.TensorViewAsComplex<T>(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAngle<T>(Tensor<T> input)
    {
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
    public override Tensor<T> TensorAll<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        GpuLaunchProbe.OnFallback("TensorAll: no device kernel", null);
        return base.TensorAll<T>(tensor, axes, keepDims);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAny<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        GpuLaunchProbe.OnFallback("TensorAny: no device kernel", null);
        return base.TensorAny<T>(tensor, axes, keepDims);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorArange<T>(double start, double end, double step = 1)
    {
        GpuLaunchProbe.OnFallback("TensorArange: no device kernel", null);
        return base.TensorArange<T>(start, end, step);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRange<T>(double start, double end, double step = 1)
    {
        GpuLaunchProbe.OnFallback("TensorRange: no device kernel", null);
        return base.TensorRange<T>(start, end, step);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogspace<T>(double start, double end, int steps, double logBase = 10)
    {
        GpuLaunchProbe.OnFallback("TensorLogspace: no device kernel", null);
        return base.TensorLogspace<T>(start, end, steps, logBase);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorZerosLike<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorZerosLike: no device kernel", null);
        return base.TensorZerosLike<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorOnesLike<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorOnesLike: no device kernel", null);
        return base.TensorOnesLike<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFullLike<T>(Tensor<T> tensor, T value)
    {
        GpuLaunchProbe.OnFallback("TensorFullLike: no device kernel", null);
        return base.TensorFullLike<T>(tensor, value);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorEmptyLike<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorEmptyLike: no device kernel", null);
        return base.TensorEmptyLike<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandLike<T>(Tensor<T> tensor, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorRandLike: no device kernel", null);
        return base.TensorRandLike<T>(tensor, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandnLike<T>(Tensor<T> tensor, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorRandnLike: no device kernel", null);
        return base.TensorRandnLike<T>(tensor, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandint<T>(long low, long high, int[] shape, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorRandint: no device kernel", null);
        return base.TensorRandint<T>(low, high, shape, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandintLike<T>(Tensor<T> tensor, long low, long high, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorRandintLike: no device kernel", null);
        return base.TensorRandintLike<T>(tensor, low, high, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRandperm<T>(int n, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorRandperm: no device kernel", null);
        return base.TensorRandperm<T>(n, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHannWindow<T>(int length, bool periodic = true)
    {
        GpuLaunchProbe.OnFallback("TensorHannWindow: no device kernel", null);
        return base.TensorHannWindow<T>(length, periodic);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHammingWindow<T>(int length, bool periodic = true, double alpha = 0.54, double beta = 0.46)
    {
        GpuLaunchProbe.OnFallback("TensorHammingWindow: no device kernel", null);
        return base.TensorHammingWindow<T>(length, periodic, alpha, beta);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBlackmanWindow<T>(int length, bool periodic = true)
    {
        GpuLaunchProbe.OnFallback("TensorBlackmanWindow: no device kernel", null);
        return base.TensorBlackmanWindow<T>(length, periodic);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBartlettWindow<T>(int length, bool periodic = true)
    {
        GpuLaunchProbe.OnFallback("TensorBartlettWindow: no device kernel", null);
        return base.TensorBartlettWindow<T>(length, periodic);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorKaiserWindow<T>(int length, bool periodic = true, double beta = 12)
    {
        GpuLaunchProbe.OnFallback("TensorKaiserWindow: no device kernel", null);
        return base.TensorKaiserWindow<T>(length, periodic, beta);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTrilIndices<T>(int row, int col, int offset = 0)
    {
        GpuLaunchProbe.OnFallback("TensorTrilIndices: no device kernel", null);
        return base.TensorTrilIndices<T>(row, col, offset);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTriuIndices<T>(int row, int col, int offset = 0)
    {
        GpuLaunchProbe.OnFallback("TensorTriuIndices: no device kernel", null);
        return base.TensorTriuIndices<T>(row, col, offset);
    }

    /// <inheritdoc/>
    public override Tensor<T>[] TensorUnravelIndex<T>(Tensor<T> indices, int[] shape)
    {
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
    public override Tensor<T> TensorQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false, QuantileInterpolation interpolation = QuantileInterpolation.Linear)
    {
        GpuLaunchProbe.OnFallback("TensorQuantile: no device kernel", null);
        return base.TensorQuantile<T>(tensor, q, dim, keepDim, interpolation);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNanQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false, QuantileInterpolation interpolation = QuantileInterpolation.Linear)
    {
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
        GpuLaunchProbe.OnFallback("TensorGridSample3D: no device kernel", null);
        return base.TensorGridSample3D<T>(input, grid, mode, padding, alignCorners);
    }

    // ---- CpuEngine.TorchMisc.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorCelu<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        GpuLaunchProbe.OnFallback("TensorCelu: no device kernel", null);
        return base.TensorCelu<T>(tensor, alpha);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHardtanh<T>(Tensor<T> tensor, double minValue = -1.0, double maxValue = 1.0)
    {
        GpuLaunchProbe.OnFallback("TensorHardtanh: no device kernel", null);
        return base.TensorHardtanh<T>(tensor, minValue, maxValue);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogSigmoid<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorLogSigmoid: no device kernel", null);
        return base.TensorLogSigmoid<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSoftsign<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorSoftsign: no device kernel", null);
        return base.TensorSoftsign<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSoftmin<T>(Tensor<T> tensor, int axis = -1)
        => base.TensorSoftmin<T>(tensor, axis);

    /// <inheritdoc/>
    public override Tensor<T> TensorRrelu<T>(Tensor<T> tensor, double lower = 1.0 / 8, double upper = 1.0 / 3, bool training = false, int? seed = null)
    {
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
        GpuLaunchProbe.OnFallback("TensorMsort: no device kernel", null);
        return base.TensorMsort<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorDiagflat<T>(Tensor<T> tensor, int offset = 0)
    {
        GpuLaunchProbe.OnFallback("TensorDiagflat: no device kernel", null);
        return base.TensorDiagflat<T>(tensor, offset);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorDiagonalScatter<T>(Tensor<T> input, Tensor<T> src, int offset = 0, int dim1 = 0, int dim2 = 1)
    {
        GpuLaunchProbe.OnFallback("TensorDiagonalScatter: no device kernel", null);
        return base.TensorDiagonalScatter<T>(input, src, offset, dim1, dim2);
    }

    // ---- CpuEngine.TorchPool.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveAvgPool1D<T>(Tensor<T> input, int outputSize)
    {
        GpuLaunchProbe.OnFallback("TensorAdaptiveAvgPool1D: no device kernel", null);
        return base.TensorAdaptiveAvgPool1D<T>(input, outputSize);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveAvgPool3D<T>(Tensor<T> input, int[] outputSize)
    {
        GpuLaunchProbe.OnFallback("TensorAdaptiveAvgPool3D: no device kernel", null);
        return base.TensorAdaptiveAvgPool3D<T>(input, outputSize);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveMaxPool1D<T>(Tensor<T> input, int outputSize)
    {
        GpuLaunchProbe.OnFallback("TensorAdaptiveMaxPool1D: no device kernel", null);
        return base.TensorAdaptiveMaxPool1D<T>(input, outputSize);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveMaxPool3D<T>(Tensor<T> input, int[] outputSize)
    {
        GpuLaunchProbe.OnFallback("TensorAdaptiveMaxPool3D: no device kernel", null);
        return base.TensorAdaptiveMaxPool3D<T>(input, outputSize);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLpPool<T>(Tensor<T> input, double power, int[] kernelSize, int[]? stride = null)
    {
        GpuLaunchProbe.OnFallback("TensorLpPool: no device kernel", null);
        return base.TensorLpPool<T>(input, power, kernelSize, stride);
    }

    // ---- CpuEngine.TorchRandom.cs ----

    /// <inheritdoc/>
    public override Tensor<T> TensorBernoulli<T>(Tensor<T> probabilities, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorBernoulli: no device kernel", null);
        return base.TensorBernoulli<T>(probabilities, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBinomial<T>(Tensor<T> count, Tensor<T> probability, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorBinomial: no device kernel", null);
        return base.TensorBinomial<T>(count, probability, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorPoisson<T>(Tensor<T> rates, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorPoisson: no device kernel", null);
        return base.TensorPoisson<T>(rates, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNormal<T>(Tensor<T> mean, Tensor<T> std, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorNormal: noise drawn on the host", null);
        return base.TensorNormal<T>(mean, std, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorUniform<T>(int[] shape, double low = 0, double high = 1, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorUniform: no device kernel", null);
        return base.TensorUniform<T>(shape, low, high, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorCauchy<T>(int[] shape, double median = 0, double sigma = 1, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorCauchy: no device kernel", null);
        return base.TensorCauchy<T>(shape, median, sigma, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorExponential<T>(int[] shape, double rate = 1, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorExponential: no device kernel", null);
        return base.TensorExponential<T>(shape, rate, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGeometric<T>(int[] shape, double p, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorGeometric: no device kernel", null);
        return base.TensorGeometric<T>(shape, p, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogNormal<T>(int[] shape, double mean = 1, double std = 2, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorLogNormal: no device kernel", null);
        return base.TensorLogNormal<T>(shape, mean, std, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMultinomial<T>(Tensor<T> probabilities, int numSamples, bool replacement = false, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorMultinomial: no device kernel", null);
        return base.TensorMultinomial<T>(probabilities, numSamples, replacement, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorAlphaDropout: noise drawn on the host", null);
        return base.TensorAlphaDropout<T>(tensor, p, training, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFeatureAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorFeatureAlphaDropout: noise drawn on the host", null);
        return base.TensorFeatureAlphaDropout<T>(tensor, p, training, seed);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChannelDropout<T>(Tensor<T> tensor, double p, bool training, int spatialDims, int? seed = null)
    {
        GpuLaunchProbe.OnFallback("TensorChannelDropout: noise drawn on the host", null);
        return base.TensorChannelDropout<T>(tensor, p, training, spatialDims, seed);
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
        GpuLaunchProbe.OnFallback("TensorNonzeroStatic: no device kernel", null);
        return base.TensorNonzeroStatic<T>(tensor, size, fillValue);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorUniqueDim<T>(Tensor<T> tensor, int dim)
    {
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
        GpuLaunchProbe.OnFallback("TensorErf: no device kernel", null);
        return base.TensorErf<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogit<T>(Tensor<T> tensor, double? eps = null)
    {
        GpuLaunchProbe.OnFallback("TensorLogit: no device kernel", null);
        return base.TensorLogit<T>(tensor, eps);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSinc<T>(Tensor<T> tensor)
    {
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
        GpuLaunchProbe.OnFallback("TensorSignbit: no device kernel", null);
        return base.TensorSignbit<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIsPosInf<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorIsPosInf: no device kernel", null);
        return base.TensorIsPosInf<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIsNegInf<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorIsNegInf: no device kernel", null);
        return base.TensorIsNegInf<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIsReal<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorIsReal: no device kernel", null);
        return base.TensorIsReal<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGreaterEqual<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorGreaterEqual: no device kernel", null);
        return base.TensorGreaterEqual<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLessEqual<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorLessEqual: no device kernel", null);
        return base.TensorLessEqual<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHeaviside<T>(Tensor<T> tensor, Tensor<T> values)
    {
        GpuLaunchProbe.OnFallback("TensorHeaviside: no device kernel", null);
        return base.TensorHeaviside<T>(tensor, values);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFmax<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorFmax: no device kernel", null);
        return base.TensorFmax<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFmin<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorFmin: no device kernel", null);
        return base.TensorFmin<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorFloorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorFloorDivide: no device kernel", null);
        return base.TensorFloorDivide<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorGcd<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorGcd: no device kernel", null);
        return base.TensorGcd<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLcm<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorLcm: no device kernel", null);
        return base.TensorLcm<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseAnd<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorBitwiseAnd: no device kernel", null);
        return base.TensorBitwiseAnd<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseOr<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorBitwiseOr: no device kernel", null);
        return base.TensorBitwiseOr<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseXor<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorBitwiseXor: no device kernel", null);
        return base.TensorBitwiseXor<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseNot<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorBitwiseNot: no device kernel", null);
        return base.TensorBitwiseNot<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseLeftShift<T>(Tensor<T> a, Tensor<T> b)
    {
        GpuLaunchProbe.OnFallback("TensorBitwiseLeftShift: no device kernel", null);
        return base.TensorBitwiseLeftShift<T>(a, b);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBitwiseRightShift<T>(Tensor<T> a, Tensor<T> b)
    {
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
        GpuLaunchProbe.OnFallback("TensorIgamma: no device kernel", null);
        return base.TensorIgamma<T>(a, x);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIgammac<T>(Tensor<T> a, Tensor<T> x)
    {
        GpuLaunchProbe.OnFallback("TensorIgammac: no device kernel", null);
        return base.TensorIgammac<T>(a, x);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMvlgamma<T>(Tensor<T> tensor, int p)
    {
        GpuLaunchProbe.OnFallback("TensorMvlgamma: no device kernel", null);
        return base.TensorMvlgamma<T>(tensor, p);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorEntr<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorEntr: no device kernel", null);
        return base.TensorEntr<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorErfcx<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorErfcx: no device kernel", null);
        return base.TensorErfcx<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNdtr<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorNdtr: no device kernel", null);
        return base.TensorNdtr<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLogNdtr<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorLogNdtr: no device kernel", null);
        return base.TensorLogNdtr<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNdtri<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorNdtri: no device kernel", null);
        return base.TensorNdtri<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselJ0<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorBesselJ0: no device kernel", null);
        return base.TensorBesselJ0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselJ1<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorBesselJ1: no device kernel", null);
        return base.TensorBesselJ1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselY0<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorBesselY0: no device kernel", null);
        return base.TensorBesselY0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBesselY1<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorBesselY1: no device kernel", null);
        return base.TensorBesselY1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselI0<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorModifiedBesselI0: no device kernel", null);
        return base.TensorModifiedBesselI0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselI1<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorModifiedBesselI1: no device kernel", null);
        return base.TensorModifiedBesselI1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselK0<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorModifiedBesselK0: no device kernel", null);
        return base.TensorModifiedBesselK0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorModifiedBesselK1<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorModifiedBesselK1: no device kernel", null);
        return base.TensorModifiedBesselK1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorScaledModifiedBesselK0<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorScaledModifiedBesselK0: no device kernel", null);
        return base.TensorScaledModifiedBesselK0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorScaledModifiedBesselK1<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorScaledModifiedBesselK1: no device kernel", null);
        return base.TensorScaledModifiedBesselK1<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSphericalBesselJ0<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorSphericalBesselJ0: no device kernel", null);
        return base.TensorSphericalBesselJ0<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAiryAi<T>(Tensor<T> tensor)
    {
        GpuLaunchProbe.OnFallback("TensorAiryAi: no device kernel", null);
        return base.TensorAiryAi<T>(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialT: no device kernel", null);
        return base.TensorChebyshevPolynomialT<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialU: no device kernel", null);
        return base.TensorChebyshevPolynomialU<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialV: no device kernel", null);
        return base.TensorChebyshevPolynomialV<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorChebyshevPolynomialW: no device kernel", null);
        return base.TensorChebyshevPolynomialW<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialT: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialT<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialU: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialU<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialV: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialV<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorShiftedChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorShiftedChebyshevPolynomialW: no device kernel", null);
        return base.TensorShiftedChebyshevPolynomialW<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHermitePolynomialH<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorHermitePolynomialH: no device kernel", null);
        return base.TensorHermitePolynomialH<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHermitePolynomialHe<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorHermitePolynomialHe: no device kernel", null);
        return base.TensorHermitePolynomialHe<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLaguerrePolynomialL<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorLaguerrePolynomialL: no device kernel", null);
        return base.TensorLaguerrePolynomialL<T>(x, n);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLegendrePolynomialP<T>(Tensor<T> x, Tensor<T> n)
    {
        GpuLaunchProbe.OnFallback("TensorLegendrePolynomialP: no device kernel", null);
        return base.TensorLegendrePolynomialP<T>(x, n);
    }
}
