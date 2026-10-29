// Copyright (c) CrowdTrainer LLC. All rights reserved.
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The <see cref="IGpuBatchExecution"/> operations (and the fused two-grid meshgrid) for backends that run the shared
/// batch kernels through a launcher: Metal (MSL translations) and WebGPU (WGSL ports). One copy of the dispatch, so the
/// Dawn parity suites, which drive this class with the WGSL ports on Dawn, exercise the code those backends ship.
/// Every launch binds as <see cref="PortableGenericFft.Launcher"/> documents: buffers and scalars in argument order.
/// </summary>
internal sealed class PortableBatchExecution
{
    /// <summary>A full reduction over one buffer: one workgroup of the kernel's fixed width.</summary>
    public delegate void FullReduceLauncher(IGpuBuffer input, IGpuBuffer output, int size, int mode, float maxVal);

    private readonly PortableGenericFft.Launcher _launch;
    private readonly FullReduceLauncher _fullReduce;

    public PortableBatchExecution(PortableGenericFft.Launcher launch, FullReduceLauncher fullReduce)
    {
        _launch = launch ?? throw new ArgumentNullException(nameof(launch));
        _fullReduce = fullReduce ?? throw new ArgumentNullException(nameof(fullReduce));
    }
    public void ReduceProduct(IGpuBuffer input, IGpuBuffer output, int size) => _fullReduce(input, output, size, K.FullProduct, 0f);
    public void ReduceNormL2(IGpuBuffer input, IGpuBuffer output, int size) => _fullReduce(input, output, size, K.FullNormL2, 0f);
    public void ReduceSumOfSquares(IGpuBuffer input, IGpuBuffer output, int size) => _fullReduce(input, output, size, K.FullSumOfSquares, 0f);
    public void ReduceMaxMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => _fullReduce(input, output, size, K.FullMaxMagnitude, 0f);
    public void ReduceMinMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => _fullReduce(input, output, size, K.FullMinMagnitude, 0f);
    public void ReduceLogSumExp(IGpuBuffer input, IGpuBuffer output, float maxVal, int size) => _fullReduce(input, output, size, K.FullLogSumExp, maxVal);

    public void VarianceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _launch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowVariance);
    public void StdAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _launch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowStd);
    public void ProductAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _launch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowProduct);
    public void NormAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _launch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowNorm);
    public void LogSumExpAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _launch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowLogSumExp);
    public void ReduceLogVariance(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _launch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowLogVariance);
    public void CumSumAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapCumSum);
    public void NormalizeL2(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapNormalizeL2);
    public void LogSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapLogSoftmax);
    public void TaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapTaylorSoftmax);
    public void SphericalSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapSphericalSoftmax);
    public void Sparsemax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapSparsemax);
    public void GumbelSoftmax(IGpuBuffer logits, IGpuBuffer output, int outerSize, int innerSize, float temperature, ulong seed)
        => _launch("bx_gumbel_softmax", outerSize, logits, output, outerSize, innerSize, temperature, seed);

    public void ScalarMinusTensor(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => _launch("bx_scalar", size, input, output, scalar, size, K.ScalarMinusTensor);
    public void DivScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => _launch("bx_scalar", size, input, output, scalar, size, K.DivideByScalar);
    public void RsqrtKernel(IGpuBuffer input, IGpuBuffer output, int size)
        => _launch("bx_unary", size, input, output, size, K.UnaryRsqrt);
    public void ReluDerivative(IGpuBuffer input, IGpuBuffer output, int size)
        => _launch("bx_unary", size, input, output, size, K.UnaryReluDerivative);
    public void SigmoidDerivative(IGpuBuffer sigmoidOutput, IGpuBuffer output, int size)
        => _launch("bx_unary", size, sigmoidOutput, output, size, K.UnarySigmoidDerivative);
    public void TanhDerivative(IGpuBuffer tanhOutput, IGpuBuffer output, int size)
        => _launch("bx_unary", size, tanhOutput, output, size, K.UnaryTanhDerivative);
    public void SinCosKernel(IGpuBuffer input, IGpuBuffer sinOutput, IGpuBuffer cosOutput, int size)
        => _launch("bx_sincos", size, input, sinOutput, cosOutput, size);

    public void ReduceSumBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _launch("bx_reduce_backward", outerSize * reduceSize, gradOutput, gradOutput, gradOutput, gradOutput, gradInput, outerSize, reduceSize, K.BackwardSum);
    public void ReduceMaxBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer maxValues, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _launch("bx_reduce_backward", outerSize * reduceSize, gradOutput, input, maxValues, maxValues, gradInput, outerSize, reduceSize, K.BackwardMax);
    public void ReduceVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _launch("bx_reduce_backward", outerSize * reduceSize, gradOutput, input, means, means, gradInput, outerSize, reduceSize, K.BackwardVariance);
    public void ReduceLogVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer variances, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _launch("bx_reduce_backward", outerSize * reduceSize, gradOutput, input, means, variances, gradInput, outerSize, reduceSize, K.BackwardLogVariance);

    public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_broadcast_first", outerSize * innerSize, a, b, output, outerSize, innerSize, K.BroadcastAdd);
    public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => _launch("bx_broadcast_first", outerSize * innerSize, a, b, output, outerSize, innerSize, K.BroadcastMul);

    public void GluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _launch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateGlu);
    public void GeGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _launch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateGeGlu);
    public void ReGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _launch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateReGlu);
    public void SwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _launch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateSwiGlu);

    public void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int aInnerSize, int bInnerSize)
        => _launch("bx_concat_axis", outerSize * (aInnerSize + bInnerSize), a, b, output, outerSize, aInnerSize, bInnerSize);
    public void SliceLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int inputInnerSize, int start, int sliceSize)
        => _launch("bx_slice_last_axis", outerSize * sliceSize, input, output, outerSize, inputInnerSize, start, sliceSize);
    public void SetSliceLastAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int outputInnerSize, int start, int sliceSize)
        => _launch("bx_set_slice_last_axis", outerSize * sliceSize, output, values, outerSize, outputInnerSize, start, sliceSize);
    public void SliceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int stride, int index)
    {
        if (index < 0 || index >= axisSize) throw new ArgumentOutOfRangeException(nameof(index));
        _launch("bx_slice_axis", outerSize * stride, input, output, outerSize, axisSize, stride, index);
    }
    public void SetSliceAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int axisSize, int stride, int index)
    {
        if (index < 0 || index >= axisSize) throw new ArgumentOutOfRangeException(nameof(index));
        _launch("bx_set_slice_axis", outerSize * stride, output, values, outerSize, axisSize, stride, index);
    }
    public void Stack2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => _launch("bx_stack2", size, a, b, output, size);
    public void Pad2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue)
        => _launch("bx_window_2d", batch * channels * outH * outW, input, output, batch, channels, inH, inW, outH, outW, padTop, padLeft, padValue, 0);
    public void Pad2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft)
        => _launch("bx_window_2d", batch * channels * inH * inW, gradOutput, gradInput, batch, channels, inH, inW, outH, outW, padTop, padLeft, 0f, 1);
    public void Crop2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => _launch("bx_window_2d", batch * channels * outH * outW, input, output, batch, channels, inH, inW, outH, outW, offsetH, offsetW, 0f, 2);
    public void Crop2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => _launch("bx_window_2d", batch * channels * inH * inW, gradOutput, gradInput, batch, channels, inH, inW, outH, outW, offsetH, offsetW, 0f, 3);
    public void TileLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => _launch("bx_tile_last_axis", outerSize * innerSize * repeats, input, output, outerSize, innerSize, repeats);
    public void RepeatElements(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => _launch("bx_repeat_elements", outerSize * innerSize * repeats, input, output, outerSize, innerSize, repeats);
    public void LinspaceKernel(IGpuBuffer output, float start, float step, int size)
        => _launch("bx_linspace", size, output, start, step, size);
    public void DiagKernel(IGpuBuffer input, IGpuBuffer output, int n)
        => _launch("bx_diag", n * n, input, output, n);
    public void ExtractDiagKernel(IGpuBuffer input, IGpuBuffer output, int n, int cols)
        => _launch("bx_extract_diag", Math.Min(n, cols), input, output, n, cols);
    public void TriangularMask(IGpuBuffer output, int rows, int cols, int diagonal, float maskValue)
        => _launch("bx_triangular_mask", rows * cols, output, rows, cols, diagonal, maskValue);
    public void IndexSelect(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int numIndices, int innerSize)
        => _launch("bx_index_select", numIndices * innerSize, input, indices, output, numIndices, innerSize);

    public void CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
        => _launch("bx_row_loss", batchSize, predictions, targets, loss, batchSize, numClasses, 0);
    public void MseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
        => _launch("bx_row_loss", batchSize, predictions, targets, loss, batchSize, numFeatures, 1);
    public void DropoutMask(IGpuBuffer mask, int size, float keepProb, ulong seed)
        => _launch("bx_dropout_mask", size, mask, size, keepProb, seed);
    public void GaussianNoise(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
        => _launch("bx_gaussian_noise", size, output, size, mean, stdDev, seed);

    public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int M, int N)
        => _launch("bx_batch_outer_product", batchSize * M * N, a, b, output, batchSize, M, N);

    public void Meshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0, IGpuBuffer output1,
        int n0, int n1, bool xy)
    {
        if (n0 <= 0) throw new ArgumentOutOfRangeException(nameof(n0));
        if (n1 <= 0) throw new ArgumentOutOfRangeException(nameof(n1));
        _launch("bx_meshgrid2d", checked(n0 * n1), source0, source1, output0, output1, n0, n1, xy ? 1 : 0);
    }
}
