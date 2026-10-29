// Copyright (c) CrowdTrainer LLC. All rights reserved.
// HIP IGpuBatchExecution (#1142): the shared CUDA C batch kernels, compiled by hiprtc.
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IGpuBatchExecution
{
    private IntPtr _batchExecutionModule;
    private IntPtr _torchParityModule;
    private int _batchDepth;
    private int _secondaryStreamOwnerThreadId;

    internal unsafe void LaunchBatch(string name, int threads, params object[] args)
    {
        if (threads <= 0) return;
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException($"HIP kernel not found: {name}");
        ulong* slots = stackalloc ulong[args.Length];
        void** pointers = stackalloc void*[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case IGpuBuffer buffer: *(IntPtr*)(slots + i) = buffer.Handle; break;
                case int value: *(int*)(slots + i) = value; break;
                case float value: *(float*)(slots + i) = value; break;
                case ulong value: slots[i] = value; break;
                default:
                    throw new ArgumentException($"Unsupported argument type {args[i]?.GetType().Name ?? "null"} for {name}.");
            }
            pointers[i] = slots + i;
        }
        uint grid = (uint)((threads + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, pointers);
    }

    private unsafe void FullReduce(IGpuBuffer input, IGpuBuffer output, int size, int mode, float maxVal)
    {
        if (!_kernelCache.TryGetValue("bx_full_reduce", out var kernel))
            throw new GpuCapabilityUnavailableException("HIP kernel not found: bx_full_reduce");
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &size; args[3] = &mode; args[4] = &maxVal;
        LaunchKernel(kernel, 1, K.FullReduceBlockSize, args);
    }

    public bool SupportsBatchExecution => true;

    /// <inheritdoc/>
    public IGpuBuffer AllocateWorkspaceBuffer(int totalElements) => AllocateBuffer(totalElements);

    /// <summary>
    /// Launches go to one in-order stream and never wait on the host, so a batch is already one uninterrupted
    /// submission; <see cref="EndBatch"/> waits for it like Vulkan's submit-and-wait.
    /// </summary>
    public void BeginBatch()
    {
        if (Interlocked.CompareExchange(ref _batchDepth, 1, 0) != 0)
            throw new InvalidOperationException("Already in batch mode.");
    }

    public void EndBatch()
    {
        if (Interlocked.CompareExchange(ref _batchDepth, 0, 1) != 1)
            throw new InvalidOperationException("Not in batch mode.");
        Synchronize();
    }

    /// <summary>In-order stream: every launch already observes the writes of the launches before it.</summary>
    public void InsertBarrier(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
    }

    /// <summary>
    /// Work recorded between the calls stays on the in-order stream, so it is ordered after earlier launches;
    /// <see cref="EndSecondaryStream"/> waits for it, matching Vulkan's secondary submit.
    /// </summary>
    public void BeginSecondaryStream()
    {
        int thread = Environment.CurrentManagedThreadId;
        if (Interlocked.CompareExchange(ref _secondaryStreamOwnerThreadId, thread, 0) != 0)
            throw new InvalidOperationException("Secondary stream is already owned by another thread.");
    }

    public void EndSecondaryStream()
    {
        int thread = Environment.CurrentManagedThreadId;
        if (Interlocked.CompareExchange(ref _secondaryStreamOwnerThreadId, 0, thread) != thread)
            throw new InvalidOperationException("Not in a secondary stream on this thread.");
        Synchronize();
    }

    public void ReduceProduct(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullProduct, 0f);
    public void ReduceNormL2(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullNormL2, 0f);
    public void ReduceSumOfSquares(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullSumOfSquares, 0f);
    public void ReduceMaxMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullMaxMagnitude, 0f);
    public void ReduceMinMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullMinMagnitude, 0f);
    public void ReduceLogSumExp(IGpuBuffer input, IGpuBuffer output, float maxVal, int size) => FullReduce(input, output, size, K.FullLogSumExp, maxVal);

    public void VarianceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => LaunchBatch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowVariance);
    public void StdAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => LaunchBatch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowStd);
    public void ProductAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => LaunchBatch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowProduct);
    public void NormAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => LaunchBatch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowNorm);
    public void LogSumExpAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => LaunchBatch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowLogSumExp);
    public void ReduceLogVariance(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => LaunchBatch("bx_row_reduce", outerSize, input, output, outerSize, reduceSize, K.RowLogVariance);
    public void CumSumAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapCumSum);
    public void NormalizeL2(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapNormalizeL2);
    public void LogSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapLogSoftmax);
    public void TaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapTaylorSoftmax);
    public void SphericalSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapSphericalSoftmax);
    public void Sparsemax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_row_map", outerSize, input, output, outerSize, innerSize, K.MapSparsemax);
    public void GumbelSoftmax(IGpuBuffer logits, IGpuBuffer output, int outerSize, int innerSize, float temperature, ulong seed)
        => LaunchBatch("bx_gumbel_softmax", outerSize, logits, output, outerSize, innerSize, temperature, seed);

    public void ScalarMinusTensor(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => LaunchBatch("bx_scalar", size, input, output, scalar, size, K.ScalarMinusTensor);
    public void DivScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => LaunchBatch("bx_scalar", size, input, output, scalar, size, K.DivideByScalar);
    public void RsqrtKernel(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchBatch("bx_unary", size, input, output, size, K.UnaryRsqrt);
    public void ReluDerivative(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchBatch("bx_unary", size, input, output, size, K.UnaryReluDerivative);
    public void SigmoidDerivative(IGpuBuffer sigmoidOutput, IGpuBuffer output, int size)
        => LaunchBatch("bx_unary", size, sigmoidOutput, output, size, K.UnarySigmoidDerivative);
    public void TanhDerivative(IGpuBuffer tanhOutput, IGpuBuffer output, int size)
        => LaunchBatch("bx_unary", size, tanhOutput, output, size, K.UnaryTanhDerivative);
    public void SinCosKernel(IGpuBuffer input, IGpuBuffer sinOutput, IGpuBuffer cosOutput, int size)
        => LaunchBatch("bx_sincos", size, input, sinOutput, cosOutput, size);

    public void ReduceSumBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => LaunchBatch("bx_reduce_backward", outerSize * reduceSize, gradOutput, gradOutput, gradOutput, gradOutput, gradInput, outerSize, reduceSize, K.BackwardSum);
    public void ReduceMaxBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer maxValues, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => LaunchBatch("bx_reduce_backward", outerSize * reduceSize, gradOutput, input, maxValues, maxValues, gradInput, outerSize, reduceSize, K.BackwardMax);
    public void ReduceVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => LaunchBatch("bx_reduce_backward", outerSize * reduceSize, gradOutput, input, means, means, gradInput, outerSize, reduceSize, K.BackwardVariance);
    public void ReduceLogVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer variances, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => LaunchBatch("bx_reduce_backward", outerSize * reduceSize, gradOutput, input, means, variances, gradInput, outerSize, reduceSize, K.BackwardLogVariance);

    public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_broadcast_first", outerSize * innerSize, a, b, output, outerSize, innerSize, K.BroadcastAdd);
    public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchBatch("bx_broadcast_first", outerSize * innerSize, a, b, output, outerSize, innerSize, K.BroadcastMul);

    public void GluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => LaunchBatch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateGlu);
    public void GeGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => LaunchBatch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateGeGlu);
    public void ReGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => LaunchBatch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateReGlu);
    public void SwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => LaunchBatch("bx_gated_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim, K.GateSwiGlu);

    public void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int aInnerSize, int bInnerSize)
        => LaunchBatch("bx_concat_axis", outerSize * (aInnerSize + bInnerSize), a, b, output, outerSize, aInnerSize, bInnerSize);
    public void SliceLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int inputInnerSize, int start, int sliceSize)
        => LaunchBatch("bx_slice_last_axis", outerSize * sliceSize, input, output, outerSize, inputInnerSize, start, sliceSize);
    public void SetSliceLastAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int outputInnerSize, int start, int sliceSize)
        => LaunchBatch("bx_set_slice_last_axis", outerSize * sliceSize, output, values, outerSize, outputInnerSize, start, sliceSize);
    public void SliceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int stride, int index)
    {
        if (index < 0 || index >= axisSize) throw new ArgumentOutOfRangeException(nameof(index));
        LaunchBatch("bx_slice_axis", outerSize * stride, input, output, outerSize, axisSize, stride, index);
    }
    public void SetSliceAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int axisSize, int stride, int index)
    {
        if (index < 0 || index >= axisSize) throw new ArgumentOutOfRangeException(nameof(index));
        LaunchBatch("bx_set_slice_axis", outerSize * stride, output, values, outerSize, axisSize, stride, index);
    }
    public void Stack2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBatch("bx_stack2", size, a, b, output, size);
    public void Pad2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue)
        => LaunchBatch("bx_window_2d", batch * channels * outH * outW, input, output, batch, channels, inH, inW, outH, outW, padTop, padLeft, padValue, 0);
    public void Pad2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft)
        => LaunchBatch("bx_window_2d", batch * channels * inH * inW, gradOutput, gradInput, batch, channels, inH, inW, outH, outW, padTop, padLeft, 0f, 1);
    public void Crop2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => LaunchBatch("bx_window_2d", batch * channels * outH * outW, input, output, batch, channels, inH, inW, outH, outW, offsetH, offsetW, 0f, 2);
    public void Crop2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => LaunchBatch("bx_window_2d", batch * channels * inH * inW, gradOutput, gradInput, batch, channels, inH, inW, outH, outW, offsetH, offsetW, 0f, 3);
    public void TileLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => LaunchBatch("bx_tile_last_axis", outerSize * innerSize * repeats, input, output, outerSize, innerSize, repeats);
    public void RepeatElements(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => LaunchBatch("bx_repeat_elements", outerSize * innerSize * repeats, input, output, outerSize, innerSize, repeats);
    public void LinspaceKernel(IGpuBuffer output, float start, float step, int size)
        => LaunchBatch("bx_linspace", size, output, start, step, size);
    public void DiagKernel(IGpuBuffer input, IGpuBuffer output, int n)
        => LaunchBatch("bx_diag", n * n, input, output, n);
    public void ExtractDiagKernel(IGpuBuffer input, IGpuBuffer output, int n, int cols)
        => LaunchBatch("bx_extract_diag", Math.Min(n, cols), input, output, n, cols);
    public void TriangularMask(IGpuBuffer output, int rows, int cols, int diagonal, float maskValue)
        => LaunchBatch("bx_triangular_mask", rows * cols, output, rows, cols, diagonal, maskValue);
    public void IndexSelect(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int numIndices, int innerSize)
        => LaunchBatch("bx_index_select", numIndices * innerSize, input, indices, output, numIndices, innerSize);

    public void CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
        => LaunchBatch("bx_row_loss", batchSize, predictions, targets, loss, batchSize, numClasses, 0);
    public void MseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
        => LaunchBatch("bx_row_loss", batchSize, predictions, targets, loss, batchSize, numFeatures, 1);
    public void DropoutMask(IGpuBuffer mask, int size, float keepProb, ulong seed)
        => LaunchBatch("bx_dropout_mask", size, mask, size, keepProb, seed);
    public void GaussianNoise(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
        => LaunchBatch("bx_gaussian_noise", size, output, size, mean, stdDev, seed);

    public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int M, int N)
        => LaunchBatch("bx_batch_outer_product", batchSize * M * N, a, b, output, batchSize, M, N);
}
