// Copyright (c) AiDotNet. All rights reserved.
// Metal IGpuBatchExecution (#1142): the shared batch kernels, translated to MSL and compiled with strict math.
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IGpuBatchExecution
{
    private const string BatchExecutionLibName = "BatchExecution";
    private readonly object _batchExecutionLibraryLock = new object();
    private IntPtr _batchExecutionLibrary;
    private int _batchDepth;
    private int _secondaryStreamOwnerThreadId;

    private MetalPipelineState GetBatchPipeline(string name)
    {
        if (_batchExecutionLibrary == IntPtr.Zero)
        {
            lock (_batchExecutionLibraryLock)
            {
                if (_batchExecutionLibrary == IntPtr.Zero)
                    _batchExecutionLibrary = _shaderLibrary.CompileLibrary(
                        BatchExecutionLibName, MetalBatchExecutionKernels.Source, requireStrictFloatingPoint: true);
            }
        }
        return GetPipeline(BatchExecutionLibName, _batchExecutionLibrary, name);
    }

    private void LaunchBatch(string name, int threads, params object[] args)
    {
        if (threads <= 0) return;
        ThrowIfDisposed();
        var pipe = GetBatchPipeline(name);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        BindBatchArguments(in enc, name, args);
        var (groups, perGroup) = pipe.Calculate1DDispatch(threads);
        enc.DispatchThreadgroups(groups, perGroup);
    }

    private static void BindBatchArguments(in ScopedComputeEncoder enc, string name, object[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            ulong index = (ulong)i;
            switch (args[i])
            {
                case MetalGpuBuffer buffer: enc.SetBuffer(buffer, i); break;
                case int value: enc.SetBytes(value, index); break;
                case float value: enc.SetBytes(value, index); break;
                case ulong value: enc.SetBytes(value, index); break;
                default:
                    throw new ArgumentException($"Unsupported argument {args[i]?.GetType().Name ?? "null"} for {name}; buffers must be MetalGpuBuffer.");
            }
        }
    }

    private void FullReduce(IGpuBuffer input, IGpuBuffer output, int size, int mode, float maxVal)
    {
        ThrowIfDisposed();
        var pipe = GetBatchPipeline("bx_full_reduce");
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        BindBatchArguments(in enc, "bx_full_reduce", new object[] { input, output, size, mode, maxVal });
        // One threadgroup of the kernel's fixed width: the tree reduction indexes a 256-wide scratch array.
        enc.DispatchThreadgroups(new MTLSize(1, 1, 1), new MTLSize((ulong)K.FullReduceBlockSize, 1, 1));
    }

    public bool SupportsBatchExecution => true;

    /// <inheritdoc/>
    public IGpuBuffer AllocateWorkspaceBuffer(int totalElements) => AllocateBuffer(totalElements);

    /// <summary>
    /// Encoders go to one command queue that executes in submission order, so a batch is already one ordered
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

    /// <summary>The queue executes in order: every encoder already observes the writes of the ones before it.</summary>
    public void InsertBarrier(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
    }

    /// <summary>
    /// Work between the calls stays on the in-order queue, so it is ordered after earlier work;
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

