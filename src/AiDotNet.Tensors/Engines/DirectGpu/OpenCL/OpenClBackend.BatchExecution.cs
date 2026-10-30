// Copyright (c) AiDotNet. All rights reserved.
// OpenCL IGpuBatchExecution (#1142, #916). The fused kernels were compiled but never reachable; this wires them.
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IGpuBatchExecution
    {
        private int _batchDepth;
        private int _secondaryStreamOwnerThreadId;

        private DirectOpenClKernel GetBatchKernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var k))
                throw new GpuCapabilityUnavailableException($"OpenCL kernel not found: {name}.");
            return k;
        }

        // Float and byte buffers (16-bit FFT storage) both expose their cl_mem as Handle.
        private static IntPtr BatchHandle(IGpuBuffer buffer) => buffer.Handle;

        private void LaunchBatch(string name, int globalSize, params object[] args)
        {
            if (globalSize <= 0) return;
            var kernel = GetBatchKernel(name);
            for (int i = 0; i < args.Length; i++)
            {
                uint index = (uint)i;
                switch (args[i])
                {
                    case IGpuBuffer buffer: kernel.SetArg(index, BatchHandle(buffer)); break;
                    case int value: kernel.SetArg(index, value); break;
                    case uint value: kernel.SetArg(index, value); break;
                    case float value: kernel.SetArg(index, value); break;
                    case ulong value: kernel.SetArg(index, value); break;
                    default:
                        throw new ArgumentException($"Unsupported argument type {args[i]?.GetType().Name ?? "null"} for {name}.");
                }
            }
            kernel.Execute1D(globalSize, CalculateOptimalWorkGroupSize1D(globalSize));
        }

        private void FullReduce(IGpuBuffer input, IGpuBuffer output, int size, int mode, float maxVal)
        {
            int local = CalculateOptimalWorkGroupSize1D(BatchExecutionKernels.FullReduceGroupSize);
            int pow2 = 1;
            while (pow2 * 2 <= local) pow2 *= 2;
            var kernel = GetBatchKernel("bx_full_reduce");
            kernel.SetArg(0, BatchHandle(input));
            kernel.SetArg(1, BatchHandle(output));
            kernel.SetArg(2, size);
            kernel.SetArg(3, mode);
            kernel.SetArg(4, maxVal);
            kernel.Execute1D(pow2, pow2);
        }

        public bool SupportsBatchExecution => true;

        /// <inheritdoc/>
        public IGpuBuffer AllocateWorkspaceBuffer(int totalElements) => AllocateBuffer(totalElements);

        /// <summary>
        /// The command queue is in-order and launches never wait on the host, so a batch is already a
        /// single uninterrupted submission; <see cref="EndBatch"/> waits for it like Vulkan's submit-and-wait.
        /// </summary>
        public void BeginBatch()
        {
            if (Interlocked.Increment(ref _batchDepth) != 1)
            {
                Interlocked.Decrement(ref _batchDepth);
                throw new InvalidOperationException("Already in batch mode.");
            }
        }

        public void EndBatch()
        {
            if (Interlocked.CompareExchange(ref _batchDepth, 0, 1) != 1)
                throw new InvalidOperationException("Not in batch mode.");
            Synchronize();
        }

        /// <summary>In-order queue: every launch already observes the writes of the launches before it.</summary>
        public void InsertBarrier(IGpuBuffer buffer)
        {
            if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        }

        /// <summary>
        /// Work recorded between the calls runs on the same in-order queue, so it stays ordered after earlier
        /// launches; <see cref="EndSecondaryStream"/> waits for it, matching Vulkan's secondary submit.
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

        public void ReduceProduct(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, 2, 0f);
        public void ReduceNormL2(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, 1, 0f);
        public void ReduceSumOfSquares(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, 0, 0f);
        public void ReduceMaxMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, 3, 0f);
        public void ReduceMinMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, 4, 0f);
        public void ReduceLogSumExp(IGpuBuffer input, IGpuBuffer output, float maxVal, int size) => FullReduce(input, output, size, 5, maxVal);

        public void VarianceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
            => LaunchBatch("variance_axis", outerSize, input, output, outerSize, reduceSize);
        public void StdAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
            => LaunchBatch("std_axis", outerSize, input, output, outerSize, reduceSize);
        public void ProductAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
            => LaunchBatch("product_axis", outerSize, input, output, outerSize, reduceSize);
        public void NormAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
            => LaunchBatch("norm_axis", outerSize, input, output, outerSize, reduceSize);
        public void LogSumExpAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
            => LaunchBatch("logsumexp_axis", outerSize, input, output, outerSize, reduceSize);
        public void CumSumAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("cumsum_axis", outerSize, input, output, outerSize, innerSize);
        public void ScalarMinusTensor(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
            => LaunchBatch("scalar_minus_tensor", size, input, output, scalar, size);
        public void NormalizeL2(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("normalize_l2", outerSize, input, output, outerSize, innerSize);
        public void ReduceSumBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int outerSize, int reduceSize)
            => LaunchBatch("reduce_sum_backward", outerSize * reduceSize, gradOutput, gradInput, outerSize, reduceSize);
        public void ReduceMaxBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer maxValues, IGpuBuffer gradInput, int outerSize, int reduceSize)
            => LaunchBatch("bx_reduce_max_backward", outerSize * reduceSize, gradOutput, input, maxValues, gradInput, outerSize, reduceSize);
        public void ReduceVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer gradInput, int outerSize, int reduceSize)
            => LaunchBatch("bx_reduce_variance_backward", outerSize * reduceSize, gradOutput, input, means, gradInput, outerSize, reduceSize);
        public void ReduceLogVariance(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
            => LaunchBatch("bx_reduce_log_variance", outerSize, input, output, outerSize, reduceSize);
        public void ReduceLogVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer variances, IGpuBuffer gradInput, int outerSize, int reduceSize)
            => LaunchBatch("bx_reduce_log_variance_backward", outerSize * reduceSize, gradOutput, input, means, variances, gradInput, outerSize, reduceSize);

        public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("broadcast_add_first", outerSize * innerSize, a, b, output, outerSize, innerSize);
        public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("broadcast_mul_first", outerSize * innerSize, a, b, output, outerSize, innerSize);
        public void DivScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
            => LaunchBatch("div_scalar", size, input, output, scalar, size);
        public void RsqrtKernel(IGpuBuffer input, IGpuBuffer output, int size)
            => LaunchBatch("rsqrt_kernel", size, input, output, size);
        public void SinCosKernel(IGpuBuffer input, IGpuBuffer sinOutput, IGpuBuffer cosOutput, int size)
            => LaunchBatch("bx_sincos", size, input, sinOutput, cosOutput, size);

        public void GluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
            => LaunchBatch("glu_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim);
        public void GeGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
            => LaunchBatch("geglu_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim);
        public void ReGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
            => LaunchBatch("reglu_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim);
        public void SwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
            => LaunchBatch("swiglu_backward", outerSize * halfDim, gradOutput, input, gradInput, outerSize, halfDim);
        public void ReluDerivative(IGpuBuffer input, IGpuBuffer output, int size)
            => LaunchBatch("relu_derivative", size, input, output, size);
        public void SigmoidDerivative(IGpuBuffer sigmoidOutput, IGpuBuffer output, int size)
            => LaunchBatch("sigmoid_derivative", size, sigmoidOutput, output, size);
        public void TanhDerivative(IGpuBuffer tanhOutput, IGpuBuffer output, int size)
            => LaunchBatch("tanh_derivative", size, tanhOutput, output, size);

        public void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int aInnerSize, int bInnerSize)
            => LaunchBatch("concat_axis", outerSize * (aInnerSize + bInnerSize), a, b, output, outerSize, aInnerSize, bInnerSize);
        public void SliceLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int inputInnerSize, int start, int sliceSize)
            => LaunchBatch("slice_last_axis", outerSize * sliceSize, input, output, outerSize, inputInnerSize, start, sliceSize);
        public void SetSliceLastAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int outputInnerSize, int start, int sliceSize)
            => LaunchBatch("set_slice_last_axis", outerSize * sliceSize, output, values, outerSize, outputInnerSize, start, sliceSize);
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
            => LaunchBatch("stack_2", size, a, b, output, size);
        public void Pad2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue)
            => LaunchBatch("pad_2d", batch * channels * outH * outW, input, output, batch, channels, inH, inW, outH, outW, padTop, padLeft, padValue);
        public void Pad2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft)
            => LaunchBatch("pad_2d_backward", batch * channels * inH * inW, gradOutput, gradInput, batch, channels, inH, inW, outH, outW, padTop, padLeft);
        public void TileLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
            => LaunchBatch("tile_last_axis", outerSize * innerSize * repeats, input, output, outerSize, innerSize, repeats);
        public void RepeatElements(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
            => LaunchBatch("repeat_elements", outerSize * innerSize * repeats, input, output, outerSize, innerSize, repeats);
        public void Crop2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
            => LaunchBatch("crop_2d", batch * channels * outH * outW, input, output, batch, channels, inH, inW, outH, outW, offsetH, offsetW);
        public void Crop2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
            => LaunchBatch("bx_crop_2d_backward", batch * channels * inH * inW, gradOutput, gradInput, batch, channels, inH, inW, outH, outW, offsetH, offsetW);
        public void LinspaceKernel(IGpuBuffer output, float start, float step, int size)
            => LaunchBatch("linspace_kernel", size, output, start, step, size);
        public void DiagKernel(IGpuBuffer input, IGpuBuffer output, int n)
            => LaunchBatch("diag_kernel", n * n, input, output, n);
        public void ExtractDiagKernel(IGpuBuffer input, IGpuBuffer output, int n, int cols)
            => LaunchBatch("extract_diag_kernel", Math.Min(n, cols), input, output, n, cols);
        public void TriangularMask(IGpuBuffer output, int rows, int cols, int diagonal, float maskValue)
            => LaunchBatch("triangular_mask", rows * cols, output, rows, cols, diagonal, maskValue);
        public void IndexSelect(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int numIndices, int innerSize)
            => LaunchBatch("index_select", numIndices * innerSize, input, indices, output, numIndices, innerSize,
                innerSize > 0 ? input.Size / innerSize : 0);

        public void CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
            => LaunchBatch("cross_entropy_loss", batchSize, predictions, targets, loss, batchSize, numClasses);
        public void MseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
            => LaunchBatch("bx_mse_loss_rows", batchSize, predictions, targets, loss, batchSize, numFeatures);
        public void DropoutMask(IGpuBuffer mask, int size, float keepProb, ulong seed)
            => LaunchBatch("dropout_mask", size, mask, size, keepProb, seed);
        public void GaussianNoise(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
            => LaunchBatch("gaussian_noise", size, output, size, mean, stdDev, seed);

        public void LogSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("log_softmax", outerSize, input, output, outerSize, innerSize);
        public void GumbelSoftmax(IGpuBuffer logits, IGpuBuffer output, int outerSize, int innerSize, float temperature, ulong seed)
            => LaunchBatch("gumbel_softmax", outerSize, logits, output, outerSize, innerSize, temperature, seed);
        public void Sparsemax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("bx_sparsemax", outerSize, input, output, outerSize, innerSize);
        public void TaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("taylor_softmax", outerSize, input, output, outerSize, innerSize);
        public void SphericalSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
            => LaunchBatch("spherical_softmax", outerSize, input, output, outerSize, innerSize);
        public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int M, int N)
            => LaunchBatch("batch_outer_product", batchSize * M * N, a, b, output, batchSize, M, N);
    }
}
