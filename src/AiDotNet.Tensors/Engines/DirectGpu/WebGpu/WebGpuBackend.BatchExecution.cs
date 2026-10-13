#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
// WebGPU IGpuBatchExecution (#1142): WGSL ports of the shared batch kernels. Like the other synchronous WebGPU
// capabilities, each operation blocks on its async dispatch (safe on the native wgpu path; a single-threaded
// JS event-loop host should call through the engine's async surface instead).
using System.Collections.Generic;
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;
using S = AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBatchExecutionKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IGpuBatchExecution
{
    private const string BatchModuleKey = "BatchExecution:";
    private int _batchDepth;
    private int _secondaryStreamOwnerThreadId;

    /// <summary>
    /// Buffers bind in argument order; every scalar becomes one 4-byte field of the parameter struct in argument
    /// order, a <see cref="ulong"/> seed two (low word first).
    /// </summary>
    private async Task DispatchBatchAsync(string name, int workgroups, object[] args)
    {
        if (!S.ByKernelName.TryGetValue(name, out var source) && !TorchParityModules.Value.TryGetValue(name, out source))
            throw new GpuCapabilityUnavailableException($"WebGPU batch kernel not found: {name}");
        var buffers = new List<WebGpuBuffer>();
        var fields = new List<float>();
        foreach (var arg in args)
        {
            switch (arg)
            {
                case IGpuBuffer buffer: buffers.Add(AsWgpu(buffer)); break;
                case int value: fields.Add(BitConverter.Int32BitsToSingle(value)); break;
                case float value: fields.Add(value); break;
                case ulong value:
                    fields.Add(BitConverter.Int32BitsToSingle(unchecked((int)(uint)value)));
                    fields.Add(BitConverter.Int32BitsToSingle(unchecked((int)(uint)(value >> 32))));
                    break;
                default:
                    throw new ArgumentException($"Unsupported argument type {arg?.GetType().Name ?? "null"} for {name}.");
            }
        }
        var pipe = await GetOrCreatePipelineAsync(BatchModuleKey + name, source, "main");
        // Uniform buffers are sized in 16-byte units.
        var uniform = new float[Math.Max(4, (fields.Count + 3) / 4 * 4)];
        fields.CopyTo(uniform);
        using var u = new WebGpuBuffer(uniform, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(pipe, buffers.ToArray());
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, u.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    private void LaunchBatch(string name, int threads, params object[] args)
    {
        if (threads <= 0) return;
        var (workgroups, _) = _device.CalculateWorkgroups1D(threads);
        DispatchBatchAsync(name, workgroups, args).GetAwaiter().GetResult();
    }

    /// <summary>One workgroup: the kernel's tree reduction spans its 256 invocations.</summary>
    private void FullReduce(IGpuBuffer input, IGpuBuffer output, int size, int mode, float maxVal)
        => DispatchBatchAsync("bx_full_reduce", 1, new object[] { input, output, size, mode, maxVal }).GetAwaiter().GetResult();

    public bool SupportsBatchExecution => true;

    /// <inheritdoc/>
    public IGpuBuffer AllocateWorkspaceBuffer(int totalElements) => AllocateBuffer(totalElements);

    /// <summary>Every dispatch already waits for its submission, so a batch is ordered and complete at its end.</summary>
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

    /// <summary>Every dispatch waits for its submission, so later work already observes earlier writes.</summary>
    public void InsertBarrier(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
    }

    /// <summary>Work between the calls is ordered after earlier work and complete when the stream ends.</summary>
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
    // The kernel-launching operations live in PortableBatchExecution, shared with Metal / WebGPU and driven by the Dawn parity suites.
    private PortableBatchExecution? _portableBatch;
    private PortableBatchExecution Portable => _portableBatch ??= new PortableBatchExecution(LaunchBatch, FullReduce);

    public void ReduceProduct(IGpuBuffer input, IGpuBuffer output, int size)
        => Portable.ReduceProduct(input, output, size);
    public void ReduceNormL2(IGpuBuffer input, IGpuBuffer output, int size)
        => Portable.ReduceNormL2(input, output, size);
    public void ReduceSumOfSquares(IGpuBuffer input, IGpuBuffer output, int size)
        => Portable.ReduceSumOfSquares(input, output, size);
    public void ReduceMaxMagnitude(IGpuBuffer input, IGpuBuffer output, int size)
        => Portable.ReduceMaxMagnitude(input, output, size);
    public void ReduceMinMagnitude(IGpuBuffer input, IGpuBuffer output, int size)
        => Portable.ReduceMinMagnitude(input, output, size);
    public void ReduceLogSumExp(IGpuBuffer input, IGpuBuffer output, float maxVal, int size)
        => Portable.ReduceLogSumExp(input, output, maxVal, size);
    public void VarianceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => Portable.VarianceAxis(input, output, outerSize, reduceSize);
    public void StdAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => Portable.StdAxis(input, output, outerSize, reduceSize);
    public void ProductAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => Portable.ProductAxis(input, output, outerSize, reduceSize);
    public void NormAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => Portable.NormAxis(input, output, outerSize, reduceSize);
    public void LogSumExpAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => Portable.LogSumExpAxis(input, output, outerSize, reduceSize);
    public void ReduceLogVariance(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => Portable.ReduceLogVariance(input, output, outerSize, reduceSize);
    public void CumSumAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.CumSumAxis(input, output, outerSize, innerSize);
    public void NormalizeL2(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.NormalizeL2(input, output, outerSize, innerSize);
    public void LogSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.LogSoftmax(input, output, outerSize, innerSize);
    public void TaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.TaylorSoftmax(input, output, outerSize, innerSize);
    public void SphericalSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.SphericalSoftmax(input, output, outerSize, innerSize);
    public void Sparsemax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.Sparsemax(input, output, outerSize, innerSize);
    public void GumbelSoftmax(IGpuBuffer logits, IGpuBuffer output, int outerSize, int innerSize, float temperature, ulong seed)
        => Portable.GumbelSoftmax(logits, output, outerSize, innerSize, temperature, seed);
    public void ScalarMinusTensor(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => Portable.ScalarMinusTensor(input, output, scalar, size);
    public void DivScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => Portable.DivScalar(input, output, scalar, size);
    public void RsqrtKernel(IGpuBuffer input, IGpuBuffer output, int size)
        => Portable.RsqrtKernel(input, output, size);
    public void ReluDerivative(IGpuBuffer input, IGpuBuffer output, int size)
        => Portable.ReluDerivative(input, output, size);
    public void SigmoidDerivative(IGpuBuffer sigmoidOutput, IGpuBuffer output, int size)
        => Portable.SigmoidDerivative(sigmoidOutput, output, size);
    public void TanhDerivative(IGpuBuffer tanhOutput, IGpuBuffer output, int size)
        => Portable.TanhDerivative(tanhOutput, output, size);
    public void SinCosKernel(IGpuBuffer input, IGpuBuffer sinOutput, IGpuBuffer cosOutput, int size)
        => Portable.SinCosKernel(input, sinOutput, cosOutput, size);
    public void ReduceSumBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => Portable.ReduceSumBackward(gradOutput, gradInput, outerSize, reduceSize);
    public void ReduceMaxBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer maxValues, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => Portable.ReduceMaxBackward(gradOutput, input, maxValues, gradInput, outerSize, reduceSize);
    public void ReduceVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => Portable.ReduceVarianceBackward(gradOutput, input, means, gradInput, outerSize, reduceSize);
    public void ReduceLogVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer variances, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => Portable.ReduceLogVarianceBackward(gradOutput, input, means, variances, gradInput, outerSize, reduceSize);
    public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.BroadcastAddFirst(a, b, output, outerSize, innerSize);
    public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => Portable.BroadcastMulFirst(a, b, output, outerSize, innerSize);
    public void GluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => Portable.GluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void GeGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => Portable.GeGluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void ReGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => Portable.ReGluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void SwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => Portable.SwiGluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int aInnerSize, int bInnerSize)
        => Portable.ConcatAxis(a, b, output, outerSize, aInnerSize, bInnerSize);
    public void SliceLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int inputInnerSize, int start, int sliceSize)
        => Portable.SliceLastAxis(input, output, outerSize, inputInnerSize, start, sliceSize);
    public void SetSliceLastAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int outputInnerSize, int start, int sliceSize)
        => Portable.SetSliceLastAxis(output, values, outerSize, outputInnerSize, start, sliceSize);
    public void SliceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int stride, int index)
        => Portable.SliceAxis(input, output, outerSize, axisSize, stride, index);
    public void SetSliceAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int axisSize, int stride, int index)
        => Portable.SetSliceAxis(output, values, outerSize, axisSize, stride, index);
    public void Stack2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => Portable.Stack2(a, b, output, size);
    public void Pad2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue)
        => Portable.Pad2D(input, output, batch, channels, inH, inW, outH, outW, padTop, padLeft, padValue);
    public void Pad2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft)
        => Portable.Pad2DBackward(gradOutput, gradInput, batch, channels, inH, inW, outH, outW, padTop, padLeft);
    public void Crop2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => Portable.Crop2D(input, output, batch, channels, inH, inW, outH, outW, offsetH, offsetW);
    public void Crop2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => Portable.Crop2DBackward(gradOutput, gradInput, batch, channels, inH, inW, outH, outW, offsetH, offsetW);
    public void TileLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => Portable.TileLastAxis(input, output, outerSize, innerSize, repeats);
    public void RepeatElements(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => Portable.RepeatElements(input, output, outerSize, innerSize, repeats);
    public void LinspaceKernel(IGpuBuffer output, float start, float step, int size)
        => Portable.LinspaceKernel(output, start, step, size);
    public void DiagKernel(IGpuBuffer input, IGpuBuffer output, int n)
        => Portable.DiagKernel(input, output, n);
    public void ExtractDiagKernel(IGpuBuffer input, IGpuBuffer output, int n, int cols)
        => Portable.ExtractDiagKernel(input, output, n, cols);
    public void TriangularMask(IGpuBuffer output, int rows, int cols, int diagonal, float maskValue)
        => Portable.TriangularMask(output, rows, cols, diagonal, maskValue);
    public void IndexSelect(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int numIndices, int innerSize)
        => Portable.IndexSelect(input, indices, output, numIndices, innerSize);
    public void CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
        => Portable.CrossEntropyLoss(predictions, targets, loss, batchSize, numClasses);
    public void MseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
        => Portable.MseLoss(predictions, targets, loss, batchSize, numFeatures);
    public void DropoutMask(IGpuBuffer mask, int size, float keepProb, ulong seed)
        => Portable.DropoutMask(mask, size, keepProb, seed);
    public void GaussianNoise(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
        => Portable.GaussianNoise(output, size, mean, stdDev, seed);
    public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int M, int N)
        => Portable.BatchOuterProduct(a, b, output, batchSize, M, N);

}
#endif

