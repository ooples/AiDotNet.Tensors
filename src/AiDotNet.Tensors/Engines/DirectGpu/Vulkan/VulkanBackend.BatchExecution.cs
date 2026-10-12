// Copyright (c) AiDotNet. All rights reserved.
// Vulkan IGpuBatchExecution operations backed by the shared batch-execution kernels (#1142). The per-op shaders
// these replace left their outputs unwritten (no push constants reached them), read gated inputs in the wrong
// layout, or ran whole-tensor reductions on one thread.
using System.Collections.Generic;
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;
using S = AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBatchExecutionKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IMeshgridBackend
{
    private static uint Bits(int value) => unchecked((uint)value);

    // The descriptor set each dispatch recorded into the open batch bound, held until the batch has executed.
    private readonly List<(VulkanComputePipeline Pipeline, IntPtr DescriptorSet)> _batchDescriptorSets =
        new List<(VulkanComputePipeline, IntPtr)>();

    /// <summary>
    /// A recorded dispatch reads its descriptor set only when the batch is submitted, so each one keeps its own:
    /// the pipeline's current set is retained for it, and reusing the same kernel on other buffers later in the
    /// batch writes a different set (see <see cref="VulkanComputePipeline.RetainDescriptorSet"/>).
    /// </summary>
    private IntPtr RetainBatchDescriptorSet(VulkanComputePipeline pipeline)
    {
        IntPtr set = pipeline.RetainDescriptorSet();
        _batchDescriptorSets.Add((pipeline, set));
        return set;
    }

    /// <summary>Hands the batch's descriptor sets back once it has executed.</summary>
    private void ReleaseBatchDescriptorSets()
    {
        foreach (var (pipeline, set) in _batchDescriptorSets) pipeline.ReleaseDescriptorSet(set);
        _batchDescriptorSets.Clear();
    }

    /// <summary>Dispatches recorded into the open batch, for tests.</summary>
    internal int BatchDispatchCount => _batchDispatchCount;

    private void FullReduce(IGpuBuffer input, IGpuBuffer output, int size, int mode, float maxVal)
        => GlslNaryOp(S.FullReduce, new[] { input, output }, K.FullReduceBlockSize, new[] { Bits(size), Bits(mode), FloatBits(maxVal) });

    private void RowReduce(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int mode)
        => GlslNaryOp(S.RowReduce, new[] { input, output }, outerSize, new[] { Bits(outerSize), Bits(innerSize), Bits(mode) });

    private void RowMap(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int mode)
        => GlslNaryOp(S.RowMap, new[] { input, output }, outerSize, new[] { Bits(outerSize), Bits(innerSize), Bits(mode) });

    private void ReduceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer stat, IGpuBuffer stat2, IGpuBuffer gradInput,
        int outerSize, int reduceSize, int mode)
        => GlslNaryOp(S.ReduceBackward, new[] { gradOutput, input, stat, stat2, gradInput }, outerSize * reduceSize,
            new[] { Bits(outerSize), Bits(reduceSize), Bits(mode) });

    private void GatedBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim, int mode)
        => GlslNaryOp(S.GatedBackward, new[] { gradOutput, input, gradInput }, outerSize * halfDim,
            new[] { Bits(outerSize), Bits(halfDim), Bits(mode) });

    private void Window2D(IGpuBuffer input, IGpuBuffer output, int threads, int batch, int channels, int inH, int inW,
        int outH, int outW, int h0, int w0, float padValue, int mode)
        => GlslNaryOp(S.Window2D, new[] { input, output }, threads, new[]
        {
            Bits(batch), Bits(channels), Bits(inH), Bits(inW), Bits(outH), Bits(outW), Bits(h0), Bits(w0), FloatBits(padValue), Bits(mode)
        });

    public void ReduceProduct(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullProduct, 0f);
    public void ReduceNormL2(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullNormL2, 0f);
    public void ReduceSumOfSquares(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullSumOfSquares, 0f);
    public void ReduceMaxMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullMaxMagnitude, 0f);
    public void ReduceMinMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => FullReduce(input, output, size, K.FullMinMagnitude, 0f);
    public void ReduceLogSumExp(IGpuBuffer input, IGpuBuffer output, float maxVal, int size) => FullReduce(input, output, size, K.FullLogSumExp, maxVal);

    public void LogSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => RowMap(input, output, outerSize, innerSize, K.MapLogSoftmax);
    public void TaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => RowMap(input, output, outerSize, innerSize, K.MapTaylorSoftmax);
    public void SphericalSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => RowMap(input, output, outerSize, innerSize, K.MapSphericalSoftmax);
    public void Sparsemax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => RowMap(input, output, outerSize, innerSize, K.MapSparsemax);
    public void GumbelSoftmax(IGpuBuffer logits, IGpuBuffer output, int outerSize, int innerSize, float temperature, ulong seed)
        => GlslNaryOp(S.GumbelSoftmax, new[] { logits, output }, outerSize,
            new[] { Bits(outerSize), Bits(innerSize), FloatBits(temperature), (uint)seed, (uint)(seed >> 32) });

    public void ReduceMaxBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer maxValues, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => ReduceBackward(gradOutput, input, maxValues, maxValues, gradInput, outerSize, reduceSize, K.BackwardMax);
    public void ReduceVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => ReduceBackward(gradOutput, input, means, means, gradInput, outerSize, reduceSize, K.BackwardVariance);
    public void ReduceLogVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer variances, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => ReduceBackward(gradOutput, input, means, variances, gradInput, outerSize, reduceSize, K.BackwardLogVariance);

    public void GluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => GatedBackward(gradOutput, input, gradInput, outerSize, halfDim, K.GateGlu);
    public void GeGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => GatedBackward(gradOutput, input, gradInput, outerSize, halfDim, K.GateGeGlu);
    public void ReGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => GatedBackward(gradOutput, input, gradInput, outerSize, halfDim, K.GateReGlu);
    public void SwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => GatedBackward(gradOutput, input, gradInput, outerSize, halfDim, K.GateSwiGlu);

    public void Pad2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue)
        => Window2D(input, output, batch * channels * outH * outW, batch, channels, inH, inW, outH, outW, padTop, padLeft, padValue, 0);
    public void Pad2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft)
        => Window2D(gradOutput, gradInput, batch * channels * inH * inW, batch, channels, inH, inW, outH, outW, padTop, padLeft, 0f, 1);
    public void Crop2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => Window2D(input, output, batch * channels * outH * outW, batch, channels, inH, inW, outH, outW, offsetH, offsetW, 0f, 2);
    public void Crop2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => Window2D(gradOutput, gradInput, batch * channels * inH * inW, batch, channels, inH, inW, outH, outW, offsetH, offsetW, 0f, 3);
    public void TileLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => GlslNaryOp(S.TileLastAxis, new[] { input, output }, outerSize * innerSize * repeats, new[] { Bits(outerSize), Bits(innerSize), Bits(repeats) });
    public void RepeatElements(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => GlslNaryOp(S.RepeatElements, new[] { input, output }, outerSize * innerSize * repeats, new[] { Bits(outerSize), Bits(innerSize), Bits(repeats) });
    public void LinspaceKernel(IGpuBuffer output, float start, float step, int size)
        => GlslNaryOp(S.Linspace, new[] { output }, size, new[] { FloatBits(start), FloatBits(step), Bits(size) });
    public void DiagKernel(IGpuBuffer input, IGpuBuffer output, int n)
        => GlslNaryOp(S.Diag, new[] { input, output }, n * n, new[] { Bits(n) });
    public void ExtractDiagKernel(IGpuBuffer input, IGpuBuffer output, int n, int cols)
        => GlslNaryOp(S.ExtractDiag, new[] { input, output }, Math.Min(n, cols), new[] { Bits(n), Bits(cols) });
    public void TriangularMask(IGpuBuffer output, int rows, int cols, int diagonal, float maskValue)
        => GlslNaryOp(S.TriangularMask, new[] { output }, rows * cols, new[] { Bits(rows), Bits(cols), Bits(diagonal), FloatBits(maskValue) });

    public void CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
        => GlslNaryOp(S.RowLoss, new[] { predictions, targets, loss }, batchSize, new[] { Bits(batchSize), Bits(numClasses), 0u });
    public void DropoutMask(IGpuBuffer mask, int size, float keepProb, ulong seed)
        => GlslNaryOp(S.DropoutMask, new[] { mask }, size, new[] { Bits(size), FloatBits(keepProb), (uint)seed, (uint)(seed >> 32) });
    public void GaussianNoise(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
        => GlslNaryOp(S.GaussianNoise, new[] { output }, size,
            new[] { Bits(size), FloatBits(mean), FloatBits(stdDev), (uint)seed, (uint)(seed >> 32) });

    public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int M, int N)
        => GlslNaryOp(S.BatchOuterProduct, new[] { a, b, output }, batchSize * M * N, new[] { Bits(batchSize), Bits(M), Bits(N) });

    public void Meshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0, IGpuBuffer output1,
        int n0, int n1, bool xy)
    {
        if (n0 <= 0) throw new ArgumentOutOfRangeException(nameof(n0));
        if (n1 <= 0) throw new ArgumentOutOfRangeException(nameof(n1));
        GlslNaryOp(S.Meshgrid2D, new[] { source0, source1, output0, output1 }, checked(n0 * n1), new[] { Bits(n0), Bits(n1), xy ? 1u : 0u });
    }

}
