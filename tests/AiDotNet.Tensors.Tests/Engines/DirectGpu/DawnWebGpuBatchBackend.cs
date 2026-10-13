#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// WebGPU's batch execution, fused meshgrid and generic FFT run as their WGSL ports on Dawn (#917, #924, #965). The
/// operations go through the same <see cref="PortableBatchExecution"/> and <see cref="PortableGenericFft"/> calls
/// <c>WebGpuBackend</c> makes, with <see cref="DawnWgslLauncher"/> in place of its browser dispatch, so the parity
/// suites check the code and the shaders WebGPU ships. Buffers live on an OpenCL host, which the launcher copies to
/// and from Dawn around each dispatch.
/// <para>Not verified here, and forwarded to the OpenCL host instead: batch and stream state (WebGPU's waits on every
/// submission), <c>Copy</c> (a buffer copy), and <c>CosineSimilarity</c>, <c>PairwiseDistance</c>, <c>PairwiseDistanceSquared</c>,
/// <c>PixelShuffle</c> and <c>PixelShuffleBackward</c>, which WebGPU dispatches through other WGSL modules.</para>
/// </summary>
internal sealed class DawnWebGpuBatchBackend : DelegatingGpuBackend, IGpuBatchExecution, IMeshgridBackend, IGenericFftBackend, IFftBackend
{
    private readonly OpenClBackend _host;
    private readonly PortableGenericFft.Launcher _launch;
    private readonly PortableBatchExecution _portable;
    private readonly GenericFftReadiness _fftReadiness;

    public DawnWebGpuBatchBackend(OpenClBackend host, DawnWgslLauncher dawn) : base(host)
    {
        _host = host;
        _launch = dawn.For(host);
        // WebGPU's readiness gate, asking Dawn's compiler as WebGpuBackend asks the browser's.
        _fftReadiness = new GenericFftReadiness(dawn.Compile);
        // WebGpuBackend.FullReduce dispatches one workgroup; one thread rounds up to exactly that.
        _portable = new PortableBatchExecution(_launch,
            (input, output, size, mode, maxVal) => _launch("bx_full_reduce", 1, input, output, size, mode, maxVal));
    }

    // ---- forwarded to the host (see the summary) ----
    public bool SupportsBatchExecution => true;
    public IGpuBuffer AllocateWorkspaceBuffer(int totalElements) => _host.AllocateWorkspaceBuffer(totalElements);
    public void BeginBatch() => _host.BeginBatch();
    public void EndBatch() => _host.EndBatch();
    public void InsertBarrier(IGpuBuffer buffer) => _host.InsertBarrier(buffer);
    public void BeginSecondaryStream() => _host.BeginSecondaryStream();
    public void EndSecondaryStream() => _host.EndSecondaryStream();
    public void CosineSimilarity(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
        => _host.CosineSimilarity(a, b, output, batchSize, dim);
    public void PairwiseDistance(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => _host.PairwiseDistance(a, b, output, m, n, dim);
    public void PairwiseDistanceSquared(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => _host.PairwiseDistanceSquared(a, b, output, m, n, dim);
    public void PixelShuffle(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int scale)
        => _host.PixelShuffle(input, output, batch, channels, inH, inW, scale);
    public void PixelShuffleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int scale)
        => _host.PixelShuffleBackward(gradOutput, gradInput, batch, channels, inH, inW, scale);

    // ---- WebGPU's radix-2 BatchedFFT, as WebGpuBackend.BatchedFFT runs a power-of-two length ----
    public override void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int n, bool inverse)
    {
        if (n <= 0 || (n & (n - 1)) != 0)
            throw new NotSupportedException("The Dawn harness runs WebGPU's radix-2 path only; other lengths use fft_1d.");
        PortableGenericFft.BatchedRadix2(this, _launch, inputReal, inputImag, outputReal, outputImag, batch, n, inverse);
    }

    // ---- WebGPU's generic FFT, as WebGpuBackend.CapabilityParity calls it ----
    bool IGenericFftBackend.PrefersGenericFftForPowerOfTwo => false;
    internal GenericFftReadiness FftReadiness => _fftReadiness;
    bool IGenericFftBackend.SupportsFftElementType(FftElementType type) => _fftReadiness.IsReady(type);
    void IGenericFftBackend.ConvertFloatToFftStorage(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
    {
        _fftReadiness.Require(type, "WebGPU");
        PortableGenericFft.ToStorage(this, _launch, source, destination, count, type);
    }
    void IGenericFftBackend.ConvertFftStorageToFloat(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
    {
        _fftReadiness.Require(type, "WebGPU");
        PortableGenericFft.FromStorage(this, _launch, source, destination, count, type);
    }
    void IGenericFftBackend.LaunchFftGeneric(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type)
    {
        _fftReadiness.Require(type, "WebGPU");
        PortableGenericFft.Launch(this, _launch, real, imaginary, batch, n, inverse, type);
    }
    void IFftBackend.LaunchFft(IGpuBuffer buffer, int batchCount, int n, bool inverse)
        => PortableGenericFft.LaunchInterleaved(this, this, buffer, batchCount, n, inverse);

    // ---- WebGPU's batch execution and meshgrid: PortableBatchExecution, as WebGpuBackend delegates ----
    public void ReduceProduct(IGpuBuffer input, IGpuBuffer output, int size)
        => _portable.ReduceProduct(input, output, size);
    public void ReduceNormL2(IGpuBuffer input, IGpuBuffer output, int size)
        => _portable.ReduceNormL2(input, output, size);
    public void ReduceSumOfSquares(IGpuBuffer input, IGpuBuffer output, int size)
        => _portable.ReduceSumOfSquares(input, output, size);
    public void ReduceMaxMagnitude(IGpuBuffer input, IGpuBuffer output, int size)
        => _portable.ReduceMaxMagnitude(input, output, size);
    public void ReduceMinMagnitude(IGpuBuffer input, IGpuBuffer output, int size)
        => _portable.ReduceMinMagnitude(input, output, size);
    public void ReduceLogSumExp(IGpuBuffer input, IGpuBuffer output, float maxVal, int size)
        => _portable.ReduceLogSumExp(input, output, maxVal, size);
    public void VarianceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _portable.VarianceAxis(input, output, outerSize, reduceSize);
    public void StdAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _portable.StdAxis(input, output, outerSize, reduceSize);
    public void ProductAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _portable.ProductAxis(input, output, outerSize, reduceSize);
    public void NormAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _portable.NormAxis(input, output, outerSize, reduceSize);
    public void LogSumExpAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _portable.LogSumExpAxis(input, output, outerSize, reduceSize);
    public void ReduceLogVariance(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
        => _portable.ReduceLogVariance(input, output, outerSize, reduceSize);
    public void CumSumAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.CumSumAxis(input, output, outerSize, innerSize);
    public void NormalizeL2(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.NormalizeL2(input, output, outerSize, innerSize);
    public void LogSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.LogSoftmax(input, output, outerSize, innerSize);
    public void TaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.TaylorSoftmax(input, output, outerSize, innerSize);
    public void SphericalSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.SphericalSoftmax(input, output, outerSize, innerSize);
    public void Sparsemax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.Sparsemax(input, output, outerSize, innerSize);
    public void GumbelSoftmax(IGpuBuffer logits, IGpuBuffer output, int outerSize, int innerSize, float temperature, ulong seed)
        => _portable.GumbelSoftmax(logits, output, outerSize, innerSize, temperature, seed);
    public void ScalarMinusTensor(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => _portable.ScalarMinusTensor(input, output, scalar, size);
    public void DivScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => _portable.DivScalar(input, output, scalar, size);
    public void RsqrtKernel(IGpuBuffer input, IGpuBuffer output, int size)
        => _portable.RsqrtKernel(input, output, size);
    public void ReluDerivative(IGpuBuffer input, IGpuBuffer output, int size)
        => _portable.ReluDerivative(input, output, size);
    public void SigmoidDerivative(IGpuBuffer sigmoidOutput, IGpuBuffer output, int size)
        => _portable.SigmoidDerivative(sigmoidOutput, output, size);
    public void TanhDerivative(IGpuBuffer tanhOutput, IGpuBuffer output, int size)
        => _portable.TanhDerivative(tanhOutput, output, size);
    public void SinCosKernel(IGpuBuffer input, IGpuBuffer sinOutput, IGpuBuffer cosOutput, int size)
        => _portable.SinCosKernel(input, sinOutput, cosOutput, size);
    public void ReduceSumBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _portable.ReduceSumBackward(gradOutput, gradInput, outerSize, reduceSize);
    public void ReduceMaxBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer maxValues, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _portable.ReduceMaxBackward(gradOutput, input, maxValues, gradInput, outerSize, reduceSize);
    public void ReduceVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _portable.ReduceVarianceBackward(gradOutput, input, means, gradInput, outerSize, reduceSize);
    public void ReduceLogVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer variances, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => _portable.ReduceLogVarianceBackward(gradOutput, input, means, variances, gradInput, outerSize, reduceSize);
    public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.BroadcastAddFirst(a, b, output, outerSize, innerSize);
    public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
        => _portable.BroadcastMulFirst(a, b, output, outerSize, innerSize);
    public void GluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _portable.GluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void GeGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _portable.GeGluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void ReGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _portable.ReGluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void SwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
        => _portable.SwiGluBackward(gradOutput, input, gradInput, outerSize, halfDim);
    public void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int aInnerSize, int bInnerSize)
        => _portable.ConcatAxis(a, b, output, outerSize, aInnerSize, bInnerSize);
    public void SliceLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int inputInnerSize, int start, int sliceSize)
        => _portable.SliceLastAxis(input, output, outerSize, inputInnerSize, start, sliceSize);
    public void SetSliceLastAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int outputInnerSize, int start, int sliceSize)
        => _portable.SetSliceLastAxis(output, values, outerSize, outputInnerSize, start, sliceSize);
    public void SliceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int stride, int index)
        => _portable.SliceAxis(input, output, outerSize, axisSize, stride, index);
    public void SetSliceAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int axisSize, int stride, int index)
        => _portable.SetSliceAxis(output, values, outerSize, axisSize, stride, index);
    public void Stack2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => _portable.Stack2(a, b, output, size);
    public void Pad2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue)
        => _portable.Pad2D(input, output, batch, channels, inH, inW, outH, outW, padTop, padLeft, padValue);
    public void Pad2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft)
        => _portable.Pad2DBackward(gradOutput, gradInput, batch, channels, inH, inW, outH, outW, padTop, padLeft);
    public void Crop2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => _portable.Crop2D(input, output, batch, channels, inH, inW, outH, outW, offsetH, offsetW);
    public void Crop2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
        => _portable.Crop2DBackward(gradOutput, gradInput, batch, channels, inH, inW, outH, outW, offsetH, offsetW);
    public void TileLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => _portable.TileLastAxis(input, output, outerSize, innerSize, repeats);
    public void RepeatElements(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
        => _portable.RepeatElements(input, output, outerSize, innerSize, repeats);
    public void LinspaceKernel(IGpuBuffer output, float start, float step, int size)
        => _portable.LinspaceKernel(output, start, step, size);
    public void DiagKernel(IGpuBuffer input, IGpuBuffer output, int n)
        => _portable.DiagKernel(input, output, n);
    public void ExtractDiagKernel(IGpuBuffer input, IGpuBuffer output, int n, int cols)
        => _portable.ExtractDiagKernel(input, output, n, cols);
    public void TriangularMask(IGpuBuffer output, int rows, int cols, int diagonal, float maskValue)
        => _portable.TriangularMask(output, rows, cols, diagonal, maskValue);
    public void IndexSelect(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int numIndices, int innerSize)
        => _portable.IndexSelect(input, indices, output, numIndices, innerSize);
    public void CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
        => _portable.CrossEntropyLoss(predictions, targets, loss, batchSize, numClasses);
    public void MseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
        => _portable.MseLoss(predictions, targets, loss, batchSize, numFeatures);
    public void DropoutMask(IGpuBuffer mask, int size, float keepProb, ulong seed)
        => _portable.DropoutMask(mask, size, keepProb, seed);
    public void GaussianNoise(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
        => _portable.GaussianNoise(output, size, mean, stdDev, seed);
    public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int M, int N)
        => _portable.BatchOuterProduct(a, b, output, batchSize, M, N);
    public void Meshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0, IGpuBuffer output1, int n0, int n1, bool xy)
        => _portable.Meshgrid2DPair(source0, source1, output0, output1, n0, n1, xy);
}
#endif