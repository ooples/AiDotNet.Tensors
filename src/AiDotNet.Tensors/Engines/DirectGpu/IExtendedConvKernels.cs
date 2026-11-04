namespace AiDotNet.Tensors.Engines.DirectGpu;

// Optional #775 convolution / pooling / interpolation / mesh / scatter GPU kernels that only some
// backends provide. Split into per-FAMILY capability interfaces so a backend can opt into one family
// at a time (implement the sub-interface + its kernels) instead of the whole surface at once —
// <see cref="DirectGpuTensorEngine"/> checks the specific family interface per op and falls back to the
// CPU when the backend does not implement it. Kept OFF <see cref="IDirectGpuBackend"/> deliberately:
// default interface methods are unavailable on net471, so a backend without these kernels must not be
// forced to carry stubs. Mirrors the existing capability-interface convention (see
// <see cref="IFusedAdvancedKernels"/>). <see cref="IExtendedConvKernels"/> is the composite of every
// family; OpenCL implements all of them via that single declaration.

/// <summary>3D average pooling forward + backward (NCDHW) (#775).</summary>
internal interface IPool3DKernels
{
    /// <summary>3D average pooling (NCDHW). OpenCL implements this (#775).</summary>
    /// <param name="countIncludePad">1 to divide by the full window size, 0 by the valid-element count.</param>
    void AvgPool3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW, int countIncludePad);

    /// <summary>Backward pass for 3D average pooling (NCDHW). Parallelizes over input elements (#775).</summary>
    void AvgPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW, int countIncludePad);
}

/// <summary>Conv3D backward w.r.t. input and weights (NCDHW, no dilation) (#775).</summary>
internal interface IConv3DBackwardKernels
{
    /// <summary>Conv3D backward w.r.t. input (NCDHW, no dilation) (#775).</summary>
    void Conv3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW);

    /// <summary>Conv3D backward w.r.t. weights (NCDHW, no dilation) (#775).</summary>
    void Conv3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW);
}

/// <summary>Depthwise Conv2D backward w.r.t. input and weights (NCHW, oc = ic*M + m) (#775).</summary>
internal interface IDepthwiseConv2DBackwardKernels
{
    /// <summary>Depthwise Conv2D backward w.r.t. input (NCHW, oc = ic*M + m) (#775).</summary>
    void DepthwiseConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW);

    /// <summary>Depthwise Conv2D backward w.r.t. weights (NCHW) (#775).</summary>
    void DepthwiseConv2DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW);
}

/// <summary>Trilinear interpolation forward + backward of a [D,H,W,C] grid at [P,3] positions (#775).</summary>
internal interface ITrilinearInterpolationKernels
{
    /// <summary>Trilinear interpolation of a [D,H,W,C] grid at [P,3] positions -> [P,C] (#775).</summary>
    void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps);

    /// <summary>Trilinear-interpolate backward w.r.t. the grid -> [D,H,W,C] (#775).</summary>
    void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps);
}

/// <summary>ConvTranspose3D forward + backward (NCDHW, weights [inC,outC,kD,kH,kW]) (#775).</summary>
internal interface IConvTranspose3DKernels
{
    /// <summary>ConvTranspose3D forward (NCDHW, weights [inC,outC,kD,kH,kW]) (#775).</summary>
    void ConvTranspose3D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW);

    /// <summary>ConvTranspose3D backward w.r.t. input (#775).</summary>
    void ConvTranspose3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW);

    /// <summary>ConvTranspose3D backward w.r.t. weights (#775).</summary>
    void ConvTranspose3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW);
}

/// <summary>SpiralConv (mesh convolution) forward + backward (#775).</summary>
internal interface ISpiralConvKernels
{
    /// <summary>SpiralConv (mesh conv): weights [outC, inC*spiralLength] -> [V,outC] (#775).</summary>
    void SpiralConv(IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer biases, IGpuBuffer output, int v, int inC, int spiralLength, int outC);

    /// <summary>SpiralConv backward w.r.t. vertex features -> [V,inC] (#775).</summary>
    void SpiralConvBackwardInput(IGpuBuffer gradOutput, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer gradVertexFeatures, int v, int inC, int spiralLength, int outC);

    /// <summary>SpiralConv backward w.r.t. weights -> [outC, inC*spiralLength] (#775).</summary>
    void SpiralConvBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices,
        IGpuBuffer gradWeights, int v, int inC, int spiralLength, int outC);
}

/// <summary>Limits of <see cref="IRectSliceKernels"/>. A class rather than interface constants: net471 has no
/// static interface members.</summary>
internal static class RectSliceLimits
{
    /// <summary>Largest rank <see cref="IRectSliceKernels.RectSlice"/> handles.</summary>
    internal const int MaxRank = 8;
}
/// <summary>
/// Validation and metadata shared by every <see cref="IRectSliceKernels"/> implementation, so each backend launches
/// from the same checks. An out-of-bounds launch is a sticky error on CUDA and HIP that poisons the context for every
/// engine in the process, so the buffers are checked against the shapes, not just the shapes themselves.
/// </summary>
internal static class RectSliceGeometry
{
    /// <summary>
    /// Fills <paramref name="outDims"/>, <paramref name="fullStrides"/> and <paramref name="starts"/> (each at least
    /// <see cref="RectSliceLimits.MaxRank"/> long; unused entries are zero) and returns the rank and element count.
    /// </summary>
    internal static void Build(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length,
        int[] outDims, int[] fullStrides, int[] starts, out int rank, out int total)
    {
        if (full is null) throw new ArgumentNullException(nameof(full));
        if (slice is null) throw new ArgumentNullException(nameof(slice));
        if (fullShape is null) throw new ArgumentNullException(nameof(fullShape));
        if (start is null) throw new ArgumentNullException(nameof(start));
        if (length is null) throw new ArgumentNullException(nameof(length));
        rank = fullShape.Length;
        if (rank < 1 || rank > RectSliceLimits.MaxRank || start.Length != rank || length.Length != rank)
            throw new ArgumentException($"RectSlice supports rank 1..{RectSliceLimits.MaxRank} with matching start/length.");
        Array.Clear(outDims, 0, outDims.Length);
        Array.Clear(fullStrides, 0, fullStrides.Length);
        Array.Clear(starts, 0, starts.Length);
        int stride = 1;
        total = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            // Compared as start > shape - length: start + length can wrap past int.MaxValue and pass.
            if (fullShape[d] < 1 || start[d] < 0 || length[d] < 1 || start[d] > fullShape[d] - length[d])
                throw new ArgumentOutOfRangeException(nameof(start), $"Axis {d}: [{start[d]}, +{length[d]}) is outside {fullShape[d]}.");
            outDims[d] = length[d];
            fullStrides[d] = stride;
            starts[d] = start[d];
            stride = checked(stride * fullShape[d]);
            total = checked(total * length[d]);
        }
        if (full.Size < stride)
            throw new ArgumentException($"RectSlice: the full buffer holds {full.Size} elements, the shape needs {stride}.", nameof(full));
        if (slice.Size < total)
            throw new ArgumentException($"RectSlice: the slice buffer holds {slice.Size} elements, the slice needs {total}.", nameof(slice));
    }

    /// <summary><see cref="Build(IGpuBuffer, IGpuBuffer, int[], int[], int[], int[], int[], int[], out int, out int)"/>
    /// into the fixed buffers of a by-value kernel parameter struct (CUDA, HIP).</summary>
    internal static unsafe void Build(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length,
        int* outDims, int* fullStrides, int* starts, out int rank, out int total)
    {
        var d = new int[RectSliceLimits.MaxRank];
        var s = new int[RectSliceLimits.MaxRank];
        var o = new int[RectSliceLimits.MaxRank];
        Build(full, slice, fullShape, start, length, d, s, o, out rank, out total);
        for (int i = 0; i < RectSliceLimits.MaxRank; i++) { outDims[i] = d[i]; fullStrides[i] = s[i]; starts[i] = o[i]; }
    }
}

/// <summary>
/// Rectangular N-d slice gather/scatter in one launch (rank &lt;= 8). Replaces the per-contiguous-row
/// device copy loop, which issued one memcpy per row: a [64, 32, 7, 7] height slice was 4,096 API calls.
/// </summary>
/// <remarks>
/// Implemented by all six GPU backends (CUDA, HIP, Metal, OpenCL, Vulkan, WebGPU), each validating through
/// <see cref="RectSliceGeometry"/>. The engine still checks <c>backend is IRectSliceKernels</c>, so a future backend
/// without it keeps the per-row device copy, which gives the same result with more launches.
/// </remarks>
internal interface IRectSliceKernels
{
    /// <summary>
    /// <paramref name="scatter"/> false: <paramref name="slice"/> = <paramref name="full"/>[start : start + length].
    /// <paramref name="scatter"/> true: writes <paramref name="slice"/> into that window of <paramref name="full"/>
    /// (other elements untouched). <paramref name="fullShape"/> is the contiguous row-major shape of <paramref name="full"/>.
    /// </summary>
    void RectSlice(IGpuBuffer full, IGpuBuffer slice, int[] fullShape, int[] start, int[] length, bool scatter);
}
/// <summary>
/// Multi-tensor reductions for a training step's global gradient-norm clip and finiteness check: one launch over
/// every gradient instead of one chain per tensor (~630 launches per N-BEATS step on the per-tensor loop).
/// </summary>
/// <remarks>
/// Implemented by all six GPU backends, every one to <see cref="Gpu.DeterministicGradientNorm"/>'s integer definition,
/// so the record, the coefficient and the scaled gradients have the same bits on every backend and on the CPU. CUDA and
/// HIP launch once over every tensor through a device table of buffer addresses; OpenCL, Metal, Vulkan and WebGPU
/// cannot address an arbitrary buffer from a table, so they launch one chunk reduction per tensor. A second launch
/// combines the chunk partials.
/// </remarks>
/// <summary>The norm record and the clip coefficient in one launch (CUDA, HIP). <see cref="MultiTensorArgs.NormAndClipScale"/>
/// uses it where implemented and the separate <see cref="IMultiTensorKernels"/> launches elsewhere; the bits are the
/// same either way.</summary>
internal interface IFusedNormClipKernels
{
    /// <summary>Writes <paramref name="record"/> as <see cref="IMultiTensorKernels.MultiTensorSumOfSquares"/> and, when
    /// <paramref name="scale"/> is not null, the coefficient as <see cref="IMultiTensorKernels.ClipScaleFromRecord"/>.</summary>
    void MultiTensorNormAndClipScale(System.Collections.Generic.IReadOnlyList<IGpuBuffer> tensors,
        System.Collections.Generic.IReadOnlyList<int> sizes, IGpuBuffer record, float maxNorm, IGpuBuffer? scale,
        Gpu.NonFiniteGradientPolicy policy);
}

internal interface IMultiTensorKernels
{
    /// <summary>Writes the deterministic norm record (<see cref="Gpu.DeterministicGradientNorm"/>, at least
    /// <see cref="Gpu.DeterministicGradientNorm.RecordWords"/> slots; its first two hold the double sum of squares)
    /// of every element of every tensor to <paramref name="sumOfSquares"/>: the same bits on every backend.</summary>
    void MultiTensorSumOfSquares(System.Collections.Generic.IReadOnlyList<IGpuBuffer> tensors,
        System.Collections.Generic.IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares);

    /// <summary>Writes PyTorch's clip_grad_norm_ coefficient, min(1, maxNorm / (norm + 1e-6)), to
    /// <paramref name="scale"/> and the record on the device, from a <see cref="MultiTensorSumOfSquares"/> record;
    /// <paramref name="policy"/> sets it for a non-finite element.</summary>
    void ClipScaleFromRecord(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale, Gpu.NonFiniteGradientPolicy policy);

    /// <summary><see cref="ClipScaleFromRecord"/> with <see cref="Gpu.NonFiniteGradientPolicy.LeaveUnscaled"/>.</summary>
    void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale);

    /// <summary>Scales every tensor in place by the device scalar <paramref name="scale"/>.</summary>
    void MultiTensorScaleByDeviceScalar(System.Collections.Generic.IReadOnlyList<IGpuBuffer> tensors,
        System.Collections.Generic.IReadOnlyList<int> sizes, IGpuBuffer scale);

    /// <summary>The clip's scale pass: <see cref="MultiTensorScaleByDeviceScalar"/>, and with
    /// <paramref name="zeroNonFinite"/> (<see cref="Gpu.NonFiniteGradientPolicy.ZeroNonFinite"/>) every NaN or infinite
    /// element becomes +0 instead.</summary>
    void MultiTensorClipScale(System.Collections.Generic.IReadOnlyList<IGpuBuffer> tensors,
        System.Collections.Generic.IReadOnlyList<int> sizes, IGpuBuffer scale, bool zeroNonFinite);

    /// <summary>Downloads a <see cref="MultiTensorSumOfSquares"/> result (the layout is this backend's) as a
    /// double. A host read: callers defer it until the value is actually wanted.</summary>
    double ReadSumOfSquares(IGpuBuffer sumOfSquares);
}

/// <summary>Argument checks every <see cref="IMultiTensorKernels"/> implementation shares.</summary>
internal static class MultiTensorArgs
{
    /// <summary>The norm record and, when <paramref name="scale"/> is not null, the clip coefficient: one launch on a
    /// backend with <see cref="IFusedNormClipKernels"/>, else the record reduction then the coefficient.</summary>
    internal static void NormAndClipScale(IMultiTensorKernels multi, System.Collections.Generic.IReadOnlyList<IGpuBuffer> tensors,
        System.Collections.Generic.IReadOnlyList<int> sizes, IGpuBuffer record, float maxNorm, IGpuBuffer? scale,
        Gpu.NonFiniteGradientPolicy policy)
    {
        if (multi is IFusedNormClipKernels fused)
        {
            fused.MultiTensorNormAndClipScale(tensors, sizes, record, maxNorm, scale, policy);
            return;
        }
        multi.MultiTensorSumOfSquares(tensors, sizes, record);
        if (scale is not null) multi.ClipScaleFromRecord(record, maxNorm, scale, policy);
    }

    /// <summary>Threads of a grid-strided scale dispatch (Vulkan, WebGPU): 65535 groups of 256, the spec minimum of
    /// the X workgroup count, so the dispatch is legal on every device and each thread loops over the rest.</summary>
    internal const int MaxStridedScaleThreads = 65535 * 256;

    /// <summary>The work-group size of the per-tensor reduction on OpenCL, Metal, Vulkan and WebGPU.</summary>
    internal const int ReductionGroupSize = 256;

    internal static void Validate(System.Collections.Generic.IReadOnlyList<IGpuBuffer> tensors,
        System.Collections.Generic.IReadOnlyList<int> sizes)
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        if (sizes is null) throw new ArgumentNullException(nameof(sizes));
        if (sizes.Count != tensors.Count) throw new ArgumentException("Every tensor needs a size.", nameof(sizes));
        for (int t = 0; t < tensors.Count; t++)
        {
            if (tensors[t] is null) throw new ArgumentException($"Tensor {t} is null.", nameof(tensors));
            if (sizes[t] <= 0) throw new ArgumentOutOfRangeException(nameof(sizes), "Every tensor size must be positive.");
            if (tensors[t].Size < sizes[t]) throw new ArgumentException($"Tensor {t}'s buffer is smaller than its size.", nameof(tensors));
        }
    }

    /// <summary>The double a sum-of-squares buffer holds as two float slots (low word first): the layout of the CUDA,
    /// HIP, OpenCL, Metal and WebGPU reductions.</summary>
    internal static double DoubleFromSlots(float[] slots)
    {
        if (slots is null || slots.Length < 2) throw new ArgumentException("The sum of squares needs two float slots.", nameof(slots));
        var bytes = new byte[8];
        Array.Copy(BitConverter.GetBytes(slots[0]), 0, bytes, 0, 4);
        Array.Copy(BitConverter.GetBytes(slots[1]), 0, bytes, 4, 4);
        if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToDouble(bytes, 0);
    }

    /// <summary>A norm record (<see cref="Gpu.DeterministicGradientNorm.RecordWords"/> words) from its float slots.</summary>
    internal static uint[] RecordFromSlots(float[] slots)
    {
        if (slots is null || slots.Length < Gpu.DeterministicGradientNorm.RecordWords)
            throw new ArgumentException("A norm record needs RecordWords slots.", nameof(slots));
        var words = new uint[Gpu.DeterministicGradientNorm.RecordWords];
        Buffer.BlockCopy(slots, 0, words, 0, words.Length * sizeof(uint));
        return words;
    }

    internal static void ValidateRecordBuffer(IGpuBuffer record)
    {
        if (record is null) throw new ArgumentNullException(nameof(record));
        if (record.Size < Gpu.DeterministicGradientNorm.RecordWords)
            throw new ArgumentException($"A norm record needs {Gpu.DeterministicGradientNorm.RecordWords} slots.", nameof(record));
    }

    internal static void ValidateSumBuffer(IGpuBuffer sumOfSquares)
    {
        if (sumOfSquares is null) throw new ArgumentNullException(nameof(sumOfSquares));
        if (sumOfSquares.Size < 2) throw new ArgumentException("The sum of squares needs two float slots.", nameof(sumOfSquares));
    }
}
/// <summary>Adaptive max pooling 2D (NCHW) (#775).</summary>
internal interface IAdaptiveMaxPool2DKernels
{
    /// <summary>Adaptive max pooling 2D (NCHW) -> [batch, channels, outHeight, outWidth] (#775).</summary>
    void AdaptiveMaxPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth);
}

/// <summary>Gaussian-splat covariance + spherical-harmonics color eval/backward (#775).</summary>
internal interface IGaussianSplatKernels
{
    /// <summary>3D Gaussian-splat covariance: rotations [N,4] (quaternion), scales [N,3] -> [N,6] upper
    /// triangular of R*S^2*R^T (#775).</summary>
    void GaussianCovariance(IGpuBuffer rotations, IGpuBuffer scales, IGpuBuffer covariances, int numGaussians);

    /// <summary>Spherical-harmonics color eval: shCoefficients [N,basisCount,numChannels], viewDirections
    /// [N or 1,3] -> colors [N,numChannels] (#775).</summary>
    void SphericalHarmonics(IGpuBuffer shCoefficients, IGpuBuffer viewDirections, IGpuBuffer output,
        int numPoints, int basisCount, int numChannels, int degree, int broadcastDir);

    /// <summary>SH backward w.r.t. coefficients -> shGrad [N,basisCount,numChannels] (#775).</summary>
    void SphericalHarmonicsBackward(IGpuBuffer shCoefficients, IGpuBuffer viewDirections,
        IGpuBuffer outputGradient, IGpuBuffer shGrad,
        int numPoints, int basisCount, int numChannels, int degree, int broadcastDir);
}

/// <summary>GNN scatter-reduce-along-dim-0 forward + backward family (gather-form, deterministic) (#775).</summary>
internal interface IScatterRowsKernels
{
    /// <summary>GNN scatter-add (index_add) along dim 0: source [srcDimSize,innerSize] + per-row indices
    /// -> output [outDimSize,innerSize] (#775).</summary>
    void ScatterAddRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int srcDimSize, int innerSize, int outDimSize);

    /// <summary>GNN scatter-mean along dim 0 (scatter-add / per-output-row count) (#775).</summary>
    void ScatterMeanRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int srcDimSize, int innerSize, int outDimSize);

    /// <summary>GNN scatter-max along dim 0; empty output rows -> -INFINITY (#775).</summary>
    void ScatterMaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int srcDimSize, int innerSize, int outDimSize);

    /// <summary>GNN scatter-softmax (softmax within each index-group); output has the source shape (#775).</summary>
    void ScatterSoftmaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int srcDimSize, int innerSize, int numGroups);

    /// <summary>ScatterAdd backward (gather) -> gradSource [srcDimSize, innerSize] (#775).</summary>
    void ScatterAddBackwardRows(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradSource,
        int srcDimSize, int innerSize, int outDimSize);

    /// <summary>ScatterMean backward (gather / count) -> gradSource [srcDimSize, innerSize] (#775).</summary>
    void ScatterMeanBackwardRows(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer counts,
        IGpuBuffer gradSource, int srcDimSize, int innerSize, int outDimSize);

    /// <summary>ScatterMax backward: route each output element's grad to its argmax source row (#775).</summary>
    void ScatterMaxBackwardRows(IGpuBuffer gradOutput, IGpuBuffer argmax, IGpuBuffer gradSource,
        int srcDimSize, int innerSize, int outDimSize);

    /// <summary>ScatterSoftmax backward (softmax jacobian within each index-group) (#775).</summary>
    void ScatterSoftmaxBackwardRows(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer indices,
        IGpuBuffer gradSource, int srcDimSize, int innerSize, int numGroups);
}

/// <summary>
/// Composite of every #775 extended-kernel family. A backend that implements the full surface (e.g.
/// OpenCL) declares this single interface; the engine's per-family dispatch still matches because this
/// composite derives from each family interface. Backends adding support incrementally implement the
/// individual family interfaces instead.
/// </summary>
internal interface IExtendedConvKernels :
    IPool3DKernels,
    IConv3DBackwardKernels,
    IDepthwiseConv2DBackwardKernels,
    ITrilinearInterpolationKernels,
    IConvTranspose3DKernels,
    ISpiralConvKernels,
    IAdaptiveMaxPool2DKernels,
    IGaussianSplatKernels,
    IScatterRowsKernels
{
}
