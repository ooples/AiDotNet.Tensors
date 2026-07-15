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
