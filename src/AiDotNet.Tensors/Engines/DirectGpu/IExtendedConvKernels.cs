namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional #775 convolution / pooling / interpolation / mesh GPU kernels that only some backends
/// provide. A backend opts in by implementing this interface; <see cref="DirectGpuTensorEngine"/>
/// checks <c>backend is IExtendedConvKernels</c> and, when the backend does not implement it, falls
/// back to the CPU. Kept OFF <see cref="IDirectGpuBackend"/> deliberately: default interface methods
/// are unavailable on net471, so a backend without these kernels must not be forced to carry stubs.
/// This mirrors the existing capability-interface convention (see <see cref="IFusedAdvancedKernels"/>).
/// </summary>
internal interface IExtendedConvKernels
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

    /// <summary>Conv3D backward w.r.t. input (NCDHW, no dilation) (#775).</summary>
    void Conv3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW);

    /// <summary>Conv3D backward w.r.t. weights (NCDHW, no dilation) (#775).</summary>
    void Conv3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW);

    /// <summary>Depthwise Conv2D backward w.r.t. input (NCHW, oc = ic*M + m) (#775).</summary>
    void DepthwiseConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW);

    /// <summary>Depthwise Conv2D backward w.r.t. weights (NCHW) (#775).</summary>
    void DepthwiseConv2DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW);

    /// <summary>Trilinear interpolation of a [D,H,W,C] grid at [P,3] positions -> [P,C] (#775).</summary>
    void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps);

    /// <summary>Trilinear-interpolate backward w.r.t. the grid -> [D,H,W,C] (#775).</summary>
    void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps);

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

    /// <summary>SpiralConv (mesh conv): weights [outC, inC*spiralLength] -> [V,outC] (#775).</summary>
    void SpiralConv(IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer biases, IGpuBuffer output, int v, int inC, int spiralLength, int outC);

    /// <summary>SpiralConv backward w.r.t. vertex features -> [V,inC] (#775).</summary>
    void SpiralConvBackwardInput(IGpuBuffer gradOutput, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer gradVertexFeatures, int v, int inC, int spiralLength, int outC);

    /// <summary>SpiralConv backward w.r.t. weights -> [outC, inC*spiralLength] (#775).</summary>
    void SpiralConvBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices,
        IGpuBuffer gradWeights, int v, int inC, int spiralLength, int outC);

    /// <summary>Adaptive max pooling 2D (NCHW) -> [batch, channels, outHeight, outWidth] (#775).</summary>
    void AdaptiveMaxPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth);

    /// <summary>3D Gaussian-splat covariance: rotations [N,4] (quaternion), scales [N,3] -> [N,6] upper
    /// triangular of R*S^2*R^T (#775).</summary>
    void GaussianCovariance(IGpuBuffer rotations, IGpuBuffer scales, IGpuBuffer covariances, int numGaussians);
}
