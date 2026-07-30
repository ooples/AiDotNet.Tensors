using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #775: HIP implementations of the per-family extended-conv capability interfaces. Added incrementally
// (interface + kernels) so a partially-ported backend opts into exactly the families it can run; the
// engine routes the rest to the CPU. Kernels live in the existing compiled modules (trilinear in the
// convolution module); this partial only wires the launch.
public sealed partial class HipBackend : ITrilinearInterpolationKernels, IConvTranspose3DKernels, ISpiralConvKernels, IAdaptiveMaxPool2DKernels, IConv3DBackwardKernels, IPool3DKernels, IDepthwiseConv2DBackwardKernels, IGaussianSplatKernels
{
    public unsafe void GaussianCovariance(IGpuBuffer rotations, IGpuBuffer scales, IGpuBuffer covariances, int numGaussians)
    {
        if (numGaussians <= 0) return;
        if (!_kernelCache.TryGetValue("gaussian_covariance", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: gaussian_covariance");
        uint gridDim = (uint)((numGaussians + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr r = rotations.Handle, s = scales.Handle, c = covariances.Handle;
        int ng = numGaussians;
        void** args = stackalloc void*[4];
        args[0] = &r; args[1] = &s; args[2] = &c; args[3] = &ng;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SphericalHarmonics(IGpuBuffer shCoefficients, IGpuBuffer viewDirections, IGpuBuffer output,
        int numPoints, int basisCount, int numChannels, int degree, int broadcastDir)
    {
        int total = checked(numPoints * numChannels);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spherical_harmonics", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: spherical_harmonics");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr sh = shCoefficients.Handle, vd = viewDirections.Handle, o = output.Handle;
        int np = numPoints, bc = basisCount, nc = numChannels, deg = degree, bd = broadcastDir;
        void** args = stackalloc void*[8];
        args[0] = &sh; args[1] = &vd; args[2] = &o; args[3] = &np;
        args[4] = &bc; args[5] = &nc; args[6] = &deg; args[7] = &bd;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SphericalHarmonicsBackward(IGpuBuffer shCoefficients, IGpuBuffer viewDirections,
        IGpuBuffer outputGradient, IGpuBuffer shGrad,
        int numPoints, int basisCount, int numChannels, int degree, int broadcastDir)
    {
        int total = checked(numPoints * basisCount * numChannels);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spherical_harmonics_backward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: spherical_harmonics_backward");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr sh = shCoefficients.Handle, vd = viewDirections.Handle, og = outputGradient.Handle, sg = shGrad.Handle;
        int np = numPoints, bc = basisCount, nc = numChannels, deg = degree, bd = broadcastDir;
        void** args = stackalloc void*[9];
        args[0] = &sh; args[1] = &vd; args[2] = &og; args[3] = &sg; args[4] = &np;
        args[5] = &bc; args[6] = &nc; args[7] = &deg; args[8] = &bd;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public void DepthwiseConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW) =>
        LaunchDepthwise2DGrid("depthwise_conv2d_backward_input", checked(n * inC * h * w),
            gradOutput, kernel, gradInput, n, inC, h, w, m, outH, outW, kH, kW, strideH, strideW, padH, padW);

    public void DepthwiseConv2DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW) =>
        LaunchDepthwise2DGrid("depthwise_conv2d_backward_weights", checked(inC * m * kH * kW),
            gradOutput, input, gradKernel, n, inC, h, w, m, outH, outW, kH, kW, strideH, strideW, padH, padW);

    private unsafe void LaunchDepthwise2DGrid(string kernelName, int total,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW)
    {
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pa = a.Handle, pb = b.Handle, pc = c.Handle;
        int vN = n, vInC = inC, vH = h, vW = w, vM = m, vOH = outH, vOW = outW, vKH = kH, vKW = kW;
        int vSH = strideH, vSW = strideW, vPH = padH, vPW = padW;
        void** args = stackalloc void*[16];
        args[0] = &pa; args[1] = &pb; args[2] = &pc;
        args[3] = &vN; args[4] = &vInC; args[5] = &vH; args[6] = &vW; args[7] = &vM;
        args[8] = &vOH; args[9] = &vOW; args[10] = &vKH; args[11] = &vKW;
        args[12] = &vSH; args[13] = &vSW; args[14] = &vPH; args[15] = &vPW;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void AvgPool3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW, int countIncludePad)
    {
        if (batch * channels * outDepth <= 0) return;
        if (!_kernelCache.TryGetValue("avgpool3d", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: avgpool3d");
        const int blockSize = 8;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);
        IntPtr i = input.Handle, o = output.Handle;
        int vB = batch, vC = channels, vID = inDepth, vIH = inHeight, vIW = inWidth;
        int vOD = outDepth, vOH = outHeight, vOW = outWidth, vKD = kernelD, vKH = kernelH, vKW = kernelW;
        int vSD = strideD, vSH = strideH, vSW = strideW, vCIP = countIncludePad;
        void** args = stackalloc void*[17];
        args[0] = &i; args[1] = &o; args[2] = &vB; args[3] = &vC;
        args[4] = &vID; args[5] = &vIH; args[6] = &vIW;
        args[7] = &vOD; args[8] = &vOH; args[9] = &vOW;
        args[10] = &vKD; args[11] = &vKH; args[12] = &vKW;
        args[13] = &vSD; args[14] = &vSH; args[15] = &vSW; args[16] = &vCIP;
        LaunchKernel3D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, 1u, args);
        Synchronize();
    }

    public unsafe void AvgPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW, int countIncludePad)
    {
        if (batch * channels * inDepth <= 0) return;
        if (!_kernelCache.TryGetValue("avgpool3d_backward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: avgpool3d_backward");
        const int blockSize = 8;
        uint gridX = (uint)((inWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((inHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * inDepth);
        IntPtr go = gradOutput.Handle, gi = gradInput.Handle;
        int vB = batch, vC = channels, vID = inDepth, vIH = inHeight, vIW = inWidth;
        int vOD = outDepth, vOH = outHeight, vOW = outWidth, vKD = kernelD, vKH = kernelH, vKW = kernelW;
        int vSD = strideD, vSH = strideH, vSW = strideW, vCIP = countIncludePad;
        void** args = stackalloc void*[17];
        args[0] = &go; args[1] = &gi; args[2] = &vB; args[3] = &vC;
        args[4] = &vID; args[5] = &vIH; args[6] = &vIW;
        args[7] = &vOD; args[8] = &vOH; args[9] = &vOW;
        args[10] = &vKD; args[11] = &vKH; args[12] = &vKW;
        args[13] = &vSD; args[14] = &vSH; args[15] = &vSW; args[16] = &vCIP;
        LaunchKernel3D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, 1u, args);
        Synchronize();
    }

    public void Conv3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConv3DGrid("conv3d_backward_input", checked(n * inC * inD * inH * inW),
            gradOutput, weights, gradInput, n, inC, inD, inH, inW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public void Conv3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConv3DGrid("conv3d_backward_weights", checked(outC * inC * kD * kH * kW),
            gradOutput, input, gradKernel, n, inC, inD, inH, inW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public unsafe void AdaptiveMaxPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        int total = checked(batch * channels * outHeight * outWidth);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("adaptive_max_pool2d", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: adaptive_max_pool2d");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr i = input.Handle, o = output.Handle;
        int vB = batch, vC = channels, vIH = inHeight, vIW = inWidth, vOH = outHeight, vOW = outWidth;
        void** args = stackalloc void*[8];
        args[0] = &i; args[1] = &o; args[2] = &vB; args[3] = &vC;
        args[4] = &vIH; args[5] = &vIW; args[6] = &vOH; args[7] = &vOW;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(p * c);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("trilinear_interpolate", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: trilinear_interpolate");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr g = grid.Handle, pos = positions.Handle, o = output.Handle;
        int dd = d, hh = h, ww = w, cc = c, pp = p;
        float eps = upperEps;
        void** args = stackalloc void*[9];
        args[0] = &g; args[1] = &pos; args[2] = &o; args[3] = &dd; args[4] = &hh;
        args[5] = &ww; args[6] = &cc; args[7] = &pp; args[8] = &eps;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(d * h * w * c);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("trilinear_interpolate_backward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: trilinear_interpolate_backward");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr go = gradOutput.Handle, pos = positions.Handle, gg = gradGrid.Handle;
        int dd = d, hh = h, ww = w, cc = c, pp = p;
        float eps = upperEps;
        void** args = stackalloc void*[9];
        args[0] = &go; args[1] = &pos; args[2] = &gg; args[3] = &dd; args[4] = &hh;
        args[5] = &ww; args[6] = &cc; args[7] = &pp; args[8] = &eps;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public void ConvTranspose3D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConv3DGrid("conv_transpose3d", checked(n * outC * outD * outH * outW),
            input, weights, output, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public void ConvTranspose3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConv3DGrid("conv_transpose3d_backward_input", checked(n * inC * iD * iH * iW),
            gradOutput, weights, gradInput, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public void ConvTranspose3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConv3DGrid("conv_transpose3d_backward_weights", checked(inC * outC * kD * kH * kW),
            gradOutput, input, gradWeights, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public unsafe void SpiralConv(IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer biases, IGpuBuffer output, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(v * outC);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spiral_conv", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: spiral_conv");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr vf = vertexFeatures.Handle, si = spiralIndices.Handle, w = weights.Handle, b = biases.Handle, o = output.Handle;
        int vv = v, vInC = inC, vSL = spiralLength, vOutC = outC;
        void** args = stackalloc void*[9];
        args[0] = &vf; args[1] = &si; args[2] = &w; args[3] = &b; args[4] = &o;
        args[5] = &vv; args[6] = &vInC; args[7] = &vSL; args[8] = &vOutC;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SpiralConvBackwardInput(IGpuBuffer gradOutput, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer gradVertexFeatures, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(v * inC);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spiral_conv_backward_input", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: spiral_conv_backward_input");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr go = gradOutput.Handle, si = spiralIndices.Handle, w = weights.Handle, gvf = gradVertexFeatures.Handle;
        int vv = v, vInC = inC, vSL = spiralLength, vOutC = outC;
        void** args = stackalloc void*[8];
        args[0] = &go; args[1] = &si; args[2] = &w; args[3] = &gvf;
        args[4] = &vv; args[5] = &vInC; args[6] = &vSL; args[7] = &vOutC;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SpiralConvBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices,
        IGpuBuffer gradWeights, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(outC * inC * spiralLength);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spiral_conv_backward_weights", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: spiral_conv_backward_weights");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr go = gradOutput.Handle, vf = vertexFeatures.Handle, si = spiralIndices.Handle, gw = gradWeights.Handle;
        int vv = v, vInC = inC, vSL = spiralLength, vOutC = outC;
        void** args = stackalloc void*[8];
        args[0] = &go; args[1] = &vf; args[2] = &si; args[3] = &gw;
        args[4] = &vv; args[5] = &vInC; args[6] = &vSL; args[7] = &vOutC;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    private unsafe void LaunchConv3DGrid(string kernelName, int total,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pa = a.Handle, pb = b.Handle, pc = c.Handle;
        int vN = n, vInC = inC, vID = iD, vIH = iH, vIW = iW, vOutC = outC, vOutD = outD, vOutH = outH, vOutW = outW;
        int vKD = kD, vKH = kH, vKW = kW, vSD = strideD, vSH = strideH, vSW = strideW, vPD = padD, vPH = padH, vPW = padW;
        void** args = stackalloc void*[21];
        args[0] = &pa; args[1] = &pb; args[2] = &pc;
        args[3] = &vN; args[4] = &vInC; args[5] = &vID; args[6] = &vIH; args[7] = &vIW;
        args[8] = &vOutC; args[9] = &vOutD; args[10] = &vOutH; args[11] = &vOutW;
        args[12] = &vKD; args[13] = &vKH; args[14] = &vKW;
        args[15] = &vSD; args[16] = &vSH; args[17] = &vSW;
        args[18] = &vPD; args[19] = &vPH; args[20] = &vPW;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }
}
