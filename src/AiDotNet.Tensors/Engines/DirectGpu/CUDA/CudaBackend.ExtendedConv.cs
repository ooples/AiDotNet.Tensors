using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// #775: CUDA implementations of the per-family extended-conv capability interfaces. Each family is
// added incrementally (interface + kernels) so a partially-ported backend still opts into exactly the
// families it can run; the engine routes the rest to the CPU. Kernels live in the existing compiled
// modules (trilinear in the convolution module); this partial only wires the launch.
public sealed partial class CudaBackend : ITrilinearInterpolationKernels, IConvTranspose3DKernels, ISpiralConvKernels
{
    public unsafe void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(p * c);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("trilinear_interpolate", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: trilinear_interpolate");
        using var _ = PushContext();
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr g = grid.Handle, pos = positions.Handle, o = output.Handle;
        int dd = d, hh = h, ww = w, cc = c, pp = p;
        float eps = upperEps;
        void** args = stackalloc void*[9];
        args[0] = &g; args[1] = &pos; args[2] = &o; args[3] = &dd; args[4] = &hh;
        args[5] = &ww; args[6] = &cc; args[7] = &pp; args[8] = &eps;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }

    public unsafe void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(d * h * w * c);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("trilinear_interpolate_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: trilinear_interpolate_backward");
        using var _ = PushContext();
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr go = gradOutput.Handle, pos = positions.Handle, gg = gradGrid.Handle;
        int dd = d, hh = h, ww = w, cc = c, pp = p;
        float eps = upperEps;
        void** args = stackalloc void*[9];
        args[0] = &go; args[1] = &pos; args[2] = &gg; args[3] = &dd; args[4] = &hh;
        args[5] = &ww; args[6] = &cc; args[7] = &pp; args[8] = &eps;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }

    public void ConvTranspose3D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConvTranspose3D("conv_transpose3d", checked(n * outC * outD * outH * outW),
            input, weights, output, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public void ConvTranspose3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConvTranspose3D("conv_transpose3d_backward_input", checked(n * inC * iD * iH * iW),
            gradOutput, weights, gradInput, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public void ConvTranspose3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        LaunchConvTranspose3D("conv_transpose3d_backward_weights", checked(inC * outC * kD * kH * kW),
            gradOutput, input, gradWeights, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public unsafe void SpiralConv(IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer biases, IGpuBuffer output, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(v * outC);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spiral_conv", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: spiral_conv");
        using var _ = PushContext();
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr vf = vertexFeatures.Handle, si = spiralIndices.Handle, w = weights.Handle, b = biases.Handle, o = output.Handle;
        int vv = v, vInC = inC, vSL = spiralLength, vOutC = outC;
        void** args = stackalloc void*[9];
        args[0] = &vf; args[1] = &si; args[2] = &w; args[3] = &b; args[4] = &o;
        args[5] = &vv; args[6] = &vInC; args[7] = &vSL; args[8] = &vOutC;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }

    public unsafe void SpiralConvBackwardInput(IGpuBuffer gradOutput, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer gradVertexFeatures, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(v * inC);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spiral_conv_backward_input", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: spiral_conv_backward_input");
        using var _ = PushContext();
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr go = gradOutput.Handle, si = spiralIndices.Handle, w = weights.Handle, gvf = gradVertexFeatures.Handle;
        int vv = v, vInC = inC, vSL = spiralLength, vOutC = outC;
        void** args = stackalloc void*[8];
        args[0] = &go; args[1] = &si; args[2] = &w; args[3] = &gvf;
        args[4] = &vv; args[5] = &vInC; args[6] = &vSL; args[7] = &vOutC;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }

    public unsafe void SpiralConvBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices,
        IGpuBuffer gradWeights, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(outC * inC * spiralLength);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("spiral_conv_backward_weights", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: spiral_conv_backward_weights");
        using var _ = PushContext();
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr go = gradOutput.Handle, vf = vertexFeatures.Handle, si = spiralIndices.Handle, gw = gradWeights.Handle;
        int vv = v, vInC = inC, vSL = spiralLength, vOutC = outC;
        void** args = stackalloc void*[8];
        args[0] = &go; args[1] = &vf; args[2] = &si; args[3] = &gw;
        args[4] = &vv; args[5] = &vInC; args[6] = &vSL; args[7] = &vOutC;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
    }

    private unsafe void LaunchConvTranspose3D(string kernelName, int total,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
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
    }
}
