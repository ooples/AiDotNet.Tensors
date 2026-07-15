#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #775: WebGPU implementations of the per-family extended-conv capability interfaces. Added
// incrementally (interface + WGSL kernels) so a partially-ported backend opts into exactly the families
// it can run; the engine routes the rest to the CPU. WGSL kernels live in WebGpuExtendedConvKernels; this
// partial only wires the dispatch via Dispatch{N}BufferAsync. Scalar params are packed into a float[]
// uniform block (ints reinterpreted as their float bit pattern, floats stored directly, padded to a
// 16-byte multiple to satisfy WGSL uniform alignment).
public sealed partial class WebGpuBackend : ITrilinearInterpolationKernels, IConvTranspose3DKernels, ISpiralConvKernels,
    IAdaptiveMaxPool2DKernels, IConv3DBackwardKernels
{
    private static float Bits(int value) => BitConverter.Int32BitsToSingle(value);

    public void Conv3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(n * inC * inD * inH * inW);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtConv3DBwdIn", WebGpuExtendedConvKernels.Conv3DBackwardInput, "main",
            gradOutput, weights, gradInput,
            Conv3DUniforms(n, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW),
            total).GetAwaiter().GetResult();
    }

    public void Conv3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int inD, int inH, int inW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(outC * inC * kD * kH * kW);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtConv3DBwdW", WebGpuExtendedConvKernels.Conv3DBackwardWeights, "main",
            gradOutput, input, gradKernel,
            Conv3DUniforms(n, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW),
            total).GetAwaiter().GetResult();
    }

    public void AdaptiveMaxPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        int total = checked(batch * channels * outHeight * outWidth);
        if (total <= 0) return;
        Dispatch2BufferAsync("ExtAdaptiveMaxPool2D", WebGpuExtendedConvKernels.AdaptiveMaxPool2D, "main",
            input, output,
            [Bits(batch), Bits(channels), Bits(inHeight), Bits(inWidth), Bits(outHeight), Bits(outWidth), 0f, 0f],
            total).GetAwaiter().GetResult();
    }

    public void SpiralConv(IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer biases, IGpuBuffer output, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(v * outC);
        if (total <= 0) return;
        Dispatch5BufferAsync("ExtSpiralConv", WebGpuExtendedConvKernels.SpiralConv, "main",
            vertexFeatures, spiralIndices, weights, biases, output,
            [Bits(v), Bits(inC), Bits(spiralLength), Bits(outC)], total).GetAwaiter().GetResult();
    }

    public void SpiralConvBackwardInput(IGpuBuffer gradOutput, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer gradVertexFeatures, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(v * inC);
        if (total <= 0) return;
        Dispatch4BufferAsync("ExtSpiralConvBwdIn", WebGpuExtendedConvKernels.SpiralConvBackwardInput, "main",
            gradOutput, spiralIndices, weights, gradVertexFeatures,
            [Bits(v), Bits(inC), Bits(spiralLength), Bits(outC)], total).GetAwaiter().GetResult();
    }

    public void SpiralConvBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices,
        IGpuBuffer gradWeights, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(outC * inC * spiralLength);
        if (total <= 0) return;
        Dispatch4BufferAsync("ExtSpiralConvBwdW", WebGpuExtendedConvKernels.SpiralConvBackwardWeights, "main",
            gradOutput, vertexFeatures, spiralIndices, gradWeights,
            [Bits(v), Bits(inC), Bits(spiralLength), Bits(outC)], total).GetAwaiter().GetResult();
    }

    private static float[] Conv3DUniforms(int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        [Bits(n), Bits(inC), Bits(iD), Bits(iH), Bits(iW), Bits(outC), Bits(outD), Bits(outH), Bits(outW),
         Bits(kD), Bits(kH), Bits(kW), Bits(strideD), Bits(strideH), Bits(strideW), Bits(padD), Bits(padH), Bits(padW), 0f, 0f];

    public void ConvTranspose3D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(n * outC * outD * outH * outW);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtConvT3D", WebGpuExtendedConvKernels.ConvTranspose3D, "main",
            input, weights, output,
            Conv3DUniforms(n, inC, iD, iH, iW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW),
            total).GetAwaiter().GetResult();
    }

    public void ConvTranspose3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(n * inC * iD * iH * iW);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtConvT3DBwdIn", WebGpuExtendedConvKernels.ConvTranspose3DBackwardInput, "main",
            gradOutput, weights, gradInput,
            Conv3DUniforms(n, inC, iD, iH, iW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW),
            total).GetAwaiter().GetResult();
    }

    public void ConvTranspose3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        int total = checked(inC * outC * kD * kH * kW);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtConvT3DBwdW", WebGpuExtendedConvKernels.ConvTranspose3DBackwardWeights, "main",
            gradOutput, input, gradWeights,
            Conv3DUniforms(n, inC, iD, iH, iW, outC, outD, outH, outW, kD, kH, kW, strideD, strideH, strideW, padD, padH, padW),
            total).GetAwaiter().GetResult();
    }

    public void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(p * c);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtTrilinear", WebGpuExtendedConvKernels.TrilinearInterpolate, "main",
            grid, positions, output,
            [Bits(d), Bits(h), Bits(w), Bits(c), Bits(p), upperEps, 0f, 0f], total).GetAwaiter().GetResult();
    }

    public void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(d * h * w * c);
        if (total <= 0) return;
        Dispatch3BufferAsync("ExtTrilinearBackward", WebGpuExtendedConvKernels.TrilinearInterpolateBackward, "main",
            gradOutput, positions, gradGrid,
            [Bits(d), Bits(h), Bits(w), Bits(c), Bits(p), upperEps, 0f, 0f], total).GetAwaiter().GetResult();
    }
}
#endif
