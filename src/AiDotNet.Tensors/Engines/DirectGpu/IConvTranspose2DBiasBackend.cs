// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The transposed-convolution forward plus a per-output-channel bias: y[n, c] = convTranspose2d(x, k)[n, c] + bias[c],
/// with the values of ConvTranspose2D followed by Conv2DBiasAdd. A backend whose transposed route can add the bias as it
/// writes the output (CUDA's single-GEMM overlap-add) does so, saving the bias pass's read and write of the output; the
/// others run their transposed forward and their bias kernel.
/// </summary>
internal interface IConvTranspose2DBiasBackend
{
    void ConvTranspose2DBias(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int outputPadH, int outputPadW);
}
