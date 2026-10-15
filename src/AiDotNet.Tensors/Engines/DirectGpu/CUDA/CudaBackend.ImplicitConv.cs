using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Conv2D forward for narrow layers as an implicit GEMM (CudaImplicitConvKernels): the im2col matrix is gathered from the
// input inside the kernel, so none is written.
public sealed partial class CudaBackend
{
    // AIDOTNET_IMPLICIT_CONV=0: every convolution takes the tuned / generic kernels as before.
    private static readonly bool ImplicitConvEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_IMPLICIT_CONV") != "0";

    private const int ImplicitConvTilePixels = 64, ImplicitConvTileChannels = 32, ImplicitConvThreads = 64;

    /// <summary>
    /// Conv2D forward through conv2d_implicit_gemm for a layer with at most 64 output channels and a reduction
    /// (inChannels * kH * kW) from 32 to 2048 over at least 1024 output pixels -- the shapes where the im2col route
    /// spent as long writing the im2col matrix as multiplying it. False -- nothing launched -- otherwise.
    /// </summary>
    internal unsafe bool TryConv2DImplicitGemm(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        long pixels = (long)batch * outHeight * outWidth;
        long reduction = (long)inChannels * kernelH * kernelW;
        if (!ImplicitConvEnabled || !IsAvailable || batch <= 0 || outHeight <= 0 || outWidth <= 0 || outChannels <= 0
            || outChannels > 64 || reduction < 32 || reduction > 2048 || pixels < 1024
            || (pixels + ImplicitConvTilePixels - 1) / ImplicitConvTilePixels > int.MaxValue
            || strideH < 1 || strideW < 1 || dilationH < 1 || dilationW < 1
            || !_kernelCache.TryGetValue("conv2d_implicit_gemm", out var implicitKernel))
            return false;
        using var _ = PushContext();
        IntPtr x = input.Handle, w = kernel.Handle, y = output.Handle;
        int b = batch, ic = inChannels, ih = inHeight, iw = inWidth, oc = outChannels, oh = outHeight, ow = outWidth;
        int kh = kernelH, kw = kernelW, sh = strideH, sw = strideW, ph = padH, pw = padW, dh = dilationH, dw = dilationW;
        void** args = stackalloc void*[18];
        args[0] = &x; args[1] = &w; args[2] = &y; args[3] = &b; args[4] = &ic; args[5] = &ih; args[6] = &iw;
        args[7] = &oc; args[8] = &oh; args[9] = &ow; args[10] = &kh; args[11] = &kw; args[12] = &sh; args[13] = &sw;
        args[14] = &ph; args[15] = &pw; args[16] = &dh; args[17] = &dw;
        LaunchKernel2D(implicitKernel, (uint)((pixels + ImplicitConvTilePixels - 1) / ImplicitConvTilePixels),
            (uint)((outChannels + ImplicitConvTileChannels - 1) / ImplicitConvTileChannels), 1, ImplicitConvThreads, 1, args);
        return true;
    }
}
