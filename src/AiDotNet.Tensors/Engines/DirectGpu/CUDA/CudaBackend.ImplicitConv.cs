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
    /// The shapes conv2d_implicit_gemm takes: at most 64 output channels and a reduction (inChannels * kH * kW) from 32
    /// to 2048 over at least 1024 output pixels. Shared by <see cref="TryConv2DImplicitGemm"/> and its tuned-slot
    /// applicability.
    /// </summary>
    private static bool ImplicitConvAdmits(int batch, int inChannels, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int dilationH, int dilationW)
    {
        long pixels = (long)batch * outHeight * outWidth;
        long reduction = (long)inChannels * kernelH * kernelW;
        return ImplicitConvEnabled && batch > 0 && outHeight > 0 && outWidth > 0 && outChannels > 0
            && outChannels <= 64 && reduction >= 32 && reduction <= 2048 && pixels >= 1024
            && (pixels + ImplicitConvTilePixels - 1) / ImplicitConvTilePixels <= int.MaxValue
            && strideH >= 1 && strideW >= 1 && dilationH >= 1 && dilationW >= 1;
    }

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
        if (!IsAvailable || !ImplicitConvAdmits(batch, inChannels, outChannels, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW, dilationH, dilationW)
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

    /// <summary>
    /// A convolution's weight gradient through conv2d_weight_grad_implicit: dW[oc][(ic, r, s)] summed over every output
    /// pixel straight from the input and dY, split over the pixels with the per-tile last-block reduction. The im2col
    /// route wrote a [K, rows] matrix, ran one GEMM per image into partials and summed them (~170 us per step for the
    /// parity CNN's second layer). Partials use the thread's cuBLAS workspace, in this stream's order. False -- nothing
    /// launched -- when the shape, workspace or kernel does not qualify.
    /// </summary>
    internal unsafe bool TryConv2DBackwardKernelImplicitNarrow(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        long rows = (long)batch * outHeight * outWidth;
        long reduction = (long)inChannels * kernelH * kernelW;
        if (!ImplicitConvEnabled || !IsAvailable || batch <= 0 || outHeight <= 0 || outWidth <= 0 || outChannels <= 0
            || outChannels > 256 || reduction < 32 || reduction > 4096 || rows < 4096 || rows > int.MaxValue
            || strideH < 1 || strideW < 1 || dilationH < 1 || dilationW < 1
            || !_kernelCache.TryGetValue("conv2d_weight_grad_implicit", out var weightGradKernel))
            return false;
        if (!CurrentCublasHasWorkspace || _threadCublas?.Value is not { Workspace: { } workspace, SkinnyCounters: { } counters })
            return false;
        int tilesK = (int)((reduction + 63) / 64), tilesOc = (outChannels + 31) / 32, tiles = tilesK * tilesOc;
        if (tiles > SkinnyCounterSlots) return false;
        // ~64 blocks in all, each split a whole number of the block's 64-row passes.
        int splits = (int)Math.Max(1, Math.Min(64 / tiles, rows / 256));
        long rowsPerSplit = (rows + splits - 1) / splits;
        rowsPerSplit = (rowsPerSplit + 63) / 64 * 64;
        splits = (int)((rows + rowsPerSplit - 1) / rowsPerSplit);
        if ((long)splits * tiles * 2048 > CublasWorkspaceFloats) return false;

        using var _ = PushContext();
        IntPtr x = input.Handle, dy = gradOutput.Handle, dw = gradKernel.Handle, p = workspace.Handle, cnt = counters.Handle;
        int b = batch, ic = inChannels, ih = inHeight, iw = inWidth, oc = outChannels, oh = outHeight, ow = outWidth;
        int kh = kernelH, kw = kernelW, sh = strideH, sw = strideW, ph = padH, pw = padW, dh = dilationH, dwl = dilationW;
        int rps = (int)rowsPerSplit;
        void** args = stackalloc void*[21];
        args[0] = &x; args[1] = &dy; args[2] = &dw; args[3] = &p; args[4] = &cnt; args[5] = &b; args[6] = &ic;
        args[7] = &ih; args[8] = &iw; args[9] = &oc; args[10] = &oh; args[11] = &ow; args[12] = &kh; args[13] = &kw;
        args[14] = &sh; args[15] = &sw; args[16] = &ph; args[17] = &pw; args[18] = &dh; args[19] = &dwl; args[20] = &rps;
        LaunchKernel2D(weightGradKernel, (uint)tilesK, (uint)tilesOc, (uint)splits, 256, 1, args);
        return true;
    }
}