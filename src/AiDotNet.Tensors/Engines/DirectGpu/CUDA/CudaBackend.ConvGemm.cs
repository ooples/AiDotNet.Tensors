using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Conv2D backward as im2col + cuBLAS GEMM, the formulation cuDNN's implicit-GEMM algorithms implement. The direct
// conv2d_backward_kernel / conv2d_backward_input kernels loop per output element: on the PyTorch-parity CNN they were
// 72% of a training step's GPU time (weight gradient 2.1 ms per layer vs cuDNN's ~0.12 ms).
//
// Layouts (row-major): input/gradInput [B, Cin, H, W]; gradOutput [B, Cout, L] with L = outH*outW; kernel and
// gradKernel [Cout, P] with P = Cin*kH*kW; im2col output col [B, P, L] (CudaConvolutionKernels im2col/col2im).
public sealed partial class CudaBackend
{
    private IGpuBuffer? _convColScratch;
    private IGpuBuffer? _convWeightPartials;
    private IGpuBuffer? _convOnes;
    private int _convOnesLength;

    /// <summary>AIDOTNET_CONV_GEMM_BACKWARD=0 restores the direct kernels.</summary>
    private static readonly bool s_convGemmBackward =
        System.Environment.GetEnvironmentVariable("AIDOTNET_CONV_GEMM_BACKWARD") != "0";

    private IGpuBuffer? TryConvScratch(ref IGpuBuffer? slot, long floats)
    {
        // Grown only outside a capture (shapes are fixed per plan, so the eager warmup steps size it); a captured
        // graph bakes the pointer, so it must stay the same buffer afterwards. Null when it cannot be sized (too large
        // for one buffer, or undersized during a capture, where allocating would abort it): the caller falls through
        // to the direct kernel, as it does when the ones-vector cannot be built.
        if (slot is null || slot.Size < floats)
        {
            if (floats > int.MaxValue || IsStreamCapturing())
                return null;
            slot?.Dispose();
            slot = AllocateBuffer((int)floats);
        }
        return slot;
    }

    private bool ConvGemmUsable()
        => s_convGemmBackward && (!IsStreamCapturing() || CurrentCublasHasWorkspace);

    private unsafe void LaunchIm2Col(IGpuBuffer input, IGpuBuffer col,
        int batch, int channels, int height, int width, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outH, int outW)
    {
        if (!_kernelCache.TryGetValue("im2col", out var kernel))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: im2col");
        IntPtr inPtr = input.Handle, outPtr = col.Handle;
        int totalPatches = batch * outH * outW;
        void** args = stackalloc void*[16];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &batch; args[3] = &channels; args[4] = &height; args[5] = &width;
        args[6] = &kernelH; args[7] = &kernelW; args[8] = &strideH; args[9] = &strideW;
        args[10] = &padH; args[11] = &padW; args[12] = &dilationH; args[13] = &dilationW;
        args[14] = &outH; args[15] = &outW;
        LaunchKernel(kernel, (uint)((totalPatches + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    private unsafe void LaunchCol2Im(IGpuBuffer col, IGpuBuffer output,
        int batch, int channels, int height, int width, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outH, int outW)
    {
        if (!_kernelCache.TryGetValue("col2im", out var kernel))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: col2im");
        IntPtr inPtr = col.Handle, outPtr = output.Handle;
        int total = batch * channels * height * width;
        void** args = stackalloc void*[16];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &batch; args[3] = &channels; args[4] = &height; args[5] = &width;
        args[6] = &kernelH; args[7] = &kernelW; args[8] = &strideH; args[9] = &strideW;
        args[10] = &padH; args[11] = &padW; args[12] = &dilationH; args[13] = &dilationW;
        args[14] = &outH; args[15] = &outW;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    /// <summary>
    /// #1313: the implicit-GEMM weight gradient; AIDOTNET_CONV_IMPLICIT_WGRAD=0 starts it off (im2col + GEMM instead).
    /// Settable so the A/B benchmark can interleave both paths in one process.
    /// </summary>
    internal bool UseImplicitConvWgrad { get; set; } =
        System.Environment.GetEnvironmentVariable("AIDOTNET_CONV_IMPLICIT_WGRAD") != "0";

    private IGpuBuffer? _convWgradPartials;

    // Tile and split-K shape of conv2d_wgrad_implicit_gemm (WG_BM / WG_BN / WG_BK in CudaConvolutionKernels).
    private const int WgradTile = 64, WgradTileK = 16;

    /// <summary>
    /// #1313: gradKernel[Cout, P] as an implicit GEMM over K = batch * outH * outW, gathering the input instead of
    /// materializing an im2col buffer. When the 64 x 64 output tiles alone cannot fill the device, K is split across
    /// blocks and the slices are summed in a fixed order afterwards. Overwrites gradKernel. False when the kernels
    /// are missing or the partials buffer cannot be sized (the caller then uses the im2col path).
    /// </summary>
    private unsafe bool TryConv2DBackwardKernelImplicitGemm(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        if (!s_convGemmBackward || !UseImplicitConvWgrad || batch <= 0) return false;
        if (!_kernelCache.TryGetValue("conv2d_wgrad_implicit_gemm", out var gemm) ||
            !_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out var reduce))
            return false;
        long kTotal = (long)batch * outHeight * outWidth;
        long p = (long)inChannels * kernelH * kernelW;
        long outputs = outChannels * p;
        if (kTotal <= 0 || kTotal > int.MaxValue || outputs > int.MaxValue) return false;

        int tilesN = (int)((p + WgradTile - 1) / WgradTile), tilesM = (outChannels + WgradTile - 1) / WgradTile;
        int splits = ImplicitGemmSplits(tilesN * tilesM, kTotal, blocksPerSm: 4, out int kPerSplit);

        using var _ = PushContext();
        IGpuBuffer destination = gradKernel;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _convWgradPartials, (long)splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }

        IntPtr inPtr = input.Handle, dyPtr = gradOutput.Handle, outPtr = destination.Handle;
        void** args = stackalloc void*[20];
        args[0] = &inPtr; args[1] = &dyPtr; args[2] = &outPtr;
        args[3] = &batch; args[4] = &inChannels; args[5] = &inHeight; args[6] = &inWidth;
        args[7] = &outChannels; args[8] = &outHeight; args[9] = &outWidth;
        args[10] = &kernelH; args[11] = &kernelW; args[12] = &strideH; args[13] = &strideW;
        args[14] = &padH; args[15] = &padW; args[16] = &dilationH; args[17] = &dilationW; args[18] = &kPerSplit;
        LaunchKernel3D(gemm, (uint)tilesN, (uint)tilesM, (uint)splits, (uint)DefaultBlockSize, 1, 1, args, 0);

        if (splits > 1)
        {
            IntPtr partPtr = destination.Handle, gkPtr = gradKernel.Handle;
            int count = (int)outputs;
            void** rargs = stackalloc void*[4];
            rargs[0] = &partPtr; rargs[1] = &gkPtr; rargs[2] = &count; rargs[3] = &splits;
            LaunchKernel(reduce, (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, rargs);
        }
        return true;
    }

    /// <summary>
    /// Split-K count for the implicit-GEMM conv kernels: enough blocks for <paramref name="blocksPerSm"/> per SM, but
    /// never a split shorter than eight K tiles. kPerSplit is a multiple of the K tile.
    /// </summary>
    private int ImplicitGemmSplits(int tiles, long kTotal, int blocksPerSm, out int kPerSplit)
    {
        int blockTarget = Math.Max(1, blocksPerSm * Math.Max(1, _multiProcessorCount));
        int maxSplits = (int)Math.Max(1, kTotal / (8L * WgradTileK));
        int splits = Math.Min(maxSplits, Math.Max(1, (blockTarget + tiles - 1) / tiles));
        kPerSplit = (int)(((kTotal + splits - 1) / splits + WgradTileK - 1) / WgradTileK * WgradTileK);
        return (int)((kTotal + kPerSplit - 1) / kPerSplit);
    }

    /// <summary>
    /// #1313: the implicit-GEMM input gradient; AIDOTNET_CONV_IMPLICIT_DGRAD=0 starts it off (GEMM + col2im instead).
    /// Settable like <see cref="UseImplicitConvWgrad"/>.
    /// </summary>
    internal bool UseImplicitConvDgrad { get; set; } =
        System.Environment.GetEnvironmentVariable("AIDOTNET_CONV_IMPLICIT_DGRAD") != "0";

    private IGpuBuffer? _convDgradPartials;

    /// <summary>Split-K target of the implicit input gradient, in blocks per SM (see <see cref="ImplicitGemmSplits"/>).</summary>
    internal int ConvDgradBlocksPerSm { get; set; } = 4;

    /// <summary>
    /// #1313: strided input gradient through conv2d_dgrad_implicit_gemm_c16_phase, one grid layer (blockIdx.z) per
    /// sub-pixel phase; each phase writes its own disjoint pixels of gradInput, so there is no split and no reduction.
    /// </summary>
    private unsafe bool TryConv2DBackwardInputPhaseGemm(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("conv2d_dgrad_implicit_gemm_c16_phase", out var gemm)) return false;
        // The largest phase has ceil(H / sH) x ceil(W / sW) pixels per image; smaller phases exit their spare blocks.
        long columns = (long)batch * ((inHeight + strideH - 1) / strideH) * ((inWidth + strideW - 1) / strideW);
        if ((long)batch * inChannels * inHeight * inWidth > int.MaxValue) return false;
        int tilesN = (int)((columns + WgradTile - 1) / WgradTile), tilesM = (inChannels + WgradTile - 1) / WgradTile;

        using var _ = PushContext();
        IntPtr dyPtr = gradOutput.Handle, wPtr = kernel.Handle, outPtr = gradInput.Handle;
        void** args = stackalloc void*[17];
        args[0] = &dyPtr; args[1] = &wPtr; args[2] = &outPtr;
        args[3] = &batch; args[4] = &inChannels; args[5] = &inHeight; args[6] = &inWidth;
        args[7] = &outChannels; args[8] = &outHeight; args[9] = &outWidth;
        args[10] = &kernelH; args[11] = &kernelW; args[12] = &strideH; args[13] = &strideW;
        args[14] = &padH; args[15] = &padW;
        LaunchKernel3D(gemm, (uint)tilesN, (uint)tilesM, (uint)(strideH * strideW), (uint)DefaultBlockSize, 1, 1, args, 0);
        return true;
    }

    /// <summary>
    /// #1313: gradInput[B, Cin, H, W] as an implicit GEMM over K = Cout * kH * kW, gathering gradOutput instead of
    /// writing a col buffer and scattering it back with col2im. Split-K and its fixed-order sum as in the weight
    /// gradient. Overwrites gradInput. False when the kernels are missing or the partials buffer cannot be sized.
    /// </summary>
    private unsafe bool TryConv2DBackwardInputImplicitGemm(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        if (!s_convGemmBackward || !UseImplicitConvDgrad || batch <= 0) return false;
        // Cout % 16 == 0 keeps every K tile on one kernel tap, which the _c16 kernels exploit.
        bool c16 = outChannels % WgradTileK == 0;
        if (c16 && (strideH > 1 || strideW > 1) && dilationH == 1 && dilationW == 1 && kernelH >= strideH && kernelW >= strideW)
            return TryConv2DBackwardInputPhaseGemm(gradOutput, kernel, gradInput, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);
        if (!_kernelCache.TryGetValue(c16 ? "conv2d_dgrad_implicit_gemm_c16" : "conv2d_dgrad_implicit_gemm", out var gemm) ||
            !_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out var reduce))
            return false;
        long nTotal = (long)batch * inChannels * inHeight * inWidth;
        long columns = (long)batch * inHeight * inWidth;
        long kTotal = (long)outChannels * kernelH * kernelW;
        if (nTotal <= 0 || kTotal <= 0 || nTotal > int.MaxValue) return false;

        int tilesN = (int)((columns + WgradTile - 1) / WgradTile), tilesM = (inChannels + WgradTile - 1) / WgradTile;
        // The partials of a split cost an extra write and read of the activation-sized dX, which pays only when the tiles
        // cannot cover the SMs once (measured on ResNet-18: 64x64 @32 slows with any split; 512x512 @4 gains 25% at four
        // blocks per SM).
        int tiles = tilesN * tilesM;
        int splits = tiles >= _multiProcessorCount ? 1 : ImplicitGemmSplits(tiles, kTotal, ConvDgradBlocksPerSm, out int _unusedKPerSplit);
        int kPerSplit = (int)(((kTotal + splits - 1) / splits + WgradTileK - 1) / WgradTileK * WgradTileK);
        splits = (int)((kTotal + kPerSplit - 1) / kPerSplit);

        using var _ = PushContext();
        IGpuBuffer destination = gradInput;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _convDgradPartials, (long)splits * nTotal);
            if (partials is null) return false;
            destination = partials;
        }

        IntPtr dyPtr = gradOutput.Handle, wPtr = kernel.Handle, outPtr = destination.Handle;
        void** args = stackalloc void*[20];
        args[0] = &dyPtr; args[1] = &wPtr; args[2] = &outPtr;
        args[3] = &batch; args[4] = &inChannels; args[5] = &inHeight; args[6] = &inWidth;
        args[7] = &outChannels; args[8] = &outHeight; args[9] = &outWidth;
        args[10] = &kernelH; args[11] = &kernelW; args[12] = &strideH; args[13] = &strideW;
        args[14] = &padH; args[15] = &padW; args[16] = &dilationH; args[17] = &dilationW; args[18] = &kPerSplit;
        LaunchKernel3D(gemm, (uint)tilesN, (uint)tilesM, (uint)splits, (uint)DefaultBlockSize, 1, 1, args, 0);

        if (splits > 1)
        {
            IntPtr partPtr = destination.Handle, gxPtr = gradInput.Handle;
            int count = (int)nTotal;
            void** rargs = stackalloc void*[4];
            rargs[0] = &partPtr; rargs[1] = &gxPtr; rargs[2] = &count; rargs[3] = &splits;
            LaunchKernel(reduce, (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, rargs);
        }
        return true;
    }

    /// <summary>gradKernel[Cout,P] = sum_b gradOutput_b[Cout,L] * col_b[P,L]^T. Overwrites gradKernel.</summary>
    private bool TryConv2DBackwardKernelGemm(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        // #1323's general implicit GEMM first; #1274's narrow-layer kernel (conv2d_weight_grad_implicit) is next for the
        // shapes it admits whenever the first declines (switched off, or its split-K partials cannot be sized).
        if (TryConv2DBackwardKernelImplicitGemm(input, gradOutput, gradKernel, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW))
            return true;
        if (TryConv2DBackwardKernelImplicitNarrow(input, gradOutput, gradKernel, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW))
            return true;
        if (!ConvGemmUsable() || batch <= 0) return false;
        int L = outHeight * outWidth, P = inChannels * kernelH * kernelW;
        using var _ = PushContext();
        var col = TryConvScratch(ref _convColScratch, (long)batch * P * L);
        var partials = TryConvScratch(ref _convWeightPartials, (long)batch * outChannels * P);
        if (col is null || partials is null)
            return false;
        if (_convOnes is null || _convOnesLength < batch)
        {
            if (IsStreamCapturing()) return false;
            _convOnes?.Dispose();
            _convOnes = AllocateBuffer(batch);
            Fill(_convOnes, 1f, batch);
            _convOnesLength = batch;
        }

        LaunchIm2Col(input, col, batch, inChannels, inHeight, inWidth, kernelH, kernelW,
            strideH, strideW, padH, padW, dilationH, dilationW, outHeight, outWidth);

        ApplyDeterministicGemmMathMode();
        float one = 1f, zero = 0f;
        // Column-major: partials_b[P x Cout] = col_b^T-view(T)[P x L] * gradOutput_b-view[L x Cout].
        BlasSgemmStridedBatched(
            CublasOperation.Transpose, CublasOperation.None, P, outChannels, L,
            ref one, col.Handle, L, (long)P * L, gradOutput.Handle, L, (long)outChannels * L,
            ref zero, partials.Handle, P, (long)outChannels * P, batch, "cublasSgemmStridedBatched(conv dW)");
        // Sum over the batch: gradKernel[N2] = partials[N2 x B] * ones[B], N2 = Cout*P.
        int n2 = outChannels * P;
        BlasSgemm(
            CublasOperation.None, CublasOperation.None, n2, 1, batch,
            ref one, partials.Handle, n2, _convOnes.Handle, batch,
            ref zero, gradKernel.Handle, n2, "cublasSgemm(conv dW batch sum)");
        return true;
    }

    /// <summary>gradInput = col2im(kernel^T * gradOutput_b) per batch item. Overwrites gradInput.</summary>
    private bool TryConv2DBackwardInputGemm(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        if (TryConv2DBackwardInputImplicitGemm(gradOutput, kernel, gradInput, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW))
            return true;
        if (!ConvGemmUsable() || batch <= 0) return false;
        int L = outHeight * outWidth, P = inChannels * kernelH * kernelW;
        using var _ = PushContext();
        var col = TryConvScratch(ref _convColScratch, (long)batch * P * L);
        if (col is null)
            return false;

        ApplyDeterministicGemmMathMode();
        float one = 1f, zero = 0f;
        // Column-major: col_b[L x P] = gradOutput_b-view[L x Cout] * kernel-view(T)[Cout x P].
        BlasSgemmStridedBatched(
            CublasOperation.None, CublasOperation.Transpose, L, P, outChannels,
            ref one, gradOutput.Handle, L, (long)outChannels * L, kernel.Handle, P, 0L,
            ref zero, col.Handle, L, (long)P * L, batch, "cublasSgemmStridedBatched(conv dX)");

        LaunchCol2Im(col, gradInput, batch, inChannels, inHeight, inWidth, kernelH, kernelW,
            strideH, strideW, padH, padW, dilationH, dilationW, outHeight, outWidth);
        return true;
    }
}
