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
            throw new InvalidOperationException("CUDA kernel not found: im2col");
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
            throw new InvalidOperationException("CUDA kernel not found: col2im");
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

    /// <summary>gradKernel[Cout,P] = sum_b gradOutput_b[Cout,L] * col_b[P,L]^T. Overwrites gradKernel.</summary>
    private bool TryConv2DBackwardKernelGemm(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
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
        CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSgemmStridedBatched(_cublasHandle,
            CublasOperation.Transpose, CublasOperation.None, P, outChannels, L,
            ref one, col.Handle, L, (long)P * L, gradOutput.Handle, L, (long)outChannels * L,
            ref zero, partials.Handle, P, (long)outChannels * P, batch), "cublasSgemmStridedBatched(conv dW)");
        // Sum over the batch: gradKernel[N2] = partials[N2 x B] * ones[B], N2 = Cout*P.
        int n2 = outChannels * P;
        CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSgemm(_cublasHandle,
            CublasOperation.None, CublasOperation.None, n2, 1, batch,
            ref one, partials.Handle, n2, _convOnes.Handle, batch,
            ref zero, gradKernel.Handle, n2), "cublasSgemm(conv dW batch sum)");
        return true;
    }

    /// <summary>gradInput = col2im(kernel^T * gradOutput_b) per batch item. Overwrites gradInput.</summary>
    private bool TryConv2DBackwardInputGemm(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        if (!ConvGemmUsable() || batch <= 0) return false;
        int L = outHeight * outWidth, P = inChannels * kernelH * kernelW;
        using var _ = PushContext();
        var col = TryConvScratch(ref _convColScratch, (long)batch * P * L);
        if (col is null)
            return false;

        ApplyDeterministicGemmMathMode();
        float one = 1f, zero = 0f;
        // Column-major: col_b[L x P] = gradOutput_b-view[L x Cout] * kernel-view(T)[Cout x P].
        CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSgemmStridedBatched(_cublasHandle,
            CublasOperation.None, CublasOperation.Transpose, L, P, outChannels,
            ref one, gradOutput.Handle, L, (long)outChannels * L, kernel.Handle, P, 0L,
            ref zero, col.Handle, L, (long)P * L, batch), "cublasSgemmStridedBatched(conv dX)");

        LaunchCol2Im(col, gradInput, batch, inChannels, inHeight, inWidth, kernelH, kernelW,
            strideH, strideW, padH, padW, dilationH, dilationW, outHeight, outWidth);
        return true;
    }
}
