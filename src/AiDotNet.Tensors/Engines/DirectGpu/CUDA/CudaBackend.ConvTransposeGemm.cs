// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The transposed-convolution forward as ONE cuBLAS GEMM and an overlap-add, the formulation cuDNN's transposed
// (backward-data) GEMM algorithms use. With x regrouped from [B, Cin, L] to [Cin, B*L],
//   col[Cout*kH*kW, B*L] = W^T x        (W is [Cin, Cout*kH*kW] row-major, the stored transposed kernel)
// and each output element gathers the (kH/sH)*(kW/sW) columns that land on it. The batch is part of the GEMM's N, so
// the 4x4 stem of a generator (L = 16) still makes a well-shaped GEMM; the strided-batched col2im route keeps one GEMM
// per sample with N = L and the implicit-GEMM data-gradient engine tiles the output pixels, both of which leave most of
// the tile idle at small L. Measured on the 64x64 DCGAN generator (RTX 1000 Ada, batch 16): the 4x4 -> 8x8
// 512 -> 256 stage 345 -> 208 us, the 64 -> 3 RGB head 105 -> 94 us.
public sealed partial class CudaBackend
{
    private IGpuBuffer? _convTransposeXt;
    private IGpuBuffer? _convTransposeCol;

    /// <summary>AIDOTNET_CONV_TRANSPOSE_GEMM=0 turns the single-GEMM transposed forward off.</summary>
    private static readonly bool s_convTransposeGemm =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_TRANSPOSE_GEMM") != "0";

    // The column buffer is bounded (the batch is processed in chunks beyond it): 8M floats, 32 MiB.
    private const long ConvTransposeColCap = 8L << 20;

    /// <summary>
    /// The transposed forward through one GEMM per batch chunk; false (nothing launched) when cuBLAS is not in use,
    /// the scratch cannot be sized (an undersized buffer during a capture), or a kernel is missing.
    /// </summary>
    internal unsafe bool TryConvTranspose2DGemm(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output, IGpuBuffer? bias,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (!s_convTransposeGemm || !UsesCublas || !ConvGemmUsable() || batch <= 0) return false;
        if (!_kernelCache.TryGetValue("nchw_to_cnhw", out var regroup) ||
            !_kernelCache.TryGetValue("col2im_cnhw_bias", out var overlapAdd))
            return false;
        int L = inHeight * inWidth;
        long P = (long)outChannels * kernelH * kernelW;
        if (L <= 0 || P <= 0 || P > int.MaxValue) return false;
        // Where it wins: a small input plane (the generator stem, L <= 16, where the per-sample routes run N = L) or a
        // narrow output (Cout*kH*kW <= 64, the RGB head, where the implicit-GEMM routes tile few output channels).
        // Between those the implicit-GEMM data-gradient engine is faster: on the 64x64 DCGAN generator at batch 16,
        // 8x8 -> 16x16 (256 -> 128) ran 211 us there vs 253 us here, and 16x16 -> 32x32 (128 -> 64) 257 vs 361 us.
        if (L > 16 && P > 64) return false;
        int chunk = (int)Math.Min(batch, ConvTransposeColCap / (P * L));
        if (chunk == 0) return false;
        long colFloats = P * chunk * L, xtFloats = (long)inChannels * chunk * L;
        if (colFloats > int.MaxValue || xtFloats > int.MaxValue) return false;

        using var _ = PushContext();
        var xt = TryConvScratch(ref _convTransposeXt, xtFloats);
        var col = xt is null ? null : TryConvScratch(ref _convTransposeCol, colFloats);
        if (xt is null || col is null) return false;

        ApplyDeterministicGemmMathMode();
        float one = 1f, zero = 0f;
        int p = (int)P;
        long inSample = (long)inChannels * L, outSample = (long)outChannels * outHeight * outWidth;
        IntPtr xtPtr = xt.Handle, colPtr = col.Handle, biasPtr = bias?.Handle ?? IntPtr.Zero;
        void** ra = stackalloc void*[5];
        void** oa = stackalloc void*[16];
        for (int b0 = 0; b0 < batch; b0 += chunk)
        {
            int nb = Math.Min(chunk, batch - b0);
            int n = nb * L;
            IntPtr inPtr = input.Handle + (nint)(b0 * inSample * sizeof(float));
            IntPtr outPtr = output.Handle + (nint)(b0 * outSample * sizeof(float));

            int c = inChannels;
            ra[0] = &inPtr; ra[1] = &xtPtr; ra[2] = &nb; ra[3] = &c; ra[4] = &L;
            long regroupTotal = (long)nb * inChannels * L;
            LaunchKernel(regroup, (uint)((regroupTotal + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, ra);

            // Column-major: colT[n x P] = xtT[n x Cin] * W[Cin x P], i.e. the row-major col[P x n] = W^T xt, with the
            // row-major xt [Cin, n] (lda n) and W [Cin, P] (ldb P) read as their column-major transposes.
            BlasSgemm(CublasOperation.None, CublasOperation.Transpose, n, p, inChannels,
                ref one, xtPtr, n, kernel.Handle, p, ref zero, colPtr, n, "cublasSgemm(conv transpose)");

            int oc = outChannels, oh = outHeight, ow = outWidth, kh = kernelH, kw = kernelW, sh = strideH, sw = strideW,
                ph = padH, pw = padW, ih = inHeight, iw = inWidth;
            oa[0] = &colPtr; oa[1] = &biasPtr; oa[2] = &outPtr; oa[3] = &nb; oa[4] = &oc; oa[5] = &oh; oa[6] = &ow;
            oa[7] = &kh; oa[8] = &kw; oa[9] = &sh; oa[10] = &sw; oa[11] = &ph; oa[12] = &pw; oa[13] = &ih; oa[14] = &iw;
            long outTotal = (long)nb * outSample;
            LaunchKernel(overlapAdd, (uint)((outTotal + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, oa);
        }
        return true;
    }
}
