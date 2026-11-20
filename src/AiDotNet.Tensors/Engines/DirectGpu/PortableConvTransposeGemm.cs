// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The transposed-convolution forward as one GEMM per batch chunk and an overlap-add on HIP, Metal, OpenCL, Vulkan and
/// WebGPU: the host side the five share (CUDA's own is <c>CudaBackend.TryConvTranspose2DGemm</c>, with cuBLAS). With x
/// regrouped from [B, Cin, L] to [Cin, B*L], <c>col[Cout*kH*kW, B*L] = W^T x</c> (W the stored [Cin, Cout*kH*kW]
/// transposed kernel, through the backend's <see cref="ITransposedAGemm"/>), and each output gathers the columns that land
/// on it (<see cref="ConvTransposeGemmKernelTemplate"/>). Each backend supplies a <see cref="Launcher"/> for its dispatch.
/// </summary>
/// <remarks>
/// The incumbent on these backends is one thread per output looping over every input channel and tap, with no reuse of
/// x or W; the GEMM reuses both through the backend's tiled matrix multiply. The column buffer is bounded
/// (<see cref="ColCap"/>): a larger batch runs in chunks.
/// </remarks>
internal static class PortableConvTransposeGemm
{
    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="ConvTransposeGemmKernelTemplate.GroupSize"/>.</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    /// <summary>AIDOTNET_CONV_TRANSPOSE_GEMM=0 turns the single-GEMM transposed forward off (on CUDA too).</summary>
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("AIDOTNET_CONV_TRANSPOSE_GEMM") != "0";

    /// <summary>The column buffer's bound in floats (32 MiB), as CUDA's.</summary>
    internal const long ColCap = 8L << 20;

    // Every API's guaranteed 1-D grid (WebGPU's 65535 workgroups per dimension).
    private const long MaxGroups = 65535;

    private const int G = ConvTransposeGemmKernelTemplate.GroupSize;

    /// <summary>
    /// The batch chunk the forward runs in, or 0 when the shape is not served: a non-positive extent, or an index past
    /// int range (the kernels index in 32 bits) or a launch past <see cref="MaxGroups"/> even for one sample.
    /// </summary>
    internal static int Chunk(int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight,
        int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (batch <= 0 || inChannels <= 0 || inHeight <= 0 || inWidth <= 0 || outChannels <= 0 || outHeight <= 0
            || outWidth <= 0 || kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0)
            return 0;
        long L = (long)inHeight * inWidth;
        long P = (long)outChannels * kernelH * kernelW;
        long inSample = inChannels * L, outSample = (long)outChannels * outHeight * outWidth;
        if ((long)batch * inSample > int.MaxValue || (long)batch * outSample > int.MaxValue) return 0;
        long chunk = Math.Max(1, Math.Min(batch, ColCap / Math.Max(1, P * L)));
        // Each launch of a chunk stays within the grid.
        chunk = Math.Min(chunk, MaxGroups * G / Math.Max(1, Math.Max(inSample, outSample)));
        if (chunk < 1 || P * chunk * L > int.MaxValue) return 0;
        return (int)chunk;
    }

    /// <summary>
    /// y = the transposed convolution of x by W ([Cin, Cout, kH, kW]), plus <paramref name="bias"/>[c] when given (added
    /// in the overlap-add); overwrites y. False, nothing launched, when the shape is not served (see
    /// <see cref="Chunk"/>) or the route is turned off.
    /// </summary>
    internal static bool TryForward(IDirectGpuBackend backend, Launcher launch, IGpuBuffer input, IGpuBuffer kernel,
        IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight,
        int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, IGpuBuffer? bias = null)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (launch is null) throw new ArgumentNullException(nameof(launch));
        if (!Enabled || backend is not ITransposedAGemm gemm) return false;
        int chunk = Chunk(batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW,
            strideH, strideW, padH, padW);
        if (chunk == 0) return false;
        int L = inHeight * inWidth;
        int P = outChannels * kernelH * kernelW;
        int inSample = inChannels * L, outSample = outChannels * outHeight * outWidth;
        using var xt = backend.AllocateBuffer(inChannels * chunk * L);
        using var col = backend.AllocateBuffer(P * chunk * L);
        for (int b0 = 0; b0 < batch; b0 += chunk)
        {
            int nb = Math.Min(chunk, batch - b0);
            int n = nb * L;
            int regroupTotal = nb * inSample;
            launch(ConvTransposeGemmKernelTemplate.Regroup, (regroupTotal + G - 1) / G, new[] { input, xt },
                new[] { nb, inChannels, L, b0 * inSample });
            // Row-major col[P, n] = W[Cin, P]^T xt[Cin, n].
            gemm.MatMulTransposedA(kernel, xt, col, P, n, inChannels);
            int outTotal = nb * outSample;
            launch(bias is null ? ConvTransposeGemmKernelTemplate.OverlapAdd : ConvTransposeGemmKernelTemplate.OverlapAddBias,
                (outTotal + G - 1) / G, bias is null ? new[] { col, output } : new[] { col, output, bias },
                new[] { nb, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
                    inHeight, inWidth, b0 * outSample });
        }
        return true;
    }
}
