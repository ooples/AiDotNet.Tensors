// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// The tuned-kernel registry's row and column variants for HIP (#1113): CUDA's one template
/// (<see cref="CudaTunedRowKernels"/>) compiled by hipRTC, with the warp shuffles in HIP's form. The template
/// shuffles within a power-of-two width of at most 32 lanes, so <c>__shfl_down(v, o, L)</c> / <c>__shfl(v, 0, L)</c>
/// keep its exact reduction order on wave32 (RDNA) and wave64 (CDNA) alike; the 32-bit full masks of the CUDA form
/// would name only half a wave64.
/// </summary>
internal static class HipTunedRowKernels
{
    internal static int[] RowLanes => CudaTunedRowKernels.RowLanes;
    internal static int[] ColumnRowLanes => CudaTunedRowKernels.ColumnRowLanes;

    internal static string SoftmaxName(int lanes) => CudaTunedRowKernels.SoftmaxName(lanes);
    internal static string SoftmaxBackwardName(int lanes) => CudaTunedRowKernels.SoftmaxBackwardName(lanes);
    internal static string LayerNormName(int lanes) => CudaTunedRowKernels.LayerNormName(lanes);
    internal static string LayerNormBackwardName(int lanes) => CudaTunedRowKernels.LayerNormBackwardName(lanes);
    internal static string LayerNormGradParamsName(int rowLanes) => CudaTunedRowKernels.LayerNormGradParamsName(rowLanes);
    internal static string RmsNormGradGammaName(int rowLanes) => CudaTunedRowKernels.RmsNormGradGammaName(rowLanes);

    /// <summary>im2col into [P, B * L] (P = Cin*kH*kW, L = outH*outW) for the registry's im2col + GEMM convolution
    /// candidate: the batch sits inside each row, so ONE GEMM covers every batch item (HIP's own im2col writes [B, P, L]).</summary>
    internal const string Im2ColName = "tk_im2col_pbl";

    private const string Im2ColSource = @"
extern ""C"" __global__ __launch_bounds__(64) void tk_im2col_pbl(const float* __restrict__ input, float* __restrict__ col,
    int batch, int channels, int height, int width, int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{
    // One thread per element of col [P, B * L], in col's order (see TunedRowKernelTemplate's tk_im2col_body).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l = outH * outW, n = batch * l;
    if (idx >= channels * kernelH * kernelW * n) return;
    int p = idx / n, j = idx - p * n;
    int b = j / l, s = j - b * l, oh = s / outW, ow = s - oh * outW;
    int kw = p % kernelW, ckh = p / kernelW, kh = ckh % kernelH, c = ckh / kernelH;
    int ih = oh * strideH - padH + kh * dilationH, iw = ow * strideW - padW + kw * dilationW;
    col[idx] = (ih >= 0 && ih < height && iw >= 0 && iw < width) ? input[((b * channels + c) * height + ih) * width + iw] : 0.0f;
}
";

    public static string GetSource()
    {
        string source = (CudaTunedRowKernels.GetSource() + Im2ColSource)
            .Replace("__shfl_down_sync(0xffffffffu, ", "__shfl_down(")
            .Replace("__shfl_sync(0xffffffffu, ", "__shfl(");
        // Any other spelling would reach hipRTC untranslated; fail here, where the template changed.
        if (source.Contains("_sync("))
            throw new InvalidOperationException("CudaTunedRowKernels uses a sync intrinsic HipTunedRowKernels does not translate.");
        return source;
    }

    public static string[] GetKernelNames()
    {
        var names = new System.Collections.Generic.List<string>(CudaTunedRowKernels.GetKernelNames()) { Im2ColName };
        return names.ToArray();
    }
}