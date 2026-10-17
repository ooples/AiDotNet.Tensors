using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The six Winograd tile transforms a backend writes in its own kernel language. <see cref="WinogradF23Pipeline"/> runs
/// them around the backend's <see cref="IDirectGpuBackend.BatchedGemm"/>, so the GEMMs, the admission and the scratch
/// sizing are shared and only these elementwise kernels differ per backend. Layouts and transforms are those of the
/// CUDA routes (<c>CUDA/Kernels/CudaConvWinogradKernels.cs</c>): every transformed operand is sixteen contiguous planes,
/// one per Winograd point, which is exactly <c>BatchedGemm</c>'s concatenated-batch layout.
/// </summary>
internal interface IWinogradF23Transforms
{
    /// <summary>
    /// U[16][K][C] = G g G^T of each 3x3 filter of <paramref name="w"/> ([K][C][3][3]). With <paramref name="flip"/>,
    /// U'[16][C][K] of the filter rotated 180 degrees and channel-transposed, which turns the input gradient into a
    /// forward Winograd convolution over dY.
    /// </summary>
    void Filter(IGpuBuffer w, IGpuBuffer u, int k, int c, bool flip);

    /// <summary>V[16][C][P] = B^T d B for the 4x4 input patch of each 2x2 output tile p (zero outside the image).</summary>
    void Input(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH, int tilesW);

    /// <summary>Y = A^T m A per tile of M[16][K][P], writing only the outputs inside outH x outW.</summary>
    void Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH, int tilesW);

    /// <summary>Weight gradient, F(3x3, 2x2): U[16][K][P] = A g A^T of each 2x2 dY tile (zero past the edge).</summary>
    void GradOutputTiles(IGpuBuffer dy, IGpuBuffer u, int n, int k, int outH, int outW, int tilesH, int tilesW);

    /// <summary>Weight gradient: V[16][P][C], the input transform stored tile-major so it is the GEMM's P x C operand.</summary>
    void InputTilesTileMajor(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH, int tilesW);

    /// <summary>
    /// Weight gradient: dW[K][C][3][3] = G^T M G of M[16][splits][K][C], summing the split-K partials in order (one
    /// split where the GEMM did not split the tile reduction). Overwrites the gradient.
    /// </summary>
    void FilterGradient(IGpuBuffer m, IGpuBuffer dw, int k, int c, int splits);
}

/// <summary>
/// Winograd F(2x2, 3x3) forward and input gradient, and F(3x3, 2x2) weight gradient (Lavin and Gray, "Fast Algorithms
/// for Convolutional Neural Networks", CVPR 2016), for 3x3 stride-1 dilation-1 convolutions with equal padding up to 2,
/// on any backend that supplies <see cref="IWinogradF23Transforms"/>. Each call is three transforms and one batched
/// GEMM of sixteen matrices.
/// </summary>
/// <remarks>
/// SCRATCH IS PER CALL, as every backend's im2col route already does it. Vulkan and WebGPU record into deferred or
/// batched command buffers, so a scratch buffer cached here and resized on the next shape could be freed while a
/// recorded dispatch still reads it. The backends' buffer types already make a disposed-while-recorded buffer safe,
/// and the tuned slot measures the allocation as part of the route, so the cost is in the number it competes on.
/// </remarks>
internal static class WinogradF23Pipeline
{
    /// <summary>
    /// The shapes these routes compute, over a tuned-slot shape key (forward and backward share the layout: [5] kH,
    /// [6] kW, [7] sH, [8] sW, [9] pH, [10] pW, [11] dH, [12] dW, [13] outH, [14] outW).
    /// </summary>
    internal static bool AdmitsShape(TunedShape s)
        => s[5] == 3 && s[6] == 3 && s[7] == 1 && s[8] == 1 && s[11] == 1 && s[12] == 1 && s[9] == s[10] && s[9] <= 2
           && s[13] > 0 && s[14] > 0;

    internal static bool Admits(in TunedConv2DArgs a)
        => a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 1 && a.StrideW == 1 && a.DilationH == 1 && a.DilationW == 1
           && a.PadH == a.PadW && a.PadH <= 2 && a.OutHeight > 0 && a.OutWidth > 0;

    internal static bool AdmitsBackward(in TunedConv2DBackwardArgs a)
        => a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 1 && a.StrideW == 1 && a.DilationH == 1 && a.DilationW == 1
           && a.PadH == a.PadW && a.PadH <= 2 && a.OutHeight > 0 && a.OutWidth > 0;

    /// <summary>y[n][K] = conv(x[n][C], w[K][C]). False, with nothing launched, when the shape is not admitted.</summary>
    internal static bool TryForward(IDirectGpuBackend backend, IWinogradF23Transforms transforms, in TunedConv2DArgs a)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (transforms is null) throw new ArgumentNullException(nameof(transforms));
        if (!Admits(a)) return false;
        return Run(backend, transforms, a.Input, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.PadH, flip: false,
            a.Kernel, a.OutChannels, a.InChannels, a.Output, a.OutChannels, a.OutHeight, a.OutWidth);
    }

    /// <summary>
    /// dX = conv(dY, w') with w' the 180-degree-rotated, channel-transposed filter and padding 2 - pad: the forward
    /// pipeline over dY (K channels, outH x outW) producing dX (C channels, H x W).
    /// </summary>
    internal static bool TryBackwardData(IDirectGpuBackend backend, IWinogradF23Transforms transforms,
        in TunedConv2DBackwardArgs a)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (transforms is null) throw new ArgumentNullException(nameof(transforms));
        if (a.WeightGradient || !AdmitsBackward(a)) return false;
        return Run(backend, transforms, a.First, a.Batch, a.OutChannels, a.OutHeight, a.OutWidth, 2 - a.PadH, flip: true,
            a.Second, a.OutChannels, a.InChannels, a.Output, a.InChannels, a.InHeight, a.InWidth);
    }

    /// <summary>
    /// dW[K][C][3][3] through F(3x3, 2x2): dY tiles U[16][K][P], input tiles V[16][P][C], sixteen K x C GEMMs reducing
    /// over the P tiles, and the output transform. Overwrites the gradient.
    /// </summary>
    internal static bool TryBackwardFilter(IDirectGpuBackend backend, IWinogradF23Transforms transforms,
        in TunedConv2DBackwardArgs a)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (transforms is null) throw new ArgumentNullException(nameof(transforms));
        if (!a.WeightGradient || !AdmitsBackward(a)) return false;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, outH = a.OutHeight, outW = a.OutWidth;
        int tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (Planes(k, p) > int.MaxValue || Planes(c, p) > int.MaxValue || Planes(k, c) > int.MaxValue) return false;
        int pi = (int)p;
        using var u = backend.AllocateBuffer((int)Planes(k, p));
        using var v = backend.AllocateBuffer((int)Planes(c, p));
        using var m = backend.AllocateBuffer((int)Planes(k, c));
        transforms.GradOutputTiles(a.Second, u, n, k, outH, outW, tilesH, tilesW);
        transforms.InputTilesTileMajor(a.First, v, n, c, a.InHeight, a.InWidth, a.PadH, tilesH, tilesW);
        // M[x] (K x C) = U[x] (K x P) * V[x] (P x C), one GEMM per Winograd point.
        backend.BatchedGemm(u, v, m, k, c, pi, WinogradPoints);
        transforms.FilterGradient(m, a.Output, k, c, splits: 1);
        return true;
    }

    /// <summary>F(2x2, 3x3) has a 4x4 transform, so sixteen points and sixteen GEMMs.</summary>
    private const int WinogradPoints = 16;

    private static long Planes(long rows, long cols) => WinogradPoints * rows * cols;

    // y[n][outC][outH][outW] = conv3x3_s1(x[n][inC][h][w], padding pad): the filter transform of weights (forward layout
    // [fwdK][fwdC][3][3], flipped for the input gradient) into U[16][outC][inC], the input transform, the sixteen GEMMs
    // and the output transform.
    private static bool Run(IDirectGpuBackend backend, IWinogradF23Transforms transforms, IGpuBuffer x, int n, int inC,
        int h, int w, int pad, bool flip, IGpuBuffer weights, int fwdK, int fwdC, IGpuBuffer y, int outC, int outH, int outW)
    {
        int tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (Planes(inC, p) > int.MaxValue || Planes(outC, p) > int.MaxValue || Planes(outC, inC) > int.MaxValue)
            return false;
        int pi = (int)p;
        using var u = backend.AllocateBuffer((int)Planes(outC, inC));
        using var v = backend.AllocateBuffer((int)Planes(inC, p));
        using var m = backend.AllocateBuffer((int)Planes(outC, p));
        transforms.Filter(weights, u, fwdK, fwdC, flip);
        transforms.Input(x, v, n, inC, h, w, pad, tilesH, tilesW);
        // M[x] (outC x P) = U[x] (outC x inC) * V[x] (inC x P), one GEMM per Winograd point.
        backend.BatchedGemm(u, v, m, outC, pi, inC, WinogradPoints);
        transforms.Output(m, y, n, outC, outH, outW, tilesH, tilesW);
        return true;
    }
}