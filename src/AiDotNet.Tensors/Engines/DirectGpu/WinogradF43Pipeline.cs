// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The three Winograd F(4x4, 3x3) tile transforms a backend writes in its own kernel language
/// (<see cref="WinogradF43KernelTemplate"/>). <see cref="WinogradF43Pipeline"/> runs them around the backend's
/// <see cref="IDirectGpuBackend.BatchedGemm"/>.
/// </summary>
internal interface IWinogradF43Transforms
{
    /// <summary>U[36][K][C] = G g G^T of each 3x3 filter of <paramref name="w"/> ([K][C][3][3]).</summary>
    void FilterF43(IGpuBuffer w, IGpuBuffer u, int k, int c);

    /// <summary>V[36][C][P] = B^T d B for the 6x6 input patch of each 4x4 output tile p (zero outside the image).</summary>
    void InputF43(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH, int tilesW);

    /// <summary>
    /// Y = A^T m A per tile of M[36][K][P], writing only the outputs inside outH x outW, with the convolution's epilogue:
    /// <paramref name="epilogue"/> bit 0 adds <paramref name="bias"/>[k] (null when the bit is clear), bit 1 applies ReLU.
    /// </summary>
    void OutputF43(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH, int tilesW, IGpuBuffer? bias,
        int epilogue);
}

/// <summary>
/// Winograd F(4x4, 3x3) forward convolution (#1432; OpenCL's route, #1431) for 3x3 stride-1 dilation-1 convolutions with
/// equal padding up to 2, on any backend that supplies <see cref="IWinogradF43Transforms"/>: three transforms and one
/// batched GEMM of 36 matrices. Against F(2x2, 3x3) it does 4x instead of 2.25x fewer multiplies than direct, for
/// transforms over 36 points instead of 16. Scratch is per call, as in <see cref="WinogradF23Pipeline"/>.
/// </summary>
internal static class WinogradF43Pipeline
{
    /// <summary>y[n][K] = conv(x[n][C], w[K][C]). False, with nothing launched, when the shape is not admitted.</summary>
    internal static bool TryForward(IDirectGpuBackend backend, IWinogradF43Transforms transforms, in TunedConv2DArgs a)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (transforms is null) throw new ArgumentNullException(nameof(transforms));
        if (!WinogradF23Pipeline.Admits(a)) return false;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, outH = a.OutHeight, outW = a.OutWidth;
        int tilesH = (outH + 3) / 4, tilesW = (outW + 3) / 4;
        long p = (long)n * tilesH * tilesW;
        long uCount = (long)WinogradF43KernelTemplate.Points * k * c;
        long vCount = (long)WinogradF43KernelTemplate.Points * c * p;
        long mCount = (long)WinogradF43KernelTemplate.Points * k * p;
        // Every index in the transforms is 32-bit (GLSL and WGSL have no portable 64-bit integer).
        if (uCount > int.MaxValue || vCount > int.MaxValue || mCount > int.MaxValue) return false;
        int pi = (int)p;
        using var u = backend.AllocateBuffer((int)uCount);
        using var v = backend.AllocateBuffer((int)vCount);
        using var m = backend.AllocateBuffer((int)mCount);
        transforms.FilterF43(a.Kernel, u, k, c);
        transforms.InputF43(a.Input, v, n, c, a.InHeight, a.InWidth, a.PadH, tilesH, tilesW);
        // M[x] (K x P) = U[x] (K x C) * V[x] (C x P), one GEMM per Winograd point.
        backend.BatchedGemm(u, v, m, k, pi, c, WinogradF43KernelTemplate.Points);
        transforms.OutputF43(m, a.Output, n, k, outH, outW, tilesH, tilesW, a.Bias, ConvEpilogue.Flags(a));
        return true;
    }
}
