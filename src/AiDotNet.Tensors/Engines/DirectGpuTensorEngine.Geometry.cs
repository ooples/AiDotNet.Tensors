// Copyright (c) AiDotNet. All rights reserved.
// GPU dispatch for the geometry / sampling ops added by Issue #217:
// Interpolate, PadNd, GridSample (extended), AffineGrid3D. Falls through
// to CpuEngine via inherited base.* when the backend does not implement
// IGeometryBackend, T != float, or the shape / mode combination isn't on
// the GPU's supported surface (1D/3D Interpolate, rank ≠ 4 PadNd, etc.).

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> Interpolate<T>(Tensor<T> input, int[] sizes, InterpolateMode mode, bool alignCorners = false)
    {
        // GPU surface: T=float, 4D NCHW, modes Nearest/Bilinear/Bicubic/Area.
        if (typeof(T) == typeof(float)
            && input.Rank == 4
            && sizes.Length == 2
            && IsGpuInterpolateMode(mode))
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IGeometryBackend geom)
                {
                    int N = input._shape[0], C = input._shape[1];
                    int Hin = input._shape[2], Win = input._shape[3];
                    int Hout = sizes[0], Wout = sizes[1];
                    if (N * C * Hout * Wout == 0)
                        return new Tensor<T>(new[] { N, C, Hout, Wout });
                    using var inBuf = GetOrAllocateBuffer(backend, input);
                    var outBuf = AllocateOutputBuffer(backend, N * C * Hout * Wout);
                    try
                    {
                        geom.Interpolate2D(inBuf.Buffer, outBuf.Buffer,
                            N, C, Hin, Win, Hout, Wout, (int)mode, alignCorners);
                        var arr = FinishGpuOp<T>(backend, outBuf, N * C * Hout * Wout);
                        return new Tensor<T>(arr, new[] { N, C, Hout, Wout });
                    }
                    catch { outBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.Interpolate(input, sizes, mode, alignCorners);
    }

    /// <inheritdoc/>
    public override Tensor<T> InterpolateByScale<T>(Tensor<T> input, double[] scaleFactors, InterpolateMode mode, bool alignCorners = false)
        => base.InterpolateByScale(input, scaleFactors, mode, alignCorners);

    /// <inheritdoc/>
    public override Tensor<T> PadNd<T>(Tensor<T> input, int[] pad, PadMode mode, T value = default!)
    {
        // GPU surface: T=float, rank 4 (NCHW), any pad pattern within the 8-int envelope.
        if (typeof(T) == typeof(float)
            && input.Rank == 4
            && pad.Length <= 8)
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IGeometryBackend geom)
                {
                    int N = input._shape[0], C = input._shape[1];
                    int Hin = input._shape[2], Win = input._shape[3];
                    // Expand pad to 8 ints (innermost-first PyTorch order → per-axis pairs).
                    int[] p = new int[8];
                    Array.Copy(pad, p, pad.Length);
                    int padW0 = p[0], padW1 = p[1];
                    int padH0 = p[2], padH1 = p[3];
                    int padC0 = p[4], padC1 = p[5];
                    int padN0 = p[6], padN1 = p[7];

                    int Nout = N + padN0 + padN1;
                    int Cout = C + padC0 + padC1;
                    int Hout = Hin + padH0 + padH1;
                    int Wout = Win + padW0 + padW1;
                    int outLen = Nout * Cout * Hout * Wout;
                    if (outLen == 0)
                        return new Tensor<T>(new[] { Nout, Cout, Hout, Wout });

                    float padValFloat = (T)(object)value! is float f ? f : 0.0f;
                    using var inBuf = GetOrAllocateBuffer(backend, input);
                    var outBuf = AllocateOutputBuffer(backend, outLen);
                    try
                    {
                        geom.Pad4D(inBuf.Buffer, outBuf.Buffer,
                            N, C, Hin, Win,
                            padN0, padN1, padC0, padC1, padH0, padH1, padW0, padW1,
                            (int)mode, padValFloat);
                        var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                        return new Tensor<T>(arr, new[] { Nout, Cout, Hout, Wout });
                    }
                    catch { outBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.PadNd(input, pad, mode, value);
    }

    /// <inheritdoc/>
    public override Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid,
        GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        if (typeof(T) == typeof(float)
            && input.Rank == 4 && grid.Rank == 4 && grid._shape[3] == 2)
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IGeometryBackend geom)
                {
                    int N = input._shape[0], H = input._shape[1], W = input._shape[2], C = input._shape[3];
                    int outH = grid._shape[1], outW = grid._shape[2];
                    int outLen = N * outH * outW * C;
                    if (outLen == 0) return new Tensor<T>(new[] { N, outH, outW, C });
                    using var inBuf = GetOrAllocateBuffer(backend, input);
                    using var grBuf = GetOrAllocateBuffer(backend, grid);
                    var outBuf = AllocateOutputBuffer(backend, outLen);
                    try
                    {
                        geom.GridSample2D(inBuf.Buffer, grBuf.Buffer, outBuf.Buffer,
                            N, H, W, C, outH, outW, (int)mode, (int)padding, alignCorners);
                        var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                        return new Tensor<T>(arr, new[] { N, outH, outW, C });
                    }
                    catch { outBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.GridSample(input, grid, mode, padding, alignCorners);
    }

    /// <inheritdoc/>
    public override Tensor<T> AffineGrid3D<T>(Tensor<T> theta, int outputDepth, int outputHeight, int outputWidth, bool alignCorners = false)
    {
        if (typeof(T) == typeof(float)
            && theta.Rank == 3 && theta._shape[1] == 3 && theta._shape[2] == 4)
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IGeometryBackend geom)
                {
                    int N = theta._shape[0];
                    int outLen = N * outputDepth * outputHeight * outputWidth * 3;
                    if (outLen == 0) return new Tensor<T>(new[] { N, outputDepth, outputHeight, outputWidth, 3 });
                    using var tBuf = GetOrAllocateBuffer(backend, theta);
                    var gBuf = AllocateOutputBuffer(backend, outLen);
                    try
                    {
                        geom.AffineGrid3D(tBuf.Buffer, gBuf.Buffer,
                            N, outputDepth, outputHeight, outputWidth, alignCorners);
                        var arr = FinishGpuOp<T>(backend, gBuf, outLen);
                        return new Tensor<T>(arr, new[] { N, outputDepth, outputHeight, outputWidth, 3 });
                    }
                    catch { gBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.AffineGrid3D(theta, outputDepth, outputHeight, outputWidth, alignCorners);
    }

    /// <summary>
    /// GPU Interpolate2D kernel supports nearest, bilinear, bicubic and
    /// area. Linear (1D) and trilinear (3D) fall through to CPU because
    /// they need 3D / 5D tensor handling respectively.
    /// </summary>
    private static bool IsGpuInterpolateMode(InterpolateMode mode)
        => mode == InterpolateMode.Nearest
           || mode == InterpolateMode.Bilinear
           || mode == InterpolateMode.Bicubic
           || mode == InterpolateMode.Area;
}
