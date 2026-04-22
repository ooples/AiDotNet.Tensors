// Copyright (c) AiDotNet. All rights reserved.
// CPU reference implementations of the geometry / sampling ops added by
// Issue #217: Interpolate (6 modes × 1D/2D/3D), PadNd (4 modes × any rank),
// GridSample (bilinear / nearest / bicubic × zeros / border / reflection
// × align_corners), AffineGrid3D. The pre-existing narrow
// GridSample(input, grid) and AffineGrid(theta, H, W) APIs stay as
// torchvision-default shims that route through here.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // ========================================================================
    // Interpolate
    // ========================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> Interpolate<T>(Tensor<T> input, int[] sizes, InterpolateMode mode, bool alignCorners = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (sizes is null) throw new ArgumentNullException(nameof(sizes));
        if (input.Rank < 3)
            throw new ArgumentException("Interpolate requires rank >= 3 ([N, C, ...] where ... is 1/2/3 spatial dims).");
        int spatialRank = input.Rank - 2;
        if (sizes.Length != spatialRank)
            throw new ArgumentException($"sizes length ({sizes.Length}) must equal spatial rank ({spatialRank}).");
        ValidateMode(mode, spatialRank);

        int N = input._shape[0], C = input._shape[1];
        var outShape = new int[input.Rank];
        outShape[0] = N; outShape[1] = C;
        for (int i = 0; i < spatialRank; i++)
        {
            if (sizes[i] <= 0) throw new ArgumentException("sizes must be positive.");
            outShape[2 + i] = sizes[i];
        }
        var output = new Tensor<T>(outShape);
        if (input.Length == 0 || output.Length == 0) return output;

        var ops = MathHelper.GetNumericOperations<T>();
        switch (mode)
        {
            case InterpolateMode.Nearest:
                InterpolateNearest(input, output, alignCorners: false);
                break;
            case InterpolateMode.Linear:
            case InterpolateMode.Bilinear:
            case InterpolateMode.Trilinear:
                InterpolateLinearFamily(input, output, alignCorners, ops);
                break;
            case InterpolateMode.Bicubic:
                InterpolateBicubic2D(input, output, alignCorners, ops);
                break;
            case InterpolateMode.Area:
                InterpolateArea(input, output, ops);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
        return output;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> InterpolateByScale<T>(Tensor<T> input, double[] scaleFactors, InterpolateMode mode, bool alignCorners = false)
    {
        if (scaleFactors is null) throw new ArgumentNullException(nameof(scaleFactors));
        int spatialRank = input.Rank - 2;
        if (scaleFactors.Length != spatialRank)
            throw new ArgumentException($"scaleFactors length ({scaleFactors.Length}) must equal spatial rank ({spatialRank}).");
        var sizes = new int[spatialRank];
        for (int i = 0; i < spatialRank; i++)
        {
            if (!(scaleFactors[i] > 0))
                throw new ArgumentException("scaleFactors must be positive.");
            sizes[i] = (int)Math.Floor(input._shape[2 + i] * scaleFactors[i]);
            if (sizes[i] < 1) sizes[i] = 1;
        }
        return Interpolate(input, sizes, mode, alignCorners);
    }

    private static void ValidateMode(InterpolateMode mode, int spatialRank)
    {
        bool ok = mode switch
        {
            InterpolateMode.Nearest => true,
            InterpolateMode.Area => true,
            InterpolateMode.Linear => spatialRank == 1,
            InterpolateMode.Bilinear => spatialRank == 2,
            InterpolateMode.Bicubic => spatialRank == 2,
            InterpolateMode.Trilinear => spatialRank == 3,
            _ => false,
        };
        if (!ok)
            throw new ArgumentException($"Mode {mode} is not valid for spatial rank {spatialRank}.");
    }

    /// <summary>
    /// Nearest-neighbour interpolation — shared across 1D/2D/3D via
    /// per-axis index rounding. Matches PyTorch's <c>mode='nearest'</c>
    /// which uses <c>floor</c> (not rounding) of the fractional source
    /// coordinate.
    /// </summary>
    private static void InterpolateNearest<T>(Tensor<T> input, Tensor<T> output, bool alignCorners)
    {
        int rank = input.Rank;
        int N = input._shape[0], C = input._shape[1];
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        int spatial = rank - 2;

        // Pre-compute per-axis floor-scale factors (input / output). We
        // iterate linearly over output and compute source offset.
        var srcDims = new int[spatial]; var dstDims = new int[spatial];
        for (int i = 0; i < spatial; i++) { srcDims[i] = input._shape[2 + i]; dstDims[i] = output._shape[2 + i]; }

        int dstSpatialCount = 1;
        for (int i = 0; i < spatial; i++) dstSpatialCount *= dstDims[i];
        int srcSpatialCount = 1;
        for (int i = 0; i < spatial; i++) srcSpatialCount *= srcDims[i];

        var dstIdx = new int[spatial];
        var srcStride = new int[spatial];
        srcStride[spatial - 1] = 1;
        for (int i = spatial - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * srcDims[i + 1];

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int srcBase = (n * C + c) * srcSpatialCount;
            int dstBase = (n * C + c) * dstSpatialCount;
            // Iterate output spatial.
            for (int k = 0; k < dstSpatialCount; k++)
            {
                int tmp = k;
                for (int i = spatial - 1; i >= 0; i--)
                {
                    dstIdx[i] = tmp % dstDims[i];
                    tmp /= dstDims[i];
                }
                int srcOff = 0;
                for (int i = 0; i < spatial; i++)
                {
                    double s = dstDims[i] > 1
                        ? (double)dstIdx[i] * srcDims[i] / dstDims[i]
                        : 0.0;
                    int si = (int)Math.Floor(s);
                    if (si >= srcDims[i]) si = srcDims[i] - 1;
                    srcOff += si * srcStride[i];
                }
                dst[dstBase + k] = src[srcBase + srcOff];
            }
        }
    }

    /// <summary>
    /// Linear-family interpolation (linear 1D / bilinear 2D / trilinear 3D).
    /// Shared implementation: each output sample mixes 2^spatial source
    /// corners via multilinear weights. align_corners controls the mapping
    /// between output and source index — align_corners=true stretches so
    /// corners coincide; align_corners=false treats pixels as points
    /// (torchvision default).
    /// </summary>
    private static void InterpolateLinearFamily<T>(Tensor<T> input, Tensor<T> output, bool alignCorners, Interfaces.INumericOperations<T> ops)
    {
        int spatial = input.Rank - 2;
        int N = input._shape[0], C = input._shape[1];
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();

        var srcDims = new int[spatial]; var dstDims = new int[spatial];
        for (int i = 0; i < spatial; i++) { srcDims[i] = input._shape[2 + i]; dstDims[i] = output._shape[2 + i]; }
        var srcStride = new int[spatial];
        srcStride[spatial - 1] = 1;
        for (int i = spatial - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * srcDims[i + 1];
        int srcSpatial = 1; for (int i = 0; i < spatial; i++) srcSpatial *= srcDims[i];
        int dstSpatial = 1; for (int i = 0; i < spatial; i++) dstSpatial *= dstDims[i];
        int corners = 1 << spatial;

        var lo = new int[spatial]; var hi = new int[spatial]; var frac = new double[spatial];
        var dstIdx = new int[spatial];

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int srcBase = (n * C + c) * srcSpatial;
            int dstBase = (n * C + c) * dstSpatial;
            for (int k = 0; k < dstSpatial; k++)
            {
                int tmp = k;
                for (int i = spatial - 1; i >= 0; i--) { dstIdx[i] = tmp % dstDims[i]; tmp /= dstDims[i]; }
                // Per-axis fractional source coord.
                for (int i = 0; i < spatial; i++)
                {
                    double s = SourceCoordinate(dstIdx[i], dstDims[i], srcDims[i], alignCorners);
                    int l = (int)Math.Floor(s);
                    if (l < 0) l = 0;
                    int h = l + 1;
                    if (h >= srcDims[i]) { h = srcDims[i] - 1; l = Math.Min(l, h); }
                    lo[i] = l; hi[i] = h; frac[i] = s - l;
                    if (frac[i] < 0) frac[i] = 0;
                    if (frac[i] > 1) frac[i] = 1;
                }

                double acc = 0.0;
                for (int corner = 0; corner < corners; corner++)
                {
                    int srcOff = 0;
                    double weight = 1.0;
                    for (int i = 0; i < spatial; i++)
                    {
                        bool takeHi = ((corner >> i) & 1) == 1;
                        srcOff += (takeHi ? hi[i] : lo[i]) * srcStride[i];
                        weight *= takeHi ? frac[i] : (1.0 - frac[i]);
                    }
                    acc += weight * ops.ToDouble(src[srcBase + srcOff]);
                }
                dst[dstBase + k] = ops.FromDouble(acc);
            }
        }
    }

    private static double SourceCoordinate(int dstIdx, int dstSize, int srcSize, bool alignCorners)
    {
        if (dstSize <= 1) return 0.0;
        return alignCorners
            ? (double)dstIdx * (srcSize - 1) / (dstSize - 1)
            // pixel-centre convention (torchvision default): map output centre
            // to corresponding source centre.
            : ((dstIdx + 0.5) * srcSize / dstSize) - 0.5;
    }

    /// <summary>
    /// Bicubic 2D. Uses the torchvision Catmull-Rom cubic kernel
    /// (a = -0.75), separable along H and W. Clamps out-of-range source
    /// indices to the boundary rather than zero-padding, matching
    /// torchvision's behaviour.
    /// </summary>
    private static void InterpolateBicubic2D<T>(Tensor<T> input, Tensor<T> output, bool alignCorners, Interfaces.INumericOperations<T> ops)
    {
        int N = input._shape[0], C = input._shape[1];
        int H = input._shape[2], W = input._shape[3];
        int outH = output._shape[2], outW = output._shape[3];
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int srcBase = (n * C + c) * H * W;
            int dstBase = (n * C + c) * outH * outW;
            for (int y = 0; y < outH; y++)
            {
                double sy = SourceCoordinate(y, outH, H, alignCorners);
                int y0 = (int)Math.Floor(sy);
                double ty = sy - y0;
                double[] wy = CubicWeights(ty);
                for (int x = 0; x < outW; x++)
                {
                    double sx = SourceCoordinate(x, outW, W, alignCorners);
                    int x0 = (int)Math.Floor(sx);
                    double tx = sx - x0;
                    double[] wx = CubicWeights(tx);
                    double acc = 0.0;
                    for (int yy = 0; yy < 4; yy++)
                    {
                        int yi = Math.Clamp(y0 - 1 + yy, 0, H - 1);
                        double rowAcc = 0.0;
                        for (int xx = 0; xx < 4; xx++)
                        {
                            int xi = Math.Clamp(x0 - 1 + xx, 0, W - 1);
                            rowAcc += wx[xx] * ops.ToDouble(src[srcBase + yi * W + xi]);
                        }
                        acc += wy[yy] * rowAcc;
                    }
                    dst[dstBase + y * outW + x] = ops.FromDouble(acc);
                }
            }
        }
    }

    /// <summary>Catmull-Rom cubic (a = −0.75), the torchvision default.</summary>
    private static double[] CubicWeights(double t)
    {
        // Kernel evaluated at offsets {1+t, t, 1-t, 2-t}.
        const double a = -0.75;
        double t1 = 1.0 + t, t2 = t, t3 = 1.0 - t, t4 = 2.0 - t;
        return new[]
        {
            CubicKernel(t1, a),
            CubicKernel(t2, a),
            CubicKernel(t3, a),
            CubicKernel(t4, a),
        };
    }

    private static double CubicKernel(double d, double a)
    {
        double ad = Math.Abs(d);
        if (ad < 1.0) return ((a + 2.0) * ad - (a + 3.0)) * ad * ad + 1.0;
        if (ad < 2.0) return a * ((ad - 5.0) * ad + 8.0) * ad - 4.0 * a;
        return 0.0;
    }

    /// <summary>
    /// Area interpolation — averages every source pixel covered by each
    /// output cell. Handles both downsampling (multi-source-per-dst, the
    /// common case) and upsampling (degenerate — equivalent to nearest-
    /// neighbour when srcSize == dstSize).
    /// </summary>
    private static void InterpolateArea<T>(Tensor<T> input, Tensor<T> output, Interfaces.INumericOperations<T> ops)
    {
        int spatial = input.Rank - 2;
        int N = input._shape[0], C = input._shape[1];
        var src = input.AsSpan();
        var dst = output.AsWritableSpan();

        var srcDims = new int[spatial]; var dstDims = new int[spatial];
        for (int i = 0; i < spatial; i++) { srcDims[i] = input._shape[2 + i]; dstDims[i] = output._shape[2 + i]; }
        var srcStride = new int[spatial];
        srcStride[spatial - 1] = 1;
        for (int i = spatial - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * srcDims[i + 1];
        int srcSpatial = 1; for (int i = 0; i < spatial; i++) srcSpatial *= srcDims[i];
        int dstSpatial = 1; for (int i = 0; i < spatial; i++) dstSpatial *= dstDims[i];

        var dstIdx = new int[spatial];
        var loI = new int[spatial]; var hiI = new int[spatial];

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int srcBase = (n * C + c) * srcSpatial;
            int dstBase = (n * C + c) * dstSpatial;
            for (int k = 0; k < dstSpatial; k++)
            {
                int tmp = k;
                for (int i = spatial - 1; i >= 0; i--) { dstIdx[i] = tmp % dstDims[i]; tmp /= dstDims[i]; }
                int cellCount = 1;
                for (int i = 0; i < spatial; i++)
                {
                    double lo = (double)dstIdx[i] * srcDims[i] / dstDims[i];
                    double hi = (double)(dstIdx[i] + 1) * srcDims[i] / dstDims[i];
                    loI[i] = (int)Math.Floor(lo);
                    hiI[i] = Math.Max(loI[i] + 1, (int)Math.Ceiling(hi));
                    if (hiI[i] > srcDims[i]) hiI[i] = srcDims[i];
                    cellCount *= (hiI[i] - loI[i]);
                }
                double acc = 0.0;
                SumRegion(src, srcBase, srcStride, loI, hiI, spatial, ops, ref acc, new int[spatial], 0);
                dst[dstBase + k] = ops.FromDouble(acc / Math.Max(1, cellCount));
            }
        }
    }

    private static void SumRegion<T>(ReadOnlySpan<T> src, int srcBase, int[] stride,
        int[] lo, int[] hi, int spatial, Interfaces.INumericOperations<T> ops,
        ref double acc, int[] coord, int axis)
    {
        if (axis == spatial)
        {
            int off = 0;
            for (int i = 0; i < spatial; i++) off += coord[i] * stride[i];
            acc += ops.ToDouble(src[srcBase + off]);
            return;
        }
        for (int i = lo[axis]; i < hi[axis]; i++)
        {
            coord[axis] = i;
            SumRegion(src, srcBase, stride, lo, hi, spatial, ops, ref acc, coord, axis + 1);
        }
    }

    // ========================================================================
    // PadNd
    // ========================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> PadNd<T>(Tensor<T> input, int[] pad, PadMode mode, T value = default!)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (pad is null) throw new ArgumentNullException(nameof(pad));
        if ((pad.Length & 1) != 0)
            throw new ArgumentException("pad length must be even (before/after pairs).");
        int padAxes = pad.Length / 2;
        if (padAxes > input.Rank)
            throw new ArgumentException($"pad covers {padAxes} axes but input has only {input.Rank}.");

        // PyTorch's pad order is innermost-first. Convert to per-axis (before, after)
        // from axis 0 to rank-1. Axes 0..rank-padAxes-1 get (0, 0).
        var before = new int[input.Rank];
        var after = new int[input.Rank];
        for (int i = 0; i < padAxes; i++)
        {
            int axis = input.Rank - 1 - i;
            before[axis] = pad[i * 2];
            after[axis] = pad[i * 2 + 1];
            if (before[axis] < 0 || after[axis] < 0)
                throw new ArgumentException("pad amounts must be non-negative.");
        }

        var outShape = new int[input.Rank];
        for (int i = 0; i < input.Rank; i++)
            outShape[i] = input._shape[i] + before[i] + after[i];
        var output = new Tensor<T>(outShape);
        if (output.Length == 0) return output;

        var src = input.AsSpan();
        var dst = output.AsWritableSpan();
        var ops = MathHelper.GetNumericOperations<T>();
        T fill = mode == PadMode.Constant ? value : ops.Zero;

        // Compute strides for both src and dst in row-major order.
        var srcStride = new int[input.Rank];
        var dstStride = new int[input.Rank];
        srcStride[input.Rank - 1] = 1;
        dstStride[input.Rank - 1] = 1;
        for (int i = input.Rank - 2; i >= 0; i--)
        {
            srcStride[i] = srcStride[i + 1] * input._shape[i + 1];
            dstStride[i] = dstStride[i + 1] * outShape[i + 1];
        }

        // Iterate output linearly, map each output index back to source
        // via the padding mode. Any axis where the mapped index is
        // out-of-range triggers the constant fill (for Constant mode) or
        // boundary arithmetic for the other modes.
        var outIdx = new int[input.Rank];
        for (int k = 0; k < output.Length; k++)
        {
            int tmp = k;
            for (int i = input.Rank - 1; i >= 0; i--)
            {
                outIdx[i] = tmp % outShape[i];
                tmp /= outShape[i];
            }
            int srcOff = 0;
            bool inBounds = true;
            for (int i = 0; i < input.Rank; i++)
            {
                int local = outIdx[i] - before[i];
                int extent = input._shape[i];
                if (local < 0 || local >= extent)
                {
                    if (mode == PadMode.Constant) { inBounds = false; break; }
                    local = MapBoundary(local, extent, mode);
                }
                srcOff += local * srcStride[i];
            }
            dst[k] = inBounds ? src[srcOff] : fill;
        }
        return output;
    }

    private static int MapBoundary(int idx, int extent, PadMode mode)
    {
        switch (mode)
        {
            case PadMode.Replicate:
                if (idx < 0) return 0;
                if (idx >= extent) return extent - 1;
                return idx;
            case PadMode.Reflect:
            {
                // Reflect without repeating the boundary pixel (pytorch
                // default). Period = 2*(extent-1).
                if (extent == 1) return 0;
                int period = 2 * (extent - 1);
                int r = ((idx % period) + period) % period;
                return r < extent ? r : period - r;
            }
            case PadMode.Circular:
            {
                int r = ((idx % extent) + extent) % extent;
                return r;
            }
            default:
                return 0;
        }
    }

    // ========================================================================
    // GridSample (extended) + AffineGrid3D
    // ========================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid,
        GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        if (input.Rank != 4) throw new ArgumentException("GridSample expects NHWC input [N, H, W, C].");
        if (grid.Rank != 4 || grid._shape[3] != 2) throw new ArgumentException("grid must be [N, outH, outW, 2].");
        if (input._shape[0] != grid._shape[0]) throw new ArgumentException("batch dim of input and grid must match.");

        int N = input._shape[0], H = input._shape[1], W = input._shape[2], C = input._shape[3];
        int outH = grid._shape[1], outW = grid._shape[2];
        var output = new Tensor<T>(new[] { N, outH, outW, C });
        if (output.Length == 0) return output;

        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var g = grid.AsSpan();
        var dst = output.AsWritableSpan();

        for (int n = 0; n < N; n++)
        for (int oy = 0; oy < outH; oy++)
        for (int ox = 0; ox < outW; ox++)
        {
            int gOff = ((n * outH + oy) * outW + ox) * 2;
            double gx = ops.ToDouble(g[gOff]);
            double gy = ops.ToDouble(g[gOff + 1]);
            // Normalised [-1, 1] -> fractional source index.
            double sx = NormalizedToPixel(gx, W, alignCorners);
            double sy = NormalizedToPixel(gy, H, alignCorners);

            if (mode == GridSampleMode.Nearest)
            {
                int nx = (int)Math.Round(sx), ny = (int)Math.Round(sy);
                for (int c = 0; c < C; c++)
                    dst[((n * outH + oy) * outW + ox) * C + c] =
                        SampleSafe(src, n, ny, nx, c, H, W, C, padding, ops);
            }
            else if (mode == GridSampleMode.Bilinear)
            {
                int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
                int x1 = x0 + 1, y1 = y0 + 1;
                double fx = sx - x0, fy = sy - y0;
                for (int c = 0; c < C; c++)
                {
                    double v00 = ops.ToDouble(SampleSafe(src, n, y0, x0, c, H, W, C, padding, ops));
                    double v01 = ops.ToDouble(SampleSafe(src, n, y0, x1, c, H, W, C, padding, ops));
                    double v10 = ops.ToDouble(SampleSafe(src, n, y1, x0, c, H, W, C, padding, ops));
                    double v11 = ops.ToDouble(SampleSafe(src, n, y1, x1, c, H, W, C, padding, ops));
                    double v =
                        v00 * (1 - fx) * (1 - fy) +
                        v01 * fx * (1 - fy) +
                        v10 * (1 - fx) * fy +
                        v11 * fx * fy;
                    dst[((n * outH + oy) * outW + ox) * C + c] = ops.FromDouble(v);
                }
            }
            else // Bicubic
            {
                int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
                double fx = sx - x0, fy = sy - y0;
                double[] wx = CubicWeights(fx), wy = CubicWeights(fy);
                for (int c = 0; c < C; c++)
                {
                    double acc = 0.0;
                    for (int yy = 0; yy < 4; yy++)
                    {
                        int yi = y0 - 1 + yy;
                        double rowAcc = 0.0;
                        for (int xx = 0; xx < 4; xx++)
                        {
                            int xi = x0 - 1 + xx;
                            rowAcc += wx[xx] * ops.ToDouble(SampleSafe(src, n, yi, xi, c, H, W, C, padding, ops));
                        }
                        acc += wy[yy] * rowAcc;
                    }
                    dst[((n * outH + oy) * outW + ox) * C + c] = ops.FromDouble(acc);
                }
            }
        }
        return output;
    }

    /// <summary>
    /// Normalised grid coord (range [-1, 1]) → fractional source pixel
    /// index. <paramref name="alignCorners"/> switches between the two
    /// conventions; matches torchvision.
    /// </summary>
    private static double NormalizedToPixel(double coord, int size, bool alignCorners)
    {
        if (alignCorners) return (coord + 1.0) * 0.5 * (size - 1);
        return ((coord + 1.0) * size - 1.0) * 0.5;
    }

    private static T SampleSafe<T>(ReadOnlySpan<T> src, int n, int y, int x, int c,
        int H, int W, int C, GridSamplePadding padding, Interfaces.INumericOperations<T> ops)
    {
        switch (padding)
        {
            case GridSamplePadding.Zeros:
                if ((uint)y >= H || (uint)x >= W) return ops.Zero;
                break;
            case GridSamplePadding.Border:
                y = Math.Clamp(y, 0, H - 1);
                x = Math.Clamp(x, 0, W - 1);
                break;
            case GridSamplePadding.Reflection:
                y = ReflectIndex(y, H);
                x = ReflectIndex(x, W);
                break;
        }
        return src[((n * H + y) * W + x) * C + c];
    }

    private static int ReflectIndex(int i, int extent)
    {
        if (extent == 1) return 0;
        int period = 2 * (extent - 1);
        int r = ((i % period) + period) % period;
        return r < extent ? r : period - r;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> AffineGrid3D<T>(Tensor<T> theta, int outputDepth, int outputHeight, int outputWidth, bool alignCorners = false)
    {
        if (theta.Rank != 3 || theta._shape[1] != 3 || theta._shape[2] != 4)
            throw new ArgumentException("theta must be [N, 3, 4].");
        int N = theta._shape[0];
        var grid = new Tensor<T>(new[] { N, outputDepth, outputHeight, outputWidth, 3 });
        if (grid.Length == 0) return grid;

        var ops = MathHelper.GetNumericOperations<T>();
        var t = theta.AsSpan();
        var g = grid.AsWritableSpan();

        for (int n = 0; n < N; n++)
        {
            int tBase = n * 12; // 3 × 4 affine matrix
            for (int d = 0; d < outputDepth; d++)
            {
                double z = GridCoordinate(d, outputDepth, alignCorners);
                for (int h = 0; h < outputHeight; h++)
                {
                    double y = GridCoordinate(h, outputHeight, alignCorners);
                    for (int w = 0; w < outputWidth; w++)
                    {
                        double x = GridCoordinate(w, outputWidth, alignCorners);
                        int gBase = (((n * outputDepth + d) * outputHeight + h) * outputWidth + w) * 3;
                        for (int row = 0; row < 3; row++)
                        {
                            double v =
                                ops.ToDouble(t[tBase + row * 4]) * x +
                                ops.ToDouble(t[tBase + row * 4 + 1]) * y +
                                ops.ToDouble(t[tBase + row * 4 + 2]) * z +
                                ops.ToDouble(t[tBase + row * 4 + 3]);
                            g[gBase + row] = ops.FromDouble(v);
                        }
                    }
                }
            }
        }
        return grid;
    }

    private static double GridCoordinate(int idx, int size, bool alignCorners)
    {
        if (size <= 1) return 0.0;
        return alignCorners
            ? -1.0 + 2.0 * idx / (size - 1)
            : -1.0 + (2.0 * idx + 1.0) / size;
    }
}
