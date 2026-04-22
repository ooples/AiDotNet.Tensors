// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for the geometry / sampling ops added by Issue #217.
// Covers Interpolate (nearest/bilinear/area/bicubic on 4D NCHW),
// Pad (4 modes on 4D NCHW), GridSample (bilinear/nearest/bicubic with
// zeros/border/reflection padding on 4D NHWC), and AffineGrid3D.
// Non-float tensors and rank ≠ 4 inputs fall back to CpuEngine.
//
// Mode ints map onto the InterpolateMode / PadMode / GridSampleMode /
// GridSamplePadding enums in the managed layer:
//   InterpolateMode: 0=Nearest 1=Linear 2=Bilinear 3=Bicubic 4=Trilinear 5=Area
//   PadMode:         0=Constant 1=Reflect 2=Replicate 3=Circular
//   GridSampleMode:  0=Bilinear 1=Nearest 2=Bicubic
//   GridSamplePadding: 0=Zeros 1=Border 2=Reflection

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    public static class CudaGeometryKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "geometry_interpolate_2d",
            "geometry_pad_4d",
            "geometry_grid_sample_2d",
            "geometry_affine_grid_3d",
        };

        public static string GetSource() => @"
#include <math.h>

// ----------------------------------------------------------------------------
// Shared device helpers.
// ----------------------------------------------------------------------------

__device__ __forceinline__ double source_coord(int dstIdx, int dstSize, int srcSize, int alignCorners)
{
    if (dstSize <= 1) return 0.0;
    if (alignCorners) return (double)dstIdx * (srcSize - 1) / (dstSize - 1);
    return ((double)dstIdx + 0.5) * srcSize / dstSize - 0.5;
}

__device__ __forceinline__ double cubic_kernel(double d, double a)
{
    double ad = fabs(d);
    if (ad < 1.0) return ((a + 2.0) * ad - (a + 3.0)) * ad * ad + 1.0;
    if (ad < 2.0) return a * ((ad - 5.0) * ad + 8.0) * ad - 4.0 * a;
    return 0.0;
}

__device__ __forceinline__ int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__device__ __forceinline__ int reflect_index(int i, int extent)
{
    if (extent == 1) return 0;
    int period = 2 * (extent - 1);
    int r = ((i % period) + period) % period;
    return r < extent ? r : period - r;
}

__device__ __forceinline__ int pad_boundary(int idx, int extent, int mode)
{
    // mode: 1=reflect, 2=replicate, 3=circular.
    // Guard zero-sized axes: reflect/replicate/circular all collapse to
    // index 0 so the kernel doesn't divide-by-zero or index past end.
    if (extent <= 0) return 0;
    if (mode == 2) {
        if (idx < 0) return 0;
        if (idx >= extent) return extent - 1;
        return idx;
    }
    if (mode == 1) return reflect_index(idx, extent);
    int r = ((idx % extent) + extent) % extent;
    return r;
}

// ----------------------------------------------------------------------------
// Interpolate 2D — NCHW, modes: 0=nearest, 2=bilinear, 3=bicubic, 5=area.
// One thread per output (n, c, y, x).
// ----------------------------------------------------------------------------

extern ""C"" __global__ __launch_bounds__(256) void geometry_interpolate_2d(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int Hin, int Win, int Hout, int Wout,
    int mode, int alignCorners)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (gid >= total) return;
    int x = gid % Wout;
    int tmp = gid / Wout;
    int y = tmp % Hout;
    tmp /= Hout;
    int c = tmp % C;
    int n = tmp / C;

    const float* src = input + ((n * C + c) * Hin) * Win;

    if (mode == 0) {  // Nearest
        double sy = Hout > 1 ? (double)y * Hin / Hout : 0.0;
        double sx = Wout > 1 ? (double)x * Win / Wout : 0.0;
        int yi = (int)floor(sy); if (yi >= Hin) yi = Hin - 1;
        int xi = (int)floor(sx); if (xi >= Win) xi = Win - 1;
        output[gid] = src[yi * Win + xi];
    } else if (mode == 2) {  // Bilinear
        double sy = source_coord(y, Hout, Hin, alignCorners);
        double sx = source_coord(x, Wout, Win, alignCorners);
        int y0 = (int)floor(sy); int x0 = (int)floor(sx);
        if (y0 < 0) y0 = 0; if (x0 < 0) x0 = 0;
        int y1 = y0 + 1; int x1 = x0 + 1;
        if (y1 >= Hin) { y1 = Hin - 1; if (y0 > y1) y0 = y1; }
        if (x1 >= Win) { x1 = Win - 1; if (x0 > x1) x0 = x1; }
        double fy = sy - y0; if (fy < 0) fy = 0; if (fy > 1) fy = 1;
        double fx = sx - x0; if (fx < 0) fx = 0; if (fx > 1) fx = 1;
        double v00 = src[y0 * Win + x0];
        double v01 = src[y0 * Win + x1];
        double v10 = src[y1 * Win + x0];
        double v11 = src[y1 * Win + x1];
        double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
                 + v10 * (1 - fx) * fy + v11 * fx * fy;
        output[gid] = (float)v;
    } else if (mode == 3) {  // Bicubic (Catmull-Rom, a = -0.75)
        double sy = source_coord(y, Hout, Hin, alignCorners);
        double sx = source_coord(x, Wout, Win, alignCorners);
        int y0 = (int)floor(sy); double ty = sy - y0;
        int x0 = (int)floor(sx); double tx = sx - x0;
        double wy[4] = {
            cubic_kernel(1.0 + ty, -0.75),
            cubic_kernel(ty, -0.75),
            cubic_kernel(1.0 - ty, -0.75),
            cubic_kernel(2.0 - ty, -0.75),
        };
        double wx[4] = {
            cubic_kernel(1.0 + tx, -0.75),
            cubic_kernel(tx, -0.75),
            cubic_kernel(1.0 - tx, -0.75),
            cubic_kernel(2.0 - tx, -0.75),
        };
        double acc = 0.0;
        for (int yy = 0; yy < 4; yy++) {
            int yi = clamp_i(y0 - 1 + yy, 0, Hin - 1);
            double rowAcc = 0.0;
            for (int xx = 0; xx < 4; xx++) {
                int xi = clamp_i(x0 - 1 + xx, 0, Win - 1);
                rowAcc += wx[xx] * src[yi * Win + xi];
            }
            acc += wy[yy] * rowAcc;
        }
        output[gid] = (float)acc;
    } else {  // Area (mode == 5) — overlap-weighted averaging
        double yLo = (double)y * Hin / Hout;
        double yHi = (double)(y + 1) * Hin / Hout;
        double xLo = (double)x * Win / Wout;
        double xHi = (double)(x + 1) * Win / Wout;
        int yL = (int)floor(yLo); int yH = (int)ceil(yHi);
        int xL = (int)floor(xLo); int xH = (int)ceil(xHi);
        if (yH <= yL) yH = yL + 1;
        if (xH <= xL) xH = xL + 1;
        if (yH > Hin) yH = Hin;
        if (xH > Win) xH = Win;
        double totalArea = (yHi - yLo) * (xHi - xLo);
        double acc = 0.0;
        for (int yy = yL; yy < yH; yy++) {
            double oy = fmax(0.0, fmin(yHi, (double)(yy + 1)) - fmax(yLo, (double)yy));
            if (oy <= 0.0) continue;
            for (int xx = xL; xx < xH; xx++) {
                double ox = fmax(0.0, fmin(xHi, (double)(xx + 1)) - fmax(xLo, (double)xx));
                if (ox <= 0.0) continue;
                acc += oy * ox * src[yy * Win + xx];
            }
        }
        output[gid] = (float)(totalArea > 0.0 ? acc / totalArea : 0.0);
    }
}

// ----------------------------------------------------------------------------
// Pad 4D — NCHW, modes: 0=constant, 1=reflect, 2=replicate, 3=circular.
// One thread per output element (n, c, h, w).
// ----------------------------------------------------------------------------

extern ""C"" __global__ __launch_bounds__(256) void geometry_pad_4d(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int Hin, int Win,
    int padN0, int padN1, int padC0, int padC1,
    int padH0, int padH1, int padW0, int padW1,
    int mode, float padValue)
{
    int Nout = N + padN0 + padN1;
    int Cout = C + padC0 + padC1;
    int Hout = Hin + padH0 + padH1;
    int Wout = Win + padW0 + padW1;
    int total = Nout * Cout * Hout * Wout;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int w = gid % Wout; int t1 = gid / Wout;
    int h = t1 % Hout; int t2 = t1 / Hout;
    int c = t2 % Cout; int n = t2 / Cout;

    int nn = n - padN0, cc = c - padC0, hh = h - padH0, ww = w - padW0;

    bool inB = nn >= 0 && nn < N && cc >= 0 && cc < C && hh >= 0 && hh < Hin && ww >= 0 && ww < Win;
    if (!inB && mode == 0) { output[gid] = padValue; return; }

    if (!inB) {
        if (!(nn >= 0 && nn < N)) nn = pad_boundary(nn, N, mode);
        if (!(cc >= 0 && cc < C)) cc = pad_boundary(cc, C, mode);
        if (!(hh >= 0 && hh < Hin)) hh = pad_boundary(hh, Hin, mode);
        if (!(ww >= 0 && ww < Win)) ww = pad_boundary(ww, Win, mode);
    }
    output[gid] = input[((nn * C + cc) * Hin + hh) * Win + ww];
}

// ----------------------------------------------------------------------------
// GridSample 2D — NHWC, modes: 0=bilinear, 1=nearest, 2=bicubic.
// padding: 0=zeros, 1=border, 2=reflection.
// One thread per output (n, oy, ox, c) — but we loop C in-thread to
// amortise grid-lookup work.
// ----------------------------------------------------------------------------

__device__ __forceinline__ float grid_sample_safe(const float* __restrict__ src,
    int n, int y, int x, int c,
    int H, int W, int C, int padding)
{
    if (padding == 0) {
        if ((unsigned)y >= (unsigned)H || (unsigned)x >= (unsigned)W) return 0.0f;
    } else if (padding == 1) {
        y = clamp_i(y, 0, H - 1);
        x = clamp_i(x, 0, W - 1);
    } else {
        y = reflect_index(y, H);
        x = reflect_index(x, W);
    }
    return src[((n * H + y) * W + x) * C + c];
}

extern ""C"" __global__ __launch_bounds__(256) void geometry_grid_sample_2d(
    const float* __restrict__ input, const float* __restrict__ grid,
    float* __restrict__ output,
    int N, int H, int W, int C, int outH, int outW,
    int mode, int padding, int alignCorners)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * outH * outW;
    if (gid >= total) return;
    int ox = gid % outW; int t1 = gid / outW;
    int oy = t1 % outH; int n = t1 / outH;

    int gOff = ((n * outH + oy) * outW + ox) * 2;
    double gx = grid[gOff];
    double gy = grid[gOff + 1];
    double sx = alignCorners ? (gx + 1.0) * 0.5 * (W - 1) : ((gx + 1.0) * W - 1.0) * 0.5;
    double sy = alignCorners ? (gy + 1.0) * 0.5 * (H - 1) : ((gy + 1.0) * H - 1.0) * 0.5;

    if (mode == 1) {  // Nearest
        int nx = (int)lrint(sx), ny = (int)lrint(sy);
        for (int c = 0; c < C; c++)
            output[((n * outH + oy) * outW + ox) * C + c] =
                grid_sample_safe(input, n, ny, nx, c, H, W, C, padding);
    } else if (mode == 0) {  // Bilinear
        int x0 = (int)floor(sx), y0 = (int)floor(sy);
        int x1 = x0 + 1, y1 = y0 + 1;
        double fx = sx - x0, fy = sy - y0;
        for (int c = 0; c < C; c++) {
            float v00 = grid_sample_safe(input, n, y0, x0, c, H, W, C, padding);
            float v01 = grid_sample_safe(input, n, y0, x1, c, H, W, C, padding);
            float v10 = grid_sample_safe(input, n, y1, x0, c, H, W, C, padding);
            float v11 = grid_sample_safe(input, n, y1, x1, c, H, W, C, padding);
            double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
                     + v10 * (1 - fx) * fy + v11 * fx * fy;
            output[((n * outH + oy) * outW + ox) * C + c] = (float)v;
        }
    } else {  // Bicubic
        int x0 = (int)floor(sx), y0 = (int)floor(sy);
        double fx = sx - x0, fy = sy - y0;
        double wy[4] = { cubic_kernel(1.0 + fy, -0.75), cubic_kernel(fy, -0.75),
                         cubic_kernel(1.0 - fy, -0.75), cubic_kernel(2.0 - fy, -0.75) };
        double wx[4] = { cubic_kernel(1.0 + fx, -0.75), cubic_kernel(fx, -0.75),
                         cubic_kernel(1.0 - fx, -0.75), cubic_kernel(2.0 - fx, -0.75) };
        for (int c = 0; c < C; c++) {
            double acc = 0.0;
            for (int yy = 0; yy < 4; yy++) {
                int yi = y0 - 1 + yy;
                double rowAcc = 0.0;
                for (int xx = 0; xx < 4; xx++) {
                    int xi = x0 - 1 + xx;
                    rowAcc += wx[xx] * grid_sample_safe(input, n, yi, xi, c, H, W, C, padding);
                }
                acc += wy[yy] * rowAcc;
            }
            output[((n * outH + oy) * outW + ox) * C + c] = (float)acc;
        }
    }
}

// ----------------------------------------------------------------------------
// AffineGrid 3D — theta [N, 3, 4] → grid [N, D, H, W, 3].
// One thread per output (n, d, h, w).
// ----------------------------------------------------------------------------

__device__ __forceinline__ double grid_norm_coord(int idx, int size, int alignCorners)
{
    if (size <= 1) return 0.0;
    return alignCorners ? -1.0 + 2.0 * idx / (size - 1) : -1.0 + (2.0 * idx + 1.0) / size;
}

extern ""C"" __global__ __launch_bounds__(256) void geometry_affine_grid_3d(
    const float* __restrict__ theta, float* __restrict__ grid,
    int N, int D, int H, int W, int alignCorners)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D * H * W;
    if (gid >= total) return;
    int w = gid % W; int t1 = gid / W;
    int h = t1 % H; int t2 = t1 / H;
    int d = t2 % D; int n = t2 / D;

    int tBase = n * 12;
    double x = grid_norm_coord(w, W, alignCorners);
    double y = grid_norm_coord(h, H, alignCorners);
    double z = grid_norm_coord(d, D, alignCorners);
    int gBase = (((n * D + d) * H + h) * W + w) * 3;
    for (int row = 0; row < 3; row++) {
        double v = theta[tBase + row * 4] * x
                 + theta[tBase + row * 4 + 1] * y
                 + theta[tBase + row * 4 + 2] * z
                 + theta[tBase + row * 4 + 3];
        grid[gBase + row] = (float)v;
    }
}
";
    }
}
