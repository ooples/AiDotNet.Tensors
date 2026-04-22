// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) kernels for the geometry / sampling ops
// added by Issue #217. Mirrors CudaGeometryKernels / OpenClGeometryKernels
// in coverage and mode-int mapping.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    public static class MetalGeometryKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "geometry_interpolate_2d",
            "geometry_pad_4d",
            "geometry_grid_sample_2d",
            "geometry_affine_grid_3d",
        };

        public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// ----------------------------------------------------------------------------
// Shared device helpers.
// ----------------------------------------------------------------------------

inline float source_coord_f(int dstIdx, int dstSize, int srcSize, int alignCorners) {
    if (dstSize <= 1) return 0.0;
    if (alignCorners) return float(dstIdx) * (srcSize - 1) / (dstSize - 1);
    return (float(dstIdx) + 0.5) * srcSize / dstSize - 0.5;
}

inline float cubic_kernel_f(float d, float a) {
    float ad = fabs(d);
    if (ad < 1.0) return ((a + 2.0) * ad - (a + 3.0)) * ad * ad + 1.0;
    if (ad < 2.0) return a * ((ad - 5.0) * ad + 8.0) * ad - 4.0 * a;
    return 0.0;
}

inline int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline int reflect_index(int i, int extent) {
    if (extent == 1) return 0;
    int period = 2 * (extent - 1);
    int r = ((i % period) + period) % period;
    return r < extent ? r : period - r;
}

inline int pad_boundary(int idx, int extent, int mode) {
    if (extent <= 0) return 0;
    if (mode == 2) { if (idx < 0) return 0; if (idx >= extent) return extent - 1; return idx; }
    if (mode == 1) return reflect_index(idx, extent);
    int r = ((idx % extent) + extent) % extent;
    return r;
}

// ----------------------------------------------------------------------------
// Interpolate 2D.
// ----------------------------------------------------------------------------

kernel void geometry_interpolate_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& N [[buffer(2)]], constant int& C [[buffer(3)]],
    constant int& Hin [[buffer(4)]], constant int& Win [[buffer(5)]],
    constant int& Hout [[buffer(6)]], constant int& Wout [[buffer(7)]],
    constant int& mode [[buffer(8)]], constant int& alignCorners [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    int total = N * C * Hout * Wout;
    if ((int)gid >= total) return;
    int x = (int)gid % Wout; int t1 = (int)gid / Wout;
    int y = t1 % Hout; int t2 = t1 / Hout;
    int c = t2 % C; int n = t2 / C;
    device const float* src = input + ((n * C + c) * Hin) * Win;

    if (mode == 0) {
        float sy = Hout > 1 ? (float)y * Hin / Hout : 0.0;
        float sx = Wout > 1 ? (float)x * Win / Wout : 0.0;
        int yi = (int)floor(sy); if (yi >= Hin) yi = Hin - 1;
        int xi = (int)floor(sx); if (xi >= Win) xi = Win - 1;
        output[gid] = src[yi * Win + xi];
    } else if (mode == 2) {
        float sy = source_coord_f(y, Hout, Hin, alignCorners);
        float sx = source_coord_f(x, Wout, Win, alignCorners);
        int y0 = (int)floor(sy); int x0 = (int)floor(sx);
        if (y0 < 0) y0 = 0; if (x0 < 0) x0 = 0;
        int y1 = y0 + 1; int x1 = x0 + 1;
        if (y1 >= Hin) { y1 = Hin - 1; if (y0 > y1) y0 = y1; }
        if (x1 >= Win) { x1 = Win - 1; if (x0 > x1) x0 = x1; }
        float fy = sy - y0; if (fy < 0) fy = 0; if (fy > 1) fy = 1;
        float fx = sx - x0; if (fx < 0) fx = 0; if (fx > 1) fx = 1;
        float v00 = src[y0 * Win + x0], v01 = src[y0 * Win + x1];
        float v10 = src[y1 * Win + x0], v11 = src[y1 * Win + x1];
        output[gid] = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
                    + v10 * (1 - fx) * fy + v11 * fx * fy;
    } else if (mode == 3) {
        float sy = source_coord_f(y, Hout, Hin, alignCorners);
        float sx = source_coord_f(x, Wout, Win, alignCorners);
        int y0 = (int)floor(sy); float ty = sy - y0;
        int x0 = (int)floor(sx); float tx = sx - x0;
        float wy[4] = { cubic_kernel_f(1.0 + ty, -0.75), cubic_kernel_f(ty, -0.75),
                        cubic_kernel_f(1.0 - ty, -0.75), cubic_kernel_f(2.0 - ty, -0.75) };
        float wx[4] = { cubic_kernel_f(1.0 + tx, -0.75), cubic_kernel_f(tx, -0.75),
                        cubic_kernel_f(1.0 - tx, -0.75), cubic_kernel_f(2.0 - tx, -0.75) };
        float acc = 0.0;
        for (int yy = 0; yy < 4; yy++) {
            int yi = clamp_i(y0 - 1 + yy, 0, Hin - 1);
            float rowAcc = 0.0;
            for (int xx = 0; xx < 4; xx++) {
                int xi = clamp_i(x0 - 1 + xx, 0, Win - 1);
                rowAcc += wx[xx] * src[yi * Win + xi];
            }
            acc += wy[yy] * rowAcc;
        }
        output[gid] = acc;
    } else {  // Area — overlap-weighted averaging
        float yLo = (float)y * Hin / Hout;
        float yHi = (float)(y + 1) * Hin / Hout;
        float xLo = (float)x * Win / Wout;
        float xHi = (float)(x + 1) * Win / Wout;
        int yL = (int)floor(yLo); int yH = (int)ceil(yHi);
        int xL = (int)floor(xLo); int xH = (int)ceil(xHi);
        if (yH <= yL) yH = yL + 1;
        if (xH <= xL) xH = xL + 1;
        if (yH > Hin) yH = Hin;
        if (xH > Win) xH = Win;
        float totalArea = (yHi - yLo) * (xHi - xLo);
        float acc = 0.0;
        for (int yy = yL; yy < yH; yy++) {
            float oy = max(0.0f, min(yHi, (float)(yy + 1)) - max(yLo, (float)yy));
            if (oy <= 0.0f) continue;
            for (int xx = xL; xx < xH; xx++) {
                float ox = max(0.0f, min(xHi, (float)(xx + 1)) - max(xLo, (float)xx));
                if (ox <= 0.0f) continue;
                acc += oy * ox * src[yy * Win + xx];
            }
        }
        output[gid] = totalArea > 0.0 ? acc / totalArea : 0.0;
    }
}

// ----------------------------------------------------------------------------
// Pad 4D.
// ----------------------------------------------------------------------------

kernel void geometry_pad_4d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& N [[buffer(2)]], constant int& C [[buffer(3)]],
    constant int& Hin [[buffer(4)]], constant int& Win [[buffer(5)]],
    constant int& padN0 [[buffer(6)]], constant int& padN1 [[buffer(7)]],
    constant int& padC0 [[buffer(8)]], constant int& padC1 [[buffer(9)]],
    constant int& padH0 [[buffer(10)]], constant int& padH1 [[buffer(11)]],
    constant int& padW0 [[buffer(12)]], constant int& padW1 [[buffer(13)]],
    constant int& mode [[buffer(14)]], constant float& padValue [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    int Nout = N + padN0 + padN1;
    int Cout = C + padC0 + padC1;
    int Hout = Hin + padH0 + padH1;
    int Wout = Win + padW0 + padW1;
    int total = Nout * Cout * Hout * Wout;
    if ((int)gid >= total) return;
    int w = (int)gid % Wout; int t1 = (int)gid / Wout;
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
// GridSample 2D NHWC.
// ----------------------------------------------------------------------------

inline float grid_sample_safe_msl(device const float* src, int n, int y, int x, int c,
    int H, int W, int C, int padding)
{
    if (padding == 0) {
        if ((uint)y >= (uint)H || (uint)x >= (uint)W) return 0.0;
    } else if (padding == 1) {
        y = clamp_i(y, 0, H - 1); x = clamp_i(x, 0, W - 1);
    } else {
        y = reflect_index(y, H); x = reflect_index(x, W);
    }
    return src[((n * H + y) * W + x) * C + c];
}

kernel void geometry_grid_sample_2d(
    device const float* input [[buffer(0)]],
    device const float* grid [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]], constant int& C [[buffer(6)]],
    constant int& outH [[buffer(7)]], constant int& outW [[buffer(8)]],
    constant int& mode [[buffer(9)]], constant int& padding [[buffer(10)]],
    constant int& alignCorners [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    int total = N * outH * outW;
    if ((int)gid >= total) return;
    int ox = (int)gid % outW; int t1 = (int)gid / outW;
    int oy = t1 % outH; int n = t1 / outH;
    int gOff = ((n * outH + oy) * outW + ox) * 2;
    float gx = grid[gOff];
    float gy = grid[gOff + 1];
    float sx = alignCorners ? (gx + 1.0) * 0.5 * (W - 1) : ((gx + 1.0) * W - 1.0) * 0.5;
    float sy = alignCorners ? (gy + 1.0) * 0.5 * (H - 1) : ((gy + 1.0) * H - 1.0) * 0.5;

    if (mode == 1) {
        int nx = (int)round(sx), ny = (int)round(sy);
        for (int c = 0; c < C; c++)
            output[((n * outH + oy) * outW + ox) * C + c] =
                grid_sample_safe_msl(input, n, ny, nx, c, H, W, C, padding);
    } else if (mode == 0) {
        int x0 = (int)floor(sx), y0 = (int)floor(sy);
        int x1 = x0 + 1, y1 = y0 + 1;
        float fx = sx - x0, fy = sy - y0;
        for (int c = 0; c < C; c++) {
            float v00 = grid_sample_safe_msl(input, n, y0, x0, c, H, W, C, padding);
            float v01 = grid_sample_safe_msl(input, n, y0, x1, c, H, W, C, padding);
            float v10 = grid_sample_safe_msl(input, n, y1, x0, c, H, W, C, padding);
            float v11 = grid_sample_safe_msl(input, n, y1, x1, c, H, W, C, padding);
            output[((n * outH + oy) * outW + ox) * C + c] =
                v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
              + v10 * (1 - fx) * fy + v11 * fx * fy;
        }
    } else {
        int x0 = (int)floor(sx), y0 = (int)floor(sy);
        float fx = sx - x0, fy = sy - y0;
        float wy[4] = { cubic_kernel_f(1.0 + fy, -0.75), cubic_kernel_f(fy, -0.75),
                        cubic_kernel_f(1.0 - fy, -0.75), cubic_kernel_f(2.0 - fy, -0.75) };
        float wx[4] = { cubic_kernel_f(1.0 + fx, -0.75), cubic_kernel_f(fx, -0.75),
                        cubic_kernel_f(1.0 - fx, -0.75), cubic_kernel_f(2.0 - fx, -0.75) };
        for (int c = 0; c < C; c++) {
            float acc = 0.0;
            for (int yy = 0; yy < 4; yy++) {
                int yi = y0 - 1 + yy;
                float rowAcc = 0.0;
                for (int xx = 0; xx < 4; xx++) {
                    int xi = x0 - 1 + xx;
                    rowAcc += wx[xx] * grid_sample_safe_msl(input, n, yi, xi, c, H, W, C, padding);
                }
                acc += wy[yy] * rowAcc;
            }
            output[((n * outH + oy) * outW + ox) * C + c] = acc;
        }
    }
}

// ----------------------------------------------------------------------------
// AffineGrid 3D.
// ----------------------------------------------------------------------------

inline float grid_norm_coord_f(int idx, int size, int alignCorners) {
    if (size <= 1) return 0.0;
    return alignCorners ? -1.0 + 2.0 * idx / (size - 1) : -1.0 + (2.0 * idx + 1.0) / size;
}

kernel void geometry_affine_grid_3d(
    device const float* theta [[buffer(0)]],
    device float* grid [[buffer(1)]],
    constant int& N [[buffer(2)]], constant int& D [[buffer(3)]],
    constant int& H [[buffer(4)]], constant int& W [[buffer(5)]],
    constant int& alignCorners [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = N * D * H * W;
    if ((int)gid >= total) return;
    int w = (int)gid % W; int t1 = (int)gid / W;
    int h = t1 % H; int t2 = t1 / H;
    int d = t2 % D; int n = t2 / D;
    int tBase = n * 12;
    float x = grid_norm_coord_f(w, W, alignCorners);
    float y = grid_norm_coord_f(h, H, alignCorners);
    float z = grid_norm_coord_f(d, D, alignCorners);
    int gBase = (((n * D + d) * H + h) * W + w) * 3;
    for (int row = 0; row < 3; row++) {
        grid[gBase + row] = theta[tBase + row * 4] * x
                          + theta[tBase + row * 4 + 1] * y
                          + theta[tBase + row * 4 + 2] * z
                          + theta[tBase + row * 4 + 3];
    }
}
";
    }
}
