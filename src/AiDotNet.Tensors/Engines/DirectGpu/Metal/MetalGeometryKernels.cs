// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
            "geometry_grid_sample_2d_backward_input",
            "geometry_grid_sample_2d_backward_grid",
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
// GridSample 2D, NCHW input/output.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// GridSample 2D forward and backward, every mode and padding (#1680), with PyTorch's grid_sampler_2d
// padding rules. One tap helper gives each point's pixels, weights and weight derivatives; the forward sums
// over them and the backward differentiates the same sum. One thread per sampling point, looping C.
// ----------------------------------------------------------------------------

// PyTorch's grid_sampler_compute_source_index for Border (1) and Reflection (2), with *slope = d(result)/d(s):
// Reflection reflects about -0.5 and S-0.5 (0 and S-1 with alignCorners), then both clip to [0, S-1]. A clipped
// coordinate has slope 0. Bilinear and Nearest pad the coordinate this way; Bicubic pads each tap instead.
inline float grid_sample_pad_coord(float s, int size, int padding, int alignCorners, thread float* slope)
{
    *slope = 1.0;
    if (padding == 0) return s;
    if (padding == 2) {
        if (size == 1 && alignCorners) { *slope = 0.0; return 0.0; }
        float twiceLow = alignCorners ? 0.0 : -1.0;
        float twiceHigh = alignCorners ? 2.0 * (size - 1) : 2.0 * size - 1.0;
        float mn = twiceLow * 0.5, span = (twiceHigh - twiceLow) * 0.5;
        float x = s - mn;
        if (x < 0.0) { *slope = -1.0; x = -x; }
        float extra = fmod(x, span);
        float flips = floor(x / span);
        if (fmod(flips, (float)2.0) == 0.0) s = extra + mn;
        else { s = span - extra + mn; *slope = -*slope; }
    }
    if (s <= 0.0) { *slope = 0.0; return 0.0; }
    if (s >= size - 1) { *slope = 0.0; return (float)(size - 1); }
    return s;
}

inline int grid_sample_in_bounds(int i, int extent)
{
    return (uint)i < (uint)extent ? i : -1;
}

// Bicubic's padded tap (PyTorch's compute_coordinates on an integer tap): without alignCorners Reflection
// mirrors about -0.5 and S-0.5, repeating the edge pixel (period 2S); with it, about 0 and S-1.
inline int grid_sample_bicubic_tap(int i, int extent, int padding, int alignCorners)
{
    if (padding == 0) return grid_sample_in_bounds(i, extent);
    if (padding == 1) return clamp_i(i, 0, extent - 1);
    if (alignCorners) return reflect_index(i, extent);
    int period = 2 * extent;
    int r = ((i % period) + period) % period;
    return r < extent ? r : period - 1 - r;
}

// dK/dd of cubic_kernel_f for d >= 0, which every offset the taps use is.
inline float cubic_kernel_deriv_f(float d, float a)
{
    if (d < 1.0) return (3.0 * (a + 2.0) * d - 2.0 * (a + 3.0)) * d;
    if (d < 2.0) return (3.0 * a * d - 10.0 * a) * d + 8.0 * a;
    return 0.0;
}

// The taps of one sampling point: per-axis pixel index, weight and d(weight)/d(source coord). Returns the taps
// per axis (1 nearest, 2 bilinear, 4 bicubic), or 0 when the forward samples nothing at this point (#1134).
inline int grid_sample_taps(device const float* grid, int gOff, int H, int W,
    int mode, int padding, int alignCorners,
    thread int* tx, thread int* ty, thread float* wx, thread float* wy, thread float* dwx, thread float* dwy)
{
    float gx = grid[gOff];
    float gy = grid[gOff + 1];
    float sx = alignCorners ? (gx + 1.0) * 0.5 * (W - 1) : ((gx + 1.0) * W - 1.0) * 0.5;
    float sy = alignCorners ? (gy + 1.0) * 0.5 * (H - 1) : ((gy + 1.0) * H - 1.0) * 0.5;
    const uint limBits = 0x4EFFFFFFu; // 2147483520.0f, the largest float below 2^31
    if ((as_type<uint>(sx) & 0x7FFFFFFFu) >= limBits || (as_type<uint>(sy) & 0x7FFFFFFFu) >= limBits) return 0;
    if (mode != 2) {
        float slx, sly;
        float px = grid_sample_pad_coord(sx, W, padding, alignCorners, &slx);
        float py = grid_sample_pad_coord(sy, H, padding, alignCorners, &sly);
        if (mode == 1) {
            // rint rounds half to even, as PyTorch's nearbyint.
            tx[0] = grid_sample_in_bounds((int)rint(px), W);
            ty[0] = grid_sample_in_bounds((int)rint(py), H);
            wx[0] = 1.0; wy[0] = 1.0; dwx[0] = 0.0; dwy[0] = 0.0;
            return 1;
        }
        int px0 = (int)floor(px), py0 = (int)floor(py);
        float fx = px - px0, fy = py - py0;
        wx[0] = 1.0f - fx; wx[1] = fx; dwx[0] = -slx; dwx[1] = slx;
        wy[0] = 1.0f - fy; wy[1] = fy; dwy[0] = -sly; dwy[1] = sly;
        for (int k = 0; k < 2; k++) {
            tx[k] = grid_sample_in_bounds(px0 + k, W);
            ty[k] = grid_sample_in_bounds(py0 + k, H);
        }
        return 2;
    }
    int x0 = (int)floor(sx), y0 = (int)floor(sy);
    float fx = sx - x0, fy = sy - y0;
    const float a = -0.75;
    wx[0] = cubic_kernel_f(1.0 + fx, a); wx[1] = cubic_kernel_f(fx, a);
    wx[2] = cubic_kernel_f(1.0 - fx, a); wx[3] = cubic_kernel_f(2.0 - fx, a);
    wy[0] = cubic_kernel_f(1.0 + fy, a); wy[1] = cubic_kernel_f(fy, a);
    wy[2] = cubic_kernel_f(1.0 - fy, a); wy[3] = cubic_kernel_f(2.0 - fy, a);
    // Offsets 1+t and t grow with t, 1-t and 2-t shrink: hence the sign flip on the last two.
    dwx[0] = cubic_kernel_deriv_f(1.0 + fx, a); dwx[1] = cubic_kernel_deriv_f(fx, a);
    dwx[2] = -cubic_kernel_deriv_f(1.0 - fx, a); dwx[3] = -cubic_kernel_deriv_f(2.0 - fx, a);
    dwy[0] = cubic_kernel_deriv_f(1.0 + fy, a); dwy[1] = cubic_kernel_deriv_f(fy, a);
    dwy[2] = -cubic_kernel_deriv_f(1.0 - fy, a); dwy[3] = -cubic_kernel_deriv_f(2.0 - fy, a);
    for (int k = 0; k < 4; k++) {
        tx[k] = grid_sample_bicubic_tap(x0 - 1 + k, W, padding, alignCorners);
        ty[k] = grid_sample_bicubic_tap(y0 - 1 + k, H, padding, alignCorners);
    }
    return 4;
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
    // #1134: a NaN, infinite or out-of-int-range point samples nothing (the CPU rule). Tested on the bits, so fast
    // math cannot fold the NaN case away: |x| below 2147483520 has a smaller bit pattern, NaN and infinity a larger one.
    const uint limBits = 0x4EFFFFFFu; // 2147483520.0f, the largest float below 2^31
    if ((as_type<uint>(sx) & 0x7FFFFFFFu) >= limBits || (as_type<uint>(sy) & 0x7FFFFFFFu) >= limBits) {
        for (int c = 0; c < C; c++) output[((n * C + c) * outH + oy) * outW + ox] = 0.0f;
        return;
    }

    // The backward's taps, so the forward and its derivative read the same pixels with the same weights.
    int tx[4], ty[4]; float wx[4], wy[4], dwx[4], dwy[4];
    int taps = grid_sample_taps(grid, gOff, H, W, mode, padding, alignCorners, tx, ty, wx, wy, dwx, dwy);
    for (int c = 0; c < C; c++) {
        int plane = (n * C + c) * H;
        float acc = 0.0;
        for (int j = 0; j < taps; j++) {
            if (ty[j] < 0) continue;
            int row = (plane + ty[j]) * W;
            float rowAcc = 0.0;
            for (int i = 0; i < taps; i++)
                if (tx[i] >= 0) rowAcc += wx[i] * input[row + tx[i]];
            acc += wy[j] * rowAcc;
        }
        output[((n * C + c) * outH + oy) * outW + ox] = (float)acc;
    }
}

// gradIn must be zeroed by the caller: several sampling points can read one pixel, so this accumulates.
kernel void geometry_grid_sample_2d_backward_input(
    device const float* gradOut [[buffer(0)]],
    device const float* grid [[buffer(1)]],
    device atomic_float* gradIn [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]], constant int& C [[buffer(6)]],
    constant int& outH [[buffer(7)]], constant int& outW [[buffer(8)]],
    constant int& mode [[buffer(9)]], constant int& padding [[buffer(10)]],
    constant int& alignCorners [[buffer(11)]],
    uint gidU [[thread_position_in_grid]])
{
    int gid = (int)gidU;
    if (gid >= N * outH * outW) return;
    int ox = gid % outW; int t1 = gid / outW;
    int oy = t1 % outH; int n = t1 / outH;
    int tx[4], ty[4]; float wx[4], wy[4], dwx[4], dwy[4];
    int taps = grid_sample_taps(grid, gid * 2, H, W, mode, padding, alignCorners, tx, ty, wx, wy, dwx, dwy);
    for (int c = 0; c < C; c++) {
        float g = gradOut[((n * C + c) * outH + oy) * outW + ox];
        if (g == 0.0) continue;
        int plane = (n * C + c) * H;
        for (int j = 0; j < taps; j++) {
            if (ty[j] < 0) continue;
            int row = (plane + ty[j]) * W;
            for (int i = 0; i < taps; i++)
                if (tx[i] >= 0)
                    atomic_fetch_add_explicit(&gradIn[row + tx[i]], g * wx[i] * wy[j], memory_order_relaxed);
        }
    }
}

// Writes every element of gradGrid: no accumulation, so no zeroing needed.
kernel void geometry_grid_sample_2d_backward_grid(
    device const float* gradOut [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device float* gradGrid [[buffer(3)]],
    constant int& N [[buffer(4)]], constant int& H [[buffer(5)]],
    constant int& W [[buffer(6)]], constant int& C [[buffer(7)]],
    constant int& outH [[buffer(8)]], constant int& outW [[buffer(9)]],
    constant int& mode [[buffer(10)]], constant int& padding [[buffer(11)]],
    constant int& alignCorners [[buffer(12)]],
    uint gidU [[thread_position_in_grid]])
{
    int gid = (int)gidU;
    if (gid >= N * outH * outW) return;
    int ox = gid % outW; int t1 = gid / outW;
    int oy = t1 % outH; int n = t1 / outH;
    int tx[4], ty[4]; float wx[4], wy[4], dwx[4], dwy[4];
    // Nearest is piecewise constant: its grid gradient is exactly 0. Written outright, since g * input can
    // overflow to infinity and infinity * 0 is NaN.
    if (mode == 1) { gradGrid[gid * 2] = 0.0f; gradGrid[gid * 2 + 1] = 0.0f; return; }
    int taps = grid_sample_taps(grid, gid * 2, H, W, mode, padding, alignCorners, tx, ty, wx, wy, dwx, dwy);
    float gradX = 0.0, gradY = 0.0;
    for (int c = 0; c < C; c++) {
        float g = gradOut[((n * C + c) * outH + oy) * outW + ox];
        int plane = (n * C + c) * H;
        for (int j = 0; j < taps; j++) {
            if (ty[j] < 0) continue;
            int row = (plane + ty[j]) * W;
            for (int i = 0; i < taps; i++) {
                if (tx[i] < 0) continue;
                float v = g * input[row + tx[i]];
                gradX += v * dwx[i] * wy[j];
                gradY += v * wx[i] * dwy[j];
            }
        }
    }
    // d(source coord)/d(grid coord): (S-1)/2 with alignCorners, S/2 without.
    gradGrid[gid * 2] = gradX * (alignCorners ? (W - 1) * 0.5 : W * 0.5);
    gradGrid[gid * 2 + 1] = gradY * (alignCorners ? (H - 1) * 0.5 : H * 0.5);
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
