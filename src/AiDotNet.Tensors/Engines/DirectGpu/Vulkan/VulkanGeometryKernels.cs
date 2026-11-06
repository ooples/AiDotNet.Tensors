namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL compute shaders for the geometry / sampling ops added by Issue
/// #217. Each entry is a self-contained compute shader; all use
/// <c>#version 450</c> and <c>layout(local_size_x = 256)</c>.
/// </summary>
public static class VulkanGeometryKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    private const string Helpers = @"
float source_coord(int dstIdx, int dstSize, int srcSize, int alignCorners) {
    if (dstSize <= 1) return 0.0;
    if (alignCorners != 0) return float(dstIdx) * (srcSize - 1) / (dstSize - 1);
    return (float(dstIdx) + 0.5) * srcSize / dstSize - 0.5;
}

float cubic_kernel(float d, float a) {
    float ad = abs(d);
    if (ad < 1.0) return ((a + 2.0) * ad - (a + 3.0)) * ad * ad + 1.0;
    if (ad < 2.0) return a * ((ad - 5.0) * ad + 8.0) * ad - 4.0 * a;
    return 0.0;
}

int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// GLSL leaves % undefined when an operand is negative (drivers differ: NVIDIA's truncates), so these never take the
// remainder of a negative index.
int reflect_index(int i, int extent) {
    if (extent == 1) return 0;
    int period = 2 * (extent - 1);
    // The reflection is even about 0, so |i| lands on the same pixel.
    int r = abs(i) % period;
    return r < extent ? r : period - r;
}

int pad_boundary(int idx, int extent, int mode) {
    if (extent <= 0) return 0;
    if (mode == 2) { if (idx < 0) return 0; if (idx >= extent) return extent - 1; return idx; }
    if (mode == 1) return reflect_index(idx, extent);
    return idx >= 0 ? idx % extent : extent - 1 - ((-idx - 1) % extent);
}
";

    // -----------------------------------------------------------------------
    // Interpolate 2D NCHW.
    // -----------------------------------------------------------------------

    public static string Interpolate2D => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer B { float output_[]; };
layout(push_constant) uniform P {
    int N; int C; int Hin; int Win; int Hout; int Wout; int mode; int alignCorners;
};
" + Helpers + @"
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = N * C * Hout * Wout;
    if (gid >= total) return;
    int x = gid % Wout; int t1 = gid / Wout;
    int y = t1 % Hout; int t2 = t1 / Hout;
    int c = t2 % C; int n = t2 / C;
    int srcBase = ((n * C + c) * Hin) * Win;

    if (mode == 0) {
        // floor(x * Win / Wout) in integers: a float division here is not correctly rounded (drivers may compute
        // a * (1 / b)), so an exact quotient such as 30 / 15 came out as 1.99999994 and floored to the wrong pixel.
        int yi = (y * Hin) / Hout; if (yi >= Hin) yi = Hin - 1;
        int xi = (x * Win) / Wout; if (xi >= Win) xi = Win - 1;
        output_[gid] = input_[srcBase + yi * Win + xi];
    } else if (mode == 2) {
        float sy = source_coord(y, Hout, Hin, alignCorners);
        float sx = source_coord(x, Wout, Win, alignCorners);
        int y0 = int(floor(sy)); int x0 = int(floor(sx));
        if (y0 < 0) y0 = 0; if (x0 < 0) x0 = 0;
        int y1 = y0 + 1; int x1 = x0 + 1;
        if (y1 >= Hin) { y1 = Hin - 1; if (y0 > y1) y0 = y1; }
        if (x1 >= Win) { x1 = Win - 1; if (x0 > x1) x0 = x1; }
        float fy = sy - y0; if (fy < 0.0) fy = 0.0; if (fy > 1.0) fy = 1.0;
        float fx = sx - x0; if (fx < 0.0) fx = 0.0; if (fx > 1.0) fx = 1.0;
        float v00 = input_[srcBase + y0 * Win + x0];
        float v01 = input_[srcBase + y0 * Win + x1];
        float v10 = input_[srcBase + y1 * Win + x0];
        float v11 = input_[srcBase + y1 * Win + x1];
        output_[gid] = v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy)
                     + v10 * (1.0 - fx) * fy + v11 * fx * fy;
    } else if (mode == 3) {
        float sy = source_coord(y, Hout, Hin, alignCorners);
        float sx = source_coord(x, Wout, Win, alignCorners);
        int y0 = int(floor(sy)); float ty = sy - y0;
        int x0 = int(floor(sx)); float tx = sx - x0;
        float wy[4] = float[](cubic_kernel(1.0 + ty, -0.75), cubic_kernel(ty, -0.75),
                               cubic_kernel(1.0 - ty, -0.75), cubic_kernel(2.0 - ty, -0.75));
        float wx[4] = float[](cubic_kernel(1.0 + tx, -0.75), cubic_kernel(tx, -0.75),
                               cubic_kernel(1.0 - tx, -0.75), cubic_kernel(2.0 - tx, -0.75));
        float acc = 0.0;
        for (int yy = 0; yy < 4; yy++) {
            int yi = clamp_i(y0 - 1 + yy, 0, Hin - 1);
            float rowAcc = 0.0;
            for (int xx = 0; xx < 4; xx++) {
                int xi = clamp_i(x0 - 1 + xx, 0, Win - 1);
                rowAcc += wx[xx] * input_[srcBase + yi * Win + xi];
            }
            acc += wy[yy] * rowAcc;
        }
        output_[gid] = acc;
    } else {  // Area — overlap-weighted averaging
        // Positions in units of 1/Hout (1/Wout): output cell y covers [y * Hin, (y + 1) * Hin) and input pixel yy
        // covers [yy * Hout, (yy + 1) * Hout), so every bound and overlap is an exact integer; a float division is
        // not correctly rounded here and could open the window a pixel wide with a spurious sliver of weight.
        int yLo = y * Hin, yHi = (y + 1) * Hin;
        int xLo = x * Win, xHi = (x + 1) * Win;
        int yL = yLo / Hout, yH = min((yHi + Hout - 1) / Hout, Hin);
        int xL = xLo / Wout, xH = min((xHi + Wout - 1) / Wout, Win);
        float acc = 0.0;
        for (int yy = yL; yy < yH; yy++) {
            int oy = min(yHi, (yy + 1) * Hout) - max(yLo, yy * Hout);
            if (oy <= 0) continue;
            for (int xx = xL; xx < xH; xx++) {
                int ox = min(xHi, (xx + 1) * Wout) - max(xLo, xx * Wout);
                if (ox <= 0) continue;
                acc += float(oy * ox) * input_[srcBase + yy * Win + xx];
            }
        }
        output_[gid] = acc / float(Hin * Win);
    }
}
";

    // -----------------------------------------------------------------------
    // Pad 4D.
    // -----------------------------------------------------------------------

    public static string Pad4D => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer B { float output_[]; };
layout(push_constant) uniform P {
    int N; int C; int Hin; int Win;
    int padN0; int padN1; int padC0; int padC1;
    int padH0; int padH1; int padW0; int padW1;
    int mode; float padValue;
};
" + Helpers + @"
void main() {
    int Nout = N + padN0 + padN1;
    int Cout = C + padC0 + padC1;
    int Hout = Hin + padH0 + padH1;
    int Wout = Win + padW0 + padW1;
    int total = Nout * Cout * Hout * Wout;
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= total) return;
    int w = gid % Wout; int t1 = gid / Wout;
    int h = t1 % Hout; int t2 = t1 / Hout;
    int c = t2 % Cout; int n = t2 / Cout;
    int nn = n - padN0, cc = c - padC0, hh = h - padH0, ww = w - padW0;
    bool inB = nn >= 0 && nn < N && cc >= 0 && cc < C && hh >= 0 && hh < Hin && ww >= 0 && ww < Win;
    if (!inB && mode == 0) { output_[gid] = padValue; return; }
    if (!inB) {
        if (!(nn >= 0 && nn < N)) nn = pad_boundary(nn, N, mode);
        if (!(cc >= 0 && cc < C)) cc = pad_boundary(cc, C, mode);
        if (!(hh >= 0 && hh < Hin)) hh = pad_boundary(hh, Hin, mode);
        if (!(ww >= 0 && ww < Win)) ww = pad_boundary(ww, Win, mode);
    }
    output_[gid] = input_[((nn * C + cc) * Hin + hh) * Win + ww];
}
";

    // -----------------------------------------------------------------------
    // GridSample 2D, NCHW input/output — 3 SSBO (input, grid, output).
    // -----------------------------------------------------------------------

    public static string GridSample2D => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float input_[]; };
layout(set = 0, binding = 1) readonly buffer G { float grid_[]; };
layout(set = 0, binding = 2) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P {
    int N; int H; int W; int C;
    int outH; int outW;
    int mode; int padding; int alignCorners;
};
" + Helpers + GridSampleTaps + @"
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = N * outH * outW;
    if (gid >= total) return;
    int ox = gid % outW; int t1 = gid / outW;
    int oy = t1 % outH; int n = t1 / outH;
    int gOff = ((n * outH + oy) * outW + ox) * 2;
    float gx = grid_[gOff];
    float gy = grid_[gOff + 1];
    float sx = alignCorners != 0 ? (gx + 1.0) * 0.5 * (W - 1) : ((gx + 1.0) * W - 1.0) * 0.5;
    float sy = alignCorners != 0 ? (gy + 1.0) * 0.5 * (H - 1) : ((gy + 1.0) * H - 1.0) * 0.5;
    // #1134: a NaN, infinite or out-of-int-range point samples nothing (the CPU rule). Tested on the bits: |x| below
    // 2147483520 (0x4EFFFFFF) has a smaller bit pattern, NaN and infinity a larger one.
    if ((floatBitsToUint(sx) & 0x7FFFFFFFu) >= 0x4EFFFFFFu || (floatBitsToUint(sy) & 0x7FFFFFFFu) >= 0x4EFFFFFFu) {
        for (int c = 0; c < C; c++) output_[((n * C + c) * outH + oy) * outW + ox] = 0.0;
        return;
    }

    // The backward's taps, so the forward and its derivative read the same pixels with the same weights.
    int tx[4]; int ty[4]; float wx[4]; float wy[4]; float dwx[4]; float dwy[4];
    int taps = grid_sample_taps(gOff, tx, ty, wx, wy, dwx, dwy);
    for (int c = 0; c < C; c++) {
        int plane = (n * C + c) * H;
        float acc = 0.0;
        for (int j = 0; j < taps; j++) {
            if (ty[j] < 0) continue;
            int row = (plane + ty[j]) * W;
            float rowAcc = 0.0;
            for (int i = 0; i < taps; i++)
                if (tx[i] >= 0) rowAcc += wx[i] * input_[row + tx[i]];
            acc += wy[j] * rowAcc;
        }
        output_[((n * C + c) * outH + oy) * outW + ox] = acc;
    }
}
";

    // -----------------------------------------------------------------------
    // GridSample 2D forward and backward, every mode and padding (#1680), with PyTorch's grid_sampler_2d
    // padding rules. One tap helper gives each point's pixels, weights and weight derivatives; the forward sums
    // over them and the backward differentiates the same sum. One thread per sampling point, looping C.
    // -----------------------------------------------------------------------

    private const string GridSampleBackwardPush = @"
layout(push_constant) uniform P {
    int N; int H; int W; int C;
    int outH; int outW;
    int mode; int padding; int alignCorners;
};
";

    private const string GridSampleTaps = @"
// PyTorch's grid_sampler_compute_source_index for Border (1) and Reflection (2), with slope = d(result)/d(s):
// Reflection reflects about -0.5 and S-0.5 (0 and S-1 with alignCorners), then both clip to [0, S-1]. A clipped
// coordinate has slope 0. Bilinear and Nearest pad the coordinate this way; Bicubic pads each tap instead.
float grid_sample_pad_coord(float s, int size, out float slope) {
    slope = 1.0;
    if (padding == 0) return s;
    if (padding == 2) {
        if (size == 1 && alignCorners != 0) { slope = 0.0; return 0.0; }
        float twiceLow = alignCorners != 0 ? 0.0 : -1.0;
        float twiceHigh = alignCorners != 0 ? 2.0 * float(size - 1) : 2.0 * float(size) - 1.0;
        float mn = twiceLow * 0.5, span = (twiceHigh - twiceLow) * 0.5;
        float x = s - mn;
        if (x < 0.0) { slope = -1.0; x = -x; }
        float flips = floor(x / span);
        float extra = x - flips * span;
        if (mod(flips, 2.0) == 0.0) s = extra + mn;
        else { s = span - extra + mn; slope = -slope; }
    }
    if (s <= 0.0) { slope = 0.0; return 0.0; }
    if (s >= float(size - 1)) { slope = 0.0; return float(size - 1); }
    return s;
}

int grid_sample_in_bounds(int i, int extent) {
    return uint(i) < uint(extent) ? i : -1;
}

// Bicubic's padded tap (PyTorch's compute_coordinates on an integer tap): without alignCorners Reflection
// mirrors about -0.5 and S-0.5, repeating the edge pixel (period 2S); with it, about 0 and S-1.
int grid_sample_bicubic_tap(int i, int extent) {
    if (padding == 0) return grid_sample_in_bounds(i, extent);
    if (padding == 1) return clamp_i(i, 0, extent - 1);
    if (alignCorners != 0) return reflect_index(i, extent);
    int period = 2 * extent;
    // Pixels i and -1 - i mirror each other about -0.5, so folding a negative tap first keeps % on non-negative operands.
    int r = (i < 0 ? -1 - i : i) % period;
    return r < extent ? r : period - 1 - r;
}

// dK/dd of cubic_kernel for d >= 0, which every offset the taps use is.
float cubic_kernel_deriv(float d, float a) {
    if (d < 1.0) return (3.0 * (a + 2.0) * d - 2.0 * (a + 3.0)) * d;
    if (d < 2.0) return (3.0 * a * d - 10.0 * a) * d + 8.0 * a;
    return 0.0;
}

// The taps of one sampling point: per-axis pixel index, weight and d(weight)/d(source coord). Returns the taps
// per axis (1 nearest, 2 bilinear, 4 bicubic), or 0 when the forward samples nothing at this point (#1134).
int grid_sample_taps(int gOff, out int tx[4], out int ty[4], out float wx[4], out float wy[4],
    out float dwx[4], out float dwy[4]) {
    for (int k = 0; k < 4; k++) { tx[k] = -1; ty[k] = -1; wx[k] = 0.0; wy[k] = 0.0; dwx[k] = 0.0; dwy[k] = 0.0; }
    float gx = grid_[gOff];
    float gy = grid_[gOff + 1];
    float sx = alignCorners != 0 ? (gx + 1.0) * 0.5 * (W - 1) : ((gx + 1.0) * W - 1.0) * 0.5;
    float sy = alignCorners != 0 ? (gy + 1.0) * 0.5 * (H - 1) : ((gy + 1.0) * H - 1.0) * 0.5;
    if ((floatBitsToUint(sx) & 0x7FFFFFFFu) >= 0x4EFFFFFFu || (floatBitsToUint(sy) & 0x7FFFFFFFu) >= 0x4EFFFFFFu) return 0;
    if (mode != 2) {
        float slx, sly;
        float px = grid_sample_pad_coord(sx, W, slx);
        float py = grid_sample_pad_coord(sy, H, sly);
        if (mode == 1) {
            // roundEven rounds half to even, as PyTorch's nearbyint.
            tx[0] = grid_sample_in_bounds(int(roundEven(px)), W);
            ty[0] = grid_sample_in_bounds(int(roundEven(py)), H);
            wx[0] = 1.0; wy[0] = 1.0;
            return 1;
        }
        int px0 = int(floor(px)), py0 = int(floor(py));
        float fx = px - float(px0), fy = py - float(py0);
        wx[0] = 1.0 - fx; wx[1] = fx; dwx[0] = -slx; dwx[1] = slx;
        wy[0] = 1.0 - fy; wy[1] = fy; dwy[0] = -sly; dwy[1] = sly;
        for (int k = 0; k < 2; k++) {
            tx[k] = grid_sample_in_bounds(px0 + k, W);
            ty[k] = grid_sample_in_bounds(py0 + k, H);
        }
        return 2;
    }
    int x0 = int(floor(sx)), y0 = int(floor(sy));
    float fx = sx - x0, fy = sy - y0;
    const float a = -0.75;
    wx[0] = cubic_kernel(1.0 + fx, a); wx[1] = cubic_kernel(fx, a);
    wx[2] = cubic_kernel(1.0 - fx, a); wx[3] = cubic_kernel(2.0 - fx, a);
    wy[0] = cubic_kernel(1.0 + fy, a); wy[1] = cubic_kernel(fy, a);
    wy[2] = cubic_kernel(1.0 - fy, a); wy[3] = cubic_kernel(2.0 - fy, a);
    // Offsets 1+t and t grow with t, 1-t and 2-t shrink: hence the sign flip on the last two.
    dwx[0] = cubic_kernel_deriv(1.0 + fx, a); dwx[1] = cubic_kernel_deriv(fx, a);
    dwx[2] = -cubic_kernel_deriv(1.0 - fx, a); dwx[3] = -cubic_kernel_deriv(2.0 - fx, a);
    dwy[0] = cubic_kernel_deriv(1.0 + fy, a); dwy[1] = cubic_kernel_deriv(fy, a);
    dwy[2] = -cubic_kernel_deriv(1.0 - fy, a); dwy[3] = -cubic_kernel_deriv(2.0 - fy, a);
    for (int k = 0; k < 4; k++) {
        tx[k] = grid_sample_bicubic_tap(x0 - 1 + k, W);
        ty[k] = grid_sample_bicubic_tap(y0 - 1 + k, H);
    }
    return 4;
}
";

    /// <summary>
    /// GridSample 2D input gradient (gradOut, grid, gradIn). gradIn must be zeroed first: several sampling
    /// points can read one pixel, so this accumulates with a compare-and-swap float add.
    /// </summary>
    public static string GridSample2DBackwardInput => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut_[]; };
layout(set = 0, binding = 1) readonly buffer G { float grid_[]; };
layout(set = 0, binding = 2) buffer GI { uint gradIn_[]; };
" + GridSampleBackwardPush + Helpers + GridSampleTaps + @"
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= N * outH * outW) return;
    int ox = gid % outW; int t1 = gid / outW;
    int oy = t1 % outH; int n = t1 / outH;
    int tx[4]; int ty[4]; float wx[4]; float wy[4]; float dwx[4]; float dwy[4];
    int taps = grid_sample_taps(gid * 2, tx, ty, wx, wy, dwx, dwy);
    for (int c = 0; c < C; c++) {
        float g = gradOut_[((n * C + c) * outH + oy) * outW + ox];
        if (g == 0.0) continue;
        int plane = (n * C + c) * H;
        for (int j = 0; j < taps; j++) {
            if (ty[j] < 0) continue;
            int row = (plane + ty[j]) * W;
            for (int i = 0; i < taps; i++) {
                if (tx[i] < 0) continue;
                int at = row + tx[i];
                float add = g * wx[i] * wy[j];
                uint expected = gradIn_[at];
                while (true) {
                    uint actual = atomicCompSwap(gradIn_[at], expected, floatBitsToUint(uintBitsToFloat(expected) + add));
                    if (actual == expected) break;
                    expected = actual;
                }
            }
        }
    }
}
";

    /// <summary>
    /// GridSample 2D grid gradient (gradOut, input, grid, gradGrid). Writes every element of gradGrid.
    /// </summary>
    public static string GridSample2DBackwardGrid => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut_[]; };
layout(set = 0, binding = 1) readonly buffer A { float input_[]; };
layout(set = 0, binding = 2) readonly buffer G { float grid_[]; };
layout(set = 0, binding = 3) writeonly buffer GG { float gradGrid_[]; };
" + GridSampleBackwardPush + Helpers + GridSampleTaps + @"
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= N * outH * outW) return;
    int ox = gid % outW; int t1 = gid / outW;
    int oy = t1 % outH; int n = t1 / outH;
    int tx[4]; int ty[4]; float wx[4]; float wy[4]; float dwx[4]; float dwy[4];
    // Nearest is piecewise constant: its grid gradient is exactly 0. Written outright, since g * input can
    // overflow to infinity and infinity * 0 is NaN.
    if (mode == 1) { gradGrid_[gid * 2] = 0.0; gradGrid_[gid * 2 + 1] = 0.0; return; }
    int taps = grid_sample_taps(gid * 2, tx, ty, wx, wy, dwx, dwy);
    float gradX = 0.0, gradY = 0.0;
    for (int c = 0; c < C; c++) {
        float g = gradOut_[((n * C + c) * outH + oy) * outW + ox];
        int plane = (n * C + c) * H;
        for (int j = 0; j < taps; j++) {
            if (ty[j] < 0) continue;
            int row = (plane + ty[j]) * W;
            for (int i = 0; i < taps; i++) {
                if (tx[i] < 0) continue;
                float v = g * input_[row + tx[i]];
                gradX += v * dwx[i] * wy[j];
                gradY += v * wx[i] * dwy[j];
            }
        }
    }
    // d(source coord)/d(grid coord): (S-1)/2 with alignCorners, S/2 without.
    gradGrid_[gid * 2] = gradX * (alignCorners != 0 ? float(W - 1) * 0.5 : float(W) * 0.5);
    gradGrid_[gid * 2 + 1] = gradY * (alignCorners != 0 ? float(H - 1) * 0.5 : float(H) * 0.5);
}
";

    // -----------------------------------------------------------------------
    // AffineGrid 3D.
    // -----------------------------------------------------------------------

    public static string AffineGrid3D => Header + @"
layout(set = 0, binding = 0) readonly buffer T { float theta[]; };
layout(set = 0, binding = 1) writeonly buffer G { float grid_[]; };
layout(push_constant) uniform P {
    int N; int D; int H; int W; int alignCorners;
};

float grid_norm_coord(int idx, int size) {
    if (size <= 1) return 0.0;
    return alignCorners != 0 ? -1.0 + 2.0 * idx / (size - 1) : -1.0 + (2.0 * idx + 1.0) / size;
}

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = N * D * H * W;
    if (gid >= total) return;
    int w = gid % W; int t1 = gid / W;
    int h = t1 % H; int t2 = t1 / H;
    int d = t2 % D; int n = t2 / D;
    int tBase = n * 12;
    float x = grid_norm_coord(w, W);
    float y = grid_norm_coord(h, H);
    float z = grid_norm_coord(d, D);
    int gBase = (((n * D + d) * H + h) * W + w) * 3;
    for (int row = 0; row < 3; row++) {
        grid_[gBase + row] = theta[tBase + row * 4] * x
                           + theta[tBase + row * 4 + 1] * y
                           + theta[tBase + row * 4 + 2] * z
                           + theta[tBase + row * 4 + 3];
    }
}
";
}
