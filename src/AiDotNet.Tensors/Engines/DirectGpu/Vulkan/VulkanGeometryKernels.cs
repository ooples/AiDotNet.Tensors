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

int reflect_index(int i, int extent) {
    if (extent == 1) return 0;
    int period = 2 * (extent - 1);
    int r = ((i % period) + period) % period;
    return r < extent ? r : period - r;
}

int pad_boundary(int idx, int extent, int mode) {
    if (extent <= 0) return 0;
    if (mode == 2) { if (idx < 0) return 0; if (idx >= extent) return extent - 1; return idx; }
    if (mode == 1) return reflect_index(idx, extent);
    int r = ((idx % extent) + extent) % extent;
    return r;
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
        float sy = Hout > 1 ? float(y) * Hin / Hout : 0.0;
        float sx = Wout > 1 ? float(x) * Win / Wout : 0.0;
        int yi = int(floor(sy)); if (yi >= Hin) yi = Hin - 1;
        int xi = int(floor(sx)); if (xi >= Win) xi = Win - 1;
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
        float yLo = float(y) * Hin / Hout;
        float yHi = float(y + 1) * Hin / Hout;
        float xLo = float(x) * Win / Wout;
        float xHi = float(x + 1) * Win / Wout;
        int yL = int(floor(yLo)); int yH = int(ceil(yHi));
        int xL = int(floor(xLo)); int xH = int(ceil(xHi));
        if (yH <= yL) yH = yL + 1;
        if (xH <= xL) xH = xL + 1;
        if (yH > Hin) yH = Hin;
        if (xH > Win) xH = Win;
        float totalArea = (yHi - yLo) * (xHi - xLo);
        float acc = 0.0;
        for (int yy = yL; yy < yH; yy++) {
            float oy = max(0.0, min(yHi, float(yy + 1)) - max(yLo, float(yy)));
            if (oy <= 0.0) continue;
            for (int xx = xL; xx < xH; xx++) {
                float ox = max(0.0, min(xHi, float(xx + 1)) - max(xLo, float(xx)));
                if (ox <= 0.0) continue;
                acc += oy * ox * input_[srcBase + yy * Win + xx];
            }
        }
        output_[gid] = totalArea > 0.0 ? acc / totalArea : 0.0;
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
    // GridSample 2D NHWC — 3 SSBO (input, grid, output).
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
" + Helpers + @"
float sample_safe(int n, int y, int x, int c) {
    if (padding == 0) {
        if (uint(y) >= uint(H) || uint(x) >= uint(W)) return 0.0;
    } else if (padding == 1) {
        y = clamp_i(y, 0, H - 1); x = clamp_i(x, 0, W - 1);
    } else {
        y = reflect_index(y, H); x = reflect_index(x, W);
    }
    return input_[((n * H + y) * W + x) * C + c];
}

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

    if (mode == 1) {
        int nx = int(round(sx)), ny = int(round(sy));
        for (int c = 0; c < C; c++)
            output_[((n * outH + oy) * outW + ox) * C + c] = sample_safe(n, ny, nx, c);
    } else if (mode == 0) {
        int x0 = int(floor(sx)), y0 = int(floor(sy));
        int x1 = x0 + 1, y1 = y0 + 1;
        float fx = sx - x0, fy = sy - y0;
        for (int c = 0; c < C; c++) {
            float v00 = sample_safe(n, y0, x0, c);
            float v01 = sample_safe(n, y0, x1, c);
            float v10 = sample_safe(n, y1, x0, c);
            float v11 = sample_safe(n, y1, x1, c);
            output_[((n * outH + oy) * outW + ox) * C + c] =
                v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy)
              + v10 * (1.0 - fx) * fy + v11 * fx * fy;
        }
    } else {
        int x0 = int(floor(sx)), y0 = int(floor(sy));
        float fx = sx - x0, fy = sy - y0;
        float wy[4] = float[](cubic_kernel(1.0 + fy, -0.75), cubic_kernel(fy, -0.75),
                               cubic_kernel(1.0 - fy, -0.75), cubic_kernel(2.0 - fy, -0.75));
        float wx[4] = float[](cubic_kernel(1.0 + fx, -0.75), cubic_kernel(fx, -0.75),
                               cubic_kernel(1.0 - fx, -0.75), cubic_kernel(2.0 - fx, -0.75));
        for (int c = 0; c < C; c++) {
            float acc = 0.0;
            for (int yy = 0; yy < 4; yy++) {
                int yi = y0 - 1 + yy;
                float rowAcc = 0.0;
                for (int xx = 0; xx < 4; xx++) {
                    int xi = x0 - 1 + xx;
                    rowAcc += wx[xx] * sample_safe(n, yi, xi, c);
                }
                acc += wy[yy] * rowAcc;
            }
            output_[((n * outH + oy) * outW + ox) * C + c] = acc;
        }
    }
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
