// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for the geometry / sampling ops added by
// Issue #217. Mirrors the CUDA / OpenCL / Metal / Vulkan kernels.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public static class WebGpuGeometryKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "geometry_interpolate_2d",
        "geometry_pad_4d",
        "geometry_grid_sample_2d",
        "geometry_affine_grid_3d",
    };

    private const string Helpers = @"
fn source_coord(dstIdx: i32, dstSize: i32, srcSize: i32, alignCorners: i32) -> f32 {
    if (dstSize <= 1) { return 0.0; }
    if (alignCorners != 0) { return f32(dstIdx) * f32(srcSize - 1) / f32(dstSize - 1); }
    return (f32(dstIdx) + 0.5) * f32(srcSize) / f32(dstSize) - 0.5;
}

fn cubic_kernel_f(d: f32, a: f32) -> f32 {
    let ad = abs(d);
    if (ad < 1.0) { return ((a + 2.0) * ad - (a + 3.0)) * ad * ad + 1.0; }
    if (ad < 2.0) { return a * ((ad - 5.0) * ad + 8.0) * ad - 4.0 * a; }
    return 0.0;
}

fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

fn reflect_index(i: i32, extent: i32) -> i32 {
    if (extent == 1) { return 0; }
    let period = 2 * (extent - 1);
    let r = ((i % period) + period) % period;
    if (r < extent) { return r; }
    return period - r;
}

fn pad_boundary(idx: i32, extent: i32, mode: i32) -> i32 {
    if (extent <= 0) { return 0; }
    if (mode == 2) {
        if (idx < 0) { return 0; }
        if (idx >= extent) { return extent - 1; }
        return idx;
    }
    if (mode == 1) { return reflect_index(idx, extent); }
    let r = ((idx % extent) + extent) % extent;
    return r;
}
";

    // -----------------------------------------------------------------------
    // Interpolate 2D.
    // -----------------------------------------------------------------------

    public static string Interpolate2D => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct P {
    N: i32, C: i32, Hin: i32, Win: i32,
    Hout: i32, Wout: i32, mode: i32, alignCorners: i32
};
@group(0) @binding(2) var<uniform> p : P;
" + Helpers + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.N * p.C * p.Hout * p.Wout;
    if (gid >= total) { return; }
    let x = gid % p.Wout; let t1 = gid / p.Wout;
    let y = t1 % p.Hout; let t2 = t1 / p.Hout;
    let c = t2 % p.C; let n = t2 / p.C;
    let srcBase = ((n * p.C + c) * p.Hin) * p.Win;

    if (p.mode == 0) {
        var sy : f32 = 0.0; var sx : f32 = 0.0;
        if (p.Hout > 1) { sy = f32(y) * f32(p.Hin) / f32(p.Hout); }
        if (p.Wout > 1) { sx = f32(x) * f32(p.Win) / f32(p.Wout); }
        var yi = i32(floor(sy)); if (yi >= p.Hin) { yi = p.Hin - 1; }
        var xi = i32(floor(sx)); if (xi >= p.Win) { xi = p.Win - 1; }
        output_[gid] = input_[srcBase + yi * p.Win + xi];
    } else if (p.mode == 2) {
        let sy = source_coord(y, p.Hout, p.Hin, p.alignCorners);
        let sx = source_coord(x, p.Wout, p.Win, p.alignCorners);
        var y0 = i32(floor(sy)); var x0 = i32(floor(sx));
        if (y0 < 0) { y0 = 0; } if (x0 < 0) { x0 = 0; }
        var y1 = y0 + 1; var x1 = x0 + 1;
        if (y1 >= p.Hin) { y1 = p.Hin - 1; if (y0 > y1) { y0 = y1; } }
        if (x1 >= p.Win) { x1 = p.Win - 1; if (x0 > x1) { x0 = x1; } }
        var fy = sy - f32(y0); if (fy < 0.0) { fy = 0.0; } if (fy > 1.0) { fy = 1.0; }
        var fx = sx - f32(x0); if (fx < 0.0) { fx = 0.0; } if (fx > 1.0) { fx = 1.0; }
        let v00 = input_[srcBase + y0 * p.Win + x0];
        let v01 = input_[srcBase + y0 * p.Win + x1];
        let v10 = input_[srcBase + y1 * p.Win + x0];
        let v11 = input_[srcBase + y1 * p.Win + x1];
        output_[gid] = v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy)
                     + v10 * (1.0 - fx) * fy + v11 * fx * fy;
    } else if (p.mode == 3) {
        let sy = source_coord(y, p.Hout, p.Hin, p.alignCorners);
        let sx = source_coord(x, p.Wout, p.Win, p.alignCorners);
        let y0 = i32(floor(sy)); let ty = sy - f32(y0);
        let x0 = i32(floor(sx)); let tx = sx - f32(x0);
        var wy : array<f32, 4>;
        wy[0] = cubic_kernel_f(1.0 + ty, -0.75); wy[1] = cubic_kernel_f(ty, -0.75);
        wy[2] = cubic_kernel_f(1.0 - ty, -0.75); wy[3] = cubic_kernel_f(2.0 - ty, -0.75);
        var wx : array<f32, 4>;
        wx[0] = cubic_kernel_f(1.0 + tx, -0.75); wx[1] = cubic_kernel_f(tx, -0.75);
        wx[2] = cubic_kernel_f(1.0 - tx, -0.75); wx[3] = cubic_kernel_f(2.0 - tx, -0.75);
        var acc : f32 = 0.0;
        for (var yy : i32 = 0; yy < 4; yy = yy + 1) {
            let yi = clamp_i(y0 - 1 + yy, 0, p.Hin - 1);
            var rowAcc : f32 = 0.0;
            for (var xx : i32 = 0; xx < 4; xx = xx + 1) {
                let xi = clamp_i(x0 - 1 + xx, 0, p.Win - 1);
                rowAcc = rowAcc + wx[xx] * input_[srcBase + yi * p.Win + xi];
            }
            acc = acc + wy[yy] * rowAcc;
        }
        output_[gid] = acc;
    } else {  // Area — overlap-weighted averaging
        let yLo = f32(y) * f32(p.Hin) / f32(p.Hout);
        let yHi = f32(y + 1) * f32(p.Hin) / f32(p.Hout);
        let xLo = f32(x) * f32(p.Win) / f32(p.Wout);
        let xHi = f32(x + 1) * f32(p.Win) / f32(p.Wout);
        var yL = i32(floor(yLo)); var yH = i32(ceil(yHi));
        var xL = i32(floor(xLo)); var xH = i32(ceil(xHi));
        if (yH <= yL) { yH = yL + 1; }
        if (xH <= xL) { xH = xL + 1; }
        if (yH > p.Hin) { yH = p.Hin; }
        if (xH > p.Win) { xH = p.Win; }
        let totalArea = (yHi - yLo) * (xHi - xLo);
        var acc : f32 = 0.0;
        for (var yy : i32 = yL; yy < yH; yy = yy + 1) {
            let oy = max(0.0, min(yHi, f32(yy + 1)) - max(yLo, f32(yy)));
            if (oy <= 0.0) { continue; }
            for (var xx : i32 = xL; xx < xH; xx = xx + 1) {
                let ox = max(0.0, min(xHi, f32(xx + 1)) - max(xLo, f32(xx)));
                if (ox <= 0.0) { continue; }
                acc = acc + oy * ox * input_[srcBase + yy * p.Win + xx];
            }
        }
        if (totalArea > 0.0) { output_[gid] = acc / totalArea; } else { output_[gid] = 0.0; }
    }
}
";

    // -----------------------------------------------------------------------
    // Pad 4D.
    // -----------------------------------------------------------------------

    public static string Pad4D => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct P {
    N: i32, C: i32, Hin: i32, Win: i32,
    padN0: i32, padN1: i32, padC0: i32, padC1: i32,
    padH0: i32, padH1: i32, padW0: i32, padW1: i32,
    mode: i32, padValue: f32,
};
@group(0) @binding(2) var<uniform> p : P;
" + Helpers + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let Nout = p.N + p.padN0 + p.padN1;
    let Cout = p.C + p.padC0 + p.padC1;
    let Hout = p.Hin + p.padH0 + p.padH1;
    let Wout = p.Win + p.padW0 + p.padW1;
    let total = Nout * Cout * Hout * Wout;
    let gid = i32(id.x);
    if (gid >= total) { return; }
    let w = gid % Wout; let t1 = gid / Wout;
    let h = t1 % Hout; let t2 = t1 / Hout;
    let c = t2 % Cout; let n = t2 / Cout;
    var nn = n - p.padN0; var cc = c - p.padC0;
    var hh = h - p.padH0; var ww = w - p.padW0;
    let inB = nn >= 0 && nn < p.N && cc >= 0 && cc < p.C && hh >= 0 && hh < p.Hin && ww >= 0 && ww < p.Win;
    if (!inB && p.mode == 0) { output_[gid] = p.padValue; return; }
    if (!inB) {
        if (!(nn >= 0 && nn < p.N)) { nn = pad_boundary(nn, p.N, p.mode); }
        if (!(cc >= 0 && cc < p.C)) { cc = pad_boundary(cc, p.C, p.mode); }
        if (!(hh >= 0 && hh < p.Hin)) { hh = pad_boundary(hh, p.Hin, p.mode); }
        if (!(ww >= 0 && ww < p.Win)) { ww = pad_boundary(ww, p.Win, p.mode); }
    }
    output_[gid] = input_[((nn * p.C + cc) * p.Hin + hh) * p.Win + ww];
}
";

    // -----------------------------------------------------------------------
    // GridSample 2D NHWC — 3 SSBO.
    // -----------------------------------------------------------------------

    public static string GridSample2D => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> grid_ : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
struct P {
    N: i32, H: i32, W: i32, C: i32,
    outH: i32, outW: i32, mode: i32, padding: i32, alignCorners: i32,
};
@group(0) @binding(3) var<uniform> p : P;
" + Helpers + @"
fn sample_safe(n: i32, y_in: i32, x_in: i32, c: i32) -> f32 {
    var y = y_in; var x = x_in;
    if (p.padding == 0) {
        if (u32(y) >= u32(p.H) || u32(x) >= u32(p.W)) { return 0.0; }
    } else if (p.padding == 1) {
        y = clamp_i(y, 0, p.H - 1); x = clamp_i(x, 0, p.W - 1);
    } else {
        y = reflect_index(y, p.H); x = reflect_index(x, p.W);
    }
    return input_[((n * p.H + y) * p.W + x) * p.C + c];
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.N * p.outH * p.outW;
    if (gid >= total) { return; }
    let ox = gid % p.outW; let t1 = gid / p.outW;
    let oy = t1 % p.outH; let n = t1 / p.outH;
    let gOff = ((n * p.outH + oy) * p.outW + ox) * 2;
    let gx = grid_[gOff];
    let gy = grid_[gOff + 1];
    var sx : f32; var sy : f32;
    if (p.alignCorners != 0) {
        sx = (gx + 1.0) * 0.5 * f32(p.W - 1);
        sy = (gy + 1.0) * 0.5 * f32(p.H - 1);
    } else {
        sx = ((gx + 1.0) * f32(p.W) - 1.0) * 0.5;
        sy = ((gy + 1.0) * f32(p.H) - 1.0) * 0.5;
    }

    if (p.mode == 1) {
        let nx = i32(round(sx)); let ny = i32(round(sy));
        for (var c : i32 = 0; c < p.C; c = c + 1) {
            output_[((n * p.outH + oy) * p.outW + ox) * p.C + c] = sample_safe(n, ny, nx, c);
        }
    } else if (p.mode == 0) {
        let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
        let x1 = x0 + 1; let y1 = y0 + 1;
        let fx = sx - f32(x0); let fy = sy - f32(y0);
        for (var c : i32 = 0; c < p.C; c = c + 1) {
            let v00 = sample_safe(n, y0, x0, c);
            let v01 = sample_safe(n, y0, x1, c);
            let v10 = sample_safe(n, y1, x0, c);
            let v11 = sample_safe(n, y1, x1, c);
            output_[((n * p.outH + oy) * p.outW + ox) * p.C + c] =
                v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy)
              + v10 * (1.0 - fx) * fy + v11 * fx * fy;
        }
    } else {
        let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
        let fx = sx - f32(x0); let fy = sy - f32(y0);
        var wy : array<f32, 4>;
        wy[0] = cubic_kernel_f(1.0 + fy, -0.75); wy[1] = cubic_kernel_f(fy, -0.75);
        wy[2] = cubic_kernel_f(1.0 - fy, -0.75); wy[3] = cubic_kernel_f(2.0 - fy, -0.75);
        var wx : array<f32, 4>;
        wx[0] = cubic_kernel_f(1.0 + fx, -0.75); wx[1] = cubic_kernel_f(fx, -0.75);
        wx[2] = cubic_kernel_f(1.0 - fx, -0.75); wx[3] = cubic_kernel_f(2.0 - fx, -0.75);
        for (var c : i32 = 0; c < p.C; c = c + 1) {
            var acc : f32 = 0.0;
            for (var yy : i32 = 0; yy < 4; yy = yy + 1) {
                let yi = y0 - 1 + yy;
                var rowAcc : f32 = 0.0;
                for (var xx : i32 = 0; xx < 4; xx = xx + 1) {
                    let xi = x0 - 1 + xx;
                    rowAcc = rowAcc + wx[xx] * sample_safe(n, yi, xi, c);
                }
                acc = acc + wy[yy] * rowAcc;
            }
            output_[((n * p.outH + oy) * p.outW + ox) * p.C + c] = acc;
        }
    }
}
";

    // -----------------------------------------------------------------------
    // AffineGrid 3D.
    // -----------------------------------------------------------------------

    public static string AffineGrid3D => @"
@group(0) @binding(0) var<storage, read> theta : array<f32>;
@group(0) @binding(1) var<storage, read_write> grid_ : array<f32>;
struct P { N: i32, D: i32, H: i32, W: i32, alignCorners: i32 };
@group(0) @binding(2) var<uniform> p : P;

fn grid_norm_coord_f(idx: i32, size: i32) -> f32 {
    if (size <= 1) { return 0.0; }
    if (p.alignCorners != 0) { return -1.0 + 2.0 * f32(idx) / f32(size - 1); }
    return -1.0 + (2.0 * f32(idx) + 1.0) / f32(size);
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.N * p.D * p.H * p.W;
    if (gid >= total) { return; }
    let w = gid % p.W; let t1 = gid / p.W;
    let h = t1 % p.H; let t2 = t1 / p.H;
    let d = t2 % p.D; let n = t2 / p.D;
    let tBase = n * 12;
    let x = grid_norm_coord_f(w, p.W);
    let y = grid_norm_coord_f(h, p.H);
    let z = grid_norm_coord_f(d, p.D);
    let gBase = (((n * p.D + d) * p.H + h) * p.W + w) * 3;
    for (var row : i32 = 0; row < 3; row = row + 1) {
        grid_[gBase + row] = theta[tBase + row * 4] * x
                           + theta[tBase + row * 4 + 1] * y
                           + theta[tBase + row * 4 + 2] * z
                           + theta[tBase + row * 4 + 3];
    }
}
";
}
#endif
