// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
        "geometry_grid_sample_2d_backward_input",
        "geometry_grid_sample_2d_backward_grid",
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
        // floor(x * Win / Wout) in integers: a float division here is not correctly rounded (drivers may compute
        // a * (1 / b)), so an exact quotient such as 30 / 15 came out as 1.99999994 and floored to the wrong pixel.
        var yi = (y * p.Hin) / p.Hout; if (yi >= p.Hin) { yi = p.Hin - 1; }
        var xi = (x * p.Win) / p.Wout; if (xi >= p.Win) { xi = p.Win - 1; }
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
        // Positions in units of 1/Hout (1/Wout): output cell y covers [y * Hin, (y + 1) * Hin) and input pixel yy
        // covers [yy * Hout, (yy + 1) * Hout), so every bound and overlap is an exact integer; a float division is
        // not correctly rounded here and could open the window a pixel wide with a spurious sliver of weight.
        let yLo = y * p.Hin; let yHi = (y + 1) * p.Hin;
        let xLo = x * p.Win; let xHi = (x + 1) * p.Win;
        let yL = yLo / p.Hout; let yH = min((yHi + p.Hout - 1) / p.Hout, p.Hin);
        let xL = xLo / p.Wout; let xH = min((xHi + p.Wout - 1) / p.Wout, p.Win);
        var acc : f32 = 0.0;
        for (var yy : i32 = yL; yy < yH; yy = yy + 1) {
            let oy = min(yHi, (yy + 1) * p.Hout) - max(yLo, yy * p.Hout);
            if (oy <= 0) { continue; }
            for (var xx : i32 = xL; xx < xH; xx = xx + 1) {
                let ox = min(xHi, (xx + 1) * p.Wout) - max(xLo, xx * p.Wout);
                if (ox <= 0) { continue; }
                acc = acc + f32(oy * ox) * input_[srcBase + yy * p.Win + xx];
            }
        }
        output_[gid] = acc / f32(p.Hin * p.Win);
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
    // GridSample 2D, NCHW input/output — 3 SSBO.
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
" + Helpers + GridSampleTaps + @"
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

    // #1134: a NaN, infinite or out-of-int-range point samples nothing (the CPU rule). Tested on the bits, since WGSL
    // lets an implementation assume floats are finite: |x| below 2147483520 (0x4EFFFFFF) has a smaller bit pattern.
    if ((bitcast<u32>(sx) & 0x7FFFFFFFu) >= 0x4EFFFFFFu || (bitcast<u32>(sy) & 0x7FFFFFFFu) >= 0x4EFFFFFFu) {
        for (var c : i32 = 0; c < p.C; c = c + 1) { output_[((n * p.C + c) * p.outH + oy) * p.outW + ox] = 0.0; }
        return;
    }

    // The backward's taps, so the forward and its derivative read the same pixels with the same weights.
    var t = grid_sample_taps(gOff);
    for (var c : i32 = 0; c < p.C; c = c + 1) {
        let plane = (n * p.C + c) * p.H;
        var acc : f32 = 0.0;
        for (var j : i32 = 0; j < t.count; j = j + 1) {
            if (t.ty[j] < 0) { continue; }
            let row = (plane + t.ty[j]) * p.W;
            var rowAcc : f32 = 0.0;
            for (var i : i32 = 0; i < t.count; i = i + 1) {
                if (t.tx[i] >= 0) { rowAcc = rowAcc + t.wx[i] * input_[row + t.tx[i]]; }
            }
            acc = acc + t.wy[j] * rowAcc;
        }
        output_[((n * p.C + c) * p.outH + oy) * p.outW + ox] = acc;
    }
}
";

    // -----------------------------------------------------------------------
    // GridSample 2D forward and backward, every mode and padding (#1680), with PyTorch's grid_sampler_2d
    // padding rules. One tap helper gives each point's pixels, weights and weight derivatives; the forward sums
    // over them and the backward differentiates the same sum. One thread per sampling point, looping C.
    // -----------------------------------------------------------------------

    private const string GridSampleBackwardUniform = @"
struct P {
    N: i32, H: i32, W: i32, C: i32,
    outH: i32, outW: i32, mode: i32, padding: i32, alignCorners: i32,
};
";

    private const string GridSampleTaps = @"
// PyTorch's grid_sampler_compute_source_index for Border (1) and Reflection (2), with *slope = d(result)/d(s):
// Reflection reflects about -0.5 and S-0.5 (0 and S-1 with alignCorners), then both clip to [0, S-1]. A clipped
// coordinate has slope 0. Bilinear and Nearest pad the coordinate this way; Bicubic pads each tap instead.
fn grid_sample_pad_coord(s_in: f32, size: i32, slope: ptr<function, f32>) -> f32 {
    var s = s_in;
    *slope = 1.0;
    if (p.padding == 0) { return s; }
    if (p.padding == 2) {
        if (size == 1 && p.alignCorners != 0) { *slope = 0.0; return 0.0; }
        var twiceLow : f32 = -1.0; var twiceHigh : f32 = 2.0 * f32(size) - 1.0;
        if (p.alignCorners != 0) { twiceLow = 0.0; twiceHigh = 2.0 * f32(size - 1); }
        let mn = twiceLow * 0.5; let span = (twiceHigh - twiceLow) * 0.5;
        var x = s - mn;
        if (x < 0.0) { *slope = -1.0; x = -x; }
        let flips = floor(x / span);
        let extra = x - flips * span;
        if (flips - 2.0 * floor(flips * 0.5) == 0.0) { s = extra + mn; }
        else { s = span - extra + mn; *slope = -*slope; }
    }
    if (s <= 0.0) { *slope = 0.0; return 0.0; }
    if (s >= f32(size - 1)) { *slope = 0.0; return f32(size - 1); }
    return s;
}

fn grid_sample_in_bounds(i: i32, extent: i32) -> i32 {
    if (u32(i) < u32(extent)) { return i; }
    return -1;
}

// Bicubic's padded tap (PyTorch's compute_coordinates on an integer tap): without alignCorners Reflection
// mirrors about -0.5 and S-0.5, repeating the edge pixel (period 2S); with it, about 0 and S-1.
fn grid_sample_bicubic_tap(i: i32, extent: i32) -> i32 {
    if (p.padding == 0) { return grid_sample_in_bounds(i, extent); }
    if (p.padding == 1) { return clamp_i(i, 0, extent - 1); }
    if (p.alignCorners != 0) { return reflect_index(i, extent); }
    let period = 2 * extent;
    let r = ((i % period) + period) % period;
    if (r < extent) { return r; }
    return period - 1 - r;
}

// dK/dd of cubic_kernel_f for d >= 0, which every offset the taps use is.
fn cubic_kernel_deriv_f(d: f32, a: f32) -> f32 {
    if (d < 1.0) { return (3.0 * (a + 2.0) * d - 2.0 * (a + 3.0)) * d; }
    if (d < 2.0) { return (3.0 * a * d - 10.0 * a) * d + 8.0 * a; }
    return 0.0;
}

// The taps of one sampling point: per-axis pixel index, weight and d(weight)/d(source coord). count is the taps
// per axis (1 nearest, 2 bilinear, 4 bicubic), or 0 when the forward samples nothing at this point (#1134).
struct Taps {
    count: i32,
    tx: array<i32, 4>, ty: array<i32, 4>,
    wx: array<f32, 4>, wy: array<f32, 4>,
    dwx: array<f32, 4>, dwy: array<f32, 4>,
};

fn grid_sample_taps(gOff: i32) -> Taps {
    var t : Taps;
    t.count = 0;
    for (var k : i32 = 0; k < 4; k = k + 1) {
        t.tx[k] = -1; t.ty[k] = -1; t.wx[k] = 0.0; t.wy[k] = 0.0; t.dwx[k] = 0.0; t.dwy[k] = 0.0;
    }
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
    if ((bitcast<u32>(sx) & 0x7FFFFFFFu) >= 0x4EFFFFFFu || (bitcast<u32>(sy) & 0x7FFFFFFFu) >= 0x4EFFFFFFu) { return t; }
    if (p.mode != 2) {
        var slx : f32; var sly : f32;
        let px = grid_sample_pad_coord(sx, p.W, &slx);
        let py = grid_sample_pad_coord(sy, p.H, &sly);
        if (p.mode == 1) {
            // round() rounds half to even, as PyTorch's nearbyint.
            t.count = 1;
            t.tx[0] = grid_sample_in_bounds(i32(round(px)), p.W);
            t.ty[0] = grid_sample_in_bounds(i32(round(py)), p.H);
            t.wx[0] = 1.0; t.wy[0] = 1.0;
            return t;
        }
        let px0 = i32(floor(px)); let py0 = i32(floor(py));
        let pfx = px - f32(px0); let pfy = py - f32(py0);
        t.count = 2;
        t.wx[0] = 1.0 - pfx; t.wx[1] = pfx; t.dwx[0] = -slx; t.dwx[1] = slx;
        t.wy[0] = 1.0 - pfy; t.wy[1] = pfy; t.dwy[0] = -sly; t.dwy[1] = sly;
        for (var k : i32 = 0; k < 2; k = k + 1) {
            t.tx[k] = grid_sample_in_bounds(px0 + k, p.W);
            t.ty[k] = grid_sample_in_bounds(py0 + k, p.H);
        }
        return t;
    }
    let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
    let fx = sx - f32(x0); let fy = sy - f32(y0);
    t.count = 4;
    let a = -0.75;
    t.wx[0] = cubic_kernel_f(1.0 + fx, a); t.wx[1] = cubic_kernel_f(fx, a);
    t.wx[2] = cubic_kernel_f(1.0 - fx, a); t.wx[3] = cubic_kernel_f(2.0 - fx, a);
    t.wy[0] = cubic_kernel_f(1.0 + fy, a); t.wy[1] = cubic_kernel_f(fy, a);
    t.wy[2] = cubic_kernel_f(1.0 - fy, a); t.wy[3] = cubic_kernel_f(2.0 - fy, a);
    // Offsets 1+t and t grow with t, 1-t and 2-t shrink: hence the sign flip on the last two.
    t.dwx[0] = cubic_kernel_deriv_f(1.0 + fx, a); t.dwx[1] = cubic_kernel_deriv_f(fx, a);
    t.dwx[2] = -cubic_kernel_deriv_f(1.0 - fx, a); t.dwx[3] = -cubic_kernel_deriv_f(2.0 - fx, a);
    t.dwy[0] = cubic_kernel_deriv_f(1.0 + fy, a); t.dwy[1] = cubic_kernel_deriv_f(fy, a);
    t.dwy[2] = -cubic_kernel_deriv_f(1.0 - fy, a); t.dwy[3] = -cubic_kernel_deriv_f(2.0 - fy, a);
    for (var k : i32 = 0; k < 4; k = k + 1) {
        t.tx[k] = grid_sample_bicubic_tap(x0 - 1 + k, p.W);
        t.ty[k] = grid_sample_bicubic_tap(y0 - 1 + k, p.H);
    }
    return t;
}
";

    /// <summary>
    /// GridSample 2D input gradient (gradOut, grid, gradIn). gradIn must be zeroed first: several sampling
    /// points can read one pixel, so this accumulates with a compare-and-swap float add.
    /// </summary>
    public static string GridSample2DBackwardInput => @"
@group(0) @binding(0) var<storage, read> gradOut_ : array<f32>;
@group(0) @binding(1) var<storage, read> grid_ : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradIn_ : array<atomic<u32>>;
" + GridSampleBackwardUniform + @"
@group(0) @binding(3) var<uniform> p : P;
" + Helpers + GridSampleTaps + @"
fn atomic_add_f(idx: i32, val: f32) {
    var old = atomicLoad(&gradIn_[idx]);
    loop {
        let res = atomicCompareExchangeWeak(&gradIn_[idx], old, bitcast<u32>(bitcast<f32>(old) + val));
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.N * p.outH * p.outW) { return; }
    let ox = gid % p.outW; let t1 = gid / p.outW;
    let oy = t1 % p.outH; let n = t1 / p.outH;
    var t = grid_sample_taps(gid * 2);
    for (var c : i32 = 0; c < p.C; c = c + 1) {
        let g = gradOut_[((n * p.C + c) * p.outH + oy) * p.outW + ox];
        if (g == 0.0) { continue; }
        let plane = (n * p.C + c) * p.H;
        for (var j : i32 = 0; j < t.count; j = j + 1) {
            if (t.ty[j] < 0) { continue; }
            let row = (plane + t.ty[j]) * p.W;
            for (var i : i32 = 0; i < t.count; i = i + 1) {
                if (t.tx[i] < 0) { continue; }
                atomic_add_f(row + t.tx[i], g * t.wx[i] * t.wy[j]);
            }
        }
    }
}
";

    /// <summary>
    /// GridSample 2D grid gradient (gradOut, input, grid, gradGrid). Writes every element of gradGrid.
    /// </summary>
    public static string GridSample2DBackwardGrid => @"
@group(0) @binding(0) var<storage, read> gradOut_ : array<f32>;
@group(0) @binding(1) var<storage, read> input_ : array<f32>;
@group(0) @binding(2) var<storage, read> grid_ : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradGrid_ : array<f32>;
" + GridSampleBackwardUniform + @"
@group(0) @binding(4) var<uniform> p : P;
" + Helpers + GridSampleTaps + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.N * p.outH * p.outW) { return; }
    let ox = gid % p.outW; let t1 = gid / p.outW;
    let oy = t1 % p.outH; let n = t1 / p.outH;
    // Nearest is piecewise constant: its grid gradient is exactly 0. Written outright, since g * input can
    // overflow to infinity and infinity * 0 is NaN.
    if (p.mode == 1) { gradGrid_[gid * 2] = 0.0; gradGrid_[gid * 2 + 1] = 0.0; return; }
    var t = grid_sample_taps(gid * 2);
    var gradX = 0.0; var gradY = 0.0;
    for (var c : i32 = 0; c < p.C; c = c + 1) {
        let g = gradOut_[((n * p.C + c) * p.outH + oy) * p.outW + ox];
        let plane = (n * p.C + c) * p.H;
        for (var j : i32 = 0; j < t.count; j = j + 1) {
            if (t.ty[j] < 0) { continue; }
            let row = (plane + t.ty[j]) * p.W;
            for (var i : i32 = 0; i < t.count; i = i + 1) {
                if (t.tx[i] < 0) { continue; }
                let v = g * input_[row + t.tx[i]];
                gradX = gradX + v * t.dwx[i] * t.wy[j];
                gradY = gradY + v * t.wx[i] * t.dwy[j];
            }
        }
    }
    // d(source coord)/d(grid coord): (S-1)/2 with alignCorners, S/2 without.
    var sxg = f32(p.W) * 0.5; var syg = f32(p.H) * 0.5;
    if (p.alignCorners != 0) { sxg = f32(p.W - 1) * 0.5; syg = f32(p.H - 1) * 0.5; }
    gradGrid_[gid * 2] = gradX * sxg;
    gradGrid_[gid * 2 + 1] = gradY * syg;
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
