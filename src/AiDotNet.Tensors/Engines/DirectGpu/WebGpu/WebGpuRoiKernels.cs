#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public static class WebGpuRoiKernels
{
    public static string[] GetKernelNames() => new[] { "roi_align", "roi_pool" };

    public static string RoIAlign => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> boxes : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
struct P {
    N: i32, C: i32, H: i32, W: i32, K: i32, outH: i32, outW: i32,
    spatialScale: f32, samplingRatio: i32, aligned: i32
};
@group(0) @binding(3) var<uniform> p : P;

fn bilinear_sample(planeBase: i32, y_in: f32, x_in: f32) -> f32 {
    var y = y_in; var x = x_in;
    if (y < -1.0 || y > f32(p.H) || x < -1.0 || x > f32(p.W)) { return 0.0; }
    if (y <= 0.0) { y = 0.0; }
    if (x <= 0.0) { x = 0.0; }
    var y0 = i32(y);
    var x0 = i32(x);
    var y1 = select(y0 + 1, p.H - 1, y0 + 1 >= p.H);
    var x1 = select(x0 + 1, p.W - 1, x0 + 1 >= p.W);
    if (y0 >= p.H - 1) { y0 = p.H - 1; y1 = p.H - 1; y = f32(y0); }
    if (x0 >= p.W - 1) { x0 = p.W - 1; x1 = p.W - 1; x = f32(x0); }
    let ly = y - f32(y0); let lx = x - f32(x0);
    let hy = 1.0 - ly; let hx = 1.0 - lx;
    return hy * hx * input_[planeBase + y0 * p.W + x0]
         + hy * lx * input_[planeBase + y0 * p.W + x1]
         + ly * hx * input_[planeBase + y1 * p.W + x0]
         + ly * lx * input_[planeBase + y1 * p.W + x1];
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.K * p.C * p.outH * p.outW;
    if (gid >= total) { return; }
    let pw = gid % p.outW; let t1 = gid / p.outW;
    let ph = t1 % p.outH; let t2 = t1 / p.outH;
    let c = t2 % p.C; let k = t2 / p.C;
    let n = i32(boxes[k * 5]);
    if (n < 0 || n >= p.N) { output_[gid] = 0.0; return; }
    var offset : f32 = 0.0;
    if (p.aligned != 0) { offset = 0.5; }
    let x1 = boxes[k * 5 + 1] * p.spatialScale - offset;
    let y1 = boxes[k * 5 + 2] * p.spatialScale - offset;
    let x2 = boxes[k * 5 + 3] * p.spatialScale - offset;
    let y2 = boxes[k * 5 + 4] * p.spatialScale - offset;
    var roiW = select(max(x2 - x1, 1.0), x2 - x1, p.aligned != 0);
    var roiH = select(max(y2 - y1, 1.0), y2 - y1, p.aligned != 0);
    let binH = roiH / f32(p.outH);
    let binW = roiW / f32(p.outW);
    var ry = select(i32(ceil(roiH / f32(p.outH))), p.samplingRatio, p.samplingRatio > 0);
    var rx = select(i32(ceil(roiW / f32(p.outW))), p.samplingRatio, p.samplingRatio > 0);
    if (ry < 1) { ry = 1; }
    if (rx < 1) { rx = 1; }
    let planeBase = (n * p.C + c) * p.H * p.W;
    var acc : f32 = 0.0;
    for (var iy : i32 = 0; iy < ry; iy = iy + 1) {
        let sy = y1 + f32(ph) * binH + (f32(iy) + 0.5) * binH / f32(ry);
        for (var ix : i32 = 0; ix < rx; ix = ix + 1) {
            let sx = x1 + f32(pw) * binW + (f32(ix) + 0.5) * binW / f32(rx);
            acc = acc + bilinear_sample(planeBase, sy, sx);
        }
    }
    output_[gid] = acc / f32(ry * rx);
}
";

    public static string RoIPool => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> boxes : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
struct P {
    N: i32, C: i32, H: i32, W: i32, K: i32, outH: i32, outW: i32, spatialScale: f32
};
@group(0) @binding(3) var<uniform> p : P;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.K * p.C * p.outH * p.outW;
    if (gid >= total) { return; }
    let pw = gid % p.outW; let t1 = gid / p.outW;
    let ph = t1 % p.outH; let t2 = t1 / p.outH;
    let c = t2 % p.C; let k = t2 / p.C;
    let n = i32(boxes[k * 5]);
    if (n < 0 || n >= p.N) { output_[gid] = 0.0; return; }
    let x1 = i32(round(boxes[k * 5 + 1] * p.spatialScale));
    let y1 = i32(round(boxes[k * 5 + 2] * p.spatialScale));
    let x2 = i32(round(boxes[k * 5 + 3] * p.spatialScale));
    let y2 = i32(round(boxes[k * 5 + 4] * p.spatialScale));
    var roiW = x2 - x1 + 1; if (roiW < 1) { roiW = 1; }
    var roiH = y2 - y1 + 1; if (roiH < 1) { roiH = 1; }
    let binH = f32(roiH) / f32(p.outH);
    let binW = f32(roiW) / f32(p.outW);
    var hstart = i32(floor(f32(ph) * binH)) + y1;
    var hend = i32(ceil(f32(ph + 1) * binH)) + y1;
    var wstart = i32(floor(f32(pw) * binW)) + x1;
    var wend = i32(ceil(f32(pw + 1) * binW)) + x1;
    if (hstart < 0) { hstart = 0; } if (hstart > p.H) { hstart = p.H; }
    if (hend < 0) { hend = 0; } if (hend > p.H) { hend = p.H; }
    if (wstart < 0) { wstart = 0; } if (wstart > p.W) { wstart = p.W; }
    if (wend < 0) { wend = 0; } if (wend > p.W) { wend = p.W; }
    let planeBase = (n * p.C + c) * p.H * p.W;
    if (hend <= hstart || wend <= wstart) { output_[gid] = 0.0; return; }
    var best : f32 = -3.4e38;
    for (var yy : i32 = hstart; yy < hend; yy = yy + 1) {
        for (var xx : i32 = wstart; xx < wend; xx = xx + 1) {
            let v = input_[planeBase + yy * p.W + xx];
            if (v > best) { best = v; }
        }
    }
    output_[gid] = best;
}
";
}
#endif
