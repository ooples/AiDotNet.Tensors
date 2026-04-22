// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for the vision detection ops added by Issue
// #217. Mirrors CudaDetectionKernels / HipDetectionKernels /
// MetalDetectionKernels / VulkanDetectionKernels function-for-function.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WGSL compute shader sources for the detection kernels. @workgroup_size(256)
/// + 1-D dispatch shape — pairwise IoU over n*m cells, per-box ops over n
/// cells. Bit-for-bit semantics with the CUDA / HIP / Metal / Vulkan kernels.
/// </summary>
public static class WebGpuDetectionKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "detection_box_iou",
        "detection_generalized_box_iou",
        "detection_distance_box_iou",
        "detection_complete_box_iou",
        "detection_box_area",
        "detection_box_convert",
        "detection_iou_backward_a",
        "detection_iou_backward_b",
    };

    // -----------------------------------------------------------------------
    // Pairwise IoU family — 3 SSBO bindings (a, b, output) + uniform (n, m).
    // -----------------------------------------------------------------------

    public static string BoxIou => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { n: i32, m: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.n * p.m;
    if (gid >= total) { return; }
    let i = gid / p.m;
    let j = gid % p.m;
    let ax1 = a[i * 4]; let ay1 = a[i * 4 + 1]; let ax2 = a[i * 4 + 2]; let ay2 = a[i * 4 + 3];
    let bx1 = b[j * 4]; let by1 = b[j * 4 + 1]; let bx2 = b[j * 4 + 2]; let by2 = b[j * 4 + 3];

    let aw = max(ax2 - ax1, 0.0); let ah = max(ay2 - ay1, 0.0);
    let bw = max(bx2 - bx1, 0.0); let bh = max(by2 - by1, 0.0);
    let areaA = aw * ah; let areaB = bw * bh;

    let ix1 = max(ax1, bx1); let iy1 = max(ay1, by1);
    let ix2 = min(ax2, bx2); let iy2 = min(ay2, by2);
    let iw = max(ix2 - ix1, 0.0); let ih = max(iy2 - iy1, 0.0);
    let inter = iw * ih;
    let u = areaA + areaB - inter;
    if (u > 0.0) { o[gid] = inter / u; } else { o[gid] = 0.0; }
}
";

    public static string GeneralizedBoxIou => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { n: i32, m: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.n * p.m;
    if (gid >= total) { return; }
    let i = gid / p.m;
    let j = gid % p.m;
    let ax1 = a[i * 4]; let ay1 = a[i * 4 + 1]; let ax2 = a[i * 4 + 2]; let ay2 = a[i * 4 + 3];
    let bx1 = b[j * 4]; let by1 = b[j * 4 + 1]; let bx2 = b[j * 4 + 2]; let by2 = b[j * 4 + 3];

    let areaA = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0);
    let areaB = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0);
    let ix1 = max(ax1, bx1); let iy1 = max(ay1, by1);
    let ix2 = min(ax2, bx2); let iy2 = min(ay2, by2);
    let inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0);
    let u = areaA + areaB - inter;
    var iou : f32 = 0.0;
    if (u > 0.0) { iou = inter / u; }

    let ex1 = min(ax1, bx1); let ey1 = min(ay1, by1);
    let ex2 = max(ax2, bx2); let ey2 = max(ay2, by2);
    let enclose = max(ex2 - ex1, 0.0) * max(ey2 - ey1, 0.0);
    if (enclose > 0.0) { o[gid] = iou - (enclose - u) / enclose; } else { o[gid] = iou; }
}
";

    public static string DistanceBoxIou => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { n: i32, m: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.n * p.m;
    if (gid >= total) { return; }
    let i = gid / p.m;
    let j = gid % p.m;
    let ax1 = a[i * 4]; let ay1 = a[i * 4 + 1]; let ax2 = a[i * 4 + 2]; let ay2 = a[i * 4 + 3];
    let bx1 = b[j * 4]; let by1 = b[j * 4 + 1]; let bx2 = b[j * 4 + 2]; let by2 = b[j * 4 + 3];

    let areaA = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0);
    let areaB = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0);
    let ix1 = max(ax1, bx1); let iy1 = max(ay1, by1);
    let ix2 = min(ax2, bx2); let iy2 = min(ay2, by2);
    let inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0);
    let u = areaA + areaB - inter;
    var iou : f32 = 0.0;
    if (u > 0.0) { iou = inter / u; }

    let acx = (ax1 + ax2) * 0.5; let acy = (ay1 + ay2) * 0.5;
    let bcx = (bx1 + bx2) * 0.5; let bcy = (by1 + by2) * 0.5;
    let dx = acx - bcx; let dy = acy - bcy;
    let centreSq = dx * dx + dy * dy;
    let ex1 = min(ax1, bx1); let ey1 = min(ay1, by1);
    let ex2 = max(ax2, bx2); let ey2 = max(ay2, by2);
    let ew = ex2 - ex1; let eh = ey2 - ey1;
    let diagSq = ew * ew + eh * eh;
    if (diagSq > 0.0) { o[gid] = iou - centreSq / diagSq; } else { o[gid] = iou; }
}
";

    public static string CompleteBoxIou => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { n: i32, m: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.n * p.m;
    if (gid >= total) { return; }
    let i = gid / p.m;
    let j = gid % p.m;
    let ax1 = a[i * 4]; let ay1 = a[i * 4 + 1]; let ax2 = a[i * 4 + 2]; let ay2 = a[i * 4 + 3];
    let bx1 = b[j * 4]; let by1 = b[j * 4 + 1]; let bx2 = b[j * 4 + 2]; let by2 = b[j * 4 + 3];

    let aw = ax2 - ax1; let ah = ay2 - ay1;
    let bw = bx2 - bx1; let bh = by2 - by1;
    let areaA = max(aw, 0.0) * max(ah, 0.0);
    let areaB = max(bw, 0.0) * max(bh, 0.0);
    let ix1 = max(ax1, bx1); let iy1 = max(ay1, by1);
    let ix2 = min(ax2, bx2); let iy2 = min(ay2, by2);
    let inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0);
    let u = areaA + areaB - inter;
    var iou : f32 = 0.0;
    if (u > 0.0) { iou = inter / u; }

    let acx = (ax1 + ax2) * 0.5; let acy = (ay1 + ay2) * 0.5;
    let bcx = (bx1 + bx2) * 0.5; let bcy = (by1 + by2) * 0.5;
    let dx = acx - bcx; let dy = acy - bcy;
    let centreSq = dx * dx + dy * dy;
    let ex1 = min(ax1, bx1); let ey1 = min(ay1, by1);
    let ex2 = max(ax2, bx2); let ey2 = max(ay2, by2);
    let ew = ex2 - ex1; let eh = ey2 - ey1;
    let diagSq = ew * ew + eh * eh;
    var diou : f32 = iou;
    if (diagSq > 0.0) { diou = iou - centreSq / diagSq; }

    var v : f32 = 0.0;
    var alpha : f32 = 0.0;
    if (ah > 0.0 && bh > 0.0) {
        let aspectA = atan(aw / ah);
        let aspectB = atan(bw / bh);
        let diff = aspectA - aspectB;
        let PI = 3.14159265358979323846;
        let invPiSq = 4.0 / (PI * PI);
        v = invPiSq * diff * diff;
        let denom = (1.0 - iou) + v;
        if (denom > 0.0) { alpha = v / denom; }
    }
    o[gid] = diou - alpha * v;
}
";

    // -----------------------------------------------------------------------
    // Per-box ops — 2 SSBO bindings (boxes, output).
    // -----------------------------------------------------------------------

    public static string BoxArea => @"
@group(0) @binding(0) var<storage, read> boxes : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { n: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.n) { return; }
    let w = max(boxes[gid * 4 + 2] - boxes[gid * 4], 0.0);
    let h = max(boxes[gid * 4 + 3] - boxes[gid * 4 + 1], 0.0);
    o[gid] = w * h;
}
";

    public static string BoxConvert => @"
@group(0) @binding(0) var<storage, read> boxes : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { n: i32, fromFormat: i32, toFormat: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.n) { return; }
    let oi = gid * 4;
    let v0 = boxes[oi]; let v1 = boxes[oi + 1]; let v2 = boxes[oi + 2]; let v3 = boxes[oi + 3];

    var x1 : f32; var y1 : f32; var x2 : f32; var y2 : f32;
    if (p.fromFormat == 0) { x1 = v0; y1 = v1; x2 = v2; y2 = v3; }
    else if (p.fromFormat == 1) { x1 = v0; y1 = v1; x2 = v0 + v2; y2 = v1 + v3; }
    else { let hw = v2 * 0.5; let hh = v3 * 0.5;
           x1 = v0 - hw; y1 = v1 - hh; x2 = v0 + hw; y2 = v1 + hh; }

    if (p.toFormat == 0) { o[oi] = x1; o[oi + 1] = y1; o[oi + 2] = x2; o[oi + 3] = y2; }
    else if (p.toFormat == 1) {
        o[oi] = x1; o[oi + 1] = y1;
        o[oi + 2] = x2 - x1; o[oi + 3] = y2 - y1;
    } else {
        let w = x2 - x1; let h = y2 - y1;
        o[oi] = x1 + w * 0.5; o[oi + 1] = y1 + h * 0.5;
        o[oi + 2] = w; o[oi + 3] = h;
    }
}
";

    // -----------------------------------------------------------------------
    // IoU family backward — Issue #217. Atomics-free two-kernel split:
    // *_backward_a owns rows of N, *_backward_b owns rows of M. WebGPU
    // lacks atomic<f32> so this design is mandatory (not an optimisation).
    // -----------------------------------------------------------------------

    private const string BackwardCellFn = @"
const DETECTION_INV_PI_SQ : f32 = 0.40528473456935109;  // 4 / pi^2

fn compute_cell_grads_iou(
    ax1: f32, ay1: f32, ax2: f32, ay2: f32,
    bx1: f32, by1: f32, bx2: f32, by2: f32,
    g: f32, variant: i32,
    gA: ptr<function, vec4<f32>>, gB: ptr<function, vec4<f32>>)
{
    *gA = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    *gB = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (g == 0.0) { return; }

    let awRaw = ax2 - ax1; let ahRaw = ay2 - ay1;
    let bwRaw = bx2 - bx1; let bhRaw = by2 - by1;
    let aw = max(awRaw, 0.0); let ah = max(ahRaw, 0.0);
    let bw = max(bwRaw, 0.0); let bh = max(bhRaw, 0.0);
    let areaA = aw * ah; let areaB = bw * bh;

    let ix1 = max(ax1, bx1); let ix2 = min(ax2, bx2);
    let iy1 = max(ay1, by1); let iy2 = min(ay2, by2);
    let iwRaw = ix2 - ix1; let ihRaw = iy2 - iy1;
    let iw = max(iwRaw, 0.0); let ih = max(ihRaw, 0.0);
    let inter = iw * ih;
    let u = areaA + areaB - inter;
    var iou : f32 = 0.0;
    if (u > 0.0) { iou = inter / u; }

    var gInter : f32 = 0.0;
    var gAreaA : f32 = 0.0;
    var gAreaB : f32 = 0.0;

    if (variant == 1) {
        let ex1 = min(ax1, bx1); let ex2 = max(ax2, bx2);
        let ey1 = min(ay1, by1); let ey2 = max(ay2, by2);
        let ewRaw = ex2 - ex1; let ehRaw = ey2 - ey1;
        let ew = max(ewRaw, 0.0); let eh = max(ehRaw, 0.0);
        let enclose = ew * eh;
        if (enclose > 0.0) {
            let gUnion = g / enclose;
            let gEnclose = g * (-u / (enclose * enclose));
            gAreaA = gAreaA + gUnion;
            gAreaB = gAreaB + gUnion;
            gInter = gInter - gUnion;
            let gEw = gEnclose * eh; let gEh = gEnclose * ew;
            var gEwRaw : f32 = 0.0;
            var gEhRaw : f32 = 0.0;
            if (ewRaw > 0.0) { gEwRaw = gEw; }
            if (ehRaw > 0.0) { gEhRaw = gEh; }
            let gEx1 = -gEwRaw; let gEx2 = gEwRaw;
            let gEy1 = -gEhRaw; let gEy2 = gEhRaw;
            if (ax1 <= bx1) { (*gA).x = (*gA).x + gEx1; } else { (*gB).x = (*gB).x + gEx1; }
            if (ay1 <= by1) { (*gA).y = (*gA).y + gEy1; } else { (*gB).y = (*gB).y + gEy1; }
            if (ax2 >= bx2) { (*gA).z = (*gA).z + gEx2; } else { (*gB).z = (*gB).z + gEx2; }
            if (ay2 >= by2) { (*gA).w = (*gA).w + gEy2; } else { (*gB).w = (*gB).w + gEy2; }
        }
    } else if (variant == 2 || variant == 3) {
        let acx = (ax1 + ax2) * 0.5; let acy = (ay1 + ay2) * 0.5;
        let bcx = (bx1 + bx2) * 0.5; let bcy = (by1 + by2) * 0.5;
        let dcx = acx - bcx; let dcy = acy - bcy;
        let centreSq = dcx * dcx + dcy * dcy;
        let ex1 = min(ax1, bx1); let ex2 = max(ax2, bx2);
        let ey1 = min(ay1, by1); let ey2 = max(ay2, by2);
        let ew = ex2 - ex1; let eh = ey2 - ey1;
        let diagSq = ew * ew + eh * eh;

        var gCentreSq : f32 = 0.0;
        var gDiagSq : f32 = 0.0;
        if (diagSq > 0.0) {
            gCentreSq = g * (-1.0 / diagSq);
            gDiagSq = g * centreSq / (diagSq * diagSq);
        }

        if (variant == 3 && ah > 0.0 && bh > 0.0) {
            let aspectA = atan(aw / ah);
            let aspectB = atan(bw / bh);
            let diff = aspectA - aspectB;
            let v = DETECTION_INV_PI_SQ * diff * diff;
            let denom = (1.0 - iou) + v;
            var alpha : f32 = 0.0;
            if (denom > 0.0) { alpha = v / denom; }
            let gV = g * (-alpha);
            let gDiff = gV * 2.0 * DETECTION_INV_PI_SQ * diff;
            let gAspectA = gDiff; let gAspectB = -gDiff;
            let aDen = aw * aw + ah * ah;
            let bDen = bw * bw + bh * bh;
            if (aDen > 0.0) {
                let gAw = gAspectA * (ah / aDen);
                let gAh = gAspectA * (-aw / aDen);
                if (awRaw > 0.0) { (*gA).z = (*gA).z + gAw; (*gA).x = (*gA).x - gAw; }
                if (ahRaw > 0.0) { (*gA).w = (*gA).w + gAh; (*gA).y = (*gA).y - gAh; }
            }
            if (bDen > 0.0) {
                let gBw = gAspectB * (bh / bDen);
                let gBh = gAspectB * (-bw / bDen);
                if (bwRaw > 0.0) { (*gB).z = (*gB).z + gBw; (*gB).x = (*gB).x - gBw; }
                if (bhRaw > 0.0) { (*gB).w = (*gB).w + gBh; (*gB).y = (*gB).y - gBh; }
            }
        }

        let gAcx = gCentreSq * 2.0 * dcx;
        let gAcy = gCentreSq * 2.0 * dcy;
        (*gA).x = (*gA).x + gAcx * 0.5; (*gA).z = (*gA).z + gAcx * 0.5;
        (*gA).y = (*gA).y + gAcy * 0.5; (*gA).w = (*gA).w + gAcy * 0.5;
        (*gB).x = (*gB).x - gAcx * 0.5; (*gB).z = (*gB).z - gAcx * 0.5;
        (*gB).y = (*gB).y - gAcy * 0.5; (*gB).w = (*gB).w - gAcy * 0.5;

        let gEw = gDiagSq * 2.0 * ew;
        let gEh = gDiagSq * 2.0 * eh;
        let gEx1 = -gEw; let gEx2 = gEw;
        let gEy1 = -gEh; let gEy2 = gEh;
        if (ax1 <= bx1) { (*gA).x = (*gA).x + gEx1; } else { (*gB).x = (*gB).x + gEx1; }
        if (ay1 <= by1) { (*gA).y = (*gA).y + gEy1; } else { (*gB).y = (*gB).y + gEy1; }
        if (ax2 >= bx2) { (*gA).z = (*gA).z + gEx2; } else { (*gB).z = (*gB).z + gEx2; }
        if (ay2 >= by2) { (*gA).w = (*gA).w + gEy2; } else { (*gB).w = (*gB).w + gEy2; }
    }

    if (u > 0.0) {
        let uSq = u * u;
        gInter = gInter + g * (u + inter) / uSq;
        gAreaA = gAreaA + g * (-inter) / uSq;
        gAreaB = gAreaB + g * (-inter) / uSq;
    }

    let gIw = gInter * ih; let gIh = gInter * iw;
    var gIwRaw : f32 = 0.0;
    var gIhRaw : f32 = 0.0;
    if (iwRaw > 0.0) { gIwRaw = gIw; }
    if (ihRaw > 0.0) { gIhRaw = gIh; }
    let gIx2 = gIwRaw; let gIx1 = -gIwRaw;
    let gIy2 = gIhRaw; let gIy1 = -gIhRaw;
    if (ax1 >= bx1) { (*gA).x = (*gA).x + gIx1; } else { (*gB).x = (*gB).x + gIx1; }
    if (ay1 >= by1) { (*gA).y = (*gA).y + gIy1; } else { (*gB).y = (*gB).y + gIy1; }
    if (ax2 <= bx2) { (*gA).z = (*gA).z + gIx2; } else { (*gB).z = (*gB).z + gIx2; }
    if (ay2 <= by2) { (*gA).w = (*gA).w + gIy2; } else { (*gB).w = (*gB).w + gIy2; }

    let gAw2 = gAreaA * ah; let gAh2 = gAreaA * aw;
    let gBw2 = gAreaB * bh; let gBh2 = gAreaB * bw;
    if (awRaw > 0.0) { (*gA).z = (*gA).z + gAw2; (*gA).x = (*gA).x - gAw2; }
    if (ahRaw > 0.0) { (*gA).w = (*gA).w + gAh2; (*gA).y = (*gA).y - gAh2; }
    if (bwRaw > 0.0) { (*gB).z = (*gB).z + gBw2; (*gB).x = (*gB).x - gBw2; }
    if (bhRaw > 0.0) { (*gB).w = (*gB).w + gBh2; (*gB).y = (*gB).y - gBh2; }
}
";

    public static string IouBackwardA => @"
@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
@group(0) @binding(1) var<storage, read> a : array<f32>;
@group(0) @binding(2) var<storage, read> b : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradA : array<f32>;
struct P { n: i32, m: i32, variant: i32 };
@group(0) @binding(4) var<uniform> p : P;
" + BackwardCellFn + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = i32(id.x);
    if (i >= p.n) { return; }
    let ax1 = a[i * 4]; let ay1 = a[i * 4 + 1]; let ax2 = a[i * 4 + 2]; let ay2 = a[i * 4 + 3];
    var sum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var j : i32 = 0; j < p.m; j = j + 1) {
        let bx1 = b[j * 4]; let by1 = b[j * 4 + 1]; let bx2 = b[j * 4 + 2]; let by2 = b[j * 4 + 3];
        let g = gradOutput[i * p.m + j];
        var gA : vec4<f32>; var gB : vec4<f32>;
        compute_cell_grads_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, g, p.variant, &gA, &gB);
        sum = sum + gA;
    }
    gradA[i * 4]     = sum.x;
    gradA[i * 4 + 1] = sum.y;
    gradA[i * 4 + 2] = sum.z;
    gradA[i * 4 + 3] = sum.w;
}
";

    public static string IouBackwardB => @"
@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
@group(0) @binding(1) var<storage, read> a : array<f32>;
@group(0) @binding(2) var<storage, read> b : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradB : array<f32>;
struct P { n: i32, m: i32, variant: i32 };
@group(0) @binding(4) var<uniform> p : P;
" + BackwardCellFn + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let j = i32(id.x);
    if (j >= p.m) { return; }
    let bx1 = b[j * 4]; let by1 = b[j * 4 + 1]; let bx2 = b[j * 4 + 2]; let by2 = b[j * 4 + 3];
    var sum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i : i32 = 0; i < p.n; i = i + 1) {
        let ax1 = a[i * 4]; let ay1 = a[i * 4 + 1]; let ax2 = a[i * 4 + 2]; let ay2 = a[i * 4 + 3];
        let g = gradOutput[i * p.m + j];
        var gA : vec4<f32>; var gB : vec4<f32>;
        compute_cell_grads_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, g, p.variant, &gA, &gB);
        sum = sum + gB;
    }
    gradB[j * 4]     = sum.x;
    gradB[j * 4 + 1] = sum.y;
    gradB[j * 4 + 2] = sum.z;
    gradB[j * 4 + 3] = sum.w;
}
";
}
#endif
