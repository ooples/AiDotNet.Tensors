#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    public static readonly string HouseholderNorm = Shader(
        new[] { "r:u", "rw:uHat", "rw:hhStats" }, "K: i32, D: i32, sqrtFloorBits: i32",
        Main + @"
    let K = p.K; let D = p.D;
    if (i >= K) { return; }
    let sqrtFloor = bitcast<f32>(p.sqrtFloorBits);
    var maxAbs: f32 = 0.0;
    for (var c: i32 = 0; c < D; c++) { maxAbs = max(maxAbs, abs(u[i * D + c])); }
    let rowScale = max(maxAbs, sqrtFloor * 0.0625);
    var scaledSq: f32 = 0.0;
    for (var c: i32 = 0; c < D; c++) { let s = u[i * D + c] / rowScale; scaledSq += s * s; }
    let sfr = sqrtFloor / rowScale;
    let scaledFloor = sfr * sfr;
    let denom = max(scaledSq, scaledFloor);
    let r = 1.0 / sqrt(denom);
    for (var c: i32 = 0; c < D; c++) { uHat[i * D + c] = (u[i * D + c] / rowScale) * r; }
    hhStats[2 * i] = r / rowScale;
    hhStats[2 * i + 1] = select(select(0.0, 0.5, scaledSq == scaledFloor), 1.0, scaledSq > scaledFloor);
}");

    public static readonly string HouseholderForward = Shader(
        new[] { "r:x", "r:uHat", "rw:y" }, "M: i32, K: i32, D: i32, trans: i32",
        Main + @"
    let M = p.M; let K = p.K; let D = p.D;
    if (i >= M) { return; }
    let baseOff = i * D;
    for (var c: i32 = 0; c < D; c++) { y[baseOff + c] = x[baseOff + c]; }
    for (var s: i32 = 0; s < K; s++) {
        let j = select(s, K - 1 - s, p.trans != 0);
        let uOff = j * D;
        var dot: f32 = 0.0;
        for (var c: i32 = 0; c < D; c++) { dot += y[baseOff + c] * uHat[uOff + c]; }
        for (var c: i32 = 0; c < D; c++) { let step = dot * uHat[uOff + c]; y[baseOff + c] = (y[baseOff + c] - step) - step; }
    }
}");

    public static readonly string HouseholderBackwardInput = Shader(
        new[] { "r:dy", "r:x", "r:uHat", "rw:dx", "rw:pqBuf" }, "M: i32, K: i32, D: i32, trans: i32",
        Main + @"
    let M = p.M; let K = p.K; let D = p.D;
    if (i >= M) { return; }
    let baseOff = i * D;
    let pqOff = i * 2 * K;
    for (var c: i32 = 0; c < D; c++) { dx[baseOff + c] = x[baseOff + c]; }
    for (var s: i32 = 0; s < K; s++) {
        let j = select(s, K - 1 - s, p.trans != 0);
        let uOff = j * D;
        var pv: f32 = 0.0;
        for (var c: i32 = 0; c < D; c++) { pv += dx[baseOff + c] * uHat[uOff + c]; }
        pqBuf[pqOff + s] = pv;
        for (var c: i32 = 0; c < D; c++) { let step = pv * uHat[uOff + c]; dx[baseOff + c] = (dx[baseOff + c] - step) - step; }
    }
    for (var c: i32 = 0; c < D; c++) { dx[baseOff + c] = dy[baseOff + c]; }
    for (var sr: i32 = 0; sr < K; sr++) {
        let s = K - 1 - sr;
        let j = select(s, K - 1 - s, p.trans != 0);
        let uOff = j * D;
        var qv: f32 = 0.0;
        for (var c: i32 = 0; c < D; c++) { qv += dx[baseOff + c] * uHat[uOff + c]; }
        pqBuf[pqOff + K + s] = qv;
        for (var c: i32 = 0; c < D; c++) { let step = qv * uHat[uOff + c]; dx[baseOff + c] = (dx[baseOff + c] - step) - step; }
    }
}");

    public static readonly string HouseholderBackwardReflectorsRaw = Shader(
        new[] { "r:dy", "r:x", "r:uHat", "r:pqBuf", "rw:duHat" }, "M: i32, K: i32, D: i32, trans: i32",
        Main + @"
    let M = p.M; let K = p.K; let D = p.D;
    if (i >= D) { return; }
    for (var j: i32 = 0; j < K; j++) { duHat[j * D + i] = 0.0; }
    for (var m: i32 = 0; m < M; m++) {
        let pqOff = m * 2 * K;
        var xv = x[m * D + i];
        for (var s: i32 = 0; s < K; s++) {
            let j = select(s, K - 1 - s, p.trans != 0);
            let ps = pqBuf[pqOff + s];
            let qs = pqBuf[pqOff + K + s];
            let uj = uHat[j * D + i];
            var gv = dy[m * D + i];
            for (var tr: i32 = 0; tr < K - 1 - s; tr++) {
                let t = K - 1 - tr;
                let jt = select(t, K - 1 - t, p.trans != 0);
                let stepQ = pqBuf[pqOff + K + t] * uHat[jt * D + i];
                gv = (gv - stepQ) - stepQ;
            }
            let duTerm = qs * xv + ps * gv;
            duHat[j * D + i] = (duHat[j * D + i] - duTerm) - duTerm;
            let stepP = ps * uj;
            xv = (xv - stepP) - stepP;
        }
    }
}");

    public static readonly string HouseholderBackwardReflectorsNorm = Shader(
        new[] { "r:duHat", "r:uHat", "r:hhStats", "rw:dU" }, "K: i32, D: i32",
        Main + @"
    let K = p.K; let D = p.D;
    if (i >= K) { return; }
    let invNorm = hhStats[2 * i];
    let isActive = hhStats[2 * i + 1];
    let off = i * D;
    var dot: f32 = 0.0;
    for (var c: i32 = 0; c < D; c++) { dot += duHat[off + c] * uHat[off + c]; }
    for (var c: i32 = 0; c < D; c++) { dU[off + c] = invNorm * (duHat[off + c] - isActive * dot * uHat[off + c]); }
}");
}
#endif
