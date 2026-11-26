#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    public static readonly string SmxGivensForward = Shader(
        new[] { "r:x", "r:angles", "rw:y" }, "totalHalf: i32, halfDim: i32",
        Main + @"
    let totalHalf = p.totalHalf; let halfDim = p.halfDim;
    if (i >= totalHalf) { return; }
    let m = i / halfDim;
    let col = i - m * halfDim;
    let rowOff = m * (2 * halfDim);
    let phi = angles[col];
    let c = cos(phi);
    let s = sin(phi);
    let x0 = x[rowOff + col];
    let x1 = x[rowOff + halfDim + col];
    y[rowOff + col] = x0 * c - x1 * s;
    y[rowOff + halfDim + col] = x0 * s + x1 * c;
}");

    public static readonly string SmxGivensBackwardDx = Shader(
        new[] { "r:dy", "r:angles", "rw:dx" }, "totalHalf: i32, halfDim: i32",
        Main + @"
    let totalHalf = p.totalHalf; let halfDim = p.halfDim;
    if (i >= totalHalf) { return; }
    let m = i / halfDim;
    let col = i - m * halfDim;
    let rowOff = m * (2 * halfDim);
    let phi = angles[col];
    let c = cos(phi);
    let s = sin(phi);
    let dy0 = dy[rowOff + col];
    let dy1 = dy[rowOff + halfDim + col];
    dx[rowOff + col] = dy0 * c + dy1 * s;
    dx[rowOff + halfDim + col] = -dy0 * s + dy1 * c;
}");

    public static readonly string SmxGivensBackwardDphi = Shader(
        new[] { "r:dy", "r:x", "r:angles", "rw:dAngles" }, "rows: i32, halfDim: i32",
        Main + @"
    let rows = p.rows; let halfDim = p.halfDim;
    if (i >= halfDim) { return; }
    let phi = angles[i];
    let c = cos(phi);
    let s = sin(phi);
    let dim = 2 * halfDim;
    var acc: f32 = 0.0;
    for (var m: i32 = 0; m < rows; m++) {
        let rowOff = m * dim;
        let x0 = x[rowOff + i];
        let x1 = x[rowOff + halfDim + i];
        let dy0 = dy[rowOff + i];
        let dy1 = dy[rowOff + halfDim + i];
        acc += dy0 * (-x0 * s - x1 * c) + dy1 * (x0 * c - x1 * s);
    }
    dAngles[i] = acc;
}");

    public static readonly string SmxConv2TapForward = Shader(
        new[] { "r:x", "r:w0", "r:w1", "rw:y" }, "total: i32, seqLen: i32, channels: i32",
        Main + @"
    let total = p.total; let seqLen = p.seqLen; let channels = p.channels;
    if (i >= total) { return; }
    let c = i % channels;
    let bt = i / channels;
    let t = bt % seqLen;
    var res: f32 = w0[c] * x[i];
    if (t > 0) { res += w1[c] * x[i - channels]; }
    y[i] = res;
}");

    public static readonly string SmxConv2TapBackwardDx = Shader(
        new[] { "r:dy", "r:w0", "r:w1", "rw:dx" }, "total: i32, seqLen: i32, channels: i32",
        Main + @"
    let total = p.total; let seqLen = p.seqLen; let channels = p.channels;
    if (i >= total) { return; }
    let c = i % channels;
    let bt = i / channels;
    let t = bt % seqLen;
    var res: f32 = w0[c] * dy[i];
    if (t + 1 < seqLen) { res += w1[c] * dy[i + channels]; }
    dx[i] = res;
}");

    public static readonly string SmxConv2TapBackwardDw = Shader(
        new[] { "r:dy", "r:x", "rw:dw0", "rw:dw1" }, "batch: i32, seqLen: i32, channels: i32",
        Main + @"
    let batch = p.batch; let seqLen = p.seqLen; let channels = p.channels;
    if (i >= channels) { return; }
    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    for (var b: i32 = 0; b < batch; b++) {
        let bOff = b * seqLen * channels + i;
        for (var t: i32 = 0; t < seqLen; t++) {
            let cur = bOff + t * channels;
            let dyv = dy[cur];
            a0 += dyv * x[cur];
            if (t > 0) { a1 += dyv * x[cur - channels]; }
        }
    }
    dw0[i] = a0;
    dw1[i] = a1;
}");

    public static readonly string SmxRopeL2Forward = Shader(
        new[] { "r:x", "r:cosBuf", "r:sinBuf", "rw:y" }, "totalHeads: i32, seqLen: i32, numHeads: i32, headDim: i32, startPos: i32, epsBits: i32, scaleBits: i32",
        Main + @"
    if (i >= p.totalHeads) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let postScale = bitcast<f32>(p.scaleBits);
    let bt = i / p.numHeads;
    let t = bt % p.seqLen;
    let half = p.headDim / 2;
    let cOff = (p.startPos + t) * half;
    let xOff = i * p.headDim;
    var normSq: f32 = 0.0;
    for (var pr: i32 = 0; pr < half; pr++) {
        let x0 = x[xOff + 2 * pr];
        let x1 = x[xOff + 2 * pr + 1];
        let c = cosBuf[cOff + pr];
        let s = sinBuf[cOff + pr];
        let r0 = x0 * c - x1 * s;
        let r1 = x0 * s + x1 * c;
        normSq += r0 * r0 + r1 * r1;
    }
    let sc = postScale / sqrt(normSq + eps);
    for (var pr: i32 = 0; pr < half; pr++) {
        let x0 = x[xOff + 2 * pr];
        let x1 = x[xOff + 2 * pr + 1];
        let c = cosBuf[cOff + pr];
        let s = sinBuf[cOff + pr];
        y[xOff + 2 * pr] = (x0 * c - x1 * s) * sc;
        y[xOff + 2 * pr + 1] = (x0 * s + x1 * c) * sc;
    }
}");

    public static readonly string SmxRopeL2Backward = Shader(
        new[] { "r:dy", "r:x", "r:cosBuf", "r:sinBuf", "rw:dx" }, "totalHeads: i32, seqLen: i32, numHeads: i32, headDim: i32, startPos: i32, epsBits: i32, scaleBits: i32",
        Main + @"
    if (i >= p.totalHeads) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let postScale = bitcast<f32>(p.scaleBits);
    let bt = i / p.numHeads;
    let t = bt % p.seqLen;
    let half = p.headDim / 2;
    let cOff = (p.startPos + t) * half;
    let xOff = i * p.headDim;
    var normSq: f32 = 0.0;
    for (var pr: i32 = 0; pr < half; pr++) {
        let x0 = x[xOff + 2 * pr];
        let x1 = x[xOff + 2 * pr + 1];
        let c = cosBuf[cOff + pr];
        let s = sinBuf[cOff + pr];
        let r0 = x0 * c - x1 * s;
        let r1 = x0 * s + x1 * c;
        normSq += r0 * r0 + r1 * r1;
    }
    let invNorm = 1.0 / sqrt(normSq + eps);
    var dot: f32 = 0.0;
    for (var pr: i32 = 0; pr < half; pr++) {
        let x0 = x[xOff + 2 * pr];
        let x1 = x[xOff + 2 * pr + 1];
        let c = cosBuf[cOff + pr];
        let s = sinBuf[cOff + pr];
        let u0 = (x0 * c - x1 * s) * invNorm;
        let u1 = (x0 * s + x1 * c) * invNorm;
        dot += dy[xOff + 2 * pr] * u0 + dy[xOff + 2 * pr + 1] * u1;
    }
    let sc = postScale * invNorm;
    for (var pr: i32 = 0; pr < half; pr++) {
        let x0 = x[xOff + 2 * pr];
        let x1 = x[xOff + 2 * pr + 1];
        let c = cosBuf[cOff + pr];
        let s = sinBuf[cOff + pr];
        let u0 = (x0 * c - x1 * s) * invNorm;
        let u1 = (x0 * s + x1 * c) * invNorm;
        let dr0 = sc * (dy[xOff + 2 * pr] - u0 * dot);
        let dr1 = sc * (dy[xOff + 2 * pr + 1] - u1 * dot);
        dx[xOff + 2 * pr] = dr0 * c + dr1 * s;
        dx[xOff + 2 * pr + 1] = -dr0 * s + dr1 * c;
    }
}");

    public static readonly string SmxRopeQuadForward = Shader(
        new[] { "r:x", "r:cosBuf", "r:sinBuf", "rw:y" }, "totalHeads: i32, seqLen: i32, numHeads: i32, mb: i32, outDim: i32, startPos: i32",
        Main + @"
    if (i >= p.totalHeads) { return; }
    let bt = i / p.numHeads;
    let t = bt % p.seqLen;
    let half = p.mb / 2;
    let cOff = (p.startPos + t) * half;
    let xOff = i * p.mb;
    let yOff = i * p.outDim;
    let mb2 = p.mb * p.mb;
    for (var u: i32 = 0; u < p.mb; u++) {
        let pu = u / 2;
        let xu0 = x[xOff + 2 * pu];
        let xu1 = x[xOff + 2 * pu + 1];
        let cu = cosBuf[cOff + pu];
        let su = sinBuf[cOff + pu];
        let ru = select(xu0 * su + xu1 * cu, xu0 * cu - xu1 * su, (u - 2 * pu) == 0);
        for (var v: i32 = 0; v < p.mb; v++) {
            let pv = v / 2;
            let xv0 = x[xOff + 2 * pv];
            let xv1 = x[xOff + 2 * pv + 1];
            let cv = cosBuf[cOff + pv];
            let sv = sinBuf[cOff + pv];
            let rv = select(xv0 * sv + xv1 * cv, xv0 * cv - xv1 * sv, (v - 2 * pv) == 0);
            y[yOff + u * p.mb + v] = ru * rv;
        }
    }
    for (var k: i32 = mb2; k < p.outDim; k++) { y[yOff + k] = 0.0; }
}");

    public static readonly string SmxRopeQuadBackward = Shader(
        new[] { "r:dy", "r:x", "r:cosBuf", "r:sinBuf", "rw:dx" }, "totalHeads: i32, seqLen: i32, numHeads: i32, mb: i32, outDim: i32, startPos: i32",
        Main + @"
    if (i >= p.totalHeads) { return; }
    let bt = i / p.numHeads;
    let t = bt % p.seqLen;
    let half = p.mb / 2;
    let cOff = (p.startPos + t) * half;
    let xOff = i * p.mb;
    let yOff = i * p.outDim;
    for (var pr: i32 = 0; pr < half; pr++) {
        var dr0: f32 = 0.0;
        var dr1: f32 = 0.0;
        let u0 = 2 * pr;
        let u1 = 2 * pr + 1;
        for (var v: i32 = 0; v < p.mb; v++) {
            let pv = v / 2;
            let xv0 = x[xOff + 2 * pv];
            let xv1 = x[xOff + 2 * pv + 1];
            let cv = cosBuf[cOff + pv];
            let sv = sinBuf[cOff + pv];
            let rv = select(xv0 * sv + xv1 * cv, xv0 * cv - xv1 * sv, (v - 2 * pv) == 0);
            dr0 += (dy[yOff + u0 * p.mb + v] + dy[yOff + v * p.mb + u0]) * rv;
            dr1 += (dy[yOff + u1 * p.mb + v] + dy[yOff + v * p.mb + u1]) * rv;
        }
        let cp = cosBuf[cOff + pr];
        let sp = sinBuf[cOff + pr];
        dx[xOff + u0] = dr0 * cp + dr1 * sp;
        dx[xOff + u1] = -dr0 * sp + dr1 * cp;
    }
}");

    public static readonly string SmxProjResForward = Shader(
        new[] { "r:h", "r:u", "r:gateLogit", "rw:y" }, "rows: i32, dim: i32, epsBits: i32",
        Main + @"
    if (i >= p.rows) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let g = 1.0 / (1.0 + exp(-gateLogit[0]));
    let beta = 1.0 - g;
    let off = i * p.dim;
    var dot: f32 = 0.0;
    var normSq: f32 = 0.0;
    for (var j: i32 = 0; j < p.dim; j++) {
        let hj = h[off + j];
        dot += u[off + j] * hj;
        normSq += hj * hj;
    }
    let coeff = dot / (normSq + eps);
    let scaleH = 1.0 - beta * coeff;
    for (var j: i32 = 0; j < p.dim; j++) {
        y[off + j] = u[off + j] + scaleH * h[off + j];
    }
}");

    public static readonly string SmxProjResBackwardRows = Shader(
        new[] { "r:dy", "r:h", "r:u", "r:gateLogit", "rw:dh", "rw:du", "rw:rowTerms" }, "rows: i32, dim: i32, epsBits: i32",
        Main + @"
    if (i >= p.rows) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let g = 1.0 / (1.0 + exp(-gateLogit[0]));
    let beta = 1.0 - g;
    let off = i * p.dim;
    var dot: f32 = 0.0;
    var normSq: f32 = 0.0;
    var pm: f32 = 0.0;
    for (var j: i32 = 0; j < p.dim; j++) {
        let hj = h[off + j];
        let uj = u[off + j];
        let dyj = dy[off + j];
        dot += uj * hj;
        normSq += hj * hj;
        pm += dyj * hj;
    }
    let denom = normSq + eps;
    let coeff = dot / denom;
    let scaleH = 1.0 - beta * coeff;
    let kFactor = (beta * pm) / denom;
    rowTerms[i] = coeff * pm;
    for (var j: i32 = 0; j < p.dim; j++) {
        let hj = h[off + j];
        let uj = u[off + j];
        let dyj = dy[off + j];
        du[off + j] = dyj - kFactor * hj;
        dh[off + j] = scaleH * dyj - kFactor * (uj - 2.0 * coeff * hj);
    }
}");

    public static readonly string SmxProjResBackwardDgate = Shader(
        new[] { "r:rowTerms", "r:gateLogit", "rw:dGateLogit" }, "rows: i32",
        Main + @"
    if (i >= 1) { return; }
    let g = 1.0 / (1.0 + exp(-gateLogit[0]));
    var acc: f32 = 0.0;
    for (var m: i32 = 0; m < p.rows; m++) { acc += rowTerms[m]; }
    dGateLogit[0] = g * (1.0 - g) * acc;
}");

    public static readonly string SmxSpecCircH = Shader(
        new[] { "r:specRe", "r:specIm", "rw:hTime" }, "batch: i32, dim: i32",
        Main + @"
    if (i >= p.batch * p.dim) { return; }
    let b = i / p.dim;
    let m = i - b * p.dim;
    let half = p.dim / 2;
    let nf = half + 1;
    let specOff = b * nf;
    let dimF = f32(p.dim);
    let twoPiOverD = 6.2831853071795864769 / dimF;
    var signNyq: f32 = -1.0;
    if ((m - 2 * (m / 2)) == 0) { signNyq = 1.0; }
    var sum: f32 = specRe[specOff] + signNyq * specRe[specOff + half];
    for (var f: i32 = 1; f < half; f++) {
        let fm = (f * m) % p.dim;
        let ang = twoPiOverD * f32(fm);
        sum += 2.0 * (specRe[specOff + f] * cos(ang) - specIm[specOff + f] * sin(ang));
    }
    hTime[i] = sum / dimF;
}");

    public static readonly string SmxSpecCircW = Shader(
        new[] { "r:diagScale", "r:hTime", "rw:outMat" }, "batch: i32, dim: i32",
        Main + @"
    let dd = p.dim * p.dim;
    if (i >= p.batch * dd) { return; }
    let b = i / dd;
    let rem = i - b * dd;
    let r = rem / p.dim;
    let c = rem - r * p.dim;
    let shift = (c - r + p.dim) % p.dim;
    outMat[i] = diagScale[b * p.dim + r] * hTime[b * p.dim + shift];
}");

    public static readonly string SmxSpecCircBackwardDhDd = Shader(
        new[] { "r:dOut", "r:diagScale", "r:hTime", "rw:dDiag", "rw:dhTime" }, "batch: i32, dim: i32",
        Main + @"
    if (i >= p.batch * p.dim) { return; }
    let b = i / p.dim;
    let idx = i - b * p.dim;
    let matBase = b * p.dim * p.dim;
    let vecBase = b * p.dim;
    var accDiag: f32 = 0.0;
    var accH: f32 = 0.0;
    for (var k: i32 = 0; k < p.dim; k++) {
        let shift = (k - idx + p.dim) % p.dim;
        accDiag += dOut[matBase + idx * p.dim + k] * hTime[vecBase + shift];
        let col = (k + idx) % p.dim;
        accH += dOut[matBase + k * p.dim + col] * diagScale[vecBase + k];
    }
    dDiag[i] = accDiag;
    dhTime[i] = accH;
}");

    public static readonly string SmxSpecCircBackwardDspec = Shader(
        new[] { "r:dhTime", "rw:dRe", "rw:dIm" }, "batch: i32, dim: i32",
        Main + @"
    let half = p.dim / 2;
    let nf = half + 1;
    if (i >= p.batch * nf) { return; }
    let b = i / nf;
    let f = i - b * nf;
    let vecBase = b * p.dim;
    let dimF = f32(p.dim);
    let invD = 1.0 / dimF;
    if (f == 0) {
        var s0: f32 = 0.0;
        for (var m: i32 = 0; m < p.dim; m++) { s0 += dhTime[vecBase + m]; }
        dRe[i] = s0 * invD;
        dIm[i] = 0.0;
    } else if (f == half) {
        var sn: f32 = 0.0;
        for (var m: i32 = 0; m < p.dim; m++) {
            var sgn: f32 = -1.0;
            if ((m - 2 * (m / 2)) == 0) { sgn = 1.0; }
            sn += sgn * dhTime[vecBase + m];
        }
        dRe[i] = sn * invD;
        dIm[i] = 0.0;
    } else {
        let twoPiOverD = 6.2831853071795864769 / dimF;
        var accRe: f32 = 0.0;
        var accIm: f32 = 0.0;
        for (var m: i32 = 0; m < p.dim; m++) {
            let fm = (f * m) % p.dim;
            let ang = twoPiOverD * f32(fm);
            let dhv = dhTime[vecBase + m];
            accRe += dhv * cos(ang);
            accIm -= dhv * sin(ang);
        }
        let scale = 2.0 * invD;
        dRe[i] = accRe * scale;
        dIm[i] = accIm * scale;
    }
}");

    public static readonly string SmxSigTopKMixForward = Shader(
        new[] { "r:logits", "r:expertTable", "r:routingBias", "rw:y", "rw:savedInfo" },
        "rows: i32, numExperts: i32, topK: i32, channels: i32, epsBits: i32",
        Main + @"
    if (i >= p.rows) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let logOff = i * p.numExperts;
    let strideInfo = 2 * p.topK + 1;
    let infoOff = i * strideInfo;
    var sumSel: f32 = 0.0;
    for (var r: i32 = 0; r < p.topK; r++) {
        var bestIdx: i32 = -1;
        var bestScore: f32 = 0.0;
        var bestSig: f32 = 0.0;
        for (var j: i32 = 0; j < p.numExperts; j++) {
            var already: i32 = 0;
            for (var prev: i32 = 0; prev < r; prev++) {
                if (i32(savedInfo[infoOff + prev]) == j) { already = 1; }
            }
            if (already == 0) {
                let sj = 1.0 / (1.0 + exp(-logits[logOff + j]));
                let bj = sj + routingBias[j];
                if (bestIdx < 0 || bj > bestScore) {
                    bestIdx = j;
                    bestScore = bj;
                    bestSig = sj;
                }
            }
        }
        savedInfo[infoOff + r] = f32(bestIdx);
        savedInfo[infoOff + p.topK + r] = bestSig;
        sumSel += bestSig;
    }
    savedInfo[infoOff + 2 * p.topK] = sumSel;
    var denom = eps;
    if (sumSel >= eps) { denom = sumSel; }
    for (var r: i32 = 0; r < p.topK; r++) {
        savedInfo[infoOff + p.topK + r] = savedInfo[infoOff + p.topK + r] / denom;
    }
    let yOff = i * p.channels;
    for (var c: i32 = 0; c < p.channels; c++) {
        var acc: f32 = 0.0;
        for (var r: i32 = 0; r < p.topK; r++) {
            let eIdx = i32(savedInfo[infoOff + r]);
            acc += savedInfo[infoOff + p.topK + r] * expertTable[eIdx * p.channels + c];
        }
        y[yOff + c] = acc;
    }
}");

    public static readonly string SmxSigTopKBiasUpdate = Shader(
        new[] { "r:savedInfo", "rw:routingBias" }, "rows: i32, numExperts: i32, topK: i32, rateBits: i32",
        Main + @"
    if (i >= p.numExperts) { return; }
    let rate = bitcast<f32>(p.rateBits);
    let strideInfo = 2 * p.topK + 1;
    var cnt: i32 = 0;
    for (var m: i32 = 0; m < p.rows; m++) {
        let infoOff = m * strideInfo;
        for (var r: i32 = 0; r < p.topK; r++) {
            if (i32(savedInfo[infoOff + r]) == i) { cnt += 1; }
        }
    }
    let diff = (f32(p.rows * p.topK) / f32(p.numExperts)) - f32(cnt);
    var sgn: f32 = 0.0;
    if (diff > 0.0) { sgn = 1.0; } else if (diff < 0.0) { sgn = -1.0; }
    routingBias[i] = routingBias[i] + rate * sgn;
}");

    public static readonly string SmxSigTopKMixBackwardDlogits = Shader(
        new[] { "r:dy", "r:logits", "r:expertTable", "r:savedInfo", "rw:dLogits" },
        "rows: i32, numExperts: i32, topK: i32, channels: i32, epsBits: i32",
        Main + @"
    if (i >= p.rows) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let logOff = i * p.numExperts;
    for (var j: i32 = 0; j < p.numExperts; j++) { dLogits[logOff + j] = 0.0; }
    let strideInfo = 2 * p.topK + 1;
    let infoOff = i * strideInfo;
    let yOff = i * p.channels;
    let sumSel = savedInfo[infoOff + 2 * p.topK];
    var dotDg: f32 = 0.0;
    for (var r: i32 = 0; r < p.topK; r++) {
        let eIdx = i32(savedInfo[infoOff + r]);
        let gateR = savedInfo[infoOff + p.topK + r];
        var dGate: f32 = 0.0;
        for (var c: i32 = 0; c < p.channels; c++) {
            dGate += dy[yOff + c] * expertTable[eIdx * p.channels + c];
        }
        dotDg += dGate * gateR;
    }
    for (var r: i32 = 0; r < p.topK; r++) {
        let eIdx = i32(savedInfo[infoOff + r]);
        var dGate: f32 = 0.0;
        for (var c: i32 = 0; c < p.channels; c++) {
            dGate += dy[yOff + c] * expertTable[eIdx * p.channels + c];
        }
        var dSel = dGate / eps;
        if (sumSel >= eps) { dSel = (dGate - dotDg) / sumSel; }
        let sj = 1.0 / (1.0 + exp(-logits[logOff + eIdx]));
        dLogits[logOff + eIdx] = dSel * sj * (1.0 - sj);
    }
}");

    public static readonly string SmxSigTopKMixBackwardDtable = Shader(
        new[] { "r:dy", "r:savedInfo", "rw:dTable" }, "rows: i32, numExperts: i32, topK: i32, channels: i32",
        Main + @"
    if (i >= p.numExperts * p.channels) { return; }
    let e = i / p.channels;
    let c = i - e * p.channels;
    let strideInfo = 2 * p.topK + 1;
    var acc: f32 = 0.0;
    for (var m: i32 = 0; m < p.rows; m++) {
        let infoOff = m * strideInfo;
        let dyc = dy[m * p.channels + c];
        for (var r: i32 = 0; r < p.topK; r++) {
            if (i32(savedInfo[infoOff + r]) == e) { acc += savedInfo[infoOff + p.topK + r] * dyc; }
        }
    }
    dTable[i] = acc;
}");

    public static readonly string SmxDualSwiGluForward = Shader(
        new[] { "r:guProj", "r:guGain", "r:upBase", "rw:y" }, "rows: i32, hiddenDim: i32",
        Main + @"
    if (i >= p.rows * p.hiddenDim) { return; }
    let m = i / p.hiddenDim;
    let j = i - m * p.hiddenDim;
    let off2 = m * (2 * p.hiddenDim);
    let gRaw = guProj[off2 + j];
    let uRaw = guProj[off2 + p.hiddenDim + j];
    let gGain = 1.0 + guGain[off2 + j];
    let uGain = upBase[j] + guGain[off2 + p.hiddenDim + j];
    let g = gRaw * gGain;
    let sig = 1.0 / (1.0 + exp(-g));
    y[i] = (g * sig * uRaw) * uGain;
}");

    public static readonly string SmxDualSwiGluBackwardElem = Shader(
        new[] { "r:dy", "r:guProj", "r:guGain", "r:upBase", "rw:dGuProj", "rw:dGuGain" }, "rows: i32, hiddenDim: i32",
        Main + @"
    if (i >= p.rows * p.hiddenDim) { return; }
    let m = i / p.hiddenDim;
    let j = i - m * p.hiddenDim;
    let off2 = m * (2 * p.hiddenDim);
    let dyv = dy[i];
    let gRaw = guProj[off2 + j];
    let uRaw = guProj[off2 + p.hiddenDim + j];
    let gGain = 1.0 + guGain[off2 + j];
    let uGain = upBase[j] + guGain[off2 + p.hiddenDim + j];
    let g = gRaw * gGain;
    let sig = 1.0 / (1.0 + exp(-g));
    let silu = g * sig;
    let h = silu * uRaw;
    dGuGain[off2 + p.hiddenDim + j] = dyv * h;
    let dh = dyv * uGain;
    dGuProj[off2 + p.hiddenDim + j] = dh * silu;
    let dg = (dh * uRaw) * sig * (1.0 + g * (1.0 - sig));
    dGuProj[off2 + j] = dg * gGain;
    dGuGain[off2 + j] = dg * gRaw;
}");

    public static readonly string SmxDualSwiGluBackwardDbase = Shader(
        new[] { "r:dy", "r:guProj", "r:guGain", "rw:dUpBase" }, "rows: i32, hiddenDim: i32",
        Main + @"
    if (i >= p.hiddenDim) { return; }
    let stride2 = 2 * p.hiddenDim;
    var acc: f32 = 0.0;
    for (var m: i32 = 0; m < p.rows; m++) {
        let off2 = m * stride2;
        let g = guProj[off2 + i] * (1.0 + guGain[off2 + i]);
        let sig = 1.0 / (1.0 + exp(-g));
        acc += dy[m * p.hiddenDim + i] * (g * sig * guProj[off2 + p.hiddenDim + i]);
    }
    dUpBase[i] = acc;
}");

    public static readonly string SmxBornRatioForward = Shader(
        new[] { "r:scanOut", "r:sinkLogAlpha", "rw:y" },
        "totalRows: i32, numHeads: i32, headDim: i32, scanDim: i32, hasSink: i32, epsBits: i32",
        Main + @"
    if (i >= p.totalRows * p.headDim) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let row = i / p.headDim;
    let j = i - row * p.headDim;
    let h = row % p.numHeads;
    var sinkTerm: f32 = 0.0;
    if (p.hasSink != 0) { sinkTerm = exp(sinkLogAlpha[h]); }
    let den = scanOut[row * p.scanDim + p.headDim] + eps + sinkTerm;
    y[i] = scanOut[row * p.scanDim + j] / den;
}");

    public static readonly string SmxBornRatioBackwardDscan = Shader(
        new[] { "r:dy", "r:scanOut", "r:sinkLogAlpha", "rw:dScan" },
        "totalRows: i32, numHeads: i32, headDim: i32, scanDim: i32, hasSink: i32, epsBits: i32",
        Main + @"
    if (i >= p.totalRows) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let h = i % p.numHeads;
    var sinkTerm: f32 = 0.0;
    if (p.hasSink != 0) { sinkTerm = exp(sinkLogAlpha[h]); }
    let sOff = i * p.scanDim;
    let yOff = i * p.headDim;
    let invDen = 1.0 / (scanOut[sOff + p.headDim] + eps + sinkTerm);
    var dot: f32 = 0.0;
    for (var j: i32 = 0; j < p.headDim; j++) {
        let dyj = dy[yOff + j];
        dot += dyj * scanOut[sOff + j];
        dScan[sOff + j] = dyj * invDen;
    }
    dScan[sOff + p.headDim] = -dot * invDen * invDen;
    for (var j: i32 = p.headDim + 1; j < p.scanDim; j++) { dScan[sOff + j] = 0.0; }
}");

    public static readonly string SmxBornRatioBackwardDsink = Shader(
        new[] { "r:dy", "r:scanOut", "r:sinkLogAlpha", "rw:dSink" },
        "totalRows: i32, numHeads: i32, headDim: i32, scanDim: i32, epsBits: i32",
        Main + @"
    if (i >= p.numHeads) { return; }
    let eps = bitcast<f32>(p.epsBits);
    let sinkTerm = exp(sinkLogAlpha[i]);
    var acc: f32 = 0.0;
    for (var row: i32 = i; row < p.totalRows; row += p.numHeads) {
        let sOff = row * p.scanDim;
        let yOff = row * p.headDim;
        let invDen = 1.0 / (scanOut[sOff + p.headDim] + eps + sinkTerm);
        var dot: f32 = 0.0;
        for (var j: i32 = 0; j < p.headDim; j++) { dot += dy[yOff + j] * scanOut[sOff + j]; }
        acc += -dot * invDen * invDen;
    }
    dSink[i] = acc * sinkTerm;
}");

    public static readonly string SmxBoundSigForward = Shader(
        new[] { "r:logits", "r:bias", "rw:y" }, "rows: i32, cols: i32, floorBits: i32",
        Main + @"
    if (i >= p.rows * p.cols) { return; }
    let flr = bitcast<f32>(p.floorBits);
    let c = i % p.cols;
    let sig = 1.0 / (1.0 + exp(-(logits[i] + bias[c])));
    y[i] = flr + (1.0 - flr) * sig;
}");

    public static readonly string SmxBoundSigBackwardDlog = Shader(
        new[] { "r:dy", "r:logits", "r:bias", "rw:dLogits" }, "rows: i32, cols: i32, floorBits: i32",
        Main + @"
    if (i >= p.rows * p.cols) { return; }
    let flr = bitcast<f32>(p.floorBits);
    let c = i % p.cols;
    let sig = 1.0 / (1.0 + exp(-(logits[i] + bias[c])));
    dLogits[i] = dy[i] * (1.0 - flr) * sig * (1.0 - sig);
}");

    public static readonly string SmxBoundSigBackwardDbias = Shader(
        new[] { "r:dy", "r:logits", "r:bias", "rw:dBias" }, "rows: i32, cols: i32, floorBits: i32",
        Main + @"
    if (i >= p.cols) { return; }
    let flr = bitcast<f32>(p.floorBits);
    let scale = 1.0 - flr;
    let bc = bias[i];
    var acc: f32 = 0.0;
    for (var r: i32 = 0; r < p.rows; r++) {
        let idx = r * p.cols + i;
        let sig = 1.0 / (1.0 + exp(-(logits[idx] + bc)));
        acc += dy[idx] * scale * sig * (1.0 - sig);
    }
    dBias[i] = acc;
}");
}
#endif
