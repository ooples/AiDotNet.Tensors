// Copyright (c) AiDotNet. All rights reserved.
// WebGPU (WGSL) compute shaders for the fused recurrence / LM-head ops (#1464). Real GPU kernels —
// mirror the HIP/CUDA/OpenCL/MSL/GLSL kernels and the CpuEngine math exactly. Int params come in
// via a uniform struct bound after the storage buffers. Forward only — the differentiable backward
// runs through the CpuEngine tape (the engine override is forward-only).

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static class WebGpuRecurrenceKernels
{
    public const string GlaScan = @"
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Vd: array<f32>;
@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
struct P { batch: i32, seqLen: i32, modelDim: i32, numHeads: i32, headDim: i32 };
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let MD = p.modelDim; let NH = p.numHeads; let HD = p.headDim;
    if (gid >= B*NH*HD) { return; }
    let di = gid % HD; let h = (gid/HD) % NH; let b = gid/(HD*NH); let hOff = h*HD;
    var Srow: array<f32, 256>;
    for (var ki: i32 = 0; ki < HD; ki = ki+1) { Srow[ki] = 0.0; }
    for (var t: i32 = 0; t < SL; t = t+1) {
        let baseOff = (b*SL+t)*MD + hOff;
        let g = G[(b*SL+t)*NH + h];
        let vd = Vd[baseOff+di];
        var o: f32 = 0.0;
        for (var ki: i32 = 0; ki < HD; ki = ki+1) { let s = g*Srow[ki] + vd*K[baseOff+ki]; Srow[ki] = s; o = o + s*Q[baseOff+ki]; }
        outp[baseOff+di] = o;
    }
}";

    public const string XLstmScan = @"
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Vd: array<f32>;
@group(0) @binding(3) var<storage, read> Ig: array<f32>;
@group(0) @binding(4) var<storage, read> Fg: array<f32>;
@group(0) @binding(5) var<storage, read> Og: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;
struct P { batch: i32, seqLen: i32, modelDim: i32, numHeads: i32, headDim: i32 };
@group(0) @binding(7) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let MD = p.modelDim; let NH = p.numHeads; let HD = p.headDim;
    if (gid >= B*NH) { return; }
    let h = gid % NH; let b = gid / NH; let hOff = h*HD; let hh = HD*HD;
    let kappa = 1.0/sqrt(f32(HD));
    var C: array<f32, 4096>;
    var n: array<f32, 64>;
    for (var i: i32 = 0; i < hh; i = i+1) { C[i] = 0.0; }
    for (var i: i32 = 0; i < HD; i = i+1) { n[i] = 0.0; }
    for (var t: i32 = 0; t < SL; t = t+1) {
        let baseOff = (b*SL+t)*MD + hOff; let gOff = (b*SL+t)*NH + h;
        var iv = Ig[gOff]; if (iv > 4.85e8) { iv = 4.85e8; }
        let f = Fg[gOff]; let o = Og[gOff];
        for (var di: i32 = 0; di < HD; di = di+1) {
            n[di] = f*n[di] + iv*(K[baseOff+di]*kappa);
            let vv = Vd[baseOff+di]; let srow = di*HD;
            for (var ki: i32 = 0; ki < HD; ki = ki+1) { C[srow+ki] = f*C[srow+ki] + iv*vv*(K[baseOff+ki]*kappa); }
        }
        var nq: f32 = 0.0; for (var j: i32 = 0; j < HD; j = j+1) { nq = nq + n[j]*Q[baseOff+j]; }
        var nf = abs(nq); if (nf < 1.0) { nf = 1.0; }
        for (var di: i32 = 0; di < HD; di = di+1) {
            let srow = di*HD; var num: f32 = 0.0;
            for (var ki: i32 = 0; ki < HD; ki = ki+1) { num = num + C[srow+ki]*Q[baseOff+ki]; }
            outp[baseOff+di] = o*num/nf;
        }
    }
}";

    public const string GatedDeltaScan = @"
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Vd: array<f32>;
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read> Bt: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
struct P { batch: i32, seqLen: i32, modelDim: i32, numHeads: i32, headDim: i32 };
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let MD = p.modelDim; let NH = p.numHeads; let HD = p.headDim;
    if (gid >= B*NH*HD) { return; }
    let di = gid % HD; let h = (gid/HD) % NH; let b = gid/(HD*NH); let hOff = h*HD;
    let kappa = 1.0/sqrt(f32(HD));
    var Srow: array<f32, 256>;
    for (var ki: i32 = 0; ki < HD; ki = ki+1) { Srow[ki] = 0.0; }
    for (var t: i32 = 0; t < SL; t = t+1) {
        let baseOff = (b*SL+t)*MD + hOff; let gIdx = (b*SL+t)*NH + h;
        let a = A[gIdx]; let bet = Bt[gIdx];
        var sK: f32 = 0.0; for (var ki: i32 = 0; ki < HD; ki = ki+1) { sK = sK + Srow[ki]*(K[baseOff+ki]*kappa); }
        let bd = bet*(Vd[baseOff+di] - sK);
        for (var ki: i32 = 0; ki < HD; ki = ki+1) { Srow[ki] = a*Srow[ki] + bd*(K[baseOff+ki]*kappa); }
        var o: f32 = 0.0; for (var ki: i32 = 0; ki < HD; ki = ki+1) { o = o + Srow[ki]*Q[baseOff+ki]; }
        outp[baseOff+di] = o;
    }
}";

    public const string RgLruScan = @"
@group(0) @binding(0) var<storage, read> Vd: array<f32>;
@group(0) @binding(1) var<storage, read> R: array<f32>;
@group(0) @binding(2) var<storage, read> Ig: array<f32>;
@group(0) @binding(3) var<storage, read> decayb: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
struct P { batch: i32, seqLen: i32, recDim: i32 };
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let RD = p.recDim;
    if (gid >= B*RD) { return; }
    let c = gid % RD; let b = gid / RD;
    let base = 1.0/(1.0+exp(decayb[c]));
    var h: f32 = 0.0;
    for (var t: i32 = 0; t < SL; t = t+1) {
        let off = (b*SL+t)*RD + c;
        let a = R[off]*base; let om = 1.0 - a*a;
        var s: f32 = 0.0; if (om > 0.0) { s = sqrt(om); }
        h = a*h + s*(Ig[off]*Vd[off]); outp[off] = h;
    }
}";

    public const string Rwkv4Wkv = @"
@group(0) @binding(0) var<storage, read> R: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Vd: array<f32>;
@group(0) @binding(3) var<storage, read> timeDecay: array<f32>;
@group(0) @binding(4) var<storage, read> timeFirst: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
struct P { batch: i32, seqLen: i32, modelDim: i32 };
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let MD = p.modelDim;
    if (gid >= B*MD) { return; }
    let c = gid % MD; let b = gid / MD;
    let w = -exp(timeDecay[c]); let u = timeFirst[c];
    var aa: f32 = 0.0; var bb: f32 = 0.0; var pp: f32 = -1.0e38;
    for (var t: i32 = 0; t < SL; t = t+1) {
        let off = (b*SL+t)*MD + c; let k = K[off]; let v = Vd[off];
        let ww = u+k; let q = max(pp, ww); let e1 = exp(pp-q); let e2 = exp(ww-q);
        let wkv = (e1*aa + e2*v)/(e1*bb + e2);
        outp[off] = (1.0/(1.0+exp(-R[off])))*wkv;
        let ww2 = pp+w; let q2 = max(ww2, k); let e1b = exp(ww2-q2); let e2b = exp(k-q2);
        aa = e1b*aa + e2b*v; bb = e1b*bb + e2b; pp = q2;
    }
}";

    public const string MambaScan = @"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@group(0) @binding(2) var<storage, read> aLog: array<f32>;
@group(0) @binding(3) var<storage, read> Bt: array<f32>;
@group(0) @binding(4) var<storage, read> Ct: array<f32>;
@group(0) @binding(5) var<storage, read> Dt: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;
struct P { batch: i32, seqLen: i32, innerDim: i32, stateDim: i32 };
@group(0) @binding(7) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let ID = p.innerDim; let SD = p.stateDim;
    if (gid >= B*ID) { return; }
    let di = gid % ID; let b = gid / ID; let hrow = di*SD;
    var negA: array<f32, 256>; var h: array<f32, 256>;
    for (var n: i32 = 0; n < SD; n = n+1) { negA[n] = -exp(aLog[hrow+n]); h[n] = 0.0; }
    for (var t: i32 = 0; t < SL; t = t+1) {
        let baseID = (b*SL+t)*ID; let baseSD = (b*SL+t)*SD;
        let dt = delta[baseID+di]; let xv = X[baseID+di]; var y: f32 = 0.0;
        for (var n: i32 = 0; n < SD; n = n+1) { let aBar = exp(dt*negA[n]); let hv = aBar*h[n] + dt*Bt[baseSD+n]*xv; h[n] = hv; y = y + Ct[baseSD+n]*hv; }
        outp[baseID+di] = y + Dt[di]*xv;
    }
}";

    public const string Mamba2Ssd = @"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@group(0) @binding(2) var<storage, read> aLog: array<f32>;
@group(0) @binding(3) var<storage, read> Bt: array<f32>;
@group(0) @binding(4) var<storage, read> Ct: array<f32>;
@group(0) @binding(5) var<storage, read> Dt: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;
struct P { batch: i32, seqLen: i32, innerDim: i32, numHeads: i32, headDim: i32, sd: i32 };
@group(0) @binding(7) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let ID = p.innerDim; let NH = p.numHeads; let HD = p.headDim; let SD = p.sd;
    if (gid >= B*ID) { return; }
    let flatD = gid % ID; let b = gid / ID; let hi = flatD / HD;
    let negA = -exp(aLog[hi]); let dv = Dt[hi];
    var h: array<f32, 256>;
    for (var n: i32 = 0; n < SD; n = n+1) { h[n] = 0.0; }
    for (var t: i32 = 0; t < SL; t = t+1) {
        let btInner = (b*SL+t)*ID; let btState = (b*SL+t)*SD; let btHead = (b*SL+t)*NH;
        let dt = delta[btHead+hi]; let aBar = exp(dt*negA); let xv = X[btInner+flatD]; var y: f32 = 0.0;
        for (var n: i32 = 0; n < SD; n = n+1) { let hNew = aBar*h[n] + dt*Bt[btState+n]*xv; h[n] = hNew; y = y + Ct[btState+n]*hNew; }
        outp[btInner+flatD] = y + dv*xv;
    }
}";

    public const string FusedCeIndex = @"
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> targetIds: array<f32>;
@group(0) @binding(4) var<storage, read_write> rowLoss: array<f32>;
struct P { N: i32, d: i32, vocab: i32 };
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let r = i32(gid3.x);
    let NN = p.N; let D = p.d; let VC = p.vocab;
    if (r >= NN) { return; }
    let hRow = r*D; let id = i32(targetIds[r] + 0.5);
    var mx: f32 = -1.0e38; var logitTarget: f32 = 0.0;
    for (var vv: i32 = 0; vv < VC; vv = vv+1) { var s = bias[vv]; for (var j: i32 = 0; j < D; j = j+1) { s = s + hidden[hRow+j]*weight[j*VC+vv]; } if (s > mx) { mx = s; } if (vv == id) { logitTarget = s; } }
    var sumExp: f32 = 0.0;
    for (var vv: i32 = 0; vv < VC; vv = vv+1) { var s = bias[vv]; for (var j: i32 = 0; j < D; j = j+1) { s = s + hidden[hRow+j]*weight[j*VC+vv]; } sumExp = sumExp + exp(s - mx); }
    rowLoss[r] = (mx + log(sumExp)) - logitTarget;
}";

    public const string FusedCeDense = @"
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> target: array<f32>;
@group(0) @binding(4) var<storage, read_write> rowLoss: array<f32>;
struct P { N: i32, d: i32, vocab: i32 };
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let r = i32(gid3.x);
    let NN = p.N; let D = p.d; let VC = p.vocab;
    if (r >= NN) { return; }
    let hRow = r*D; let tRow = r*VC;
    var mx: f32 = -1.0e38;
    for (var vv: i32 = 0; vv < VC; vv = vv+1) { var s = bias[vv]; for (var j: i32 = 0; j < D; j = j+1) { s = s + hidden[hRow+j]*weight[j*VC+vv]; } if (s > mx) { mx = s; } }
    var sumExp: f32 = 0.0;
    for (var vv: i32 = 0; vv < VC; vv = vv+1) { var s = bias[vv]; for (var j: i32 = 0; j < D; j = j+1) { s = s + hidden[hRow+j]*weight[j*VC+vv]; } sumExp = sumExp + exp(s - mx); }
    let lse = mx + log(sumExp); var rl: f32 = 0.0;
    for (var vv: i32 = 0; vv < VC; vv = vv+1) { var s = bias[vv]; for (var j: i32 = 0; j < D; j = j+1) { s = s + hidden[hRow+j]*weight[j*VC+vv]; } rl = rl + target[tRow+vv]*(s - lse); }
    rowLoss[r] = -rl;
}";

    // GLA BPTT backward: recompute the state trajectory, then reverse-sweep. WGSL has no float
    // atomics, so dQ/dK/dG are bound as atomic<u32> and accumulated via a bitcast CAS loop; dV is a
    // plain per-thread store. dQ/dK/dG must be pre-zeroed.
    public const string GlaRecompute = @"
@group(0) @binding(0) var<storage, read> K: array<f32>;
@group(0) @binding(1) var<storage, read> Vd: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> Straj: array<f32>;
struct P { batch: i32, seqLen: i32, modelDim: i32, numHeads: i32, headDim: i32 };
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let MD = p.modelDim; let NH = p.numHeads; let HD = p.headDim;
    if (gid >= B*NH*HD) { return; }
    let di = gid % HD; let h = (gid/HD) % NH; let b = gid/(HD*NH); let hOff = h*HD; let hh = HD*HD;
    var Srow: array<f32, 256>;
    for (var ki: i32 = 0; ki < HD; ki = ki+1) { Srow[ki] = 0.0; }
    for (var t: i32 = 0; t < SL; t = t+1) {
        let baseOff = (b*SL+t)*MD + hOff;
        let g = G[(b*SL+t)*NH + h]; let vd = Vd[baseOff+di];
        let trajBase = ((b*NH+h)*SL+t)*hh + di*HD;
        for (var ki: i32 = 0; ki < HD; ki = ki+1) { let s = g*Srow[ki] + vd*K[baseOff+ki]; Srow[ki] = s; Straj[trajBase+ki] = s; }
    }
}";

    public const string GlaBackward = @"
@group(0) @binding(0) var<storage, read> dOut: array<f32>;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> Vd: array<f32>;
@group(0) @binding(4) var<storage, read> G: array<f32>;
@group(0) @binding(5) var<storage, read> Straj: array<f32>;
@group(0) @binding(6) var<storage, read_write> dQ: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> dK: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> dV: array<f32>;
@group(0) @binding(9) var<storage, read_write> dG: array<atomic<u32>>;
struct P { batch: i32, seqLen: i32, modelDim: i32, numHeads: i32, headDim: i32 };
@group(0) @binding(10) var<uniform> p: P;
fn addDQ(idx: i32, val: f32) { var old = atomicLoad(&dQ[idx]); loop { let nv = bitcast<u32>(bitcast<f32>(old) + val); let r = atomicCompareExchangeWeak(&dQ[idx], old, nv); if (r.exchanged) { break; } old = r.old_value; } }
fn addDK(idx: i32, val: f32) { var old = atomicLoad(&dK[idx]); loop { let nv = bitcast<u32>(bitcast<f32>(old) + val); let r = atomicCompareExchangeWeak(&dK[idx], old, nv); if (r.exchanged) { break; } old = r.old_value; } }
fn addDG(idx: i32, val: f32) { var old = atomicLoad(&dG[idx]); loop { let nv = bitcast<u32>(bitcast<f32>(old) + val); let r = atomicCompareExchangeWeak(&dG[idx], old, nv); if (r.exchanged) { break; } old = r.old_value; } }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = i32(gid3.x);
    let B = p.batch; let SL = p.seqLen; let MD = p.modelDim; let NH = p.numHeads; let HD = p.headDim;
    if (gid >= B*NH*HD) { return; }
    let di = gid % HD; let h = (gid/HD) % NH; let b = gid/(HD*NH); let hOff = h*HD; let hh = HD*HD;
    var dSrow: array<f32, 256>;
    for (var ki: i32 = 0; ki < HD; ki = ki+1) { dSrow[ki] = 0.0; }
    for (var t: i32 = SL-1; t >= 0; t = t-1) {
        let baseOff = (b*SL+t)*MD + hOff; let gOff = (b*SL+t)*NH + h; let g = G[gOff];
        let trajBase = ((b*NH+h)*SL+t)*hh + di*HD;
        let sprevBase = ((b*NH+h)*SL+(t-1))*hh + di*HD;
        let dOutVal = dOut[baseOff+di];
        for (var ki: i32 = 0; ki < HD; ki = ki+1) { let sval = Straj[trajBase+ki]; addDQ(baseOff+ki, dOutVal*sval); dSrow[ki] = dSrow[ki] + dOutVal*Q[baseOff+ki]; }
        let vd = Vd[baseOff+di]; var dg: f32 = 0.0; var dVacc: f32 = 0.0;
        for (var ki: i32 = 0; ki < HD; ki = ki+1) { let dStv = dSrow[ki]; var sprev: f32 = 0.0; if (t > 0) { sprev = Straj[sprevBase+ki]; } dg = dg + dStv*sprev; addDK(baseOff+ki, dStv*vd); dVacc = dVacc + dStv*K[baseOff+ki]; }
        dV[baseOff+di] = dVacc;
        addDG(gOff, dg);
        for (var ki: i32 = 0; ki < HD; ki = ki+1) { dSrow[ki] = dSrow[ki] * g; }
    }
}";
}
