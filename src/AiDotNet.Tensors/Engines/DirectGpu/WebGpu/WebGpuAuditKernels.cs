namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// WGSL compute shaders for the missing-kernels audit ops, hand-written from the verified OpenCL/CUDA
// kernels (WGSL has no ?: operator [select()], no int-as-bool, typed var decls, atomic<u32>-only, and no
// lgamma intrinsic). Buffers are @binding(0..n-1); the scalar uniform struct is @binding(n). Validated on
// a WebGPU runtime later. One @compute fn main per op.
public static class WebGpuAuditKernels
{
    private const string Lgamma = @"
fn p210_lgamma(x: f32) -> f32 {
  var c = array<f32,9>(0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
  let xx = x - 1.0; var a = c[0]; let t = xx + 7.5;
  for (var i = 1; i < 9; i = i + 1) { a = a + c[i] / (xx + f32(i)); }
  return 0.5 * log(2.0 * 3.14159265358979) + (xx + 0.5) * log(t) - t + log(a);
}";

    public static string ReflectPad1D => @"
@group(0) @binding(0) var<storage, read> inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { batch: i32, L: i32, Lp: i32, pad: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.batch * pc.Lp) { return; }
  let j = idx % pc.Lp; let b = idx / pc.Lp; var src: i32;
  if (j < pc.pad) { src = pc.pad - j; } else if (j < pc.pad + pc.L) { src = j - pc.pad; } else { src = pc.L - 2 - (j - pc.pad - pc.L); }
  outp[idx] = inp[b * pc.L + src];
}";

    public static string StftMagPhase => @"
@group(0) @binding(0) var<storage, read> padded : array<f32>;
@group(0) @binding(1) var<storage, read> window : array<f32>;
@group(0) @binding(2) var<storage, read_write> mag : array<f32>;
@group(0) @binding(3) var<storage, read_write> phase : array<f32>;
struct P { batch: i32, Lp: i32, nFft: i32, hop: i32, numFrames: i32, numFreqs: i32 };
@group(0) @binding(4) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let total = pc.batch * pc.numFreqs * pc.numFrames; if (idx >= total) { return; }
  let frame = idx % pc.numFrames; let tmp = idx / pc.numFrames; let k = tmp % pc.numFreqs; let b = tmp / pc.numFreqs;
  let start = frame * pc.hop; let inOff = b * pc.Lp; var re = 0.0; var im = 0.0;
  let bk = -2.0 * 3.14159265358979 * f32(k) / f32(pc.nFft);
  for (var i = 0; i < pc.nFft; i = i + 1) { let x = padded[inOff + start + i] * window[i]; let a = bk * f32(i); re = re + x * cos(a); im = im + x * sin(a); }
  let outOff = b * pc.numFreqs * pc.numFrames + k * pc.numFrames + frame;
  mag[outOff] = sqrt(re * re + im * im); phase[outOff] = atan2(im, re);
}";

    public static string PhaseVocoder => @"
@group(0) @binding(0) var<storage, read> mag : array<f32>;
@group(0) @binding(1) var<storage, read> phase : array<f32>;
@group(0) @binding(2) var<storage, read_write> newMag : array<f32>;
@group(0) @binding(3) var<storage, read_write> newPhase : array<f32>;
struct P { leading: i32, nFramesV: i32, nFreqV: i32, outFrames: i32, rate: f32 };
@group(0) @binding(4) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.leading * pc.nFreqV) { return; }
  let f = idx % pc.nFreqV; let b = idx / pc.nFreqV; let stride = pc.nFramesV * pc.nFreqV; let outStride = pc.outFrames * pc.nFreqV;
  var accPhase = 0.0;
  for (var t = 0; t < pc.outFrames; t = t + 1) {
    let srcT = f32(t) * pc.rate; let t0 = i32(floor(srcT)); let t1 = min(t0 + 1, pc.nFramesV - 1); let frac = srcT - f32(t0);
    let m0 = mag[b * stride + t0 * pc.nFreqV + f]; let m1 = mag[b * stride + t1 * pc.nFreqV + f];
    newMag[b * outStride + t * pc.nFreqV + f] = (1.0 - frac) * m0 + frac * m1;
    var dp = 0.0;
    if (t0 + 1 < pc.nFramesV) { dp = phase[b * stride + (t0 + 1) * pc.nFreqV + f] - phase[b * stride + t0 * pc.nFreqV + f]; dp = dp - 2.0 * 3.14159265358979 * round(dp / (2.0 * 3.14159265358979)); }
    accPhase = accPhase + dp; newPhase[b * outStride + t * pc.nFreqV + f] = accPhase;
  }
}";

    public static string BuildSpectrum => @"
@group(0) @binding(0) var<storage, read> mag : array<f32>;
@group(0) @binding(1) var<storage, read> phase : array<f32>;
@group(0) @binding(2) var<storage, read_write> specRe : array<f32>;
@group(0) @binding(3) var<storage, read_write> specIm : array<f32>;
struct P { batch: i32, numFreqs: i32, numFrames: i32, nFft: i32 };
@group(0) @binding(4) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.batch * pc.numFrames) { return; }
  let frame = idx % pc.numFrames; let b = idx / pc.numFrames; let magOff = b * pc.numFreqs * pc.numFrames; let specOff = idx * pc.nFft;
  for (var k = 0; k < pc.nFft; k = k + 1) { specRe[specOff + k] = 0.0; specIm[specOff + k] = 0.0; }
  for (var k = 0; k < pc.numFreqs && k < pc.nFft; k = k + 1) { let m = mag[magOff + k * pc.numFrames + frame]; let p = phase[magOff + k * pc.numFrames + frame]; specRe[specOff + k] = m * cos(p); specIm[specOff + k] = m * sin(p); }
  for (var k = 1; k < pc.numFreqs - 1; k = k + 1) { let dst = pc.nFft - k; if (dst >= 0 && dst < pc.nFft && k < pc.nFft) { specRe[specOff + dst] = specRe[specOff + k]; specIm[specOff + dst] = -specIm[specOff + k]; } }
}";

    public static string IstftNormalize => @"
@group(0) @binding(0) var<storage, read_write> resultb : array<f32>;
@group(0) @binding(1) var<storage, read> windowSum : array<f32>;
struct P { total: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.total) { return; }
  let ws = windowSum[idx]; if (ws > 1e-8) { resultb[idx] = resultb[idx] / ws; }
}";

    public static string LogicalOp => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { mode: i32, n: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = i32(gid.x); if (i >= pc.n) { return; }
  let ba = a[i] != 0.0; let bb = b[i] != 0.0;
  var r = false;
  if (pc.mode == 0) { r = ba && bb; } else if (pc.mode == 1) { r = ba || bb; } else { r = ba != bb; }
  outp[i] = select(0.0, 1.0, r);
}";

    public static string LogicalNot => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { n: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = i32(gid.x); if (i >= pc.n) { return; }
  outp[i] = select(1.0, 0.0, a[i] != 0.0);
}";

    public static string ShiftedDiff => @"
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read_write> mask : array<f32>;
struct P { n: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = i32(gid.x); if (i >= pc.n) { return; }
  mask[i] = select(0.0, 1.0, i == 0 || x[i] != x[i - 1]);
}";

    public static string MasksToBoxes => @"
@group(0) @binding(0) var<storage, read> masks : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { N: i32, H: i32, W: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = i32(gid.x); if (n >= pc.N) { return; }
  var xMin = pc.W; var yMin = pc.H; var xMax = -1; var yMax = -1; let planeOff = n * pc.H * pc.W;
  for (var y = 0; y < pc.H; y = y + 1) { for (var xx = 0; xx < pc.W; xx = xx + 1) {
    if (masks[planeOff + y * pc.W + xx] != 0.0) { if (xx < xMin) { xMin = xx; } if (xx > xMax) { xMax = xx; } if (y < yMin) { yMin = y; } if (y > yMax) { yMax = y; } }
  } }
  let o = n * 4;
  if (xMax < 0) { outp[o] = 0.0; outp[o + 1] = 0.0; outp[o + 2] = 0.0; outp[o + 3] = 0.0; }
  else { outp[o] = f32(xMin); outp[o + 1] = f32(yMin); outp[o + 2] = f32(xMax); outp[o + 3] = f32(yMax); }
}";

    public static string PairwiseIou => @"
@group(0) @binding(0) var<storage, read> boxes : array<f32>;
@group(0) @binding(1) var<storage, read_write> iou : array<f32>;
struct P { N: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.N * pc.N) { return; } let j = idx % pc.N; let i = idx / pc.N;
  let ix1 = boxes[i*4]; let iy1 = boxes[i*4+1]; let ix2 = boxes[i*4+2]; let iy2 = boxes[i*4+3];
  let jx1 = boxes[j*4]; let jy1 = boxes[j*4+1]; let jx2 = boxes[j*4+2]; let jy2 = boxes[j*4+3];
  let ai = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1); let aj = max(0.0, jx2 - jx1) * max(0.0, jy2 - jy1);
  let iw = max(0.0, min(ix2, jx2) - max(ix1, jx1)); let ih = max(0.0, min(iy2, jy2) - max(iy1, jy1));
  let inter = iw * ih; let uni = ai + aj - inter; iou[idx] = select(0.0, inter / uni, uni > 0.0);
}";

    public static string Cross3 => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { outerSize: i32, innerSize: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.outerSize * pc.innerSize) { return; }
  let inner = idx % pc.innerSize; let outer = idx / pc.innerSize; let p = outer * 3 * pc.innerSize + inner;
  let a0 = a[p]; let a1 = a[p + pc.innerSize]; let a2 = a[p + 2 * pc.innerSize];
  let b0 = b[p]; let b1 = b[p + pc.innerSize]; let b2 = b[p + 2 * pc.innerSize];
  outp[p] = a1 * b2 - a2 * b1; outp[p + pc.innerSize] = a2 * b0 - a0 * b2; outp[p + 2 * pc.innerSize] = a0 * b1 - a1 * b0;
}";

    public static string Ldexp => @"
@group(0) @binding(0) var<storage, read> inp : array<f32>;
@group(0) @binding(1) var<storage, read> exponents : array<i32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.size) { return; }
  outp[idx] = ldexp(inp[idx], exponents[idx]);
}";

    public static string Kron2D => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { am: i32, an: i32, bp: i32, bq: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let outCols = pc.an * pc.bq; let total = (pc.am * pc.bp) * outCols; if (idx >= total) { return; }
  let oc = idx % outCols; let orow = idx / outCols; let i = orow / pc.bp; let k = orow % pc.bp; let j = oc / pc.bq; let l = oc % pc.bq;
  outp[idx] = a[i * pc.an + j] * b[k * pc.bq + l];
}";

    public static string SearchSorted => @"
@group(0) @binding(0) var<storage, read> seq : array<f32>;
@group(0) @binding(1) var<storage, read> values : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { seqLen: i32, numValues: i32, right: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.numValues) { return; } let v = values[idx]; var lo = 0; var hi = pc.seqLen;
  while (lo < hi) { let mid = (lo + hi) >> 1u; var cond = false; if (pc.right != 0) { cond = seq[mid] <= v; } else { cond = seq[mid] < v; } if (cond) { lo = mid + 1; } else { hi = mid; } }
  outp[idx] = f32(lo);
}";

    public static string NextAfter => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.size) { return; } let av = a[idx]; let bv = b[idx];
  let ua = bitcast<u32>(av); let ub = bitcast<u32>(bv);
  let aNan = ((ua >> 23u) & 0xFFu) == 0xFFu && (ua & 0x7FFFFFu) != 0u;
  let bNan = ((ub >> 23u) & 0xFFu) == 0xFFu && (ub & 0x7FFFFFu) != 0u;
  if (aNan || bNan) { outp[idx] = bitcast<f32>(0x7FC00000u); return; }
  if (av == bv) { outp[idx] = bv; return; }
  if (av == 0.0) { outp[idx] = bitcast<f32>(select(0x80000001u, 0x00000001u, bv > 0.0)); return; }
  var r = ua;
  if (bv > av) { if (av > 0.0) { r = r + 1u; } else { r = r - 1u; } } else { if (av > 0.0) { r = r - 1u; } else { r = r + 1u; } }
  outp[idx] = bitcast<f32>(r);
}";

    public static string IndexWrite => @"
@group(0) @binding(0) var<storage, read_write> outp : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read> source : array<f32>;
struct P { fillValue: f32, mode: i32, outerSize: i32, idxAxis: i32, innerSize: i32, dstAxis: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let total = pc.outerSize * pc.idxAxis * pc.innerSize; if (idx >= total) { return; }
  let inner = idx % pc.innerSize; let j = (idx / pc.innerSize) % pc.idxAxis; let outer = (idx / pc.innerSize) / pc.idxAxis; let dstJ = indices[j];
  if (dstJ < 0 || dstJ >= pc.dstAxis) { return; }
  let v = select(pc.fillValue, source[idx], pc.mode == 0);
  outp[(outer * pc.dstAxis + dstJ) * pc.innerSize + inner] = v;
}";

    public static string Cdist => @"
@group(0) @binding(0) var<storage, read> x1 : array<f32>;
@group(0) @binding(1) var<storage, read> x2 : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { m: i32, n: i32, d: i32, p: f32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.m * pc.n) { return; } let j = idx % pc.n; let i = idx / pc.n; var sum = 0.0;
  for (var k = 0; k < pc.d; k = k + 1) { let diff = abs(x1[i * pc.d + k] - x2[j * pc.d + k]); if (pc.p == 1.0) { sum = sum + diff; } else if (pc.p == 2.0) { sum = sum + diff * diff; } else { sum = sum + pow(diff, pc.p); } }
  outp[idx] = select(select(pow(sum, 1.0 / pc.p), sqrt(sum), pc.p == 2.0), sum, pc.p == 1.0);
}";

    public static string Pdist => @"
@group(0) @binding(0) var<storage, read> inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { n: i32, d: i32, p: f32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let flat = i32(gid.x); if (flat >= pc.n * pc.n) { return; } let j = flat % pc.n; let i = flat / pc.n; if (i >= j) { return; } var sum = 0.0;
  for (var k = 0; k < pc.d; k = k + 1) { let diff = abs(inp[i * pc.d + k] - inp[j * pc.d + k]); if (pc.p == 1.0) { sum = sum + diff; } else if (pc.p == 2.0) { sum = sum + diff * diff; } else { sum = sum + pow(diff, pc.p); } }
  let dist = select(select(pow(sum, 1.0 / pc.p), sqrt(sum), pc.p == 2.0), sum, pc.p == 1.0);
  let outIdx = i * pc.n - (i * (i + 1)) / 2 + (j - i - 1); outp[outIdx] = dist;
}";

    public static string CopyRows => @"
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
struct P { srcRowLen: i32, dstRowLen: i32, numRows: i32, copyLen: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = i32(gid.x); if (g >= pc.numRows * pc.copyLen) { return; } let i = g % pc.copyLen; let r = g / pc.copyLen; dst[r * pc.dstRowLen + i] = src[r * pc.srcRowLen + i];
}";

    public static string IotaPad => @"
@group(0) @binding(0) var<storage, read_write> idxb : array<f32>;
struct P { L: i32, Pp: i32, numRows: i32 };
@group(0) @binding(1) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = i32(gid.x); if (g >= pc.numRows * pc.Pp) { return; } let i = g % pc.Pp; idxb[g] = select(-1.0, f32(i), i < pc.L);
}";

    public static string TakeAlongDimF => @"
@group(0) @binding(0) var<storage, read> inp : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { outerSize: i32, axisOut: i32, innerSize: i32, axisIn: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let total = pc.outerSize * pc.axisOut * pc.innerSize; if (idx >= total) { return; }
  let inner = idx % pc.innerSize; let outer = (idx / pc.innerSize) / pc.axisOut; let srcJ = i32(indices[idx]);
  if (srcJ < 0 || srcJ >= pc.axisIn) { outp[idx] = 0.0; return; }
  outp[idx] = inp[(outer * pc.axisIn + srcJ) * pc.innerSize + inner];
}";

    public static string IsIn => @"
@group(0) @binding(0) var<storage, read> elements : array<f32>;
@group(0) @binding(1) var<storage, read> sortedTest : array<f32>;
@group(0) @binding(2) var<storage, read_write> mask : array<f32>;
struct P { numElements: i32, testLen: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.numElements) { return; } let v = elements[idx]; var lo = 0; var hi = pc.testLen;
  while (lo < hi) { let mid = (lo + hi) >> 1u; if (sortedTest[mid] < v) { lo = mid + 1; } else { hi = mid; } }
  mask[idx] = select(0.0, 1.0, lo < pc.testLen && sortedTest[lo] == v);
}";

    public static string CopyBlock2D => @"
@group(0) @binding(0) var<storage, read> block : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { blockRows: i32, blockCols: i32, totalCols: i32, rowOff: i32, colOff: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.blockRows * pc.blockCols) { return; } let j = idx % pc.blockCols; let i = idx / pc.blockCols;
  outp[(pc.rowOff + i) * pc.totalCols + (pc.colOff + j)] = block[i * pc.blockCols + j];
}";

    public static string Unfold => @"
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
struct P { outerSize: i32, dimSize: i32, innerSize: i32, nWindows: i32, size: i32, step: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let total = pc.outerSize * pc.nWindows * pc.innerSize * pc.size; if (idx >= total) { return; }
  let s = idx % pc.size; var tmp = idx / pc.size; let inner = tmp % pc.innerSize; tmp = tmp / pc.innerSize; let w = tmp % pc.nWindows; let outer = tmp / pc.nWindows;
  dst[idx] = src[(outer * pc.dimSize + (w * pc.step + s)) * pc.innerSize + inner];
}";

    public static string ClassifyFloat => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { mode: i32, size: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.size) { return; } let bits = bitcast<u32>(a[idx]); let expo = (bits >> 23u) & 0xFFu; let mant = bits & 0x7FFFFFu;
  let isNan = expo == 0xFFu && mant != 0u; let isInf = expo == 0xFFu && mant == 0u; var r = 0.0;
  if (pc.mode == 0) { r = select(0.0, 1.0, isNan); } else if (pc.mode == 1) { r = select(0.0, 1.0, isInf); } else { r = select(0.0, 1.0, !isNan && !isInf); }
  outp[idx] = r;
}";

    public static string HsoftmaxPaths => @"
@group(0) @binding(0) var<storage, read> acts : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { rows: i32, treeDepth: i32, numClasses: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.rows * pc.numClasses) { return; } let c = idx % pc.numClasses; let r = idx / pc.numClasses; let gbase = r * pc.treeDepth; var prob = 1.0; var node = 1;
  for (var level = 0; level < pc.treeDepth; level = level + 1) {
    let goRight = (c & (1 << u32(pc.treeDepth - level - 1))) != 0; let g = 1.0 / (1.0 + exp(-acts[gbase + level]));
    prob = prob * select(1.0 - g, g, goRight); node = node * 2 + select(0, 1, goRight); if (node >= pc.numClasses) { break; }
  }
  outp[idx] = prob;
}";

    public static string Zeta => Lgamma + @"
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> q : array<f32>;
@group(0) @binding(2) var<storage, read_write> outp : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> pc : P;
fn p210_zeta_scalar(xv: f32, qv: f32) -> f32 {
  if (xv == 1.0) { return bitcast<f32>(0x7F800000u); }
  if (qv <= 0.0 && qv == floor(qv)) { return bitcast<f32>(0x7F800000u); }
  var sum = 0.0;
  for (var k = 0; k < 12; k = k + 1) { sum = sum + pow(qv + f32(k), -xv); }
  let Nq = 12.0 + qv; let lnNq = log(Nq); let cont = exp((1.0 - xv) * lnNq) / (xv - 1.0); let halfTerm = 0.5 * exp(-xv * lnNq);
  var b2k = array<f32,8>(1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0, 5.0/66.0, -691.0/2730.0, 7.0/6.0, -3617.0/510.0);
  var fact2k = array<f32,8>(2.0, 24.0, 720.0, 40320.0, 3628800.0, 479001600.0, 87178291200.0, 20922789888000.0);
  var corr = 0.0; var xPow = exp(-xv * lnNq) / Nq; let invNq2 = 1.0 / (Nq * Nq); var rising = xv;
  for (var j = 1; j <= 8; j = j + 1) {
    if (j > 1) { rising = rising * (xv + 2.0 * f32(j - 1) - 2.0) * (xv + 2.0 * f32(j - 1) - 1.0); xPow = xPow * invNq2; }
    corr = corr + (b2k[j - 1] / fact2k[j - 1]) * rising * xPow;
  }
  return sum + cont + halfTerm + corr;
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = i32(gid.x); if (i >= pc.size) { return; } outp[i] = p210_zeta_scalar(x[i], q[i]);
}";

    public static string Polygamma => Lgamma + @"
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
struct P { n: i32, size: i32 };
@group(0) @binding(2) var<uniform> pc : P;
fn p210_polygamma_scalar(n: i32, xv: f32) -> f32 {
  if (xv <= 0.0 && xv == floor(xv)) { return bitcast<f32>(0x7F800000u); }
  var recurrence = 0.0; var xd = xv; let signN = select(-1, 1, (n & 1) == 1);
  loop { if (xd >= 10.0) { break; } recurrence = recurrence + f32(signN) * exp(p210_lgamma(f32(n + 1)) - f32(n + 1) * log(xd)); xd = xd + 1.0; }
  let lnX = log(xd);
  var asympt = exp(p210_lgamma(f32(n)) - f32(n) * lnX) + 0.5 * exp(p210_lgamma(f32(n + 1)) - f32(n + 1) * lnX);
  var b2k = array<f32,8>(1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0, 5.0/66.0, -691.0/2730.0, 7.0/6.0, -3617.0/510.0);
  let invX2 = 1.0 / (xd * xd); var xPow = exp(-f32(n) * lnX);
  for (var k = 1; k <= 8; k = k + 1) { xPow = xPow * invX2; asympt = asympt + b2k[k - 1] * exp(p210_lgamma(f32(2 * k + n)) - p210_lgamma(f32(2 * k + 1))) * xPow; }
  return recurrence + f32(signN) * asympt;
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = i32(gid.x); if (i >= pc.size) { return; } outp[i] = p210_polygamma_scalar(pc.n, x[i]);
}";

    public static string GridSampleBackwardGrid => @"
@group(0) @binding(0) var<storage, read> gradOut : array<f32>;
@group(0) @binding(1) var<storage, read> inp : array<f32>;
@group(0) @binding(2) var<storage, read> grid : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradGrid : array<f32>;
struct P { batch: i32, H: i32, W: i32, C: i32, outH: i32, outW: i32 };
@group(0) @binding(4) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.batch * pc.outH * pc.outW) { return; }
  let ow = idx % pc.outW; let tmp = idx / pc.outW; let oh = tmp % pc.outH; let b = tmp / pc.outH;
  let gridBase = ((b * pc.outH + oh) * pc.outW + ow) * 2; let gx = grid[gridBase]; let gy = grid[gridBase + 1];
  let srcH = (gy + 1.0) * 0.5 * f32(pc.H - 1); let srcW = (gx + 1.0) * 0.5 * f32(pc.W - 1);
  var gradGx = 0.0; var gradGy = 0.0;
  if (!(srcH <= -1.0 || srcH >= f32(pc.H) || srcW <= -1.0 || srcW >= f32(pc.W))) {
    let h0 = i32(floor(srcH)); let h1 = h0 + 1; let w0 = i32(floor(srcW)); let w1 = w0 + 1; let lh = srcH - f32(h0); let lw = srcW - f32(w0);
    let in00 = h0 >= 0 && h0 < pc.H && w0 >= 0 && w0 < pc.W; let in01 = h0 >= 0 && h0 < pc.H && w1 >= 0 && w1 < pc.W;
    let in10 = h1 >= 0 && h1 < pc.H && w0 >= 0 && w0 < pc.W; let in11 = h1 >= 0 && h1 < pc.H && w1 >= 0 && w1 < pc.W;
    for (var c = 0; c < pc.C; c = c + 1) {
      let v00 = select(0.0, inp[((b*pc.C+c)*pc.H+h0)*pc.W+w0], in00); let v01 = select(0.0, inp[((b*pc.C+c)*pc.H+h0)*pc.W+w1], in01);
      let v10 = select(0.0, inp[((b*pc.C+c)*pc.H+h1)*pc.W+w0], in10); let v11 = select(0.0, inp[((b*pc.C+c)*pc.H+h1)*pc.W+w1], in11);
      let dH = (1.0 - lw) * (v10 - v00) + lw * (v11 - v01); let dW = (1.0 - lh) * (v01 - v00) + lh * (v11 - v10);
      let go = gradOut[((b*pc.C+c)*pc.outH+oh)*pc.outW+ow];
      gradGx = gradGx + go * dW * f32(pc.W - 1) * 0.5; gradGy = gradGy + go * dH * f32(pc.H - 1) * 0.5;
    }
  }
  gradGrid[gridBase] = gradGx; gradGrid[gridBase + 1] = gradGy;
}";

    // ---- atomic kernels (atomic<u32> + CAS float-add helper) ----
    private const string AtomicAddF = @"
fn p210_atomicAddF(buf: ptr<storage, array<atomic<u32>>, read_write>, idx: i32, val: f32) {
  var old = atomicLoad(&(*buf)[idx]);
  loop {
    let nv = bitcast<u32>(bitcast<f32>(old) + val);
    let res = atomicCompareExchangeWeak(&(*buf)[idx], old, nv);
    if (res.exchanged) { break; }
    old = res.old_value;
  }
}";

    public static string Histc => AtomicAddF + @"
@group(0) @binding(0) var<storage, read> inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> hist : array<atomic<u32>>;
struct P { n: i32, bins: i32, mn: f32, mx: f32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); if (idx >= pc.n) { return; } let x = inp[idx]; if (!(x >= pc.mn && x <= pc.mx)) { return; }
  let bw = (pc.mx - pc.mn) / f32(pc.bins); var b = i32((x - pc.mn) / bw); if (b >= pc.bins) { b = pc.bins - 1; } if (b < 0) { b = 0; }
  p210_atomicAddF(&hist, b, 1.0);
}";

    public static string Histogramdd => AtomicAddF + @"
@group(0) @binding(0) var<storage, read> samples : array<f32>;
@group(0) @binding(1) var<storage, read_write> hist : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> bins : array<i32>;
@group(0) @binding(3) var<storage, read> mins : array<f32>;
@group(0) @binding(4) var<storage, read> maxs : array<f32>;
struct P { n: i32, d: i32 };
@group(0) @binding(5) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = i32(gid.x); if (i >= pc.n) { return; } var linIdx = 0; var valid = true;
  for (var k = 0; k < pc.d; k = k + 1) {
    let v = samples[i * pc.d + k]; let mn = mins[k]; let mx = maxs[k]; if (!(v >= mn && v <= mx)) { valid = false; break; }
    let width = (mx - mn) / f32(bins[k]); var kIdx = i32(floor((v - mn) / width)); if (kIdx >= bins[k]) { kIdx = bins[k] - 1; } if (kIdx < 0) { kIdx = 0; }
    linIdx = linIdx * bins[k] + kIdx;
  }
  if (valid) { p210_atomicAddF(&hist, linIdx, 1.0); }
}";

    public static string GridSampleBackwardInput => AtomicAddF + @"
@group(0) @binding(0) var<storage, read> gradOut : array<f32>;
@group(0) @binding(1) var<storage, read> grid : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradIn : array<atomic<u32>>;
struct P { batch: i32, H: i32, W: i32, C: i32, outH: i32, outW: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let total = pc.batch * pc.outH * pc.outW * pc.C; if (idx >= total) { return; }
  let ow = idx % pc.outW; var tmp = idx / pc.outW; let oh = tmp % pc.outH; tmp = tmp / pc.outH; let c = tmp % pc.C; let b = tmp / pc.C;
  let gridBase = ((b * pc.outH + oh) * pc.outW + ow) * 2; let gx = grid[gridBase]; let gy = grid[gridBase + 1];
  let srcH = (gy + 1.0) * 0.5 * f32(pc.H - 1); let srcW = (gx + 1.0) * 0.5 * f32(pc.W - 1);
  if (srcH <= -1.0 || srcH >= f32(pc.H) || srcW <= -1.0 || srcW >= f32(pc.W)) { return; }
  let h0 = i32(floor(srcH)); let h1 = h0 + 1; let w0 = i32(floor(srcW)); let w1 = w0 + 1; let lh = srcH - f32(h0); let lw = srcW - f32(w0); let g = gradOut[idx];
  if (h0 >= 0 && h0 < pc.H && w0 >= 0 && w0 < pc.W) { p210_atomicAddF(&gradIn, ((b*pc.C+c)*pc.H+h0)*pc.W+w0, g*(1.0-lh)*(1.0-lw)); }
  if (h0 >= 0 && h0 < pc.H && w1 >= 0 && w1 < pc.W) { p210_atomicAddF(&gradIn, ((b*pc.C+c)*pc.H+h0)*pc.W+w1, g*(1.0-lh)*lw); }
  if (h1 >= 0 && h1 < pc.H && w0 >= 0 && w0 < pc.W) { p210_atomicAddF(&gradIn, ((b*pc.C+c)*pc.H+h1)*pc.W+w0, g*lh*(1.0-lw)); }
  if (h1 >= 0 && h1 < pc.H && w1 >= 0 && w1 < pc.W) { p210_atomicAddF(&gradIn, ((b*pc.C+c)*pc.H+h1)*pc.W+w1, g*lh*lw); }
}";

    public static string IstftFromSpectrum => AtomicAddF + @"
@group(0) @binding(0) var<storage, read> specRe : array<f32>;
@group(0) @binding(1) var<storage, read> specIm : array<f32>;
@group(0) @binding(2) var<storage, read> window : array<f32>;
@group(0) @binding(3) var<storage, read_write> resultb : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> windowSum : array<atomic<u32>>;
struct P { batch: i32, numFrames: i32, nFft: i32, hop: i32, outputLength: i32, center: i32 };
@group(0) @binding(5) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let total = pc.batch * pc.numFrames * pc.nFft; if (idx >= total) { return; }
  let i = idx % pc.nFft; let tmp = idx / pc.nFft; let frame = tmp % pc.numFrames; let b = tmp / pc.numFrames; let specOff = (b * pc.numFrames + frame) * pc.nFft;
  var acc = 0.0;
  for (var k = 0; k < pc.nFft; k = k + 1) { let a = 2.0 * 3.14159265358979 * f32(k) * f32(i) / f32(pc.nFft); acc = acc + specRe[specOff + k] * cos(a) - specIm[specOff + k] * sin(a); }
  var writeStart = frame * pc.hop; if (pc.center != 0) { writeStart = max(0, frame * pc.hop - pc.nFft / 2); }
  let outIdx = writeStart + i;
  if (outIdx >= 0 && outIdx < pc.outputLength) { let w = window[i]; p210_atomicAddF(&resultb, b * pc.outputLength + outIdx, acc * (1.0 / f32(pc.nFft)) * w); p210_atomicAddF(&windowSum, b * pc.outputLength + outIdx, w * w); }
}";

    public static string ScatterReduce => @"
@group(0) @binding(0) var<storage, read_write> outp : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> source : array<f32>;
@group(0) @binding(2) var<storage, read> indexb : array<i32>;
struct P { outerSize: i32, srcDim: i32, dstDim: i32, innerSize: i32, mode: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x); let total = pc.outerSize * pc.srcDim * pc.innerSize; if (idx >= total) { return; }
  let inner = idx % pc.innerSize; let tmp = idx / pc.innerSize; let outer = tmp / pc.srcDim; let t = indexb[idx];
  if (t < 0 || t >= pc.dstDim) { return; } let dst = (outer * pc.dstDim + t) * pc.innerSize + inner; let val = source[idx];
  var old = atomicLoad(&outp[dst]);
  loop {
    let pf = bitcast<f32>(old); var nv = pf + val;
    if (pc.mode == 1) { nv = pf * val; } else if (pc.mode == 2) { nv = max(pf, val); } else if (pc.mode == 3) { nv = min(pf, val); }
    let res = atomicCompareExchangeWeak(&outp[dst], old, bitcast<u32>(nv)); if (res.exchanged) { break; } old = res.old_value;
  }
}";

    public static string BitonicStep => @"
@group(0) @binding(0) var<storage, read_write> values : array<f32>;
@group(0) @binding(1) var<storage, read_write> indices : array<f32>;
struct P { rowLen: i32, k: i32, j: i32, numRows: i32, descending: i32 };
@group(0) @binding(2) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = i32(gid.x); if (g >= pc.numRows * pc.rowLen) { return; } let i = g % pc.rowLen; let ixj = i ^ pc.j; if (ixj <= i) { return; }
  let base = (g / pc.rowLen) * pc.rowLen; let a = values[base + i]; let b = values[base + ixj];
  let ua = bitcast<u32>(a); let ub = bitcast<u32>(b);
  let ka = select(a, bitcast<f32>(0x7F800000u), ((ua >> 23u) & 0xFFu) == 0xFFu && (ua & 0x7FFFFFu) != 0u);
  let kb = select(b, bitcast<f32>(0x7F800000u), ((ub >> 23u) & 0xFFu) == 0xFFu && (ub & 0x7FFFFFu) != 0u);
  var up = (i & pc.k) == 0; if (pc.descending != 0) { up = !up; }
  let doSwap = select(ka < kb, ka > kb, up);
  if (doSwap) { values[base + i] = b; values[base + ixj] = a; let t = indices[base + i]; indices[base + i] = indices[base + ixj]; indices[base + ixj] = t; }
}";

    public static string Rwkv7Forward => @"
@group(0) @binding(0) var<storage, read> R : array<f32>;
@group(0) @binding(1) var<storage, read> K : array<f32>;
@group(0) @binding(2) var<storage, read> V : array<f32>;
@group(0) @binding(3) var<storage, read> A : array<f32>;
@group(0) @binding(4) var<storage, read> B : array<f32>;
@group(0) @binding(5) var<storage, read_write> outp : array<f32>;
@group(0) @binding(6) var<storage, read_write> Sbuf : array<f32>;
struct P { batch: i32, seqLen: i32, modelDim: i32, numHeads: i32, headDim: i32 };
@group(0) @binding(7) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let bh = i32(gid.x); if (bh >= pc.batch * pc.numHeads) { return; } let b = bh / pc.numHeads; let h = bh % pc.numHeads; let hOff = h * pc.headDim; let hh = pc.headDim * pc.headDim; let sBase = bh * hh;
  for (var i = 0; i < hh; i = i + 1) { Sbuf[sBase + i] = 0.0; }
  for (var t = 0; t < pc.seqLen; t = t + 1) {
    let baseOff = (b * pc.seqLen + t) * pc.modelDim + hOff;
    for (var di = 0; di < pc.headDim; di = di + 1) {
      let ga = 1.0 / (1.0 + exp(-A[baseOff + di])); let gbk = (1.0 / (1.0 + exp(-B[baseOff + di]))) * K[baseOff + di]; let srow = di * pc.headDim;
      for (var vi = 0; vi < pc.headDim; vi = vi + 1) { Sbuf[sBase + srow + vi] = ga * Sbuf[sBase + srow + vi] + gbk * V[baseOff + vi]; }
    }
    for (var di = 0; di < pc.headDim; di = di + 1) {
      let srow = di * pc.headDim; var sk = 0.0;
      for (var vi = 0; vi < pc.headDim; vi = vi + 1) { sk = sk + Sbuf[sBase + srow + vi] * K[baseOff + vi]; }
      outp[baseOff + di] = (1.0 / (1.0 + exp(-R[baseOff + di]))) * sk;
    }
  }
}";
}
