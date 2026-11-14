@group(0) @binding(0) var<storage, read> grad : array<f32>;
@group(0) @binding(1) var<storage, read> magnitude : array<f32>;
@group(0) @binding(2) var<storage, read> phase : array<f32>;
@group(0) @binding(3) var<storage, read_write> frameGrad : array<f32>;
struct P { batch: i32, nFft: i32, numFrames: i32, numFreqs: i32, phaseTarget: i32 };
@group(0) @binding(4) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  if (idx >= pc.batch * pc.numFrames * pc.nFft) { return; }
  let nFft = pc.nFft; let numFrames = pc.numFrames;
  let i = idx % nFft; let tmp = idx / nFft; let frame = tmp % numFrames; let b = tmp / numFrames;
  let specOff = b * pc.numFreqs * numFrames + frame;
  let tw = 2.0 * 3.14159265358979 / f32(nFft); var acc = 0.0; var m = 0;
  for (var k = 0; k < pc.numFreqs; k = k + 1) {
    let bin = specOff + k * numFrames; let g = grad[bin]; let p = phase[bin];
    let cp = cos(p); let sp = sin(p); var sr = g * cp; var si = g * sp;
    if (pc.phaseTarget != 0) {
      let mg = magnitude[bin]; var r = 0.0; if (mg > 1e-12) { r = g / mg; }
      sr = -r * sp; si = r * cp;
    }
    var c = 0.0; var s = 0.0;
    if (m == 0) { c = 1.0; }
    else if (2 * m == nFft) { c = -1.0; }
    else if (4 * m == nFft) { s = 1.0; }
    else if (4 * m == 3 * nFft) { s = -1.0; }
    else { var mm = m; if (2 * m > nFft) { mm = m - nFft; } let a = tw * f32(mm); c = cos(a); s = sin(a); }
    acc = acc + sr * c - si * s;
    m = m + i; if (m >= nFft) { m = m - nFft; }
  }
  frameGrad[idx] = acc;
}