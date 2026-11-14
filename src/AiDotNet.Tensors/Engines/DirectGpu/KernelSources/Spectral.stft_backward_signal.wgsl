@group(0) @binding(0) var<storage, read> frameGrad : array<f32>;
@group(0) @binding(1) var<storage, read> window : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradSignal : array<f32>;
struct P { batch: i32, sigLen: i32, pad: i32, nFft: i32, hop: i32, numFrames: i32 };
@group(0) @binding(3) var<uniform> pc : P;
fn stftOverlap(b : i32, q : i32) -> f32 {
  var fHi = q / pc.hop; if (fHi > pc.numFrames - 1) { fHi = pc.numFrames - 1; }
  var fLo = 0; if (q - pc.nFft + 1 > 0) { fLo = (q - pc.nFft + pc.hop) / pc.hop; }
  var acc = 0.0;
  for (var f = fLo; f <= fHi; f = f + 1) { let i = q - f * pc.hop; acc = acc + frameGrad[(b * pc.numFrames + f) * pc.nFft + i] * window[i]; }
  return acc;
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  let sigLen = pc.sigLen; let pad = pc.pad;
  if (idx >= pc.batch * sigLen) { return; }
  let s = idx % sigLen; let b = idx / sigLen;
  var acc = stftOverlap(b, pad + s);
  if (s >= 1 && s <= pad) { acc = acc + stftOverlap(b, pad - s); }
  if (s >= sigLen - 1 - pad && s <= sigLen - 2) { acc = acc + stftOverlap(b, 2 * sigLen - 2 + pad - s); }
  gradSignal[idx] = acc;
}