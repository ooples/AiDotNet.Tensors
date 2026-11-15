@group(0) @binding(0) var<storage, read> frameGrad : array<f32>;
@group(0) @binding(1) var<storage, read> signal : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradWindow : array<f32>;
struct P { batch: i32, sigLen: i32, pad: i32, nFft: i32, hop: i32, numFrames: i32 };
@group(0) @binding(3) var<uniform> pc : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  if (idx >= pc.nFft) { return; }
  let sigLen = pc.sigLen; let pad = pc.pad; let i = idx; var acc = 0.0;
  for (var b = 0; b < pc.batch; b = b + 1) {
    for (var f = 0; f < pc.numFrames; f = f + 1) {
      let j = f * pc.hop + i; var src = 0;
      if (j < pad) { src = pad - j; } else if (j < pad + sigLen) { src = j - pad; } else { src = sigLen - 2 - (j - pad - sigLen); }
      acc = acc + frameGrad[(b * pc.numFrames + f) * pc.nFft + i] * signal[b * sigLen + src];
    }
  }
  gradWindow[i] = acc;
}