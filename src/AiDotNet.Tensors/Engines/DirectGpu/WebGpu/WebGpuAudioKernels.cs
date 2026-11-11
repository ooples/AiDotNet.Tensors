// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public static class WebGpuAudioKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "audio_amplitude_to_db", "audio_mulaw_encoding", "audio_mulaw_decoding",
        "audio_compute_deltas", "audio_resample",
    };

    public static string AmplitudeToDB => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct P { length: i32, minAmp: f32, topDbFloor: f32, clipTopDb: i32 };
@group(0) @binding(2) var<uniform> p : P;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.length) { return; }
    var v = max(input_[gid], p.minAmp);
    var db = 20.0 * log(v) / log(10.0);
    if (p.clipTopDb != 0 && db < p.topDbFloor) { db = p.topDbFloor; }
    output_[gid] = db;
}
";

    public static string MuLawEncoding => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct P { length: i32, quantizationChannels: i32 };
@group(0) @binding(2) var<uniform> p : P;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.length) { return; }
    if (p.quantizationChannels < 2) { output_[gid] = 0.0; return; }
    let mu = f32(p.quantizationChannels - 1);
    let logMu = log(1.0 + mu);
    var x = input_[gid];
    if (x > 1.0) { x = 1.0; } else if (x < -1.0) { x = -1.0; }
    let sgn = f32(x > 0.0) - f32(x < 0.0);
    let y = sgn * log(1.0 + mu * abs(x)) / logMu;
    var q = floor((y + 1.0) * 0.5 * mu + 0.5);
    if (q < 0.0) { q = 0.0; } else if (q > mu) { q = mu; }
    output_[gid] = q;
}
";

    public static string MuLawDecoding => @"
@group(0) @binding(0) var<storage, read> input_ : array<i32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct P { length: i32, quantizationChannels: i32 };
@group(0) @binding(2) var<uniform> p : P;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.length) { return; }
    if (p.quantizationChannels < 2) { output_[gid] = 0.0; return; }
    let mu = f32(p.quantizationChannels - 1);
    let q = f32(input_[gid]);
    let y = (q / mu) * 2.0 - 1.0;
    let sgn = f32(y > 0.0) - f32(y < 0.0);
    output_[gid] = sgn * (pow(1.0 + mu, abs(y)) - 1.0) / mu;
}
";

    public static string ComputeDeltas => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct P { leading: i32, timeAxis: i32, winLength: i32 };
@group(0) @binding(2) var<uniform> p : P;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.leading * p.timeAxis;
    if (gid >= total) { return; }
    let t = gid % p.timeAxis;
    let row = gid / p.timeAxis;
    let n = p.winLength / 2;
    if (n < 1) { output_[gid] = 0.0; return; }
    var denom : f32 = 0.0;
    for (var i : i32 = 1; i <= n; i = i + 1) { denom = denom + f32(2 * i * i); }
    var acc : f32 = 0.0;
    let base_ = row * p.timeAxis;
    for (var k : i32 = 1; k <= n; k = k + 1) {
        var left = t - k; if (left < 0) { left = 0; }
        var right = t + k; if (right >= p.timeAxis) { right = p.timeAxis - 1; }
        acc = acc + f32(k) * (input_[base_ + right] - input_[base_ + left]);
    }
    output_[gid] = acc / denom;
}
";

    public static string Resample => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct P { leading: i32, inLen: i32, outLen: i32, up: i32, down: i32, halfWidth: i32 };
@group(0) @binding(2) var<uniform> p : P;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.leading * p.outLen;
    if (gid >= total) { return; }
    if (p.halfWidth < 1 || p.up <= 0 || p.down <= 0) { output_[gid] = 0.0; return; }
    let ot = gid % p.outLen;
    let row = gid / p.outLen;
    let sBase = row * p.inLen;
    var cutoff : f32;
    if (p.up > p.down) { cutoff = 1.0 / f32(p.up); } else { cutoff = 1.0 / f32(p.down); }
    // Exact source position in 32-bit integers (ot = q*up + r; the host guarantees up*down < 2^31).
    let q = ot / p.up; let r = ot - q * p.up;
    let rd = r * p.down;
    let centre = q * p.down + rd / p.up;
    let frac = f32(rd - (rd / p.up) * p.up) / f32(p.up);
    var acc : f32 = 0.0; var wSum : f32 = 0.0;
    let PI = 3.14159265358979323846;
    for (var k : i32 = -p.halfWidth; k <= p.halfWidth; k = k + 1) {
        let idx = centre + k;
        if (idx < 0 || idx >= p.inLen) { continue; }
        let tt = (f32(k) - frac) * cutoff;
        var sinc : f32;
        // Near zero, sin(x)/x by its Taylor series: a fast or relaxed sin has ~1e-7 absolute error, which turns
        // sin(x)/x for x ~ 1e-8 (a tap almost exactly on the source position) into garbage. Error below 1e-14.
        let px = PI * tt;
        if (abs(px) < 1e-3) { sinc = 1.0 - px * px * (1.0 / 6.0); } else { sinc = sin(px) / px; }
        let hann = 0.5 - 0.5 * cos(2.0 * PI * f32(k + p.halfWidth) / f32(2 * p.halfWidth));
        let w = sinc * hann;
        acc = acc + w * input_[sBase + idx];
        wSum = wSum + w;
    }
    if (wSum > 0.0) { output_[gid] = acc / wSum; } else { output_[gid] = 0.0; }
}
";
}
#endif
