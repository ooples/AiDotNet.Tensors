// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shader — radix-2 Cooley-Tukey FFT. Matches the Metal/
// Vulkan peer kernels. One workgroup per batch slice.

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu
{
    internal static class WebGpuFftKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "parity212_fft",
        };

        public static string Fft => @"
@group(0) @binding(0) var<storage, read_write> buf : array<f32>;
struct P { batchCount : i32, n : i32, inverse : i32 };
@group(0) @binding(1) var<uniform> p : P;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) li : vec3<u32>)
{
    let b = i32(wg.x);
    if (b >= p.batchCount) { return; }
    let tid = i32(li.x);
    let bs = 256;
    let n = p.n;
    let bOff = b * 2 * n;

    // Bit reversal permutation.
    var bits : i32 = 0;
    var t : i32 = n;
    loop { if (t <= 1) { break; } t = t >> 1; bits = bits + 1; }

    var i : i32 = tid;
    loop {
        if (i >= n) { break; }
        var j : i32 = 0;
        var v : i32 = i;
        for (var bb : i32 = 0; bb < bits; bb = bb + 1) { j = (j << 1) | (v & 1); v = v >> 1; }
        if (j > i) {
            let tr = buf[bOff + 2 * i];
            let ti = buf[bOff + 2 * i + 1];
            buf[bOff + 2 * i] = buf[bOff + 2 * j];
            buf[bOff + 2 * i + 1] = buf[bOff + 2 * j + 1];
            buf[bOff + 2 * j] = tr;
            buf[bOff + 2 * j + 1] = ti;
        }
        i = i + bs;
    }
    storageBarrier();
    workgroupBarrier();

    let sign = select(-1.0, 1.0, p.inverse != 0);
    var size : i32 = 2;
    loop {
        if (size > n) { break; }
        let half = size >> 1;
        let theta = sign * 2.0 * 3.14159265358979323846 / f32(size);
        let numButterflies = n >> 1;
        var bf : i32 = tid;
        loop {
            if (bf >= numButterflies) { break; }
            let group = bf / half;
            let k = bf - group * half;
            let e = group * size + k;
            let o = e + half;
            let angle = theta * f32(k);
            let wRe = cos(angle);
            let wIm = sin(angle);
            let eRe = buf[bOff + 2 * e];
            let eIm = buf[bOff + 2 * e + 1];
            let oRe = buf[bOff + 2 * o];
            let oIm = buf[bOff + 2 * o + 1];
            let tRe2 = wRe * oRe - wIm * oIm;
            let tIm2 = wRe * oIm + wIm * oRe;
            buf[bOff + 2 * e] = eRe + tRe2;
            buf[bOff + 2 * e + 1] = eIm + tIm2;
            buf[bOff + 2 * o] = eRe - tRe2;
            buf[bOff + 2 * o + 1] = eIm - tIm2;
            bf = bf + bs;
        }
        storageBarrier();
    workgroupBarrier();
        size = size << 1;
    }

    if (p.inverse != 0) {
        let invN = 1.0 / f32(n);
        var i2 : i32 = tid;
        loop {
            if (i2 >= n) { break; }
            buf[bOff + 2 * i2] = buf[bOff + 2 * i2] * invN;
            buf[bOff + 2 * i2 + 1] = buf[bOff + 2 * i2 + 1] * invN;
            i2 = i2 + bs;
        }
    }
}
";
    }
}
