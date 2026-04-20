// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language radix-2 Cooley-Tukey FFT kernel. One workgroup per
// batch slice; threads cooperate on the log₂ n butterfly stages via
// threadgroup memory. Works on interleaved float re/im pairs (same layout
// as the Linalg kernels).
//
// Limitations (documented — match the IFftBackend contract):
//   * n must be a power of two
//   * n ≤ 1024 (thread-per-butterfly scheme; larger sizes split into tiles
//     in a future pass or route to CPU Bluestein)

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    internal static class MetalFftKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "parity212_fft",
        };

        public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// One threadblock per batch slice; threads cooperate on n/2 butterflies.
// buffer layout: buffer[b * 2n + 2i + 0] = Re(x_b[i])
//                buffer[b * 2n + 2i + 1] = Im(x_b[i])
kernel void parity212_fft(
    device float* buf [[buffer(0)]],
    constant int& batchCount [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& inverse [[buffer(3)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint blockSize [[threads_per_threadgroup]])
{
    if ((int)b >= batchCount) return;
    device float* x = buf + b * 2 * n;

    // Bit-reversal permutation — each thread swaps one pair of indices.
    int bits = 0;
    for (int t = n; t > 1; t >>= 1) bits++;
    for (int i = tid; i < n; i += blockSize) {
        int j = 0;
        int v = i;
        for (int b2 = 0; b2 < bits; b2++) { j = (j << 1) | (v & 1); v >>= 1; }
        if (j > i) {
            float tr = x[2 * i];
            float ti = x[2 * i + 1];
            x[2 * i] = x[2 * j];
            x[2 * i + 1] = x[2 * j + 1];
            x[2 * j] = tr;
            x[2 * j + 1] = ti;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    float sign = inverse ? 1.0f : -1.0f;
    for (int size = 2; size <= n; size <<= 1) {
        int half = size >> 1;
        float theta = sign * 2.0f * M_PI_F / (float)size;
        // Each thread handles one butterfly per group.
        int numButterflies = n >> 1;
        for (int bf = (int)tid; bf < numButterflies; bf += blockSize) {
            int group = bf / half;
            int k = bf - group * half;
            int e = group * size + k;
            int o = e + half;
            float angle = theta * (float)k;
            float wRe = cos(angle);
            float wIm = sin(angle);
            float eRe = x[2 * e];
            float eIm = x[2 * e + 1];
            float oRe = x[2 * o];
            float oIm = x[2 * o + 1];
            float tRe = wRe * oRe - wIm * oIm;
            float tIm = wRe * oIm + wIm * oRe;
            x[2 * e] = eRe + tRe;
            x[2 * e + 1] = eIm + tIm;
            x[2 * o] = eRe - tRe;
            x[2 * o + 1] = eIm - tIm;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // Apply 1/n scaling on inverse (backward-norm convention).
    if (inverse != 0) {
        float invN = 1.0f / (float)n;
        for (int i = tid; i < n; i += blockSize) {
            x[2 * i] *= invN;
            x[2 * i + 1] *= invN;
        }
    }
}
";
    }
}
