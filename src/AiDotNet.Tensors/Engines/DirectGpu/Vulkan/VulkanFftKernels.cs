// Copyright (c) AiDotNet. All rights reserved.
// Vulkan GLSL compute shader — radix-2 Cooley-Tukey FFT matching the Metal/
// WebGPU peers. One workgroup per batch slice; threads cooperate on the
// log₂n butterfly stages.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan
{
    internal static class VulkanFftKernels
    {
        // Uses the GLSL binary-op plumbing already established by the Linalg
        // kernels — one input/output SSBO + a push-constant block.
        private const string Header = @"
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
";

        // Two-binding layout (input RO, output RW) to fit GlslUnaryOp's pipeline.
        // The first phase copies input → output; all subsequent work is in-place
        // on `output` (which makes the actual FFT in-place semantics of the
        // algorithm work out with zero extra allocations).
        public static string Fft => Header + @"
layout(set = 0, binding = 0) readonly buffer InBuf { float src[]; };
layout(set = 0, binding = 1) buffer OutBuf { float x[]; };
layout(push_constant) uniform P { int batchCount; int n; int inverse; };

void main() {
    uint b = gl_WorkGroupID.x;
    if (int(b) >= batchCount) return;
    uint tid = gl_LocalInvocationID.x;
    uint blockSize = gl_WorkGroupSize.x;
    uint bOff = b * uint(2 * n);

    // Copy input → output so subsequent in-place processing works cleanly.
    for (uint i = tid; i < uint(2 * n); i += blockSize) x[bOff + i] = src[bOff + i];
    barrier();

    // Bit reversal permutation.
    int bits = 0;
    for (int t = n; t > 1; t >>= 1) bits++;
    for (uint i = tid; i < uint(n); i += blockSize) {
        int j = 0;
        int v = int(i);
        for (int bb = 0; bb < bits; bb++) { j = (j << 1) | (v & 1); v >>= 1; }
        if (j > int(i)) {
            float tr = x[bOff + 2u * i];
            float ti = x[bOff + 2u * i + 1u];
            x[bOff + 2u * i] = x[bOff + 2u * uint(j)];
            x[bOff + 2u * i + 1u] = x[bOff + 2u * uint(j) + 1u];
            x[bOff + 2u * uint(j)] = tr;
            x[bOff + 2u * uint(j) + 1u] = ti;
        }
    }
    barrier();

    float sign = (inverse != 0) ? 1.0 : -1.0;
    for (int size = 2; size <= n; size <<= 1) {
        int half = size >> 1;
        float theta = sign * 2.0 * 3.14159265358979323846 / float(size);
        int numButterflies = n >> 1;
        for (int bf = int(tid); bf < numButterflies; bf += int(blockSize)) {
            int group = bf / half;
            int k = bf - group * half;
            int e = group * size + k;
            int o = e + half;
            float angle = theta * float(k);
            float wRe = cos(angle);
            float wIm = sin(angle);
            float eRe = x[bOff + 2u * uint(e)];
            float eIm = x[bOff + 2u * uint(e) + 1u];
            float oRe = x[bOff + 2u * uint(o)];
            float oIm = x[bOff + 2u * uint(o) + 1u];
            float tRe = wRe * oRe - wIm * oIm;
            float tIm = wRe * oIm + wIm * oRe;
            x[bOff + 2u * uint(e)] = eRe + tRe;
            x[bOff + 2u * uint(e) + 1u] = eIm + tIm;
            x[bOff + 2u * uint(o)] = eRe - tRe;
            x[bOff + 2u * uint(o) + 1u] = eIm - tIm;
        }
        barrier();
    }

    if (inverse != 0) {
        float invN = 1.0 / float(n);
        for (uint i = tid; i < uint(n); i += blockSize) {
            x[bOff + 2u * i] *= invN;
            x[bOff + 2u * i + 1u] *= invN;
        }
    }
}";
    }
}
