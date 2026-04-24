// Copyright (c) AiDotNet. All rights reserved.
// GLSL 450 compute shaders for HRR binding primitives (issue #248).
//
// Each kernel keeps work on the GPU — no download/upload. Per-cell
// phases use a 32-bit Murmur3 fmix hash seeded from (seed, cellIdx).
// This does NOT bit-match the CPU xorshift64* sequence or the CUDA
// splitmix64 sequence; each backend generates its own uniformly
// distributed phase. Determinism within a backend (same seed + same
// (V, D) → identical output) is preserved.
//
// Int buffers (keyIds / valIds) arrive as float bitcasts — see
// VulkanBackend.AllocateIntBuffer which packs via Int32BitsToSingle.
// Inside the shader we reverse this with floatBitsToInt().

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanHrrKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;

uint hrr_hash(uint seed_u, uint cell_u) {
    // Murmur3-fmix — uniform 32-bit output for any 32-bit input.
    uint z = seed_u * 0x9E3779B9u + cell_u * 0x85EBCA6Bu;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z =  z ^ (z >> 16);
    return z;
}

float hrr_phase_from_cell(int seed, uint cellIdx) {
    uint z = hrr_hash(uint(seed), cellIdx);
    // Upper 24 bits → exact-representable float in [0, 1).
    uint top24 = z >> 8;
    return float(top24) * (1.0 / 16777216.0) * 6.28318530717958647692;
}
";

    /// <summary>
    /// outReal[i], outImag[i] = cos/sin of deterministic phase derived
    /// from (seed, i). If kPsk != 0, the phase snaps to the nearest
    /// multiple of 2π/k (k-ary phase-shift keying).
    /// </summary>
    public static string UnitPhaseCodebook => Header + @"
layout(set = 0, binding = 0) writeonly buffer OR { float outReal[]; };
layout(set = 0, binding = 1) writeonly buffer OI { float outImag[]; };
layout(push_constant) uniform P {
    int seed;
    int V;
    int D;
    int kPsk;
    int k;
    int total;  // = V * D (pre-computed to avoid mul in hot path)
} p;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(p.total)) return;

    float phase = hrr_phase_from_cell(p.seed, gid);
    // Defense-in-depth: host validates (kPsk && k <= 0) → throw, so
    // the shader should never see k <= 0 with kPsk set, but guard
    // against division-by-zero anyway to avoid NaN propagation if
    // that invariant is ever broken (e.g., by a future caller that
    // bypasses the host check).
    if (p.kPsk != 0 && p.k > 0) {
        float step = 6.28318530717958647692 / float(p.k);
        phase = floor(phase / step + 0.5) * step;
    }
    outReal[gid] = cos(phase);
    outImag[gid] = sin(phase);
}
";

    /// <summary>
    /// One work-item per V. Each thread loops D and accumulates the
    /// real part of the Hermitian inner product &lt;codes[v], query&gt;:
    /// scores[v] = Σ_d (codesR[v,d]·queryR[d] + codesI[v,d]·queryI[d]).
    /// </summary>
    public static string PhaseCoherenceDecode => Header + @"
layout(set = 0, binding = 0) readonly buffer CR { float codesReal[]; };
layout(set = 0, binding = 1) readonly buffer CI { float codesImag[]; };
layout(set = 0, binding = 2) readonly buffer QR { float queryReal[]; };
layout(set = 0, binding = 3) readonly buffer QI { float queryImag[]; };
layout(set = 0, binding = 4) writeonly buffer OS { float outScores[]; };
layout(push_constant) uniform P {
    int V;
    int D;
} p;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= uint(p.V)) return;

    uint rowOff = v * uint(p.D);
    float acc = 0.0;
    for (int d = 0; d < p.D; d++) {
        acc += codesReal[rowOff + uint(d)] * queryReal[d]
             + codesImag[rowOff + uint(d)] * queryImag[d];
    }
    outScores[v] = acc;
}
";

    /// <summary>
    /// Accumulate N bindings in-place into memory[D]:
    ///   memory += Σ_n keyCode[keyIds[n]] ⊛ valPermCode[valIds[n]]
    /// where ⊛ is pointwise complex multiplication.
    /// One work-item per dimension d; each thread loops N (no cross-
    /// thread synchronization needed — every thread writes a different
    /// memory[d]).
    /// keyIds/valIds are stored as Int32BitsToSingle-packed floats
    /// (see VulkanBackend.AllocateIntBuffer); we recover the ints with
    /// floatBitsToInt().
    /// </summary>
    public static string HrrBindAccumulate => Header + @"
layout(set = 0, binding = 0) readonly buffer KR  { float keyCodeReal[]; };
layout(set = 0, binding = 1) readonly buffer KI  { float keyCodeImag[]; };
layout(set = 0, binding = 2) readonly buffer VR  { float valPermCodeReal[]; };
layout(set = 0, binding = 3) readonly buffer VI  { float valPermCodeImag[]; };
layout(set = 0, binding = 4) readonly buffer KID { float keyIds[]; };
layout(set = 0, binding = 5) readonly buffer VID { float valIds[]; };
layout(set = 0, binding = 6)          buffer MR  { float memoryReal[]; };
layout(set = 0, binding = 7)          buffer MI  { float memoryImag[]; };
// nKeys / nVals are the vocabulary sizes (codebook row counts); the
// host passes keyCodeReal.Size / D and valPermCodeReal.Size / D so
// the shader can reject out-of-range ids without reading past the
// codebook (mirrors the (uint)kId >= (uint)nKeys check in the CPU
// SIMD implementation).
layout(push_constant) uniform P {
    int N;
    int D;
    int nKeys;
    int nVals;
} p;

void main() {
    uint d = gl_GlobalInvocationID.x;
    if (d >= uint(p.D)) return;

    float accR = memoryReal[d];
    float accI = memoryImag[d];
    for (int n = 0; n < p.N; n++) {
        int kid = floatBitsToInt(keyIds[n]);
        int vid = floatBitsToInt(valIds[n]);
        // Unsigned-comparison trick rejects both negative and
        // too-large indices in one branch; out-of-range pairs
        // contribute zero (match the CPU path's exception, but
        // silently — the shader can't throw).
        if (uint(kid) >= uint(p.nKeys) || uint(vid) >= uint(p.nVals)) continue;
        uint kOff = uint(kid) * uint(p.D) + d;
        uint vOff = uint(vid) * uint(p.D) + d;
        float ar = keyCodeReal[kOff];
        float ai = keyCodeImag[kOff];
        float br = valPermCodeReal[vOff];
        float bi = valPermCodeImag[vOff];
        accR += ar * br - ai * bi;
        accI += ar * bi + ai * br;
    }
    memoryReal[d] = accR;
    memoryImag[d] = accI;
}
";
}
