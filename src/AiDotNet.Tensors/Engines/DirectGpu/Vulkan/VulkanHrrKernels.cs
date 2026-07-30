// Copyright (c) AiDotNet. All rights reserved.
// GLSL 450 compute shaders for HRR binding primitives (issue #248).
//
// Each kernel keeps work on the GPU — no download/upload. Per-cell
// phases use a 32-bit Murmur3 fmix hash seeded from (seed, cellIdx).
// The CPU reference and every GPU backend use the same hash and phase
// construction, so the random phase sequence is backend-independent.
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

uint hrr_mul_hi(uint a, uint b) {
    uint a0 = a & 0xFFFFu, a1 = a >> 16;
    uint b0 = b & 0xFFFFu, b1 = b >> 16;
    uint p0 = a0 * b0, p1 = a1 * b0, p2 = a0 * b1;
    uint carry = (p0 >> 16) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    return a1 * b1 + (p1 >> 16) + (p2 >> 16) + (carry >> 16);
}

uint hrr_quantize_turn(uint turn, uint k) {
    uint lattice = hrr_mul_hi(turn, k);
    if (turn * k >= 0x80000000u) lattice++;
    if (lattice == k) lattice = 0;
    uint quotient = 0, remainder = lattice;
    for (int i = 0; i < 32; i++) {
        remainder <<= 1;
        quotient <<= 1;
        if (remainder >= k) { remainder -= k; quotient |= 1u; }
    }
    return quotient;
}

const int hrr_cordic_angles[30] = int[30](
    536870912, 316933406, 167458907, 85004756, 42667331,
    21354465, 10679838, 5340245, 2670163, 1335087,
    667544, 333772, 166886, 83443, 41722,
    20861, 10430, 5215, 2608, 1304,
    652, 326, 163, 81, 41, 20, 10, 5, 3, 1);

vec2 hrr_sincos(uint turn) {
    uint quadrant = turn >> 30;
    uint offset = turn & 0x3FFFFFFFu;
    int z = int((quadrant == 1u || quadrant == 3u) ? 0x40000000u - offset : offset);
    int x = 652032874, y = 0;
    for (int i = 0; i < 30; i++) {
        int oldX = x;
        if (z >= 0) {
            x -= y >> i; y += oldX >> i; z -= hrr_cordic_angles[i];
        } else {
            x += y >> i; y -= oldX >> i; z += hrr_cordic_angles[i];
        }
    }
    if (quadrant == 1u || quadrant == 2u) x = -x;
    if (quadrant >= 2u) y = -y;
    return vec2(float(x), float(y)) * (1.0 / 1073741824.0);
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

    uint turn = hrr_hash(uint(p.seed), gid) & 0xFFFFFF00u;
    if (p.kPsk != 0 && p.k > 0) {
        turn = hrr_quantize_turn(turn, uint(p.k));
    }
    vec2 value = hrr_sincos(turn);
    outReal[gid] = value.x;
    outImag[gid] = value.y;
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
