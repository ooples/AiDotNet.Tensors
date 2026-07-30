// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for HRR binding primitives (issue #248).
//
// Each kernel keeps work on the GPU — no download/upload. Per-cell
// phases use a 32-bit Murmur3 fmix hash seeded from (seed, cellIdx).
// The CPU reference and every GPU backend use the same hash and phase
// construction, so the random phase sequence is backend-independent.
//
// Int buffers (keyIds / valIds) arrive as Int32BitsToSingle-packed
// floats (see WebGpuBackend.AllocateIntBuffer); shader reverses with
// bitcast<i32>().

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static class WebGpuHrrKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "hrr_unit_phase_codebook",
        "hrr_phase_coherence_decode",
        "hrr_bind_accumulate",
    };

    // Shared hash helper — emitted in front of every shader that needs it.
    private const string HashHelper = @"
fn hrr_hash(seed_u : u32, cell_u : u32) -> u32 {
    var z = seed_u * 0x9E3779B9u + cell_u * 0x85EBCA6Bu;
    z = (z ^ (z >> 16u)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13u)) * 0xC2B2AE35u;
    z =  z ^ (z >> 16u);
    return z;
}

fn hrr_mul_hi(a : u32, b : u32) -> u32 {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p0 = a0 * b0;
    let p1 = a1 * b0;
    let p2 = a0 * b1;
    let carry = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    return a1 * b1 + (p1 >> 16u) + (p2 >> 16u) + (carry >> 16u);
}

fn hrr_quantize_turn(turn : u32, k : u32) -> u32 {
    var lattice = hrr_mul_hi(turn, k);
    if (turn * k >= 0x80000000u) { lattice = lattice + 1u; }
    if (lattice == k) { lattice = 0u; }
    var quotient = 0u;
    var remainder = lattice;
    for (var i = 0; i < 32; i = i + 1) {
        remainder = remainder << 1u;
        quotient = quotient << 1u;
        if (remainder >= k) {
            remainder = remainder - k;
            quotient = quotient | 1u;
        }
    }
    return quotient;
}

const hrr_cordic_angles = array<i32, 30>(
    536870912, 316933406, 167458907, 85004756, 42667331,
    21354465, 10679838, 5340245, 2670163, 1335087,
    667544, 333772, 166886, 83443, 41722,
    20861, 10430, 5215, 2608, 1304,
    652, 326, 163, 81, 41, 20, 10, 5, 3, 1);

fn hrr_sincos(turn : u32) -> vec2<f32> {
    let quadrant = turn >> 30u;
    let offset = turn & 0x3FFFFFFFu;
    var z = i32(offset);
    if (quadrant == 1u || quadrant == 3u) { z = i32(0x40000000u - offset); }
    var x = 652032874;
    var y = 0;
    for (var i = 0; i < 30; i = i + 1) {
        let oldX = x;
        if (z >= 0) {
            x = x - (y >> u32(i));
            y = y + (oldX >> u32(i));
            z = z - hrr_cordic_angles[i];
        } else {
            x = x + (y >> u32(i));
            y = y - (oldX >> u32(i));
            z = z + hrr_cordic_angles[i];
        }
    }
    if (quadrant == 1u || quadrant == 2u) { x = -x; }
    if (quadrant >= 2u) { y = -y; }
    return vec2<f32>(f32(x), f32(y)) * (1.0 / 1073741824.0);
}
";

    /// <summary>
    /// outReal[i], outImag[i] = cos/sin of deterministic phase derived
    /// from (seed, i). If kPsk != 0, phase snaps to nearest multiple of
    /// 2π/k (k-ary phase-shift keying).
    /// </summary>
    public static string UnitPhaseCodebook => HashHelper + @"
@group(0) @binding(0) var<storage, read_write> outReal : array<f32>;
@group(0) @binding(1) var<storage, read_write> outImag : array<f32>;
struct P { seed : i32, V : i32, D : i32, kPsk : i32, k : i32, total : i32 };
@group(0) @binding(2) var<uniform> p : P;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = id.x;
    if (gid >= u32(p.total)) { return; }

    var turn = hrr_hash(bitcast<u32>(p.seed), gid) & 0xFFFFFF00u;
    if (p.kPsk != 0) {
        turn = hrr_quantize_turn(turn, u32(p.k));
    }
    let value = hrr_sincos(turn);
    outReal[gid] = value.x;
    outImag[gid] = value.y;
}
";

    /// <summary>
    /// One work-item per V. Each thread loops D and accumulates the
    /// real part of the Hermitian inner product &lt;codes[v], query&gt;:
    /// scores[v] = Σ_d (codesR[v,d]·queryR[d] + codesI[v,d]·queryI[d]).
    /// </summary>
    public static string PhaseCoherenceDecode => @"
@group(0) @binding(0) var<storage, read> codesReal : array<f32>;
@group(0) @binding(1) var<storage, read> codesImag : array<f32>;
@group(0) @binding(2) var<storage, read> queryReal : array<f32>;
@group(0) @binding(3) var<storage, read> queryImag : array<f32>;
@group(0) @binding(4) var<storage, read_write> outScores : array<f32>;
struct P { V : i32, D : i32 };
@group(0) @binding(5) var<uniform> p : P;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let v = id.x;
    if (v >= u32(p.V)) { return; }

    let rowOff = v * u32(p.D);
    var acc : f32 = 0.0;
    for (var d : i32 = 0; d < p.D; d = d + 1) {
        acc = acc + codesReal[rowOff + u32(d)] * queryReal[d]
                  + codesImag[rowOff + u32(d)] * queryImag[d];
    }
    outScores[v] = acc;
}
";

    /// <summary>
    /// Accumulate N bindings in-place into memory[D]:
    ///   memory += Σ_n keyCode[keyIds[n]] ⊛ valPermCode[valIds[n]]
    /// where ⊛ is pointwise complex multiplication. One work-item per
    /// dimension d; each thread loops N (no cross-thread sync needed —
    /// every thread writes a distinct memory[d]).
    /// keyIds/valIds stored as Int32BitsToSingle-packed floats; we
    /// recover the ints with bitcast&lt;i32&gt;().
    /// </summary>
    public static string HrrBindAccumulate => @"
@group(0) @binding(0) var<storage, read>       keyCodeReal     : array<f32>;
@group(0) @binding(1) var<storage, read>       keyCodeImag     : array<f32>;
@group(0) @binding(2) var<storage, read>       valPermCodeReal : array<f32>;
@group(0) @binding(3) var<storage, read>       valPermCodeImag : array<f32>;
@group(0) @binding(4) var<storage, read>       keyIds          : array<f32>;
@group(0) @binding(5) var<storage, read>       valIds          : array<f32>;
@group(0) @binding(6) var<storage, read_write> memoryReal      : array<f32>;
@group(0) @binding(7) var<storage, read_write> memoryImag      : array<f32>;
// nKeys / nVals are the vocabulary sizes (codebook row counts); the
// host passes keyCodeReal.Size / D and valPermCodeReal.Size / D so
// the shader can reject out-of-range ids without reading past the
// codebook. Mirrors the (uint)kId >= (uint)nKeys check in the CPU
// SIMD implementation and the equivalent guard in the Vulkan shader.
struct P { N : i32, D : i32, nKeys : i32, nVals : i32 };
@group(0) @binding(8) var<uniform> p : P;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let d = id.x;
    if (d >= u32(p.D)) { return; }

    var accR = memoryReal[d];
    var accI = memoryImag[d];
    for (var n : i32 = 0; n < p.N; n = n + 1) {
        let kid = bitcast<i32>(keyIds[n]);
        let vid = bitcast<i32>(valIds[n]);
        // Unsigned-comparison trick rejects both negative and
        // too-large indices in one branch; out-of-range pairs
        // contribute zero (match the CPU path's exception, but
        // silently — the shader can't throw).
        if (u32(kid) >= u32(p.nKeys) || u32(vid) >= u32(p.nVals)) { continue; }
        let kOff = u32(kid) * u32(p.D) + d;
        let vOff = u32(vid) * u32(p.D) + d;
        let ar = keyCodeReal[kOff];
        let ai = keyCodeImag[kOff];
        let br = valPermCodeReal[vOff];
        let bi = valPermCodeImag[vOff];
        accR = accR + ar * br - ai * bi;
        accI = accI + ar * bi + ai * br;
    }
    memoryReal[d] = accR;
    memoryImag[d] = accI;
}
";
}
#endif
