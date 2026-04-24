// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for HRR binding primitives (issue #248).
//
// Each kernel keeps work on the GPU — no download/upload. Per-cell
// phases use a 32-bit Murmur3 fmix hash seeded from (seed, cellIdx).
// Deterministic within the WebGPU backend (same seed + same (V, D) →
// identical output). Does not bit-match CPU, CUDA, or Vulkan phase
// sequences — each backend generates its own uniform distribution.
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

fn hrr_phase_from_cell(seed : i32, cellIdx : u32) -> f32 {
    let z = hrr_hash(bitcast<u32>(seed), cellIdx);
    // Upper 24 bits → exact-representable f32 in [0, 1).
    let top24 = z >> 8u;
    return f32(top24) * (1.0 / 16777216.0) * 6.28318530717958647692;
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

    var phase = hrr_phase_from_cell(p.seed, gid);
    if (p.kPsk != 0) {
        let step = 6.28318530717958647692 / f32(p.k);
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
