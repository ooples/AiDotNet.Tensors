namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Native GLSL compute kernels for the compressed-moment (bf16 / int8-block) fused optimizers on
/// Vulkan — the real GPU implementation that replaces the per-step host round-trip fallback.
/// Ported 1:1 from the WebGpu WGSL kernels and verified against
/// <see cref="AiDotNet.Tensors.Engines.DirectGpu.CompressedMomentHostFallback"/> (the CPU reference).
///
/// <para>Buffer layouts (matching the host packer):
/// bf16 moments are two 16-bit halves per 32-bit word (<c>m_bits[w]</c> holds elements 2w, 2w+1);
/// int8 quant moments are four bytes per word (M is signed with a 128 zero-point, V is unsigned).
/// The bf16 kernels dispatch ONE thread per WORD (2 elements) so the two halves of a word are never
/// written by different threads — no atomics needed. The int8 kernel dispatches ONE workgroup per
/// block (a fixed 256-wide shared-memory reduction for the per-block scales) and stores the quantized
/// bytes with an atomicCompSwap RMW since four elements share a word.</para>
/// </summary>
internal static class VulkanCompressedOptimizerKernels
{
    // Shared bf16 <-> float helpers (round-to-nearest-even, NaN-preserving) — identical to the WGSL /
    // CPU FloatToBf16Rne so the stored moments are bit-identical across backends.
    private const string Bf16Helpers = @"
float bf16_to_float(uint x) { return uintBitsToFloat(x << 16); }
uint float_to_bf16_rne(float x) {
    uint bits = floatBitsToUint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return ((bits >> 16) | 0x40u) & 0xffffu; }
    uint rounding = 0x7fffu + ((bits >> 16) & 1u);
    return ((bits + rounding) >> 16) & 0xffffu;
}
";

    /// <summary>
    /// bf16 Adam AND AdamW (selected by <c>decoupled</c>): matches CompressedMomentHostFallback.AdamBf16.
    /// One thread owns one 32-bit word = two consecutive bf16 moments.
    /// </summary>
    public static readonly string AdamBf16 = @"#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer P  { float param[]; };
layout(std430, binding = 1) readonly buffer G { float grad[]; };
layout(std430, binding = 2) buffer M  { uint m_bits[]; };
layout(std430, binding = 3) buffer V  { uint v_bits[]; };
layout(push_constant) uniform PC {
    uint size;
    uint step;
    uint decoupled;    // 1 = AdamW (decoupled weight decay), 0 = Adam (L2 into gradient)
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weightDecay;
} pc;
" + Bf16Helpers + @"
void main() {
    uint w = gl_GlobalInvocationID.x;
    uint base = w * 2u;
    if (base >= pc.size) { return; }

    float bc1 = 1.0 - pow(pc.beta1, float(pc.step));
    float bc2 = 1.0 - pow(pc.beta2, float(pc.step));
    bool firstStep = pc.step <= 1u;

    uint mword = m_bits[w];
    uint vword = v_bits[w];
    uint outM = mword;
    uint outV = vword;

    for (uint k = 0u; k < 2u; k++) {
        uint idx = base + k;
        if (idx >= pc.size) { break; }
        uint shift = k * 16u;

        if (pc.decoupled == 1u && pc.weightDecay > 0.0) {
            param[idx] *= (1.0 - pc.lr * pc.weightDecay);
        }
        float g = (pc.decoupled == 1u) ? grad[idx] : (grad[idx] + pc.weightDecay * param[idx]);

        float oldM = firstStep ? 0.0 : bf16_to_float((mword >> shift) & 0xffffu);
        float oldV = firstStep ? 0.0 : bf16_to_float((vword >> shift) & 0xffffu);
        float newM = pc.beta1 * oldM + (1.0 - pc.beta1) * g;
        float newV = pc.beta2 * oldV + (1.0 - pc.beta2) * g * g;

        float mHat = newM / bc1;
        float vHat = newV / bc2;
        param[idx] -= pc.lr * mHat / (sqrt(vHat) + pc.epsilon);

        uint mMask = 0xffffu << shift;
        outM = (outM & ~mMask) | (float_to_bf16_rne(newM) << shift);
        outV = (outV & ~mMask) | (float_to_bf16_rne(newV) << shift);
    }
    m_bits[w] = outM;
    v_bits[w] = outV;
}
";

    /// <summary>
    /// int8 block-quantized Adam: matches CompressedMomentHostFallback.Adam8Bit. One workgroup per block
    /// (up to 256 elements) does a shared-memory reduction for the new per-block M/V scales, then
    /// quantizes. M is signed (zero-point 128, /127), V is unsigned (zero-point 0, /255).
    /// </summary>
    public static readonly string Adam8Bit = @"#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer P  { float param[]; };
layout(std430, binding = 1) readonly buffer G { float grad[]; };
layout(std430, binding = 2) buffer MQ { uint m_quant[]; };  // 4 bytes/word, signed, +128 bias
layout(std430, binding = 3) buffer VQ { uint v_quant[]; };  // 4 bytes/word, unsigned
layout(std430, binding = 4) buffer MS { float m_scales[]; };
layout(std430, binding = 5) buffer VS { float v_scales[]; };
layout(push_constant) uniform PC {
    uint paramLength;
    uint blockSize;
    uint numBlocks;
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float oneMinusBeta1;
    float oneMinusBeta2;
    float biasCorrection1;
    float biasCorrection2;
} pc;

shared float s_maxM[256];
shared float s_maxV[256];

uint load_byte(uint word, uint i) { return (word >> ((i & 3u) * 8u)) & 0xffu; }

void store_byte(uint isV, uint i, uint b) {
    uint wi = i >> 2u;
    uint shift = (i & 3u) * 8u;
    uint mask = 0xffu << shift;
    uint want = (b & 0xffu) << shift;
    // RMW: four elements share a 32-bit word and are written by different threads, so CAS.
    for (;;) {
        uint oldW = (isV == 1u) ? v_quant[wi] : m_quant[wi];
        uint newW = (oldW & ~mask) | want;
        uint prev = (isV == 1u) ? atomicCompSwap(v_quant[wi], oldW, newW)
                                : atomicCompSwap(m_quant[wi], oldW, newW);
        if (prev == oldW) { break; }
    }
}

void main() {
    uint block = gl_WorkGroupID.x;
    if (block >= pc.numBlocks) { return; }
    uint lid = gl_LocalInvocationID.x;
    uint start = block * pc.blockSize;
    uint end = min(start + pc.blockSize, pc.paramLength);

    bool firstStep = (pc.biasCorrection1 <= pc.oneMinusBeta1 + 1e-7)
                  && (pc.biasCorrection2 <= pc.oneMinusBeta2 + 1e-7);
    float oldMScale = firstStep ? 0.0 : m_scales[block];
    float oldVScale = firstStep ? 0.0 : v_scales[block];

    // Pass 1: reduce max|newM|, max|newV| over the block for the new scales.
    float localMaxM = 0.0;
    float localMaxV = 0.0;
    for (uint i = start + lid; i < end; i += 256u) {
        float oldM = firstStep ? 0.0 : (float(load_byte(m_quant[i >> 2u], i)) - 128.0) * oldMScale;
        float oldV = firstStep ? 0.0 : float(load_byte(v_quant[i >> 2u], i)) * oldVScale;
        float g = grad[i];
        float newM = pc.beta1 * oldM + pc.oneMinusBeta1 * g;
        float newV = pc.beta2 * oldV + pc.oneMinusBeta2 * g * g;
        localMaxM = max(localMaxM, abs(newM));
        localMaxV = max(localMaxV, abs(newV));
    }
    s_maxM[lid] = localMaxM;
    s_maxV[lid] = localMaxV;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            s_maxM[lid] = max(s_maxM[lid], s_maxM[lid + stride]);
            s_maxV[lid] = max(s_maxV[lid], s_maxV[lid + stride]);
        }
        barrier();
    }
    float newMScale = max(s_maxM[0] / 127.0, 1e-10);
    float newVScale = max(s_maxV[0] / 255.0, 1e-10);
    if (lid == 0u) {
        m_scales[block] = newMScale;
        v_scales[block] = newVScale;
    }
    barrier();

    // Pass 2: recompute, update param, quantize with the new scales.
    for (uint i = start + lid; i < end; i += 256u) {
        float oldM = firstStep ? 0.0 : (float(load_byte(m_quant[i >> 2u], i)) - 128.0) * oldMScale;
        float oldV = firstStep ? 0.0 : float(load_byte(v_quant[i >> 2u], i)) * oldVScale;
        float g = grad[i];
        float newM = pc.beta1 * oldM + pc.oneMinusBeta1 * g;
        float newV = pc.beta2 * oldV + pc.oneMinusBeta2 * g * g;

        float mHat = newM / pc.biasCorrection1;
        float vHat = newV / pc.biasCorrection2;
        param[i] -= pc.lr * mHat / (sqrt(vHat) + pc.epsilon);

        int qm = int(round(newM / newMScale));
        qm = clamp(qm, -128, 127);
        store_byte(0u, i, uint(qm + 128));
        int qv = int(round(newV / newVScale));
        qv = clamp(qv, 0, 255);
        store_byte(1u, i, uint(qv));
    }
}
";
}
