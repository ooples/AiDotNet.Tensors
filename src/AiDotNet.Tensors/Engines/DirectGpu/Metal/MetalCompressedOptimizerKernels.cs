namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Native Metal Shading Language kernels for the compressed-moment (bf16 / int8-block) fused
/// optimizers — the real GPU implementation that replaces the per-step host round-trip fallback.
/// Ported 1:1 from the Vulkan GLSL / WebGpu WGSL kernels and verified against
/// <see cref="AiDotNet.Tensors.Engines.DirectGpu.CompressedMomentHostFallback"/> (the CPU reference).
///
/// <para>bf16 moments are two 16-bit halves per 32-bit word (one thread owns one word = 2 elements,
/// so no atomics are needed for the packed store); int8 quant moments are four bytes per word and are
/// stored with an atomic compare-exchange RMW. The int8 kernel runs one threadgroup per block with a
/// fixed 256-wide shared-memory reduction for the per-block scales.</para>
/// </summary>
internal static class MetalCompressedOptimizerKernels
{
    public const string Source = @"
#include <metal_stdlib>
using namespace metal;

static inline float bf16_to_float(uint x) { return as_type<float>(x << 16); }
static inline uint float_to_bf16_rne(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return ((bits >> 16) | 0x40u) & 0xffffu; }
    uint rounding = 0x7fffu + ((bits >> 16) & 1u);
    return ((bits + rounding) >> 16) & 0xffffu;
}

// bf16 Adam / AdamW (decoupled flag). One thread per 32-bit word (= two bf16 moments).
kernel void adam_bf16(
    device float* param       [[buffer(0)]],
    device const float* grad  [[buffer(1)]],
    device uint* m_bits       [[buffer(2)]],
    device uint* v_bits       [[buffer(3)]],
    constant uint& size       [[buffer(4)]],
    constant uint& step       [[buffer(5)]],
    constant uint& decoupled  [[buffer(6)]],
    constant float& lr        [[buffer(7)]],
    constant float& beta1     [[buffer(8)]],
    constant float& beta2     [[buffer(9)]],
    constant float& epsilon   [[buffer(10)]],
    constant float& weightDecay [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint w = gid;
    uint base = w * 2u;
    if (base >= size) { return; }

    float bc1 = 1.0f - pow(beta1, (float)step);
    float bc2 = 1.0f - pow(beta2, (float)step);
    bool firstStep = step <= 1u;

    uint mword = m_bits[w];
    uint vword = v_bits[w];
    uint outM = mword;
    uint outV = vword;

    for (uint k = 0u; k < 2u; k++) {
        uint idx = base + k;
        if (idx >= size) { break; }
        uint shift = k * 16u;

        if (decoupled == 1u && weightDecay > 0.0f) { param[idx] *= (1.0f - lr * weightDecay); }
        float g = (decoupled == 1u) ? grad[idx] : (grad[idx] + weightDecay * param[idx]);

        float oldM = firstStep ? 0.0f : bf16_to_float((mword >> shift) & 0xffffu);
        float oldV = firstStep ? 0.0f : bf16_to_float((vword >> shift) & 0xffffu);
        float newM = beta1 * oldM + (1.0f - beta1) * g;
        float newV = beta2 * oldV + (1.0f - beta2) * g * g;

        float mHat = newM / bc1;
        float vHat = newV / bc2;
        param[idx] -= lr * mHat / (sqrt(vHat) + epsilon);

        uint mask = 0xffffu << shift;
        outM = (outM & ~mask) | (float_to_bf16_rne(newM) << shift);
        outV = (outV & ~mask) | (float_to_bf16_rne(newV) << shift);
    }
    m_bits[w] = outM;
    v_bits[w] = outV;
}

static inline uint load_byte(uint word, uint i) { return (word >> ((i & 3u) * 8u)) & 0xffu; }

static inline void store_byte(device atomic_uint* buf, uint i, uint b) {
    uint wi = i >> 2u;
    uint shift = (i & 3u) * 8u;
    uint mask = 0xffu << shift;
    uint want = (b & 0xffu) << shift;
    uint expected = atomic_load_explicit(&buf[wi], memory_order_relaxed);
    for (;;) {
        uint desired = (expected & ~mask) | want;
        if (atomic_compare_exchange_weak_explicit(&buf[wi], &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) { break; }
        // expected was reloaded by the CAS on failure; retry.
    }
}

// int8 block-quantized Adam. One threadgroup per block; 256-wide shared reduction for the scales.
kernel void adam8bit(
    device float* param       [[buffer(0)]],
    device const float* grad  [[buffer(1)]],
    device atomic_uint* m_quant [[buffer(2)]],
    device atomic_uint* v_quant [[buffer(3)]],
    device float* m_scales    [[buffer(4)]],
    device float* v_scales    [[buffer(5)]],
    constant uint& paramLength [[buffer(6)]],
    constant uint& blockSize   [[buffer(7)]],
    constant uint& numBlocks   [[buffer(8)]],
    constant float& lr         [[buffer(9)]],
    constant float& beta1      [[buffer(10)]],
    constant float& beta2      [[buffer(11)]],
    constant float& epsilon    [[buffer(12)]],
    constant float& oneMinusBeta1 [[buffer(13)]],
    constant float& oneMinusBeta2 [[buffer(14)]],
    constant float& biasCorrection1 [[buffer(15)]],
    constant float& biasCorrection2 [[buffer(16)]],
    uint block [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    threadgroup float s_maxM[256];
    threadgroup float s_maxV[256];

    if (block >= numBlocks) { return; }
    uint start = block * blockSize;
    uint endIdx = min(start + blockSize, paramLength);

    bool firstStep = (biasCorrection1 <= oneMinusBeta1 + 1e-7f)
                  && (biasCorrection2 <= oneMinusBeta2 + 1e-7f);
    float oldMScale = firstStep ? 0.0f : m_scales[block];
    float oldVScale = firstStep ? 0.0f : v_scales[block];

    float localMaxM = 0.0f;
    float localMaxV = 0.0f;
    for (uint i = start + lid; i < endIdx; i += 256u) {
        float oldM = firstStep ? 0.0f : ((float)load_byte(atomic_load_explicit(&m_quant[i >> 2u], memory_order_relaxed), i) - 128.0f) * oldMScale;
        float oldV = firstStep ? 0.0f : (float)load_byte(atomic_load_explicit(&v_quant[i >> 2u], memory_order_relaxed), i) * oldVScale;
        float g = grad[i];
        float newM = beta1 * oldM + oneMinusBeta1 * g;
        float newV = beta2 * oldV + oneMinusBeta2 * g * g;
        localMaxM = max(localMaxM, fabs(newM));
        localMaxV = max(localMaxV, fabs(newV));
    }
    s_maxM[lid] = localMaxM;
    s_maxV[lid] = localMaxV;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            s_maxM[lid] = max(s_maxM[lid], s_maxM[lid + stride]);
            s_maxV[lid] = max(s_maxV[lid], s_maxV[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float newMScale = max(s_maxM[0] / 127.0f, 1e-10f);
    float newVScale = max(s_maxV[0] / 255.0f, 1e-10f);
    if (lid == 0u) {
        m_scales[block] = newMScale;
        v_scales[block] = newVScale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = start + lid; i < endIdx; i += 256u) {
        float oldM = firstStep ? 0.0f : ((float)load_byte(atomic_load_explicit(&m_quant[i >> 2u], memory_order_relaxed), i) - 128.0f) * oldMScale;
        float oldV = firstStep ? 0.0f : (float)load_byte(atomic_load_explicit(&v_quant[i >> 2u], memory_order_relaxed), i) * oldVScale;
        float g = grad[i];
        float newM = beta1 * oldM + oneMinusBeta1 * g;
        float newV = beta2 * oldV + oneMinusBeta2 * g * g;

        float mHat = newM / biasCorrection1;
        float vHat = newV / biasCorrection2;
        param[i] -= lr * mHat / (sqrt(vHat) + epsilon);

        int qm = clamp((int)round(newM / newMScale), -128, 127);
        store_byte(m_quant, i, (uint)(qm + 128));
        int qv = clamp((int)round(newV / newVScale), 0, 255);
        store_byte(v_quant, i, (uint)qv);
    }
}
";
}
