// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>CSR attention in the shared GPU model-op dialect. The backward column gather uses the
/// graph transpose so no float atomics are required. Each dispatch fits eight storage bindings.</summary>
internal static class PortableGraphAttentionKernels
{
    private const string LoadScore = """
        int j = (baseNode + cols[e]) * heads + h;
        uint2 neighbor = make_uint2((unsigned int)(neighborWords[2*j]), (unsigned int)(neighborWords[2*j+1]));
        uint2 raw = gat_add(ss, neighbor);
        uint2 value = raw;
        if (!gat_less(make_uint2(0u, 0u), raw)) { value = gat_scale(raw, slopeLo, slopeHi); }
""";

    private const string NormalizeRow = """
    int t = bx_gid();
    if (t >= total) { return; }
    int h = t % heads;
    int node = (t / heads) % nodes;
    int baseNode = (t / heads) - node;
    int batchIndex = baseNode / nodes;
    int begin = rowPtr[node];
    int end = rowPtr[node + 1];
    uint2 ss = make_uint2((unsigned int)(selfWords[2*t]), (unsigned int)(selfWords[2*t+1]));
    uint2 largest = make_uint2(0u, 0u);
    for (int e = begin; e < end; e = e + 1) {

""" + LoadScore + """

        if (e == begin || gat_less(largest, value)) { largest = value; }
    }
    float sum = 0.0f;
    for (int e = begin; e < end; e = e + 1) {

""" + LoadScore + """

        sum = sum + gat_shifted_exp(value, largest);
    }
""";

    internal const string Source = "\n" + """
// Values in the two score buffers are binary64 words. Metal and WGSL have no native
// double. Keep the full float-product range and round each sum before the max shift.
__device__ __forceinline__ uint2 gat_pack(uint2 v, int power, unsigned int sign)
{
    int bits = mo_u64_bitlen(v);
    if (bits == 0) { return make_uint2(0u, sign); }
    int biased = power + bits + 1022;
    int shift = bits - 53;
    if (biased <= 0) { shift = -1074 - power; }
    if (shift >= 64) { return make_uint2(0u, sign); }
    uint2 m = v;
    if (shift > 0) { m = mo_round_shift(v, (unsigned int)shift); }
    if (shift < 0) { m = mo_u64_shl(v, (unsigned int)(0 - shift)); }
    if (biased <= 0) { return make_uint2(m.x, sign | m.y); }
    if (m.y >= 0x200000u) { m = mo_u64_shr(m, 1u); biased = biased + 1; }
    if (biased >= 2047) { return make_uint2(0u, sign | 0x7ff00000u); }
    return make_uint2(m.x, sign | ((unsigned int)biased << 20u) | (m.y & 0xfffffu));
}

__device__ __forceinline__ uint2 gat_sig(uint2 v)
{
    unsigned int hi = v.y & 0xfffffu;
    if ((v.y & 0x7ff00000u) != 0u) { hi = hi | 0x100000u; }
    return make_uint2(v.x, hi);
}

__device__ __forceinline__ int gat_power(uint2 v)
{
    int biased = (int)((v.y >> 20u) & 2047u);
    if (biased == 0) { return -1074; }
    return biased - 1075;
}

__device__ __forceinline__ uint2 gat_neg(uint2 v) { return make_uint2(v.x, v.y ^ 0x80000000u); }

__device__ __forceinline__ bool gat_less(uint2 a, uint2 b)
{
    uint2 ma = make_uint2(a.x, a.y & 0x7fffffffu);
    uint2 mb = make_uint2(b.x, b.y & 0x7fffffffu);
    if ((ma.x | ma.y | mb.x | mb.y) == 0u) { return false; }
    if (((a.y ^ b.y) >> 31u) != 0u) { return (a.y >> 31u) != 0u; }
    if ((a.y >> 31u) != 0u) { return mo_u64_less(mb, ma); }
    return mo_u64_less(ma, mb);
}

__device__ __forceinline__ uint2 gat_add(uint2 a, uint2 b)
{
    uint2 big = a;
    uint2 small = b;
    if (mo_u64_less(make_uint2(a.x, a.y & 0x7fffffffu), make_uint2(b.x, b.y & 0x7fffffffu))) { big = b; small = a; }
    unsigned int sign = big.y & 0x80000000u;
    if ((big.y & 0x7ff00000u) == 0x7ff00000u) {
        if ((small.y ^ big.y) == 0x80000000u && small.x == big.x) { return make_uint2(0u, 0x7ff80000u); }
        return big;
    }
    int power = gat_power(big);
    uint2 hi = mo_u64_shl(gat_sig(big), 5u);
    uint2 lo = mo_shr_sticky(mo_u64_shl(gat_sig(small), 5u), power - gat_power(small));
    if (((big.y ^ small.y) >> 31u) != 0u) { return gat_pack(mo_u64_sub(hi, lo), power - 5, sign); }
    return gat_pack(mo_u64_add(hi, lo), power - 5, sign);
}

// A float product has at most 48 significant bits and is exact in the score accumulator.
__device__ __forceinline__ uint2 gat_product(float a, float b)
{
    unsigned int x = __float_as_uint(a);
    unsigned int y = __float_as_uint(b);
    int ex = (int)((x >> 23u) & 255u);
    int ey = (int)((y >> 23u) & 255u);
    unsigned int mx = x & 0x7fffffu;
    unsigned int my = y & 0x7fffffu;
    unsigned int sign = (x ^ y) & 0x80000000u;
    if (ex == 255 || ey == 255) {
        if (mo_isnan(a) || mo_isnan(b) || a == 0.0f || b == 0.0f) { return make_uint2(0u, 0x7ff80000u); }
        return make_uint2(0u, sign | 0x7ff00000u);
    }
    if (ex != 0) { mx = mx | 0x800000u; } else { ex = 1; }
    if (ey != 0) { my = my | 0x800000u; } else { ey = 1; }
    return gat_pack(mo_mul_wide(mx, my), ex + ey - 300, sign);
}

// Multiply a binary64 value by the caller's binary64 LeakyReLU slope. The upper
// 58 bits of the 106-bit product, with a sticky low bit, suffice for one rounding.
__device__ __forceinline__ uint2 gat_scale(uint2 z, int slopeLo, int slopeHi)
{
    uint2 s = make_uint2((unsigned int)slopeLo, (unsigned int)slopeHi);
    unsigned int sign = (s.y ^ z.y) & 0x80000000u;
    uint2 x = gat_sig(z);
    uint2 y = gat_sig(s);
    bool zeroX = (x.x | x.y) == 0u;
    bool zeroY = (y.x | y.y) == 0u;
    if ((s.y & 0x7ff00000u) == 0x7ff00000u || (z.y & 0x7ff00000u) == 0x7ff00000u) {
        if (zeroX || zeroY || ((s.y & 0x7ff00000u) == 0x7ff00000u && (s.x | (s.y & 0xfffffu)) != 0u)
            || ((z.y & 0x7ff00000u) == 0x7ff00000u && (z.x | (z.y & 0xfffffu)) != 0u)) { return make_uint2(0u, 0x7ff80000u); }
        return make_uint2(0u, sign | 0x7ff00000u);
    }
    if (zeroX || zeroY) { return make_uint2(0u, sign); }
    // Normalize subnormal operands before choosing the fixed 58-bit product window.
    int normalizeX = 53 - mo_u64_bitlen(x);
    int normalizeY = 53 - mo_u64_bitlen(y);
    x = mo_u64_shl(x, (unsigned int)normalizeX);
    y = mo_u64_shl(y, (unsigned int)normalizeY);
    uint2 ll = mo_mul_wide(x.x, y.x);
    uint2 lh = mo_mul_wide(x.x, y.y);
    uint2 hl = mo_mul_wide(x.y, y.x);
    uint2 hh = mo_mul_wide(x.y, y.y);
    uint2 mid = mo_u64_add(make_uint2(ll.y, 0u), mo_u64_add(make_uint2(lh.x, 0u), make_uint2(hl.x, 0u)));
    uint2 high = mo_u64_add(hh, mo_u64_add(make_uint2(mid.y + lh.y, 0u), make_uint2(hl.y, 0u)));
    uint2 window = mo_u64_shl(high, 16u);
    window.x = window.x | (mid.x >> 16u);
    if ((ll.x | (mid.x & 65535u)) != 0u) { window.x = window.x | 1u; }
    return gat_pack(window, gat_power(z) + gat_power(s) + 48 - normalizeX - normalizeY, sign);
}

__device__ __forceinline__ float gat_float(uint2 delta)
{
    if ((delta.y & 0x7ff00000u) == 0x7ff00000u) {
        if ((delta.y & 0xfffffu) != 0u || delta.x != 0u) { return mo_nan(); }
        return __uint_as_float((delta.y & 0x80000000u) | 0x7f800000u);
    }
    return mo_to_float(gat_sig(delta), gat_power(delta), delta.y & 0x80000000u);
}

__device__ __forceinline__ float gat_shifted_exp(uint2 value, uint2 largest)
{
    return expf(gat_float(gat_add(value, gat_neg(largest))));
}

__device__ __forceinline__ float gat_mask(int seed, int slot, int cutoff, float keep)
{
    unsigned int x = (unsigned int)slot * 0x9e3779b1u + (unsigned int)seed;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    if (((x ^ (x >> 16u)) >> 8u) < (unsigned int)cutoff) { return 0.0f; }
    return keep;
}

extern "C" __global__ __launch_bounds__(256) void gat_scores(const float* wh, const float* attn, int* selfWords, int* neighborWords, int total, int heads, int features)
{
    int t = bx_gid();
    if (t >= total) { return; }
    int a = (t % heads) * 2 * features;
    uint2 ss = make_uint2(0u, 0u);
    uint2 ns = make_uint2(0u, 0u);
    for (int f = 0; f < features; f = f + 1) {
        float v = wh[t * features + f];
        ss = gat_add(ss, gat_product(v, attn[a + f]));
        ns = gat_add(ns, gat_product(v, attn[a + features + f]));
    }
    selfWords[2*t] = (int)(ss.x);
    selfWords[2*t+1] = (int)(ss.y);
    neighborWords[2*t] = (int)(ns.x);
    neighborWords[2*t+1] = (int)(ns.y);
}
""" + "\n" +
"""
extern "C" __global__ __launch_bounds__(256) void gat_forward(const float* wh, const int* selfWords, const int* neighborWords, const int* rowPtr, const int* cols, float* dst, int total, int nodes, int heads, int features, int edges, int slopeLo, int slopeHi, int cutoff, float keep, int seed)
{
""" + "\n" + NormalizeRow + "\n" +
"""
    for (int f = 0; f < features; f = f + 1) { dst[t*features+f] = 0.0f; }
    for (int e = begin; e < end; e = e + 1) {
""" + "\n" + LoadScore + "\n" +
"""
        int slot = (int)(((unsigned int)batchIndex * (unsigned int)edges + (unsigned int)e) * (unsigned int)heads + (unsigned int)h);
        float c = (gat_shifted_exp(value, largest) / sum) * gat_mask(seed, slot, cutoff, keep);
        if (c != 0.0f) {
            for (int f = 0; f < features; f = f + 1) { dst[t*features+f] = dst[t*features+f] + c * wh[j*features+f]; }
        }
    }
}
""" + "\n" +
"""
extern "C" __global__ __launch_bounds__(256) void gat_coefficients(const int* selfWords, const int* neighborWords, const int* rowPtr, const int* cols, float* coeff, int total, int nodes, int heads, int edges, int slopeLo, int slopeHi)
{
""" + "\n" + NormalizeRow + "\n" +
"""
    for (int e = begin; e < end; e = e + 1) {
""" + "\n" + LoadScore + "\n" +
"""
        int slot = (batchIndex * edges + e) * heads + h;
        float probability = gat_shifted_exp(value, largest) / sum;
        // Probabilities are nonnegative. Save the activation branch in the sign bit, including for zero.
        unsigned int word = __float_as_uint(probability);
        if (!gat_less(make_uint2(0u, 0u), raw)) { word = word | 0x80000000u; }
        coeff[slot] = __uint_as_float(word);
    }
}
""" + "\n" +
"""
extern "C" __global__ __launch_bounds__(256) void gat_row_grad(const float* wh, const float* upstream, const int* rowPtr, const int* cols, float* coeff, float* dz, float* dself, int total, int nodes, int heads, int features, int edges, int slopeLo, int slopeHi, int cutoff, float keep, int seed)
{
    int t = bx_gid();
    if (t >= total) { return; }
    int h = t % heads;
    int node = (t / heads) % nodes;
    int baseNode = (t / heads) - node;
    int batchIndex = baseNode / nodes;
    int begin = rowPtr[node];
    int end = rowPtr[node + 1];
    float mean = 0.0f;
    for (int e = begin; e < end; e = e + 1) {
        int slot = (batchIndex * edges + e) * heads + h;
        int j = (baseNode + cols[e]) * heads + h;
        float dotp = 0.0f;
        float mask = gat_mask(seed, slot, cutoff, keep);
        if (mask != 0.0f) {
            for (int f = 0; f < features; f = f + 1) { dotp = dotp + upstream[t*features+f] * wh[j*features+f]; }
            dotp = dotp * mask;
        }
        dz[slot] = dotp;
        mean = mean + fabsf(coeff[slot]) * dotp;
    }
    float gs = 0.0f;
    for (int e = begin; e < end; e = e + 1) {
        int slot = (batchIndex * edges + e) * heads + h;
        float probability = fabsf(coeff[slot]);
        float g = probability * (dz[slot] - mean);
        if ((__float_as_uint(coeff[slot]) >> 31u) != 0u) {
            g = gat_float(gat_scale(gat_product(g, 1.0f), slopeLo, slopeHi));
        }
        coeff[slot] = probability * gat_mask(seed, slot, cutoff, keep);
        dz[slot] = g;
        gs = gs + g;
    }
    dself[t] = gs;
}

extern "C" __global__ __launch_bounds__(256) void gat_column_grad(const float* upstream, const float* dz, const float* coeff, const int* colPtr, const int* tRow, const int* tEdge, float* dneighbor, float* dwh, int total, int nodes, int heads, int features, int edges)
{
    int t = bx_gid();
    if (t >= total) { return; }
    int h = t % heads;
    int node = (t / heads) % nodes;
    int baseNode = (t / heads) - node;
    int batchIndex = baseNode / nodes;
    float gn = 0.0f;
    for (int f = 0; f < features; f = f + 1) { dwh[t*features+f] = 0.0f; }
    for (int k = colPtr[node]; k < colPtr[node+1]; k = k + 1) {
        int slot = (batchIndex * edges + tEdge[k]) * heads + h;
        gn = gn + dz[slot];
        float c = coeff[slot];
        if (c != 0.0f) {
            int src = ((baseNode + tRow[k]) * heads + h) * features;
            for (int f = 0; f < features; f = f + 1) {
                dwh[t*features+f] = dwh[t*features+f] + c * upstream[src+f];
            }
        }
    }
    dneighbor[t] = gn;
}

extern "C" __global__ __launch_bounds__(256) void gat_feature_grad(const float* attn, const float* dself, const float* dneighbor, float* dwh, int total, int heads, int features)
{
    int i = bx_gid();
    if (i >= total) { return; }
    int t = i / features;
    int a = (t % heads) * 2 * features + i % features;
    dwh[i] = dwh[i] + dself[t] * attn[a] + dneighbor[t] * attn[a+features];
}

extern "C" __global__ __launch_bounds__(256) void gat_attention_grad(const float* wh, const float* dself, const float* dneighbor, float* da, int total, int rows, int heads, int features)
{
    int i = bx_gid();
    if (i >= total) { return; }
    int h = i / (2 * features);
    int f = i % features;
    float sum = 0.0f;
    for (int r = 0; r < rows; r = r + 1) {
        int t = r * heads + h;
        float g = dself[t];
        if (((i / features) % 2) != 0) { g = dneighbor[t]; }
        sum = sum + g * wh[t*features+f];
    }
    da[i] = sum;
}
""";
}
