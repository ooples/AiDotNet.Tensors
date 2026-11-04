// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// The deterministic global-norm kernels in GLSL (<see cref="Gpu.DeterministicGradientNorm"/> is the reference they
/// match bit for bit), in 32-bit integers only: core Vulkan compute does not promise 64-bit integers, so 64- and
/// 128-bit values are uvec2 / uvec4 (low word first) with explicit carries.
/// </summary>
internal static class VulkanDeterministicNormKernels
{
    // Shared by every shader below; a port of DeterministicNormKernels.Common (keep the three in step).
    private const string Common = @"
uvec2 dgn_add64(uvec2 a, uvec2 b) { uint c; uint lo = uaddCarry(a.x, b.x, c); return uvec2(lo, a.y + b.y + c); }
bool dgn_ge64(uvec2 a, uvec2 b) { return a.y > b.y || (a.y == b.y && a.x >= b.x); }
uvec2 dgn_sub64(uvec2 a, uvec2 b) { uint br; uint lo = usubBorrow(a.x, b.x, br); return uvec2(lo, a.y - b.y - br); }
uvec2 dgn_shr64(uvec2 v, int sh)
{
    if (sh <= 0) return v;
    if (sh >= 64) return uvec2(0u);
    if (sh >= 32) return uvec2(v.y >> uint(sh - 32), 0u);
    return uvec2((v.x >> uint(sh)) | (v.y << uint(32 - sh)), v.y >> uint(sh));
}
uvec2 dgn_shl64(uvec2 v, int sh)
{
    if (sh <= 0) return v;
    if (sh >= 64) return uvec2(0u);
    if (sh >= 32) return uvec2(0u, v.x << uint(sh - 32));
    return uvec2(v.x << uint(sh), (v.y << uint(sh)) | (v.x >> uint(32 - sh)));
}
int dgn_bitlen32(uint x) { return findMSB(x) + 1; }
int dgn_bitlen64(uvec2 v) { return v.y != 0u ? 32 + dgn_bitlen32(v.y) : dgn_bitlen32(v.x); }
bool dgn_low64(uvec2 v, int bits)   // any of the low `bits` bits set
{
    if (bits <= 0) return false;
    if (bits >= 64) return (v.x | v.y) != 0u;
    return (dgn_shl64(v, 64 - bits).x | dgn_shl64(v, 64 - bits).y) != 0u;
}
bool dgn_bit64(uvec2 v, int bit) { return bit < 32 ? ((v.x >> uint(bit)) & 1u) != 0u : ((v.y >> uint(bit - 32)) & 1u) != 0u; }

uvec4 dgn_add128(uvec4 a, uvec4 b)
{
    uvec4 r; uint c, c1, c2;
    r.x = uaddCarry(a.x, b.x, c);
    r.y = uaddCarry(a.y, b.y, c1); r.y = uaddCarry(r.y, c, c2); c = c1 + c2;
    r.z = uaddCarry(a.z, b.z, c1); r.z = uaddCarry(r.z, c, c2); c = c1 + c2;
    r.w = a.w + b.w + c;
    return r;
}
uint dgn_word(uvec4 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : (i == 3 ? v.w : 0u))); }
uvec4 dgn_shr128(uvec4 v, int sh)
{
    if (sh <= 0) return v;
    if (sh >= 128) return uvec4(0u);
    int ws = sh >> 5, bs = sh & 31;
    uvec4 r;
    for (int i = 0; i < 4; i++)
    {
        uint lo = i + ws < 4 ? dgn_word(v, i + ws) : 0u;
        uint hi = i + ws + 1 < 4 ? dgn_word(v, i + ws + 1) : 0u;
        r[i] = bs == 0 ? lo : ((lo >> uint(bs)) | (hi << uint(32 - bs)));
    }
    return r;
}
uvec4 dgn_shl128(uvec4 v, int sh)
{
    if (sh <= 0) return v;
    if (sh >= 128) return uvec4(0u);
    int ws = sh >> 5, bs = sh & 31;
    uvec4 r;
    for (int i = 0; i < 4; i++)
    {
        uint a = i - ws >= 0 ? dgn_word(v, i - ws) : 0u;
        uint b = i - ws - 1 >= 0 ? dgn_word(v, i - ws - 1) : 0u;
        r[i] = bs == 0 ? a : ((a << uint(bs)) | (b >> uint(32 - bs)));
    }
    return r;
}
int dgn_bitlen128(uvec4 v)
{
    if (v.w != 0u) return 96 + dgn_bitlen32(v.w);
    if (v.z != 0u) return 64 + dgn_bitlen32(v.z);
    if (v.y != 0u) return 32 + dgn_bitlen32(v.y);
    return dgn_bitlen32(v.x);
}
bool dgn_low128(uvec4 v, int bits)
{
    if (bits <= 0) return false;
    if (bits >= 128) return (v.x | v.y | v.z | v.w) != 0u;
    uvec4 t = dgn_shl128(v, 128 - bits);
    return (t.x | t.y | t.z | t.w) != 0u;
}
bool dgn_bit128(uvec4 v, int bit) { return ((dgn_word(v, bit >> 5) >> uint(bit & 31)) & 1u) != 0u; }

uvec4 dgn_square(uint b, uint emax)
{
    uint e = (b >> 23) & 0xFFu;
    if (e == 0xFFu) return uvec4(0u);
    uint m = (b & 0x7FFFFFu) | (e != 0u ? 0x800000u : 0u);
    if (e == 0u) e = 1u;
    uint hi, lo;
    umulExtended(m, m, hi, lo);
    int sh = 68 - 2 * int(emax - e);
    uvec4 v = uvec4(lo, hi, 0u, 0u);
    return sh >= 0 ? dgn_shl128(v, sh) : dgn_shr128(v, -sh);
}

// floor(sqrt(a)) for a < 2^54, two bits at a time; rem = a - r^2.
uint dgn_isqrt(uvec2 a, out uint rem)
{
    uint r = 0u, q = 0u;
    for (int i = 26; i >= 0; i--)
    {
        uint two = i >= 16 ? (a.y >> uint(2 * i - 32)) & 3u : (a.x >> uint(2 * i)) & 3u;
        q = (q << 2) | two;
        uint trial = (r << 2) | 1u;
        if (q >= trial) { q -= trial; r = (r << 1) | 1u; } else r <<= 1;
    }
    rem = q;
    return r;
}

uvec2 dgn_roundshift(uvec2 m, int sh, bool sticky)
{
    if (sh > 64) return uvec2(0u);
    uvec2 q = dgn_shr64(m, sh);
    bool roundBit = dgn_bit64(m, sh - 1);
    bool rest = sticky || dgn_low64(m, sh - 1);
    if (roundBit && (rest || (q.x & 1u) != 0u)) q = dgn_add64(q, uvec2(1u, 0u));
    return q;
}

uint dgn_round_to_float(uvec2 m, int e, bool sticky)
{
    if ((m.x | m.y) == 0u) return 0u;
    int lead = e + dgn_bitlen64(m) - 1;
    int ulp = max(lead - 23, -149);
    int sh = ulp - e;
    uvec2 q = sh > 0 ? dgn_roundshift(m, sh, sticky) : dgn_shl64(m, -sh);
    if (q.y != 0u || q.x >= 0x1000000u) { q = dgn_shr64(q, 1); ulp++; }
    if (q.x < 0x800000u) return q.x;
    int biased = ulp + 23 + 127;
    if (biased >= 255) return 0x7F800000u;
    return (uint(biased) << 23) | (q.x & 0x7FFFFFu);
}

void dgn_round24(uvec2 m, int e, bool sticky, out uint rm, out int re)
{
    if ((m.x | m.y) == 0u) { rm = 0u; re = 0; return; }
    int len = dgn_bitlen64(m);
    if (len <= 24) { rm = m.x << uint(24 - len); re = e - (24 - len); return; }
    int sh = len - 24;
    uvec2 q = dgn_roundshift(m, sh, sticky);
    int ee = e + sh;
    if (q.y != 0u || q.x >= 0x1000000u) { q = dgn_shr64(q, 1); ee++; }
    rm = q.x; re = ee;
}

void dgn_addround24(uint am, int ae, uint bm, int be, out uint rm, out int re)
{
    if (am == 0u) { rm = bm; re = be; return; }
    if (bm == 0u) { rm = am; re = ae; return; }
    if (ae < be) { uint tm = am; am = bm; bm = tm; int te = ae; ae = be; be = te; }
    int d = ae - be;
    if (d > 30) { dgn_round24(uvec2((am << 3) | 1u, am >> 29), ae - 3, false, rm, re); return; }
    dgn_round24(dgn_add64(dgn_shl64(uvec2(am, 0u), d), uvec2(bm, 0u)), be, false, rm, re);
}

void dgn_sqrt(uvec4 a, int v, out uint m, out int e, out bool sticky)
{
    int len = dgn_bitlen128(a);
    if (len == 0) { m = 0u; e = 0; sticky = false; return; }
    int k = len - 54;
    if (((v + k) & 1) != 0) k++;
    uvec4 t = k >= 0 ? dgn_shr128(a, k) : dgn_shl128(a, -k);
    bool dropped = k > 0 && dgn_low128(a, k);
    uint rem;
    m = dgn_isqrt(t.xy, rem);
    e = (v + k) / 2;
    sticky = dropped || rem != 0u;
}

uint dgn_clip_coef(uint nm, int ne, bool ns, uint mb)
{
    if (mb == 0x7F800000u) return 0x3F800000u;
    uint nM, dM; int nE, dE;
    dgn_round24(uvec2(nm, 0u), ne, ns, nM, nE);
    dgn_addround24(nM, nE, 0x8637BDu, 107 - 150, dM, dE);
    uint me = (mb >> 23) & 0xFFu;
    uint mM = (mb & 0x7FFFFFu) | (me != 0u ? 0x800000u : 0u);
    int mE = (me != 0u ? int(me) : 1) - 150;
    if (mM == 0u) return 0u;
    while (mM < 0x800000u) { mM <<= 1; mE--; }
    if (mE > dE || (mE == dE && mM >= dM)) return 0x3F800000u;
    // (mM << 40) / dM by long division: the remainder stays below dM < 2^24.
    uvec2 num = dgn_shl64(uvec2(mM, 0u), 40);
    uvec2 q = uvec2(0u);
    uint rem = 0u;
    for (int i = 63; i >= 0; i--)
    {
        rem = (rem << 1) | (dgn_bit64(num, i) ? 1u : 0u);
        q = dgn_shl64(q, 1);
        if (rem >= dM) { rem -= dM; q.x |= 1u; }
    }
    return dgn_round_to_float(q, mE - 40 - dE, rem != 0u);
}

uvec2 dgn_to_double(uvec4 a, int v)
{
    int len = dgn_bitlen128(a);
    if (len == 0) return uvec2(0u);
    uvec2 m; int e;
    if (len > 53)
    {
        int sh = len - 53;
        m = dgn_shr128(a, sh).xy;
        bool roundBit = dgn_bit128(a, sh - 1);
        bool rest = sh > 1 && dgn_low128(a, sh - 1);
        if (roundBit && (rest || (m.x & 1u) != 0u)) m = dgn_add64(m, uvec2(1u, 0u));
        e = v + sh;
        if (m.y >= 0x200000u) { m = dgn_shr64(m, 1); e++; }
    }
    else { m = dgn_shl128(a, 53 - len).xy; e = v - (53 - len); }
    int lead = e + 52;
    if (lead >= -1022) return uvec2(m.x, (uint(lead + 1023) << 20) | (m.y & 0xFFFFFu));
    return dgn_shr64(m, -1022 - lead);
}

uint dgn_mul(uint g, uint s)
{
    if (s == 0x3F800000u) return g;   // unscaled: every element, NaN payloads included, keeps its bits
    uint eg = (g >> 23) & 0xFFu, es = (s >> 23) & 0xFFu;
    if (eg - 1u < 254u && es - 1u < 254u && eg + es >= 128u)
    {
        precise float p = uintBitsToFloat(g) * uintBitsToFloat(s);
        return floatBitsToUint(p);
    }
    uint sign = (g ^ s) & 0x80000000u;
    uint fg = g & 0x7FFFFFu, fs = s & 0x7FFFFFu;
    if ((eg == 0xFFu && fg != 0u) || (es == 0xFFu && fs != 0u)) return 0x7FC00000u;
    bool gzero = eg == 0u && fg == 0u, szero = es == 0u && fs == 0u;
    if (eg == 0xFFu || es == 0xFFu) return (gzero || szero) ? 0x7FC00000u : (sign | 0x7F800000u);
    if (gzero || szero) return sign;
    uint hi, lo;
    umulExtended(fg | (eg != 0u ? 0x800000u : 0u), fs | (es != 0u ? 0x800000u : 0u), hi, lo);
    int e = (eg != 0u ? int(eg) : 1) + (es != 0u ? int(es) : 1) - 300;
    return dgn_round_to_float(uvec2(lo, hi), e, false) | sign;
}
";

    /// <summary>One chunk partial per work group: binding 0 the tensor, 1 the partials; push n, tensor, first partial.</summary>
    internal const string ChunkPartials = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Xb { uint x[]; };
layout(set=0,binding=1) buffer Pb { uint partials[]; };
layout(push_constant) uniform PC { uint n; uint tensor; uint partialBase; };
shared uint su[256];
shared uvec4 sacc[256];
" + Common + @"
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint c = gl_WorkGroupID.x;
    uint start = c * 4096u;
    uint end = min(start + 4096u, n);
    uint b[16];
    uint emax = 1u, nan = 0u, inf = 0u;
    for (int j = 0; j < 16; j++)
    {
        uint i = start + uint(j) * 256u + tid;
        uint v = i < end ? x[i] : 0u;
        b[j] = v;
        uint e = (v >> 23) & 0xFFu;
        if (e == 0xFFu) { if ((v & 0x7FFFFFu) != 0u) nan++; else inf++; }
        else if (e > emax) emax = e;
    }
    su[tid] = emax;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) { if (tid < s) su[tid] = max(su[tid], su[tid + s]); barrier(); }
    emax = su[0];
    barrier();
    su[tid] = (nan << 16) | inf;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) { if (tid < s) su[tid] = su[tid] + su[tid + s]; barrier(); }
    uint counts = su[0];
    uvec4 acc = uvec4(0u);
    for (int j = 0; j < 16; j++) acc = dgn_add128(acc, dgn_square(b[j], emax));
    sacc[tid] = acc;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) { if (tid < s) sacc[tid] = dgn_add128(sacc[tid], sacc[tid + s]); barrier(); }
    if (tid == 0u)
    {
        uint o = (partialBase + c) * 8u;
        partials[o] = emax; partials[o + 1u] = counts >> 16; partials[o + 2u] = counts & 0xFFFFu; partials[o + 3u] = tensor;
        uvec4 a = sacc[0];
        partials[o + 4u] = a.x; partials[o + 5u] = a.y; partials[o + 6u] = a.z; partials[o + 7u] = a.w;
    }
}";

    /// <summary>Combines the partials into the record: binding 0 partials, 1 record; push count, headroom. One group.</summary>
    internal const string Finish = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Pb { uint p[]; };
layout(set=0,binding=1) buffer Rb { uint rec[]; };
layout(push_constant) uniform PC { uint count; uint headroom; };
shared uint su[256], su2[256], sn[256], si[256];
shared uvec4 sacc[256];
" + Common + @"
uint dgn_satadd(uint a, uint b) { uint s = a + b; return s < a ? 0xFFFFFFFFu : s; }
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint g = 0u, first = 0xFFFFFFFFu, nan = 0u, inf = 0u;
    for (uint i = tid; i < count; i += 256u)
    {
        uint o = i * 8u;
        if ((p[o + 4u] | p[o + 5u] | p[o + 6u] | p[o + 7u]) != 0u && p[o] > g) g = p[o];
        nan = dgn_satadd(nan, p[o + 1u]); inf = dgn_satadd(inf, p[o + 2u]);
        if ((p[o + 1u] | p[o + 2u]) != 0u && p[o + 3u] < first) first = p[o + 3u];
    }
    su[tid] = g; su2[tid] = first; sn[tid] = nan; si[tid] = inf;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            su[tid] = max(su[tid], su[tid + s]); su2[tid] = min(su2[tid], su2[tid + s]);
            sn[tid] = dgn_satadd(sn[tid], sn[tid + s]); si[tid] = dgn_satadd(si[tid], si[tid + s]);
        }
        barrier();
    }
    g = su[0]; first = su2[0]; nan = sn[0]; inf = si[0];
    uvec4 acc = uvec4(0u);
    if (g != 0u)
        for (uint i = tid; i < count; i += 256u)
        {
            uint o = i * 8u;
            uvec4 a = uvec4(p[o + 4u], p[o + 5u], p[o + 6u], p[o + 7u]);
            if ((a.x | a.y | a.z | a.w) == 0u) continue;
            acc = dgn_add128(acc, dgn_shr128(a, 2 * int(g - p[o]) + int(headroom)));
        }
    sacc[tid] = acc;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) { if (tid < s) sacc[tid] = dgn_add128(sacc[tid], sacc[tid + s]); barrier(); }
    if (tid == 0u)
    {
        uvec4 a = sacc[0];
        int v = g != 0u ? 2 * int(g) - 368 + int(headroom) : 0;
        rec[2] = a.x; rec[3] = a.y; rec[4] = a.z; rec[5] = a.w;
        rec[6] = uint(v); rec[7] = nan; rec[8] = inf; rec[9] = first;
        uvec2 dbl = uvec2(0u);
        uint norm;
        if (nan != 0u) { dbl = uvec2(0u, 0x7FF80000u); norm = 0x7FC00000u; }
        else if (inf != 0u) { dbl = uvec2(0u, 0x7FF00000u); norm = 0x7F800000u; }
        else
        {
            dbl = dgn_to_double(a, v);
            uint nm; int ne; bool ns;
            dgn_sqrt(a, v, nm, ne, ns);
            norm = dgn_round_to_float(uvec2(nm, 0u), ne, ns);
        }
        rec[0] = dbl.x; rec[1] = dbl.y; rec[10] = norm; rec[11] = 0x3F800000u;
        rec[12] = (nan | inf) != 0u ? 1u : 0u; rec[13] = 0u; rec[14] = 0u; rec[15] = 0u;
    }
}";

    /// <summary>The coefficient: binding 0 record, 1 scale; push maxNorm bits, policy. One invocation.</summary>
    internal const string ClipScale = @"#version 450
layout(local_size_x = 1) in;
layout(set=0,binding=0) buffer Rb { uint rec[]; };
layout(set=0,binding=1) buffer Sb { uint scale[]; };
layout(push_constant) uniform PC { uint maxNormBits; uint policy; };
" + Common + @"
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    uint s;
    if (rec[7] != 0u && policy != 2u) s = policy == 1u ? 0x7FC00000u : 0x3F800000u;
    else if (rec[8] != 0u && policy != 2u) s = policy == 1u ? 0u : 0x3F800000u;
    else
    {
        uint nm; int ne; bool ns;
        dgn_sqrt(uvec4(rec[2], rec[3], rec[4], rec[5]), int(rec[6]), nm, ne, ns);
        s = dgn_clip_coef(nm, ne, ns, maxNormBits);
    }
    rec[11] = s; rec[13] = maxNormBits;
    scale[0] = s;
}";

    /// <summary>x[i] = x[i] * scale with subnormals kept: binding 0 scale, 1 x; push n, zeroNonFinite. Grid-strided, so
    /// the dispatch stays within maxComputeWorkGroupCountX (65535 on minimum-spec devices) for any n.</summary>
    internal const string Scale = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Sb { uint scale[]; };
layout(set=0,binding=1) buffer Xb { uint x[]; };
layout(push_constant) uniform PC { uint n; uint zeroNonFinite; };
" + Common + @"
void main() {
    uint s = scale[0], stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < n; i += stride)
        x[i] = (zeroNonFinite != 0u && ((x[i] >> 23) & 0xFFu) == 0xFFu) ? 0u : dgn_mul(x[i], s);
}";
}
