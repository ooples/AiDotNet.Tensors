// The binary16 bits of the float32 f, rounded to nearest even (subnormal results included); a NaN keeps its sign and
// the top of its payload, quieted. The host twin is CpuEngine.DoubleToHalfBits.
FN uint dd_f32_to_f16(uint f)
{
    uint sgn = (f >> 16) & 32768u;
    int e = UTOI((f >> 23) & 255u);
    uint m = f & 8388607u;
    if (e == 255)
    {
        if (m != 0u) return sgn | 32256u | (m >> 13);
        return sgn | 31744u;
    }
    if (e == 0) return sgn;
    int ue = e - 127;
    if (ue > 15) return sgn | 31744u;
    if (ue >= -14)
    {
        uint hm = dd_round_shift(0u, m, 13);
        if (hm == 1024u)
        {
            hm = 0u;
            ue = ue + 1;
            if (ue > 15) return sgn | 31744u;
        }
        return sgn | (ITOU(ue + 15) << 10) | hm;
    }
    // A subnormal (or zero) result in units of 2^-24; a carry into 2^10 is exactly the smallest normal's bits.
    return sgn | dd_round_shift(0u, m | 8388608u, -ue - 1);
}

// The high word of the binary64 bits of the float32 f (exact); a NaN keeps its payload, quieted, as the x64
// conversion does.
FN uint dd_f32_to_f64_hi(uint f)
{
    uint sgn = f & 2147483648u;
    int e = UTOI((f >> 23) & 255u);
    uint m = f & 8388607u;
    if (e == 255)
    {
        if (m != 0u) return sgn | 2146959360u | (m >> 3);
        return sgn | 2146435072u;
    }
    if (e == 0)
    {
        if (m == 0u) return sgn;
        int p = dd_msb(m);
        uint v = m & ((1u << ITOU(p)) - 1u);
        int s = 52 - p;
        uint fh = 0u;
        if (s >= 32) fh = v << ITOU(s - 32); else fh = v >> ITOU(32 - s);
        return sgn | (ITOU(p + 874) << 20) | fh;
    }
    return sgn | (ITOU(e + 896) << 20) | (m >> 3);
}

// The low word of the binary64 bits of the float32 f.
FN uint dd_f32_to_f64_lo(uint f)
{
    int e = UTOI((f >> 23) & 255u);
    uint m = f & 8388607u;
    if (e == 0)
    {
        if (m == 0u) return 0u;
        int p = dd_msb(m);
        uint v = m & ((1u << ITOU(p)) - 1u);
        int s = 52 - p;
        if (s >= 32) return 0u;
        return v << ITOU(s);
    }
    return m << 29;
}

// The float32 f truncated toward zero and saturated at integer storage type tc's range (NaN as 0), in two's
// complement: the low word, or (wantHi) the high word, nonzero only for Int64.
FN uint dd_f32_to_int(uint f, int tc, int wantHi)
{
    int e = UTOI((f >> 23) & 255u);
    uint m = f & 8388607u;
    int neg = 0;
    if ((f & 2147483648u) != 0u) neg = 1;
    if (e == 255 && m != 0u) return 0u;
    uint mhi = 0u;
    uint mlo = 0u;
    int big = 0;
    if (e == 255 || e - 127 >= 63) big = 1;
    else if (e >= 127)
    {
        int ue = e - 127;
        uint sm = m | 8388608u;
        if (ue <= 23) mlo = sm >> ITOU(23 - ue);
        else
        {
            int s = ue - 23;
            if (s < 32)
            {
                mlo = sm << ITOU(s);
                mhi = sm >> ITOU(32 - s);
            }
            else mhi = sm << ITOU(s - 32);
        }
    }
    if (tc == 9)
    {
        if (big != 0)
        {
            if (wantHi != 0)
            {
                if (neg != 0) return 2147483648u;
                return 2147483647u;
            }
            if (neg != 0) return 0u;
            return 4294967295u;
        }
        if (neg != 0)
        {
            uint nlo = (~mlo) + 1u;
            uint nhi = ~mhi;
            if (nlo == 0u) nhi = nhi + 1u;
            if (wantHi != 0) return nhi;
            return nlo;
        }
        if (wantHi != 0) return mhi;
        return mlo;
    }
    if (wantHi != 0) return 0u;
    // The narrow type's largest value and the magnitude of its most negative one.
    uint maxPos = 4294967295u;
    uint maxNeg = 0u;
    if (tc == 3) { maxPos = 127u; maxNeg = 128u; }
    else if (tc == 4) { maxPos = 255u; }
    else if (tc == 5) { maxPos = 32767u; maxNeg = 32768u; }
    else if (tc == 6) { maxPos = 65535u; }
    else if (tc == 7) { maxPos = 2147483647u; maxNeg = 2147483648u; }
    int over = big;
    if (mhi != 0u) over = 1;
    uint mag = mlo;
    if (neg != 0)
    {
        if (over != 0 || mag > maxNeg) mag = maxNeg;
        return (~mag) + 1u;
    }
    if (over != 0 || mag > maxPos) mag = maxPos;
    return mag;
}

// The low word of the float32 f stored as tc.
FN uint dd_encode_lo(uint f, int tc)
{
    if (tc == 1) return f;
    if (tc == 2) return dd_f32_to_f64_lo(f);
    if (tc == 0) return dd_f32_to_f16(f);
    return dd_f32_to_int(f, tc, 0);
}

// The high word (bytes 4 to 7) of the float32 f stored as tc.
FN uint dd_encode_hi(uint f, int tc)
{
    if (tc == 2) return dd_f32_to_f64_hi(f);
    if (tc == 9) return dd_f32_to_int(f, tc, 1);
    return 0u;
}
