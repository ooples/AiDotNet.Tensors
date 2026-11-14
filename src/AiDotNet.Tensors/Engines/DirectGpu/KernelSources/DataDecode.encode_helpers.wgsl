
fn dd_f32_to_f16(f: u32) -> u32 {
    let sgn = (f >> 16u) & 32768u;
    let e = i32((f >> 23u) & 255u);
    let m = f & 8388607u;
    if (e == 255) {
        if (m != 0u) { return sgn | 32256u | (m >> 13u); }
        return sgn | 31744u;
    }
    if (e == 0) { return sgn; }
    var ue = e - 127;
    if (ue > 15) { return sgn | 31744u; }
    if (ue >= -14) {
        var hm = dd_round_shift(0u, m, 13);
        if (hm == 1024u) {
            hm = 0u;
            ue = ue + 1;
            if (ue > 15) { return sgn | 31744u; }
        }
        return sgn | (u32(ue + 15) << 10u) | hm;
    }
    return sgn | dd_round_shift(0u, m | 8388608u, -ue - 1);
}

fn dd_f32_to_f64_hi(f: u32) -> u32 {
    let sgn = f & 2147483648u;
    let e = i32((f >> 23u) & 255u);
    let m = f & 8388607u;
    if (e == 255) {
        if (m != 0u) { return sgn | 2146959360u | (m >> 3u); }
        return sgn | 2146435072u;
    }
    if (e == 0) {
        if (m == 0u) { return sgn; }
        let p = dd_msb(m);
        let v = m & ((1u << u32(p)) - 1u);
        let s = 52 - p;
        var fh = 0u;
        if (s >= 32) { fh = v << u32(s - 32); } else { fh = v >> u32(32 - s); }
        return sgn | (u32(p + 874) << 20u) | fh;
    }
    return sgn | (u32(e + 896) << 20u) | (m >> 3u);
}

fn dd_f32_to_f64_lo(f: u32) -> u32 {
    let e = i32((f >> 23u) & 255u);
    let m = f & 8388607u;
    if (e == 0) {
        if (m == 0u) { return 0u; }
        let p = dd_msb(m);
        let v = m & ((1u << u32(p)) - 1u);
        let s = 52 - p;
        if (s >= 32) { return 0u; }
        return v << u32(s);
    }
    return m << 29u;
}

fn dd_f32_to_int(f: u32, tc: i32, wantHi: bool) -> u32 {
    let e = i32((f >> 23u) & 255u);
    let m = f & 8388607u;
    let neg = (f & 2147483648u) != 0u;
    if (e == 255 && m != 0u) { return 0u; }
    var mhi = 0u;
    var mlo = 0u;
    var big = false;
    if (e == 255 || e - 127 >= 63) {
        big = true;
    } else if (e >= 127) {
        let ue = e - 127;
        let sm = m | 8388608u;
        if (ue <= 23) {
            mlo = sm >> u32(23 - ue);
        } else {
            let s = ue - 23;
            if (s < 32) {
                mlo = sm << u32(s);
                mhi = sm >> u32(32 - s);
            } else {
                mhi = sm << u32(s - 32);
            }
        }
    }
    if (tc == 9) {
        if (big) {
            if (wantHi) {
                if (neg) { return 2147483648u; }
                return 2147483647u;
            }
            if (neg) { return 0u; }
            return 4294967295u;
        }
        if (neg) {
            let nlo = (~mlo) + 1u;
            var nhi = ~mhi;
            if (nlo == 0u) { nhi = nhi + 1u; }
            if (wantHi) { return nhi; }
            return nlo;
        }
        if (wantHi) { return mhi; }
        return mlo;
    }
    if (wantHi) { return 0u; }
    var maxPos = 4294967295u;
    var maxNeg = 0u;
    if (tc == 3) { maxPos = 127u; maxNeg = 128u; }
    else if (tc == 4) { maxPos = 255u; }
    else if (tc == 5) { maxPos = 32767u; maxNeg = 32768u; }
    else if (tc == 6) { maxPos = 65535u; }
    else if (tc == 7) { maxPos = 2147483647u; maxNeg = 2147483648u; }
    let over = big || mhi != 0u;
    var mag = mlo;
    if (neg) {
        if (over || mag > maxNeg) { mag = maxNeg; }
        return (~mag) + 1u;
    }
    if (over || mag > maxPos) { mag = maxPos; }
    return mag;
}

fn dd_encode_lo(f: u32, tc: i32) -> u32 {
    if (tc == 1) { return f; }
    if (tc == 2) { return dd_f32_to_f64_lo(f); }
    if (tc == 0) { return dd_f32_to_f16(f); }
    return dd_f32_to_int(f, tc, false);
}

fn dd_encode_hi(f: u32, tc: i32) -> u32 {
    if (tc == 2) { return dd_f32_to_f64_hi(f); }
    if (tc == 9) { return dd_f32_to_int(f, tc, true); }
    return 0u;
}
