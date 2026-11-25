uint pk_fpack(uint sgn, int e, uint sig) {
    uint s = sig;
    int x = e;
    if (x <= 0) {
        uint down = uint(1 - x);
        if (down > 27u) {
            s = uint(s != 0u);
        } else {
            uint lost = s & ((1u << down) - 1u);
            s = s >> down;
            if (lost != 0u) {
                s = s | 1u;
            }
        }
        x = 1;
    }
    uint tail = s & 7u;
    uint r = s >> 3u;
    if ((tail > 4u) || ((tail == 4u) && ((r & 1u) != 0u))) {
        r = r + 1u;
    }
    if (r >= 0x1000000u) {
        r = r >> 1u;
        x = x + 1;
    }
    if (r < 0x800000u) {
        return sgn | r;
    }
    if (x >= 255) {
        return sgn | 0x7F800000u;
    }
    return (sgn | (uint(x) << 23u)) | (r & 0x7FFFFFu);
}

uint pk_fadd(uint a, uint b) {
    uint ea = (a >> 23u) & 0xFFu;
    uint eb = (b >> 23u) & 0xFFu;
    uint ma = a & 0x7FFFFFu;
    uint mb = b & 0x7FFFFFu;
    if (ea == 255u) {
        if (ma != 0u) {
            return 0x7FC00000u;
        }
        if (eb == 255u) {
            if (mb != 0u) {
                return 0x7FC00000u;
            }
            if (((a ^ b) & 0x80000000u) != 0u) {
                return 0x7FC00000u;
            }
        }
        return a;
    }
    if (eb == 255u) {
        if (mb != 0u) {
            return 0x7FC00000u;
        }
        return b;
    }
    uint big = a;
    uint sml = b;
    if ((b & 0x7FFFFFFFu) > (a & 0x7FFFFFFFu)) {
        big = b;
        sml = a;
    }
    uint sgn = big & 0x80000000u;
    uint xb = (big >> 23u) & 0xFFu;
    uint xs = (sml >> 23u) & 0xFFu;
    uint sb = (big & 0x7FFFFFu) << 3u;
    uint ss = (sml & 0x7FFFFFu) << 3u;
    if (xb != 0u) {
        sb = sb | 0x4000000u;
    } else {
        xb = 1u;
    }
    if (xs != 0u) {
        ss = ss | 0x4000000u;
    } else {
        xs = 1u;
    }
    uint d = xb - xs;
    if (d > 27u) {
        ss = uint(ss != 0u);
    } else {
        if (d > 0u) {
            uint lost = ss & ((1u << d) - 1u);
            ss = ss >> d;
            if (lost != 0u) {
                ss = ss | 1u;
            }
        }
    }
    int x = int(xb);
    uint s = 0u;
    if (((big ^ sml) & 0x80000000u) == 0u) {
        s = sb + ss;
        if (s >= 0x8000000u) {
            s = (s >> 1u) | (s & 1u);
            x = x + 1;
        }
        if (x >= 255) {
            return sgn | 0x7F800000u;
        }
        return pk_fpack(sgn, x, s);
    }
    s = sb - ss;
    if (s == 0u) {
        return 0u;
    }
    for (uint j = 0u; j < 27u; j++) {
        if ((s < 0x4000000u) && (x > 1)) {
            s = s << 1u;
            x = x - 1;
        }
    }
    return pk_fpack(sgn, x, s);
}

uint pk_fmul(uint a, uint b) {
    uint sgn = (a ^ b) & 0x80000000u;
    uint ea = (a >> 23u) & 0xFFu;
    uint eb = (b >> 23u) & 0xFFu;
    uint ma = a & 0x7FFFFFu;
    uint mb = b & 0x7FFFFFu;
    bool za = (ea == 0u) && (ma == 0u);
    bool zb = (eb == 0u) && (mb == 0u);
    if ((ea == 255u) || (eb == 255u)) {
        if (((ea == 255u) && (ma != 0u)) || ((eb == 255u) && (mb != 0u))) {
            return 0x7FC00000u;
        }
        if (za || zb) {
            return 0x7FC00000u;
        }
        return sgn | 0x7F800000u;
    }
    if (za || zb) {
        return sgn;
    }
    int xa = int(ea);
    int xb = int(eb);
    if (ea == 0u) {
        xa = 1;
        for (uint j = 0u; j < 23u; j++) {
            if ((ma & 0x800000u) == 0u) {
                ma = ma << 1u;
                xa = xa - 1;
            }
        }
    } else {
        ma = ma | 0x800000u;
    }
    if (eb == 0u) {
        xb = 1;
        for (uint j = 0u; j < 23u; j++) {
            if ((mb & 0x800000u) == 0u) {
                mb = mb << 1u;
                xb = xb - 1;
            }
        }
    } else {
        mb = mb | 0x800000u;
    }
    uint al = ma & 0xFFFu;
    uint ah = ma >> 12u;
    uint bl = mb & 0xFFFu;
    uint bh = mb >> 12u;
    uint p0 = al * bl;
    uint p1 = (al * bh) + (ah * bl);
    uint p2 = ah * bh;
    uint lo = p0 + ((p1 & 0xFFFu) << 12u);
    uint hi = (p2 + (p1 >> 12u)) + (lo >> 24u);
    lo = lo & 0xFFFFFFu;
    int x = (xa + xb) - 127;
    uint s = 0u;
    if ((hi & 0x800000u) != 0u) {
        x = x + 1;
        s = (hi << 3u) | (lo >> 21u);
        if ((lo & 0x1FFFFFu) != 0u) {
            s = s | 1u;
        }
    } else {
        s = (hi << 4u) | (lo >> 20u);
        if ((lo & 0xFFFFFu) != 0u) {
            s = s | 1u;
        }
    }
    return pk_fpack(sgn, x, s);
}

uint pk_fdiv(uint a, uint b) {
    uint sgn = (a ^ b) & 0x80000000u;
    uint ea = (a >> 23u) & 0xFFu;
    uint eb = (b >> 23u) & 0xFFu;
    uint ma = a & 0x7FFFFFu;
    uint mb = b & 0x7FFFFFu;
    bool za = (ea == 0u) && (ma == 0u);
    bool zb = (eb == 0u) && (mb == 0u);
    if (((ea == 255u) && (ma != 0u)) || ((eb == 255u) && (mb != 0u))) {
        return 0x7FC00000u;
    }
    if (ea == 255u) {
        if (eb == 255u) {
            return 0x7FC00000u;
        }
        return sgn | 0x7F800000u;
    }
    if (eb == 255u) {
        return sgn;
    }
    if (zb) {
        if (za) {
            return 0x7FC00000u;
        }
        return sgn | 0x7F800000u;
    }
    if (za) {
        return sgn;
    }
    int xa = int(ea);
    int xb = int(eb);
    if (ea == 0u) {
        xa = 1;
        for (uint j = 0u; j < 23u; j++) {
            if ((ma & 0x800000u) == 0u) {
                ma = ma << 1u;
                xa = xa - 1;
            }
        }
    } else {
        ma = ma | 0x800000u;
    }
    if (eb == 0u) {
        xb = 1;
        for (uint j = 0u; j < 23u; j++) {
            if ((mb & 0x800000u) == 0u) {
                mb = mb << 1u;
                xb = xb - 1;
            }
        }
    } else {
        mb = mb | 0x800000u;
    }
    int x = (xa - xb) + 127;
    if (ma < mb) {
        ma = ma << 1u;
        x = x - 1;
    }
    // Long division of two 24-bit significands, the quotient in [1, 2): 27 bits (24 and three to round on), and
    // whatever remains is sticky.
    uint q = 0u;
    uint r = ma;
    for (uint j = 0u; j < 27u; j++) {
        q = q << 1u;
        if (r >= mb) {
            r = r - mb;
            q = q | 1u;
        }
        r = r << 1u;
    }
    if (r != 0u) {
        q = q | 1u;
    }
    return pk_fpack(sgn, x, q);
}

// The functions below take and return binary32 bits. A device need not keep a NaN, an infinity or a subnormal through
// a float function, so those are decided here on the bits and only an ordinary operand reaches the device's function.
uint pk_usqrt(uint w) {
    uint e = (w >> 23u) & 0xFFu;
    uint m = w & 0x7FFFFFu;
    if (e == 255u) {
        if ((m != 0u) || ((w & 0x80000000u) != 0u)) {
            return 0x7FC00000u;
        }
        return w;
    }
    if ((e == 0u) && (m == 0u)) {
        return w;
    }
    if ((w & 0x80000000u) != 0u) {
        return 0x7FC00000u;
    }
    if (e == 0u) {
        // A subnormal is m 2^-149: its root is sqrt(2 m) 2^-75, and 2 m is exact in a float.
        return FBITS(sqrt(float(m << 1u)) * 2.6469779601696886e-23f);
    }
    return FBITS(sqrt(UFLOAT(w)));
}

uint pk_uexp(uint w) {
    uint e = (w >> 23u) & 0xFFu;
    uint m = w & 0x7FFFFFu;
    if (e == 255u) {
        if (m != 0u) {
            return 0x7FC00000u;
        }
        if ((w & 0x80000000u) != 0u) {
            return 0u;
        }
        return w;
    }
    if (e < 97u) {
        // Below 2^-30 in magnitude (zeros and subnormals included) the result is 1 in binary32.
        return 0x3F800000u;
    }
    float v = UFLOAT(w);
    // ln of the largest float is 88.7228: no 16-bit float lies between this bound and it.
    if (v > 88.72f) {
        return 0x7F800000u;
    }
    if (v < -104.0f) {
        return 0u;
    }
    if (v < -87.0f) {
        // The result is a binary32 subnormal, which a device may flush to zero: e^(v + 64 ln 2) is an ordinary
        // float, and its product with 2^-64 is taken on the bits.
        return pk_fmul(FBITS(exp(v + 44.3614195558365f)), 0x1F800000u);
    }
    return FBITS(exp(v));
}

uint pk_ulog(uint w) {
    uint e = (w >> 23u) & 0xFFu;
    uint m = w & 0x7FFFFFu;
    if (e == 255u) {
        if ((m != 0u) || ((w & 0x80000000u) != 0u)) {
            return 0x7FC00000u;
        }
        return w;
    }
    if ((e == 0u) && (m == 0u)) {
        return 0xFF800000u;
    }
    if ((w & 0x80000000u) != 0u) {
        return 0x7FC00000u;
    }
    if (e == 0u) {
        // A subnormal is m 2^-149: ln m - 149 ln 2.
        return FBITS(log(float(m)) - 103.27892990343185f);
    }
    return FBITS(log(UFLOAT(w)));
}

uint pk_utanh(uint w) {
    uint e = (w >> 23u) & 0xFFu;
    if ((e == 255u) && ((w & 0x7FFFFFu) != 0u)) {
        return 0x7FC00000u;
    }
    if (e >= 131u) {
        // 16 and beyond, the infinities included: 1 in binary32, with the operand's sign.
        return (w & 0x80000000u) | 0x3F800000u;
    }
    if (e < 114u) {
        // Below 2^-13 the result is the operand to far more than 16-bit precision; zeros and subnormals come back whole.
        return w;
    }
    return FBITS(tanh(UFLOAT(w)));
}

uint pk_usigmoid(uint w) {
    uint e = (w >> 23u) & 0xFFu;
    if (e == 255u) {
        if ((w & 0x7FFFFFu) != 0u) {
            return 0x7FC00000u;
        }
        if ((w & 0x80000000u) != 0u) {
            return 0u;
        }
        return 0x3F800000u;
    }
    float v = UFLOAT(w);
    if (v >= 17.0f) {
        return 0x3F800000u;
    }
    if (v <= -104.0f) {
        return 0u;
    }
    if (v < -80.0f) {
        // e^-v would overflow on its way to the quotient; here the result is e^v to binary32 precision.
        return pk_uexp(w);
    }
    return FBITS(1.0f / (1.0f + exp(-v)));
}

uint pk_w_f16(uint h) {
    uint sgn = (h & 0x8000u) << 16u;
    uint e = (h >> 10u) & 0x1Fu;
    uint m = h & 0x3FFu;
    if (e == 0u) {
        if (m == 0u) {
            return sgn;
        }
        uint p = 0u;
        uint t = m;
        for (uint j = 0u; j < 10u; j++) {
            if (t > 1u) {
                t = t >> 1u;
                p = p + 1u;
            }
        }
        return (sgn | ((p + 103u) << 23u)) | ((m << (23u - p)) & 0x7FFFFFu);
    }
    if (e == 31u) {
        return (sgn | 0x7F800000u) | (m << 13u);
    }
    return (sgn | ((e + 112u) << 23u)) | (m << 13u);
}

uint pk_n_f16(uint x) {
    uint sgn = (x >> 16u) & 0x8000u;
    uint ax = x & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) {
        return (sgn | 0x7E00u) | ((ax >> 13u) & 0x1FFu);
    }
    if (ax >= 0x47800000u) {
        return sgn | 0x7C00u;
    }
    if (ax >= 0x38800000u) {
        uint m = ax - 0x38000000u;
        return sgn | (((m + 0xFFFu) + ((m >> 13u) & 1u)) >> 13u);
    }
    if (ax <= 0x33000000u) {
        return sgn;
    }
    uint down = 126u - (ax >> 23u);
    uint mant = (ax & 0x7FFFFFu) | 0x800000u;
    uint q = mant >> down;
    uint rem = mant & ((1u << down) - 1u);
    uint mid = 1u << (down - 1u);
    if ((rem > mid) || ((rem == mid) && ((q & 1u) != 0u))) {
        q = q + 1u;
    }
    return sgn | q;
}

uint pk_w_bf16(uint h) {
    return h << 16u;
}

uint pk_n_bf16(uint x) {
    if (((x & 0x7F800000u) == 0x7F800000u) && ((x & 0x007FFFFFu) != 0u)) {
        return (x >> 16u) | 0x0040u;
    }
    return ((x + 0x7FFFu) + ((x >> 16u) & 1u)) >> 16u;
}

uint pk_relu(uint h, uint top) {
    uint m = h & 0x7FFFu;
    if ((((h & 0x8000u) != 0u) || (m == 0u)) || (m > top)) {
        return 0u;
    }
    return h;
}

float pk_gelu(float v) {
    // Past 8 in magnitude the tanh argument is beyond its clamp of 20 (0.79788 * (8 + 0.044715 * 512) = 24.6), where
    // the host's tanh is exactly 1: the result is v, or 0.5 v * 0. Decided before the cube, which overflows binary32
    // for a large finite v (a BFloat16 of 2^80) and need not then survive to the clamp on every device.
    if (v >= 8.0f) {
        return v;
    }
    if (v <= -8.0f) {
        return (0.5f * v) * 0.0f;
    }
    float z = clamp(0.7978845608028654f * (v + (((0.044715f * v) * v) * v)), -20.0f, 20.0f);
    return (0.5f * v) * (1.0f + tanh(z));
}
