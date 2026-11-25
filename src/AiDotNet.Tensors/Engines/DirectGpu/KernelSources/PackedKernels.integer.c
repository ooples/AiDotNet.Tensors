uint pk_mulhi(uint x, uint y) {
    uint x0 = x & 0xFFFFu;
    uint x1 = x >> 16u;
    uint y0 = y & 0xFFFFu;
    uint y1 = y >> 16u;
    uint p00 = x0 * y0;
    uint p01 = x0 * y1;
    uint p10 = x1 * y0;
    uint p11 = x1 * y1;
    uint mid = ((p00 >> 16u) + (p01 & 0xFFFFu)) + (p10 & 0xFFFFu);
    return ((p11 + (p01 >> 16u)) + (p10 >> 16u)) + (mid >> 16u);
}

uint pk_abs(uint x) {
    if ((x >> 31u) != 0u) {
        return 0u - x;
    }
    return x;
}

uint pk_sdiv(uint x, uint y) {
    if (y == 0u) {
        return 0u;
    }
    uint q = pk_abs(x) / pk_abs(y);
    if ((x >> 31u) != (y >> 31u)) {
        q = 0u - q;
    }
    return q;
}

uint pk_srem(uint x, uint y) {
    if (y == 0u) {
        return 0u;
    }
    uint r = pk_abs(x) % pk_abs(y);
    if ((x >> 31u) != 0u) {
        r = 0u - r;
    }
    return r;
}
