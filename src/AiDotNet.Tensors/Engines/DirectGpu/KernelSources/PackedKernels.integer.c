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
