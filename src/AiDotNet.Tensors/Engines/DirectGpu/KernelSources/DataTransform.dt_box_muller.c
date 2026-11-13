
    int i = gid + first * 256;
    if (i >= N) return;
    float r = uv[2 * i];
    float u2 = uv[2 * i + 1];
    float ang = 6.2831855f * u2;
    float z = r * COSF(ang);
    if (branch == 1) z = r * SINF(ang);
    float v = IBITS(meanBits) + IBITS(stdBits) * z;
    if (clampOn != 0)
    {
        float lo = IBITS(lowBits);
        float hi = IBITS(highBits);
        if (v < lo) v = lo;
        else if (v > hi) v = hi;
    }
    ot[i] = v;
