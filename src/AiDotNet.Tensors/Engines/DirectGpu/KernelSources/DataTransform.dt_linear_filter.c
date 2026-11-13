
    int s = gid + first * 256;
    if (s >= S) return;
    int K = NB;
    if (NA > K) K = NA;
    int o = s * L;
    int zo = s * (K - 1);
    float a0 = ac[0];
    for (int k = 0; k < K - 1; k++)
    {
        float z0 = 0.0f;
        if (hasZi != 0) z0 = zi[zo + k];
        zs[zo + k] = z0;
    }
    float g = IBITS(guardBits);
    float b0 = bc[0] / a0;
    for (int n = 0; n < L; n++)
    {
        float xn = xs[o + n];
        float yn = b0 * xn;
        if (K > 1) yn = yn + zs[zo];
        if (guardOn != 0)
        {
            if ((FBITS(yn) & 2147483647) > 2139095040) yn = 0.0f;
            else if (yn > g) yn = g;
            else if (yn < -g) yn = -g;
        }
        for (int k = 1; k < K; k++)
        {
            float bk = 0.0f;
            if (k < NB) bk = bc[k] / a0;
            float ak = 0.0f;
            if (k < NA) ak = ac[k] / a0;
            float zk = 0.0f;
            if (k < K - 1) zk = zs[zo + k];
            zs[zo + k - 1] = bk * xn - ak * yn + zk;
        }
        ys[o + n] = yn;
    }
