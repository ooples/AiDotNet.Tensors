
    int i = gid + first * 256;
    if (i >= N) return;
    float a = uv[2 * i];
    float r = uv[2 * i + 1];
    float t = IBITS(thrBits);
    if (perElement != 0) t = th[i];
    int hit = 0;
    if (above != 0)
    {
        if (a > t || (a == t && r > 0.0f)) hit = 1;
    }
    else
    {
        if (a < t || (a == t && r < 0.0f)) hit = 1;
    }
    float v = IBITS(offBits);
    if (hit != 0) v = IBITS(onBits);
    ot[i] = v;
