
    int i = gid + first * 256;
    if (i >= N) return;
    float v = xs[i];
    float t = 0.0f;
    if ((FBITS(v) & 2147483647) <= 2139095040)
    {
        t = TRUNCF(v);
        float f = v - t;
        float af = f;
        if (f < 0.0f) af = -f;
        float sg = 1.0f;
        if (v < 0.0f) sg = -1.0f;
        if (mode == 1)
        {
            float h = t * 0.5f;
            if (af > 0.5f || (af == 0.5f && h != TRUNCF(h))) t = t + sg;
        }
        else if (mode == 2)
        {
            if (af >= 0.5f) t = t + sg;
        }
        else if (mode == 3)
        {
            if (f < 0.0f) t = t - 1.0f;
        }
        else if (mode == 4)
        {
            if (f > 0.0f) t = t + 1.0f;
        }
        float lo = IBITS(lowBits);
        float hi = IBITS(highBits);
        if (t < lo) t = lo;
        else if (t > hi) t = hi;
    }
    ys[i] = t;
