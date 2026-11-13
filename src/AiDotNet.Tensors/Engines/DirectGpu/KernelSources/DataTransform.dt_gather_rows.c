
    int i = gid + first * 256;
    if (i >= K * D) return;
    int k = i / D;
    int d = i - k * D;
    int r = TOINT(ids[k]);
    float v = 0.0f;
    if (r >= 0) v = tb[r * D + d];
    ot[i] = v;
