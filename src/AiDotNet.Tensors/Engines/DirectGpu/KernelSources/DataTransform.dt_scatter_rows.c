
    int i = gid + first * 256;
    if (i >= K * D) return;
    int k = i / D;
    int d = i - k * D;
    int r = TOINT(ids[k]);
    if (r < 0) return;
    tb[r * D + d] = sr[i];
