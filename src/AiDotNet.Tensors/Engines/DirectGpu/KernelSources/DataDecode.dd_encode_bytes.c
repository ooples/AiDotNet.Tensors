
    int w = gid + firstItem;
    if (w >= nWords) return;
    int total = nItems * elemSize;
    uint word = 0u;
    for (int k = 0; k < 4; k++)
    {
        int q = w * 4 + k;
        if (q < total)
        {
            int el = q / elemSize;
            int j = q - el * elemSize;
            if (littleEndian == 0) j = elemSize - 1 - j;
            uint f = inBits[el];
            uint b = 0u;
            if (j < 4) b = (dd_encode_lo(f, typeCode) >> ITOU(j * 8)) & 255u;
            else b = (dd_encode_hi(f, typeCode) >> ITOU((j - 4) * 8)) & 255u;
            word = word | (b << ITOU(k * 8));
        }
    }
    words[w] = word;
