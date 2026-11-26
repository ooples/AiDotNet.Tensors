extern "C" __global__ __launch_bounds__(256) void hh_p_norm(const float* u, float* uHat, float* hhStats, int K, int D, int sqrtFloorBits) {
    int k = bx_gid(); if (k >= K) return;
    float sqrtFloor = __int_as_float(sqrtFloorBits);
    float maxAbs = 0.0f;
    for (int i = 0; i < D; i++) {
        float a = fabsf(u[k * D + i]);
        maxAbs = fmaxf(maxAbs, a);
    }
    float rowScale = fmaxf(maxAbs, sqrtFloor * 0.0625f);
    float scaledSq = 0.0f;
    for (int i = 0; i < D; i++) {
        float s = u[k * D + i] / rowScale;
        scaledSq += s * s;
    }
    float sfr = sqrtFloor / rowScale;
    float scaledFloor = sfr * sfr;
    float denom = fmaxf(scaledSq, scaledFloor);
    float r = 1.0f / sqrtf(denom);
    for (int i = 0; i < D; i++) {
        uHat[k * D + i] = (u[k * D + i] / rowScale) * r;
    }
    hhStats[2 * k] = r / rowScale;
    hhStats[2 * k + 1] = scaledSq > scaledFloor ? 1.0f : (scaledSq == scaledFloor ? 0.5f : 0.0f);
}

extern "C" __global__ __launch_bounds__(256) void hh_p_fwd(const float* x, const float* uHat, float* y, int M, int K, int D, int trans) {
    int m = bx_gid(); if (m >= M) return;
    int baseOff = m * D;
    for (int i = 0; i < D; i++) y[baseOff + i] = x[baseOff + i];
    for (int s = 0; s < K; s++) {
        int j = trans != 0 ? (K - 1 - s) : s;
        int uOff = j * D;
        float dot = 0.0f;
        for (int i = 0; i < D; i++) dot += y[baseOff + i] * uHat[uOff + i];
        for (int i = 0; i < D; i++) {
            float step = dot * uHat[uOff + i];
            y[baseOff + i] = (y[baseOff + i] - step) - step;
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void hh_p_bwd_dx(const float* dy, const float* x, const float* uHat, float* dx, float* pqBuf, int M, int K, int D, int trans) {
    int m = bx_gid(); if (m >= M) return;
    int baseOff = m * D;
    int pqOff = m * 2 * K;
    for (int i = 0; i < D; i++) dx[baseOff + i] = x[baseOff + i];
    for (int s = 0; s < K; s++) {
        int j = trans != 0 ? (K - 1 - s) : s;
        int uOff = j * D;
        float p = 0.0f;
        for (int i = 0; i < D; i++) p += dx[baseOff + i] * uHat[uOff + i];
        pqBuf[pqOff + s] = p;
        for (int i = 0; i < D; i++) {
            float step = p * uHat[uOff + i];
            dx[baseOff + i] = (dx[baseOff + i] - step) - step;
        }
    }
    for (int i = 0; i < D; i++) dx[baseOff + i] = dy[baseOff + i];
    for (int s = K - 1; s >= 0; s--) {
        int j = trans != 0 ? (K - 1 - s) : s;
        int uOff = j * D;
        float q = 0.0f;
        for (int i = 0; i < D; i++) q += dx[baseOff + i] * uHat[uOff + i];
        pqBuf[pqOff + K + s] = q;
        for (int i = 0; i < D; i++) {
            float step = q * uHat[uOff + i];
            dx[baseOff + i] = (dx[baseOff + i] - step) - step;
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void hh_p_bwd_du_raw(const float* dy, const float* x, const float* uHat, const float* pqBuf, float* duHat, int M, int K, int D, int trans) {
    int col = bx_gid(); if (col >= D) return;
    for (int j = 0; j < K; j++) duHat[j * D + col] = 0.0f;
    for (int m = 0; m < M; m++) {
        int pqOff = m * 2 * K;
        float xv = x[m * D + col];
        for (int s = 0; s < K; s++) {
            int j = trans != 0 ? (K - 1 - s) : s;
            float ps = pqBuf[pqOff + s];
            float qs = pqBuf[pqOff + K + s];
            float uj = uHat[j * D + col];
            float gv = dy[m * D + col];
            for (int t = K - 1; t > s; t--) {
                int jt = trans != 0 ? (K - 1 - t) : t;
                float stepQ = pqBuf[pqOff + K + t] * uHat[jt * D + col];
                gv = (gv - stepQ) - stepQ;
            }
            float duTerm = qs * xv + ps * gv;
            duHat[j * D + col] = (duHat[j * D + col] - duTerm) - duTerm;
            float stepP = ps * uj;
            xv = (xv - stepP) - stepP;
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void hh_p_bwd_du_norm(const float* duHat, const float* uHat, const float* hhStats, float* dU, int K, int D) {
    int j = bx_gid(); if (j >= K) return;
    float invNorm = hhStats[2 * j];
    float isActive = hhStats[2 * j + 1];
    int off = j * D;
    float dot = 0.0f;
    for (int i = 0; i < D; i++) dot += duHat[off + i] * uHat[off + i];
    for (int i = 0; i < D; i++) {
        dU[off + i] = invNorm * (duHat[off + i] - isActive * dot * uHat[off + i]);
    }
}
