extern "C" __global__ __launch_bounds__(256) void smx_p_givens_fwd(const float* x, const float* angles, float* y, int totalHalf, int halfDim) {
    int idx = bx_gid(); if (idx >= totalHalf) return;
    int m = idx / halfDim;
    int i = idx - m * halfDim;
    int rowOff = m * (2 * halfDim);
    float phi = angles[i];
    float c = cosf(phi);
    float s = sinf(phi);
    float x0 = x[rowOff + i];
    float x1 = x[rowOff + halfDim + i];
    y[rowOff + i] = x0 * c - x1 * s;
    y[rowOff + halfDim + i] = x0 * s + x1 * c;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_givens_bwd_dx(const float* dy, const float* angles, float* dx, int totalHalf, int halfDim) {
    int idx = bx_gid(); if (idx >= totalHalf) return;
    int m = idx / halfDim;
    int i = idx - m * halfDim;
    int rowOff = m * (2 * halfDim);
    float phi = angles[i];
    float c = cosf(phi);
    float s = sinf(phi);
    float dy0 = dy[rowOff + i];
    float dy1 = dy[rowOff + halfDim + i];
    dx[rowOff + i] = dy0 * c + dy1 * s;
    dx[rowOff + halfDim + i] = -dy0 * s + dy1 * c;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_givens_bwd_dphi(const float* dy, const float* x, const float* angles, float* dAngles, int rows, int halfDim) {
    int i = bx_gid(); if (i >= halfDim) return;
    float phi = angles[i];
    float c = cosf(phi);
    float s = sinf(phi);
    int dim = 2 * halfDim;
    float acc = 0.0f;
    for (int m = 0; m < rows; m++) {
        int rowOff = m * dim;
        float x0 = x[rowOff + i];
        float x1 = x[rowOff + halfDim + i];
        float dy0 = dy[rowOff + i];
        float dy1 = dy[rowOff + halfDim + i];
        acc += dy0 * (-x0 * s - x1 * c) + dy1 * (x0 * c - x1 * s);
    }
    dAngles[i] = acc;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_conv2_fwd(const float* x, const float* w0, const float* w1, float* y, int total, int seqLen, int channels) {
    int idx = bx_gid(); if (idx >= total) return;
    int c = idx % channels;
    int bt = idx / channels;
    int t = bt % seqLen;
    float xv = x[idx];
    float xp = t > 0 ? x[idx - channels] : 0.0f;
    y[idx] = w0[c] * xv + w1[c] * xp;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_conv2_bwd_dx(const float* dy, const float* w0, const float* w1, float* dx, int total, int seqLen, int channels) {
    int idx = bx_gid(); if (idx >= total) return;
    int c = idx % channels;
    int bt = idx / channels;
    int t = bt % seqLen;
    float dyv = dy[idx];
    float dyn = (t + 1 < seqLen) ? dy[idx + channels] : 0.0f;
    dx[idx] = w0[c] * dyv + w1[c] * dyn;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_conv2_bwd_dw(const float* dy, const float* x, float* dw0, float* dw1, int batch, int seqLen, int channels) {
    int c = bx_gid(); if (c >= channels) return;
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (int b = 0; b < batch; b++) {
        int bOff = b * seqLen * channels + c;
        for (int t = 0; t < seqLen; t++) {
            int cur = bOff + t * channels;
            float dyv = dy[cur];
            a0 += dyv * x[cur];
            if (t > 0) a1 += dyv * x[cur - channels];
        }
    }
    dw0[c] = a0;
    dw1[c] = a1;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_ropel2_fwd(const float* x, const float* cosBuf, const float* sinBuf, float* y, int totalHeads, int seqLen, int numHeads, int headDim, int startPos, int epsBits, int scaleBits) {
    int r = bx_gid(); if (r >= totalHeads) return;
    float eps = __int_as_float(epsBits);
    float postScale = __int_as_float(scaleBits);
    int bt = r / numHeads;
    int t = bt % seqLen;
    int half = headDim / 2;
    int cOff = (startPos + t) * half;
    int xOff = r * headDim;
    float normSq = 0.0f;
    for (int p = 0; p < half; p++) {
        float x0 = x[xOff + 2 * p];
        float x1 = x[xOff + 2 * p + 1];
        float c = cosBuf[cOff + p];
        float s = sinBuf[cOff + p];
        float r0 = x0 * c - x1 * s;
        float r1 = x0 * s + x1 * c;
        normSq += r0 * r0 + r1 * r1;
    }
    float sc = postScale / sqrtf(normSq + eps);
    for (int p = 0; p < half; p++) {
        float x0 = x[xOff + 2 * p];
        float x1 = x[xOff + 2 * p + 1];
        float c = cosBuf[cOff + p];
        float s = sinBuf[cOff + p];
        y[xOff + 2 * p] = (x0 * c - x1 * s) * sc;
        y[xOff + 2 * p + 1] = (x0 * s + x1 * c) * sc;
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_ropel2_bwd(const float* dy, const float* x, const float* cosBuf, const float* sinBuf, float* dx, int totalHeads, int seqLen, int numHeads, int headDim, int startPos, int epsBits, int scaleBits) {
    int r = bx_gid(); if (r >= totalHeads) return;
    float eps = __int_as_float(epsBits);
    float postScale = __int_as_float(scaleBits);
    int bt = r / numHeads;
    int t = bt % seqLen;
    int half = headDim / 2;
    int cOff = (startPos + t) * half;
    int xOff = r * headDim;
    float normSq = 0.0f;
    for (int p = 0; p < half; p++) {
        float x0 = x[xOff + 2 * p];
        float x1 = x[xOff + 2 * p + 1];
        float c = cosBuf[cOff + p];
        float s = sinBuf[cOff + p];
        float r0 = x0 * c - x1 * s;
        float r1 = x0 * s + x1 * c;
        normSq += r0 * r0 + r1 * r1;
    }
    float invNorm = 1.0f / sqrtf(normSq + eps);
    float dot = 0.0f;
    for (int p = 0; p < half; p++) {
        float x0 = x[xOff + 2 * p];
        float x1 = x[xOff + 2 * p + 1];
        float c = cosBuf[cOff + p];
        float s = sinBuf[cOff + p];
        float u0 = (x0 * c - x1 * s) * invNorm;
        float u1 = (x0 * s + x1 * c) * invNorm;
        dot += dy[xOff + 2 * p] * u0 + dy[xOff + 2 * p + 1] * u1;
    }
    float sc = postScale * invNorm;
    for (int p = 0; p < half; p++) {
        float x0 = x[xOff + 2 * p];
        float x1 = x[xOff + 2 * p + 1];
        float c = cosBuf[cOff + p];
        float s = sinBuf[cOff + p];
        float u0 = (x0 * c - x1 * s) * invNorm;
        float u1 = (x0 * s + x1 * c) * invNorm;
        float dr0 = sc * (dy[xOff + 2 * p] - u0 * dot);
        float dr1 = sc * (dy[xOff + 2 * p + 1] - u1 * dot);
        dx[xOff + 2 * p] = dr0 * c + dr1 * s;
        dx[xOff + 2 * p + 1] = -dr0 * s + dr1 * c;
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_ropequad_fwd(const float* x, const float* cosBuf, const float* sinBuf, float* y, int totalHeads, int seqLen, int numHeads, int mb, int outDim, int startPos) {
    int r = bx_gid(); if (r >= totalHeads) return;
    int bt = r / numHeads;
    int t = bt % seqLen;
    int half = mb / 2;
    int cOff = (startPos + t) * half;
    int xOff = r * mb;
    int yOff = r * outDim;
    int mb2 = mb * mb;
    for (int u = 0; u < mb; u++) {
        int pu = u / 2;
        float xu0 = x[xOff + 2 * pu];
        float xu1 = x[xOff + 2 * pu + 1];
        float cu = cosBuf[cOff + pu];
        float su = sinBuf[cOff + pu];
        float ru = (u - 2 * pu) == 0 ? (xu0 * cu - xu1 * su) : (xu0 * su + xu1 * cu);
        for (int v = 0; v < mb; v++) {
            int pv = v / 2;
            float xv0 = x[xOff + 2 * pv];
            float xv1 = x[xOff + 2 * pv + 1];
            float cv = cosBuf[cOff + pv];
            float sv = sinBuf[cOff + pv];
            float rv = (v - 2 * pv) == 0 ? (xv0 * cv - xv1 * sv) : (xv0 * sv + xv1 * cv);
            y[yOff + u * mb + v] = ru * rv;
        }
    }
    for (int k = mb2; k < outDim; k++) y[yOff + k] = 0.0f;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_ropequad_bwd(const float* dy, const float* x, const float* cosBuf, const float* sinBuf, float* dx, int totalHeads, int seqLen, int numHeads, int mb, int outDim, int startPos) {
    int r = bx_gid(); if (r >= totalHeads) return;
    int bt = r / numHeads;
    int t = bt % seqLen;
    int half = mb / 2;
    int cOff = (startPos + t) * half;
    int xOff = r * mb;
    int yOff = r * outDim;
    for (int p = 0; p < half; p++) {
        float dr0 = 0.0f;
        float dr1 = 0.0f;
        int u0 = 2 * p;
        int u1 = 2 * p + 1;
        for (int v = 0; v < mb; v++) {
            int pv = v / 2;
            float xv0 = x[xOff + 2 * pv];
            float xv1 = x[xOff + 2 * pv + 1];
            float cv = cosBuf[cOff + pv];
            float sv = sinBuf[cOff + pv];
            float rv = (v - 2 * pv) == 0 ? (xv0 * cv - xv1 * sv) : (xv0 * sv + xv1 * cv);
            dr0 += (dy[yOff + u0 * mb + v] + dy[yOff + v * mb + u0]) * rv;
            dr1 += (dy[yOff + u1 * mb + v] + dy[yOff + v * mb + u1]) * rv;
        }
        float cp = cosBuf[cOff + p];
        float sp = sinBuf[cOff + p];
        dx[xOff + u0] = dr0 * cp + dr1 * sp;
        dx[xOff + u1] = -dr0 * sp + dr1 * cp;
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_projres_fwd(const float* h, const float* u, const float* gateLogit, float* y, int rows, int dim, int epsBits) {
    int m = bx_gid(); if (m >= rows) return;
    float eps = __int_as_float(epsBits);
    float g = 1.0f / (1.0f + expf(-gateLogit[0]));
    float beta = 1.0f - g;
    int off = m * dim;
    float dot = 0.0f;
    float normSq = 0.0f;
    for (int j = 0; j < dim; j++) {
        float hj = h[off + j];
        dot += u[off + j] * hj;
        normSq += hj * hj;
    }
    float coeff = dot / (normSq + eps);
    float scaleH = 1.0f - beta * coeff;
    for (int j = 0; j < dim; j++) {
        y[off + j] = u[off + j] + scaleH * h[off + j];
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_projres_bwd_rows(const float* dy, const float* h, const float* u, const float* gateLogit, float* dh, float* du, float* rowTerms, int rows, int dim, int epsBits) {
    int m = bx_gid(); if (m >= rows) return;
    float eps = __int_as_float(epsBits);
    float g = 1.0f / (1.0f + expf(-gateLogit[0]));
    float beta = 1.0f - g;
    int off = m * dim;
    float dot = 0.0f;
    float normSq = 0.0f;
    float pm = 0.0f;
    for (int j = 0; j < dim; j++) {
        float hj = h[off + j];
        float uj = u[off + j];
        float dyj = dy[off + j];
        dot += uj * hj;
        normSq += hj * hj;
        pm += dyj * hj;
    }
    float denom = normSq + eps;
    float coeff = dot / denom;
    float scaleH = 1.0f - beta * coeff;
    float kFactor = (beta * pm) / denom;
    rowTerms[m] = coeff * pm;
    for (int j = 0; j < dim; j++) {
        float hj = h[off + j];
        float uj = u[off + j];
        float dyj = dy[off + j];
        du[off + j] = dyj - kFactor * hj;
        dh[off + j] = scaleH * dyj - kFactor * (uj - 2.0f * coeff * hj);
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_projres_bwd_dgate(const float* rowTerms, const float* gateLogit, float* dGateLogit, int rows) {
    int tid = bx_gid(); if (tid >= 1) return;
    float g = 1.0f / (1.0f + expf(-gateLogit[0]));
    float acc = 0.0f;
    for (int m = 0; m < rows; m++) acc += rowTerms[m];
    dGateLogit[0] = g * (1.0f - g) * acc;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_speccirc_h(const float* specRe, const float* specIm, float* hTime, int batch, int dim) {
    int gid = bx_gid(); if (gid >= batch * dim) return;
    int b = gid / dim;
    int m = gid - b * dim;
    int half = dim / 2;
    int nf = half + 1;
    int specOff = b * nf;
    float dimF = (float)dim;
    float twoPiOverD = 6.2831853071795864769f / dimF;
    float signNyq = (m - 2 * (m / 2)) == 0 ? 1.0f : -1.0f;
    float sum = specRe[specOff] + signNyq * specRe[specOff + half];
    for (int f = 1; f < half; f++) {
        int fm = (f * m) % dim;
        float fmF = (float)fm;
        float ang = twoPiOverD * fmF;
        sum += 2.0f * (specRe[specOff + f] * cosf(ang) - specIm[specOff + f] * sinf(ang));
    }
    hTime[gid] = sum / dimF;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_speccirc_w(const float* diagScale, const float* hTime, float* outMat, int batch, int dim) {
    int gid = bx_gid(); if (gid >= batch * dim * dim) return;
    int dd = dim * dim;
    int b = gid / dd;
    int rem = gid - b * dd;
    int r = rem / dim;
    int c = rem - r * dim;
    int shift = (c - r + dim) % dim;
    outMat[gid] = diagScale[b * dim + r] * hTime[b * dim + shift];
}

extern "C" __global__ __launch_bounds__(256) void smx_p_speccirc_bwd_dhdd(const float* dOut, const float* diagScale, const float* hTime, float* dDiag, float* dhTime, int batch, int dim) {
    int gid = bx_gid(); if (gid >= batch * dim) return;
    int b = gid / dim;
    int idx = gid - b * dim;
    int matBase = b * dim * dim;
    int vecBase = b * dim;
    float accDiag = 0.0f;
    float accH = 0.0f;
    for (int k = 0; k < dim; k++) {
        int shift = (k - idx + dim) % dim;
        accDiag += dOut[matBase + idx * dim + k] * hTime[vecBase + shift];
        int col = (k + idx) % dim;
        accH += dOut[matBase + k * dim + col] * diagScale[vecBase + k];
    }
    dDiag[gid] = accDiag;
    dhTime[gid] = accH;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_speccirc_bwd_dspec(const float* dhTime, float* dRe, float* dIm, int batch, int dim) {
    int half = dim / 2;
    int nf = half + 1;
    int gid = bx_gid(); if (gid >= batch * nf) return;
    int b = gid / nf;
    int f = gid - b * nf;
    int vecBase = b * dim;
    float dimF = (float)dim;
    float invD = 1.0f / dimF;
    if (f == 0) {
        float s0 = 0.0f;
        for (int m = 0; m < dim; m++) s0 += dhTime[vecBase + m];
        dRe[gid] = s0 * invD;
        dIm[gid] = 0.0f;
    } else if (f == half) {
        float sn = 0.0f;
        for (int m = 0; m < dim; m++) {
            float sgn = (m - 2 * (m / 2)) == 0 ? 1.0f : -1.0f;
            sn += sgn * dhTime[vecBase + m];
        }
        dRe[gid] = sn * invD;
        dIm[gid] = 0.0f;
    } else {
        float twoPiOverD = 6.2831853071795864769f / dimF;
        float accRe = 0.0f;
        float accIm = 0.0f;
        for (int m = 0; m < dim; m++) {
            int fm = (f * m) % dim;
            float fmF = (float)fm;
            float ang = twoPiOverD * fmF;
            float dhv = dhTime[vecBase + m];
            accRe += dhv * cosf(ang);
            accIm -= dhv * sinf(ang);
        }
        float scale = 2.0f * invD;
        dRe[gid] = accRe * scale;
        dIm[gid] = accIm * scale;
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_sigtopk_mix_fwd(const float* logits, const float* expertTable, const float* routingBias, float* y, float* savedInfo, int rows, int numExperts, int topK, int channels, int epsBits) {
    int m = bx_gid(); if (m >= rows) return;
    float eps = __int_as_float(epsBits);
    int logOff = m * numExperts;
    int strideInfo = 2 * topK + 1;
    int infoOff = m * strideInfo;
    float sumSel = 0.0f;
    for (int r = 0; r < topK; r++) {
        int bestIdx = -1;
        float bestScore = 0.0f;
        float bestSig = 0.0f;
        for (int j = 0; j < numExperts; j++) {
            int already = 0;
            for (int prev = 0; prev < r; prev++) {
                float pf = savedInfo[infoOff + prev];
                int pi = (int)pf;
                if (pi == j) already = 1;
            }
            if (already == 0) {
                float sj = 1.0f / (1.0f + expf(-logits[logOff + j]));
                float bj = sj + routingBias[j];
                if (bestIdx < 0 || bj > bestScore) {
                    bestIdx = j;
                    bestScore = bj;
                    bestSig = sj;
                }
            }
        }
        savedInfo[infoOff + r] = (float)bestIdx;
        savedInfo[infoOff + topK + r] = bestSig;
        sumSel += bestSig;
    }
    savedInfo[infoOff + 2 * topK] = sumSel;
    float denom = sumSel >= eps ? sumSel : eps;
    for (int r = 0; r < topK; r++) {
        savedInfo[infoOff + topK + r] = savedInfo[infoOff + topK + r] / denom;
    }
    int yOff = m * channels;
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        for (int r = 0; r < topK; r++) {
            float idxF = savedInfo[infoOff + r];
            int eIdx = (int)idxF;
            acc += savedInfo[infoOff + topK + r] * expertTable[eIdx * channels + c];
        }
        y[yOff + c] = acc;
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_sigtopk_bias_upd(const float* savedInfo, float* routingBias, int rows, int numExperts, int topK, int rateBits) {
    int e = bx_gid(); if (e >= numExperts) return;
    float rate = __int_as_float(rateBits);
    int strideInfo = 2 * topK + 1;
    int cnt = 0;
    for (int m = 0; m < rows; m++) {
        int infoOff = m * strideInfo;
        for (int r = 0; r < topK; r++) {
            float idxF = savedInfo[infoOff + r];
            int eIdx = (int)idxF;
            if (eIdx == e) cnt++;
        }
    }
    int totalSel = rows * topK;
    float totalF = (float)totalSel;
    float expF = (float)numExperts;
    float cntF = (float)cnt;
    float diff = (totalF / expF) - cntF;
    float sgn = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
    routingBias[e] = routingBias[e] + rate * sgn;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_sigtopk_mix_bwd_dlog(const float* dy, const float* logits, const float* expertTable, const float* savedInfo, float* dLogits, int rows, int numExperts, int topK, int channels, int epsBits) {
    int m = bx_gid(); if (m >= rows) return;
    float eps = __int_as_float(epsBits);
    int logOff = m * numExperts;
    for (int j = 0; j < numExperts; j++) dLogits[logOff + j] = 0.0f;
    int strideInfo = 2 * topK + 1;
    int infoOff = m * strideInfo;
    int yOff = m * channels;
    float sumSel = savedInfo[infoOff + 2 * topK];
    float dotDg = 0.0f;
    for (int r = 0; r < topK; r++) {
        float idxF = savedInfo[infoOff + r];
        int eIdx = (int)idxF;
        float gateR = savedInfo[infoOff + topK + r];
        float dGate = 0.0f;
        for (int c = 0; c < channels; c++) {
            dGate += dy[yOff + c] * expertTable[eIdx * channels + c];
        }
        dotDg += dGate * gateR;
    }
    for (int r = 0; r < topK; r++) {
        float idxF = savedInfo[infoOff + r];
        int eIdx = (int)idxF;
        float dGate = 0.0f;
        for (int c = 0; c < channels; c++) {
            dGate += dy[yOff + c] * expertTable[eIdx * channels + c];
        }
        float dSel = sumSel >= eps ? ((dGate - dotDg) / sumSel) : (dGate / eps);
        float sj = 1.0f / (1.0f + expf(-logits[logOff + eIdx]));
        dLogits[logOff + eIdx] = dSel * sj * (1.0f - sj);
    }
}

extern "C" __global__ __launch_bounds__(256) void smx_p_sigtopk_mix_bwd_dtbl(const float* dy, const float* savedInfo, float* dTable, int rows, int numExperts, int topK, int channels) {
    int gid = bx_gid(); if (gid >= numExperts * channels) return;
    int e = gid / channels;
    int c = gid - e * channels;
    int strideInfo = 2 * topK + 1;
    float acc = 0.0f;
    for (int m = 0; m < rows; m++) {
        int infoOff = m * strideInfo;
        float dyc = dy[m * channels + c];
        for (int r = 0; r < topK; r++) {
            float idxF = savedInfo[infoOff + r];
            int eIdx = (int)idxF;
            if (eIdx == e) acc += savedInfo[infoOff + topK + r] * dyc;
        }
    }
    dTable[gid] = acc;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_dualswiglu_fwd(const float* guProj, const float* guGain, const float* upBase, float* y, int rows, int hiddenDim) {
    int gid = bx_gid(); if (gid >= rows * hiddenDim) return;
    int m = gid / hiddenDim;
    int j = gid - m * hiddenDim;
    int off2 = m * (2 * hiddenDim);
    float gRaw = guProj[off2 + j];
    float uRaw = guProj[off2 + hiddenDim + j];
    float gGain = 1.0f + guGain[off2 + j];
    float uGain = upBase[j] + guGain[off2 + hiddenDim + j];
    float g = gRaw * gGain;
    float sig = 1.0f / (1.0f + expf(-g));
    y[gid] = (g * sig * uRaw) * uGain;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_dualswiglu_bwd_elem(const float* dy, const float* guProj, const float* guGain, const float* upBase, float* dGuProj, float* dGuGain, int rows, int hiddenDim) {
    int gid = bx_gid(); if (gid >= rows * hiddenDim) return;
    int m = gid / hiddenDim;
    int j = gid - m * hiddenDim;
    int off2 = m * (2 * hiddenDim);
    float dyv = dy[gid];
    float gRaw = guProj[off2 + j];
    float uRaw = guProj[off2 + hiddenDim + j];
    float gGain = 1.0f + guGain[off2 + j];
    float uGain = upBase[j] + guGain[off2 + hiddenDim + j];
    float g = gRaw * gGain;
    float sig = 1.0f / (1.0f + expf(-g));
    float silu = g * sig;
    float h = silu * uRaw;
    dGuGain[off2 + hiddenDim + j] = dyv * h;
    float dh = dyv * uGain;
    dGuProj[off2 + hiddenDim + j] = dh * silu;
    float dg = (dh * uRaw) * sig * (1.0f + g * (1.0f - sig));
    dGuProj[off2 + j] = dg * gGain;
    dGuGain[off2 + j] = dg * gRaw;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_dualswiglu_bwd_dbase(const float* dy, const float* guProj, const float* guGain, float* dUpBase, int rows, int hiddenDim) {
    int j = bx_gid(); if (j >= hiddenDim) return;
    int stride2 = 2 * hiddenDim;
    float acc = 0.0f;
    for (int m = 0; m < rows; m++) {
        int off2 = m * stride2;
        float g = guProj[off2 + j] * (1.0f + guGain[off2 + j]);
        float sig = 1.0f / (1.0f + expf(-g));
        acc += dy[m * hiddenDim + j] * (g * sig * guProj[off2 + hiddenDim + j]);
    }
    dUpBase[j] = acc;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_bornratio_fwd(const float* scanOut, const float* sinkLogAlpha, float* y, int totalRows, int numHeads, int headDim, int scanDim, int hasSink, int epsBits) {
    int gid = bx_gid(); if (gid >= totalRows * headDim) return;
    float eps = __int_as_float(epsBits);
    int row = gid / headDim;
    int j = gid - row * headDim;
    int h = row % numHeads;
    float sinkTerm = hasSink != 0 ? expf(sinkLogAlpha[h]) : 0.0f;
    float den = scanOut[row * scanDim + headDim] + eps + sinkTerm;
    y[gid] = scanOut[row * scanDim + j] / den;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_bornratio_bwd_dscan(const float* dy, const float* scanOut, const float* sinkLogAlpha, float* dScan, int totalRows, int numHeads, int headDim, int scanDim, int hasSink, int epsBits) {
    int row = bx_gid(); if (row >= totalRows) return;
    float eps = __int_as_float(epsBits);
    int h = row % numHeads;
    float sinkTerm = hasSink != 0 ? expf(sinkLogAlpha[h]) : 0.0f;
    int sOff = row * scanDim;
    int yOff = row * headDim;
    float invDen = 1.0f / (scanOut[sOff + headDim] + eps + sinkTerm);
    float dot = 0.0f;
    for (int j = 0; j < headDim; j++) {
        float dyj = dy[yOff + j];
        dot += dyj * scanOut[sOff + j];
        dScan[sOff + j] = dyj * invDen;
    }
    dScan[sOff + headDim] = -dot * invDen * invDen;
    for (int j = headDim + 1; j < scanDim; j++) dScan[sOff + j] = 0.0f;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_bornratio_bwd_dsink(const float* dy, const float* scanOut, const float* sinkLogAlpha, float* dSink, int totalRows, int numHeads, int headDim, int scanDim, int epsBits) {
    int h = bx_gid(); if (h >= numHeads) return;
    float eps = __int_as_float(epsBits);
    float sinkTerm = expf(sinkLogAlpha[h]);
    float acc = 0.0f;
    for (int row = h; row < totalRows; row += numHeads) {
        int sOff = row * scanDim;
        int yOff = row * headDim;
        float invDen = 1.0f / (scanOut[sOff + headDim] + eps + sinkTerm);
        float dot = 0.0f;
        for (int j = 0; j < headDim; j++) dot += dy[yOff + j] * scanOut[sOff + j];
        acc += -dot * invDen * invDen;
    }
    dSink[h] = acc * sinkTerm;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_boundsig_fwd(const float* logits, const float* bias, float* y, int rows, int cols, int floorBits) {
    int gid = bx_gid(); if (gid >= rows * cols) return;
    float flr = __int_as_float(floorBits);
    int c = gid % cols;
    float sig = 1.0f / (1.0f + expf(-(logits[gid] + bias[c])));
    y[gid] = flr + (1.0f - flr) * sig;
}

extern "C" __global__ __launch_bounds__(256) void smx_p_boundsig_bwd_dlog(const float* dy, const float* logits, const float* bias, float* dLogits, int rows, int cols, int floorBits) {
    int gid = bx_gid(); if (gid >= rows * cols) return;
    float flr = __int_as_float(floorBits);
    int c = gid % cols;
    float sig = 1.0f / (1.0f + expf(-(logits[gid] + bias[c])));
    dLogits[gid] = dy[gid] * (1.0f - flr) * sig * (1.0f - sig);
}

extern "C" __global__ __launch_bounds__(256) void smx_p_boundsig_bwd_dbias(const float* dy, const float* logits, const float* bias, float* dBias, int rows, int cols, int floorBits) {
    int c = bx_gid(); if (c >= cols) return;
    float flr = __int_as_float(floorBits);
    float scale = 1.0f - flr;
    float bc = bias[c];
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) {
        int idx = r * cols + c;
        float sig = 1.0f / (1.0f + expf(-(logits[idx] + bc)));
        acc += dy[idx] * scale * sig * (1.0f - sig);
    }
    dBias[c] = acc;
}
