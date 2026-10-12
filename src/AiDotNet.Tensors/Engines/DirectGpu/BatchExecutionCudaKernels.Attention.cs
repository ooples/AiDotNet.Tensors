namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    public const string FaLse = "fa_p_lse", FaOut = "fa_p_out", FaDelta = "fa_p_delta", FaDq = "fa_p_dq",
        FaDkv = "fa_p_dkv";

    /// <summary>Indices into the int32 metadata buffer every fa_p_ kernel reads (see <see cref="AttentionSource"/>).</summary>
    public const int FaMetaCount = 24;

    // #1179: fused attention for every backend without the CUDA tiles, as the CPU and CUDA kernels define it: scores
    // (q * scale) . k, an optional soft-cap cap * tanh(s / cap), causal masking j <= i + offset, an optional int32 mask
    // (0 masks) through its own strides, lse = -inf for a row with no allowed key. One thread per output element, scores
    // recomputed rather than tiled, so the source carries no per-thread arrays, atomics or helper functions and
    // translates mechanically to OpenCL, Metal and GLSL. Every integer argument travels in the meta buffer (a GLSL push
    // block holds 128 bytes, too few for the strides): B H Sq Sk D, q k v o strides (batch, head, row), mask strides
    // (batch, head, row, column), causal, causal offset, has-mask; fp holds scale and soft-cap.
    internal const string AttentionSource = @"
#define FA_META int H = meta[1]; int Sq = meta[2]; int Sk = meta[3]; int D = meta[4]; int qb = meta[5]; int qh = meta[6]; int qr = meta[7]; int kb = meta[8]; int kh = meta[9]; int kr = meta[10]; int vb = meta[11]; int vh = meta[12]; int vr = meta[13]; int ob = meta[14]; int oh = meta[15]; int orw = meta[16]; int mb = meta[17]; int mh = meta[18]; int mi = meta[19]; int mj = meta[20]; int causal = meta[21]; int coff = meta[22]; int hasMask = meta[23];
#define FA_ALLOWED(I, J) (!(causal != 0 && (J) > (I) + coff) && !(hasMask != 0 && mask[b * mb + h * mh + (I) * mi + (J) * mj] == 0))
#define FA_SCORE(I, J, S) { S = 0.0f; for (int fa_d = 0; fa_d < D; fa_d++) S += (q[b * qb + h * qh + (I) * qr + fa_d] * scale) * k[b * kb + h * kh + (J) * kr + fa_d]; }
#define FA_CAP(S, SL) { SL = 1.0f; if (cap > 0.0f) { float fa_t = tanhf(S / cap); SL = 1.0f - fa_t * fa_t; S = cap * fa_t; } }
#define FA_DP(I, J, DP) { DP = 0.0f; for (int fa_e = 0; fa_e < D; fa_e++) DP += dout[b * ob + h * oh + (I) * orw + fa_e] * v[b * vb + h * vh + (J) * vr + fa_e]; }

extern ""C"" __global__ __launch_bounds__(256) void fa_p_lse(const float* q, const float* k, const int* meta, const float* fp, const int* mask, float* lse, int total) {
    int t = bx_gid(); if (t >= total) return;
    FA_META
    float scale = fp[0]; float cap = fp[1];
    int i = t % Sq; int bh = t / Sq; int b = bh / H; int h = bh % H;
    float m = 0.0f; int seen = 0;
    for (int j = 0; j < Sk; j++) {
        if (!(FA_ALLOWED(i, j))) continue;
        float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
        if (seen == 0 || s > m) m = s;
        seen = 1;
    }
    if (seen == 0) { lse[t] = __uint_as_float(0xff800000u); return; }
    float l = 0.0f;
    for (int j = 0; j < Sk; j++) {
        if (!(FA_ALLOWED(i, j))) continue;
        float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
        l += expf(s - m);
    }
    lse[t] = m + logf(l);
}

extern ""C"" __global__ __launch_bounds__(256) void fa_p_out(const float* q, const float* k, const float* v, const int* meta, const float* fp, const int* mask, const float* lse, float* o, int total) {
    int t = bx_gid(); if (t >= total) return;
    FA_META
    float scale = fp[0]; float cap = fp[1];
    int e = t % D; int r = t / D; int i = r % Sq; int bh = r / Sq; int b = bh / H; int h = bh % H;
    float lr = lse[r]; float acc = 0.0f;
    if (lr > -3.0e38f) {
        for (int j = 0; j < Sk; j++) {
            if (!(FA_ALLOWED(i, j))) continue;
            float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
            acc += expf(s - lr) * v[b * vb + h * vh + j * vr + e];
        }
    }
    o[b * ob + h * oh + i * orw + e] = acc;
}

extern ""C"" __global__ __launch_bounds__(256) void fa_p_delta(const float* o, const float* dout, const int* meta, float* delta, int total) {
    int t = bx_gid(); if (t >= total) return;
    FA_META
    int i = t % Sq; int bh = t / Sq; int b = bh / H; int h = bh % H;
    float acc = 0.0f;
    for (int e = 0; e < D; e++) acc += dout[b * ob + h * oh + i * orw + e] * o[b * ob + h * oh + i * orw + e];
    delta[t] = acc;
}

extern ""C"" __global__ __launch_bounds__(256) void fa_p_dq(const float* q, const float* k, const float* v, const float* dout, const float* lse, const float* delta, const int* meta, const float* fp, const int* mask, float* dq, int total) {
    int t = bx_gid(); if (t >= total) return;
    FA_META
    float scale = fp[0]; float cap = fp[1];
    int e = t % D; int r = t / D; int i = r % Sq; int bh = r / Sq; int b = bh / H; int h = bh % H;
    float lr = lse[r]; float acc = 0.0f;
    if (lr > -3.0e38f) {
        float dl = delta[r];
        for (int j = 0; j < Sk; j++) {
            if (!(FA_ALLOWED(i, j))) continue;
            float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
            float dp; FA_DP(i, j, dp)
            acc += expf(s - lr) * (dp - dl) * sl * k[b * kb + h * kh + j * kr + e];
        }
    }
    dq[b * qb + h * qh + i * qr + e] = acc * scale;
}

extern ""C"" __global__ __launch_bounds__(256) void fa_p_dkv(const float* q, const float* k, const float* v, const float* dout, const float* lse, const float* delta, const int* meta, const float* fp, const int* mask, float* dk, float* dv, int total) {
    int t = bx_gid(); if (t >= total) return;
    FA_META
    float scale = fp[0]; float cap = fp[1];
    int e = t % D; int r = t / D; int j = r % Sk; int bh = r / Sk; int b = bh / H; int h = bh % H;
    float ak = 0.0f; float av = 0.0f;
    for (int i = 0; i < Sq; i++) {
        float lr = lse[bh * Sq + i];
        if (!(lr > -3.0e38f)) continue;
        if (!(FA_ALLOWED(i, j))) continue;
        float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
        float dp; FA_DP(i, j, dp)
        float p = expf(s - lr);
        av += p * dout[b * ob + h * oh + i * orw + e];
        ak += p * (dp - delta[bh * Sq + i]) * sl * q[b * qb + h * qh + i * qr + e] * scale;
    }
    dk[b * kb + h * kh + j * kr + e] = ak;
    dv[b * vb + h * vh + j * vr + e] = av;
}
";
}
