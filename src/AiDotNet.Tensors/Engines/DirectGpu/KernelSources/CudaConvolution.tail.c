
// Fusion hooks live: pro = relu(x * ps[ci] + pt[ci]) on in-bounds inputs when ps is set (an inference batch norm and
// ReLU before the convolution; padding stays 0), epi = + bias[co] (ef bit 0), ReLU (bit 1) in the store (ef = 0 with
// split-K, whose partials are summed after).
struct CeAffPro { const float* s; const float* t;
    __device__ __forceinline__ float operator()(int, int c, int, float x) const { return s ? fmaxf(fmaf(x, s[c], t[c]), 0.0f) : x; } };
struct CeBiasEpi { const float* b; int f;
    __device__ __forceinline__ float operator()(int, int c, int, float y) const { if (f & 1) y += b[c]; return (f & 2) ? fmaxf(y, 0.0f) : y; } };
#define CE_FPROP_KERNEL(NAME, BM, TAPU, MAIN) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit, \
    const float* bias, int ef, const float* ps, const float* pt) \
{ \
    CeAffPro pro; pro.s = ps; pro.t = pt; CeBiasEpi epi; epi.b = bias; epi.f = ef; \
    conv2d_fprop_ig_body<BM, BM, TAPU, CeAffPro, CeBiasEpi, MAIN>(input, kernel, out, batch, inC, H, W, \
        outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit, pro, epi); \
}
CE_FPROP_KERNEL(conv2d_fprop_ig128_tap, 128, true, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig128_gen, 128, false, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig64_tap, 64, true, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig64_gen, 64, false, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig128_tap_x3, 128, true, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig128_gen_x3, 128, false, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig64_tap_x3, 64, true, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig64_gen_x3, 64, false, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig128_tap_tf32, 128, true, 2)
CE_FPROP_KERNEL(conv2d_fprop_ig128_gen_tf32, 128, false, 2)
CE_FPROP_KERNEL(conv2d_fprop_ig64_tap_tf32, 64, true, 2)
CE_FPROP_KERNEL(conv2d_fprop_ig64_gen_tf32, 64, false, 2)

// Weight gradient on the engine core: dW[Cout, P] = dY[Cout, K] * X_gathered[K, P], P = Cin*kH*kW, K = B*oH*oW.
// Both loaders put one K element per lane (8 lanes along K, rows 32 apart), so each lane decomposes its k = (b, oh, ow)
// once per tile, dY reads coalesce along the output row, and every column's (ci, kh, kw) is fixed for the whole loop.
// Split-K slices go to out + z * Cout*P, summed in order by conv2d_wgrad_splitk_reduce.
template <int BM, int BN, int MAIN>
__device__ __forceinline__ void conv2d_wgrad_ce_body(
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + CePad<MAIN>::value) + 2 * CE_BK * (BN + CePad<MAIN>::value)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256;
    const int L = outH * outW, khw = kH * kW, P = inC * khw, K = batch * L, HW = H * W;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kBegin = blockIdx.z * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + CE_BK - 1) / CE_BK : 0;
    const int tid = threadIdx.x;
    const int lk = tid & 7, lr = tid >> 3;   // k lane; rows (A) / columns (B) lr + 32 r

    int cci[RB], ckh[RB], ckw[RB];
    bool cIn[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++)
    {
        int p = n0 + lr + 32 * r;
        cIn[r] = p < P;
        int pp = cIn[r] ? p : 0;
        cci[r] = pp / khw;
        int t = pp - cci[r] * khw;
        ckh[r] = t / kW;
        ckw[r] = t - ckh[r] * kW;
    }

    int kc = kBegin;
    float ra[RA], rb[RB];
    auto load = [&]()
    {
        int k = kc + lk;
        bool kIn = k < kEnd;
        int b = 0, oh = 0, ow = 0;
        if (kIn) { b = k / L; int l = k - b * L; oh = l / outW; ow = l - oh * outW; }
        const float* dyRow = gradOutput + (long long)b * outC * L + oh * outW + ow;
        #pragma unroll
        for (int r = 0; r < RA; r++)
        {
            int m = m0 + lr + 32 * r;
            ra[r] = (kIn && m < outC) ? dyRow[(long long)m * L] : 0.0f;
        }
        const float* xImage = input + (long long)b * inC * HW;
        int ihBase = oh * sH - pH, iwBase = ow * sW - pW;
        #pragma unroll
        for (int r = 0; r < RB; r++)
        {
            float v = 0.0f;
            if (kIn && cIn[r])
            {
                int ih = ihBase + ckh[r] * dH, iw = iwBase + ckw[r] * dW;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = xImage[(long long)cci[r] * HW + ih * W + iw];
            }
            rb[r] = v;
        }
    };
    // Concrete parameter types, not (auto as, auto bs): NVRTC 12.x (the 12.6 runtime AiDotNet.Native.CUDA ships) treats
    // a generic lambda as a host function in JIT mode and rejects the whole module, leaving the CUDA backend
    // unavailable. The tiles are the mainloop's shared buffers, rows padded by CePad<MAIN>.
    auto store = [&](float (*as)[BM + CePad<MAIN>::value], float (*bs)[BN + CePad<MAIN>::value])
    {
        #pragma unroll
        for (int r = 0; r < RA; r++) as[lk][lr + 32 * r] = ra[r];
        #pragma unroll
        for (int r = 0; r < RB; r++) bs[lk][lr + 32 * r] = rb[r];
    };
    auto advance = [&]() { kc += CE_BK; };

    float acc[TM][TN];
    ce_run_mainloop<BM, BN, MAIN>(smem, tiles, load, store, advance, acc);

    using Map = CeMap<BM, BN, MAIN>;
    constexpr int VEC = Map::VEC;
    float* slice = out + (long long)blockIdx.z * outC * P;
    const bool vec = (P % VEC) == 0;
    #pragma unroll
    for (int jg = 0; jg < TN; jg += VEC)
    {
        int nn = n0 + Map::col(jg);
        if (nn >= P) continue;
        #pragma unroll
        for (int i = 0; i < TM; i++)
        {
            int m = m0 + Map::row(i);
            if (m >= outC) continue;
            float* rowp = slice + (long long)m * P + nn;
            float v[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) v[j] = acc[i][jg + j];
            if (vec)
                ce_store_vec<VEC>(rowp, v);
            else
            {
                #pragma unroll
                for (int j = 0; j < VEC; j++)
                    if (nn + j < P) rowp[j] = v[j];
            }
        }
    }
}

// Input gradient on the engine core, Cout % 8 == 0: dX[Cin, B*H*W] = W'[Cin, K] * dY_gathered[K, B*H*W] with K ordered
// (co / 8, kh, kw, co % 8), so a K tile shares one tap. The A loader's lanes run along ci. tapMajor: the weights arrive as
// wt[kh*kW + kw][co][ci] (conv2d_dgrad_weight_tapmajor), so those lanes read contiguous floats; else the forward layout
// w[co][ci][kh][kw], where consecutive ci are kH*kW floats apart (a 1x1 kernel's layout is the same either way). The
// host relays out only when enough column tiles reread the weights to pay for the copy. PHASE = false: unit stride or any stride with per-tile divisibility checks,
// split-K slices to out + z * B*Cin*H*W. PHASE = true (strided, unit dilation): blockIdx.z is the
// sub-pixel phase (rh, rw); the block's columns are that phase's pixels and only the taps kh = rh (mod sH),
// kw = rw (mod sW) are visited, every one of which hits; phases write disjoint pixels. With split-K, each phase's K
// is cut into ceil(K_phase / kPerSplit) slices (a 4-tap phase gets four times the slices of a 1-tap one, so every grid
// layer carries the same K) and conv2d_dgrad_phase_reduce sums each pixel's slices in order.
template <int BM, int BN, bool PHASE, int MAIN>
__device__ __forceinline__ void conv2d_dgrad_ce_body(
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit, int tapMajor)
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + CePad<MAIN>::value) + 2 * CE_BK * (BN + CePad<MAIN>::value)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256, BSTEP = 256 / BN;
    const int HW = H * W, khw = kH * kW, P = inC * khw, L = outH * outW;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // Column space: every pixel (PHASE = false) or the pixels of one phase.
    // PHASE: blockIdx.z enumerates (phase, split) with phase-major order and ceil(K_phase / kPerSplit) splits per
    // phase; each split takes a CE_BK-aligned K range of its phase's taps and writes a disjoint-pixel partial slice,
    // summed in order afterwards when any phase has more than one split.
    int rh = 0, rw = 0, ih0 = 0, iw0 = 0, Hp = H, Wp = W, tapsH = kH, tapsW = kW, split = 0;
    if (PHASE)
    {
        const int phases = sH * sW;
        int phase = 0, rem = blockIdx.z;
        for (; phase < phases - 1; phase++)
        {
            const int ph = phase / sW, pw = phase - ph * sW;
            const int units = max(1, (outC * ((kH - ph + sH - 1) / sH) * ((kW - pw + sW - 1) / sW) + kPerSplit - 1) / kPerSplit);
            if (rem < units) break;
            rem -= units;
        }
        split = rem;
        rh = phase / sW; rw = phase - rh * sW;
        ih0 = ((rh - pH) % sH + sH) % sH; iw0 = ((rw - pW) % sW + sW) % sW;
        Hp = ih0 < H ? (H - ih0 + sH - 1) / sH : 0; Wp = iw0 < W ? (W - iw0 + sW - 1) / sW : 0;
        tapsH = (kH - rh + sH - 1) / sH; tapsW = (kW - rw + sW - 1) / sW;
    }
    const int plane = Hp * Wp, N = batch * plane;
    if (n0 >= N) return;
    const int K = outC * tapsH * tapsW;
    const int kBegin = (PHASE ? split : (int)blockIdx.z) * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + CE_BK - 1) / CE_BK : 0;

    constexpr int ASTEP = 256 / BM;   // A tile rows (K) per pass: lane runs along m (ci), RA passes of ASTEP rows
    const int am = tid % BM, ak = tid / BM;
    const int bn = tid % BN, bk = tid / BN;
    const int n = n0 + bn;
    const bool nIn = n < N;
    int nb = 0, ihp = 0, iwp = 0;
    if (nIn)
    {
        nb = n / plane; int q = n - nb * plane; int i = q / Wp;
        ihp = ih0 + (PHASE ? i * sH : i) + pH;
        iwp = iw0 + (PHASE ? (q - i * Wp) * sW : q - i * Wp) + pW;
    }
    const float* dyBatch = gradOutput + (long long)nb * outC * L;
    // PHASE: the phase's taps kh = rh + th * sH hit dY row (ihp - rh) / sH - th exactly (ihp - rh is a non-negative
    // multiple of sH), so each K tile's row and column are a subtraction, not two runtime divisions per lane per tile
    // (measured: those divisions were most of the phase kernel's integer instructions).
    const int ohBase = PHASE ? (ihp - rh) / sH : 0, owBase = PHASE ? (iwp - rw) / sW : 0;

    // Tile state: tap (kh, kw) and first output channel co0 (K tiles never straddle a tap: Cout % 8 == 0).
    // K order (co group of CE_BK, tap, co within the group), as in the forward: the taps of one group read the weights
    // w[co][ci][0 .. kH*kW-1] back to back, so in the forward layout each (co, ci) run stays in L1 across its taps.
    // A tapless phase (a kernel smaller than its stride, e.g. a 1x1 stride-2 downsample) has K = 0 and no tiles; it
    // only writes zeros, so its tap state is never read and must not be derived by dividing by taps or tapsW.
    const int taps = tapsH * tapsW;
    int g0 = 0, tapIndex = 0, th = 0, tw = 0;
    if (taps > 0)
    {
        g0 = kBegin / (CE_BK * taps);
        tapIndex = (kBegin - g0 * CE_BK * taps) / CE_BK;
        th = tapIndex / tapsW;
        tw = tapIndex - th * tapsW;
    }
    int co0 = g0 * CE_BK;
    int kh = PHASE ? rh + th * sH : th, kw = PHASE ? rw + tw * sW : tw;

    float ra[RA], rb[RB];
    auto load = [&]()
    {
        const int m = m0 + am;
        const int tap = kh * kW + kw;
        const float* wTap = tapMajor ? kernel + ((long long)tap * outC + co0 + ak) * inC + m
                                     : kernel + ((long long)(co0 + ak) * inC + m) * khw + tap;
        const long long rowStep = tapMajor ? (long long)ASTEP * inC : (long long)ASTEP * P;
        #pragma unroll
        for (int r = 0; r < RA; r++)
            ra[r] = m < inC ? wTap[r * rowStep] : 0.0f;
        int oh, ow;
        bool valid;
        if (PHASE)
        {
            oh = ohBase - th; ow = owBase - tw;
            valid = nIn && oh >= 0 && ow >= 0;
        }
        else
        {
            int ty_ = ihp - kh * dH, tx_ = iwp - kw * dW;
            oh = ty_; ow = tx_;
            valid = nIn && ty_ >= 0 && tx_ >= 0;
            if (valid && (sH > 1 || sW > 1))
            {
                oh = ty_ / sH; ow = tx_ / sW;
                valid = oh * sH == ty_ && ow * sW == tx_;
            }
        }
        valid = valid && oh < outH && ow < outW;
        const float* src = dyBatch + (long long)(co0 + bk) * L + (valid ? oh * outW + ow : 0);
        #pragma unroll
        for (int r = 0; r < RB; r++)
            rb[r] = valid ? src[(long long)(BSTEP * r) * L] : 0.0f;
    };
    // Concrete parameter types, not (auto as, auto bs): NVRTC 12.x (the 12.6 runtime AiDotNet.Native.CUDA ships) treats
    // a generic lambda as a host function in JIT mode and rejects the whole module, leaving the CUDA backend
    // unavailable. The tiles are the mainloop's shared buffers, rows padded by CePad<MAIN>.
    auto store = [&](float (*as)[BM + CePad<MAIN>::value], float (*bs)[BN + CePad<MAIN>::value])
    {
        #pragma unroll
        for (int r = 0; r < RA; r++) as[ak + ASTEP * r][am] = ra[r];
        #pragma unroll
        for (int r = 0; r < RB; r++) bs[bk + BSTEP * r][bn] = rb[r];
    };
    auto advance = [&]()
    {
        if (++tw == tapsW) { tw = 0; th++; }
        if (th == tapsH) { th = 0; co0 += CE_BK; }
        kh = PHASE ? rh + th * sH : th; kw = PHASE ? rw + tw * sW : tw;
    };

    float acc[TM][TN];
    ce_run_mainloop<BM, BN, MAIN>(smem, tiles, load, store, advance, acc);

    using Map = CeMap<BM, BN, MAIN>;
    constexpr int VEC = Map::VEC;
    float* slice = out + (long long)(PHASE ? split : (int)blockIdx.z) * batch * inC * HW;
    if (PHASE)
    {
        // A phase's pixels are sH apart in dX, so the register layout (a lane's 4 columns, lanes 4 columns apart) put
        // every lane of a store in its own 32-byte sector. The tile goes through shared memory 32 rows at a time instead
        // and consecutive lanes store consecutive phase pixels (32 rows * (BN + 1) floats fit the main loop's buffers).
        constexpr int CR = 32, LD = BN + 1;
        float* cs = smem;
        for (int r0 = 0; r0 < BM; r0 += CR)
        {
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                const int rr = Map::row(i) - r0;
                if (rr < 0 || rr >= CR) continue;
                #pragma unroll
                for (int j = 0; j < TN; j++) cs[rr * LD + Map::col(j)] = acc[i][j];
            }
            __syncthreads();
            for (int e = tid; e < CR * BN; e += 256)
            {
                const int rr = e / BN, nl = e - rr * BN;
                const int m = m0 + r0 + rr, nj = n0 + nl;
                if (m >= inC || nj >= N) continue;
                const int b = nj / plane, q = nj - b * plane, i2 = q / Wp;
                const int pix = (ih0 + i2 * sH) * W + iw0 + (q - i2 * Wp) * sW;
                slice[((long long)b * inC + m) * HW + pix] = cs[rr * LD + nl];
            }
        }
        return;
    }
    const bool vec = (HW % VEC) == 0;
    #pragma unroll
    for (int jg = 0; jg < TN; jg += VEC)
    {
        int nn = n0 + Map::col(jg);
        if (nn >= N) continue;
        if (vec)
        {
            int b = nn / plane, q = nn - b * plane;
            float* colp = slice + (long long)b * inC * HW + q;
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int m = m0 + Map::row(i);
                if (m >= inC) continue;
                float v[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; j++) v[j] = acc[i][jg + j];
                ce_store_vec<VEC>(colp + (long long)m * HW, v);
            }
            continue;
        }
        #pragma unroll
        for (int j = 0; j < VEC; j++)
        {
            int nj = nn + j;
            if (nj >= N) break;
            int b = nj / plane, q = nj - b * plane, i2 = q / Wp;
            int pix = PHASE ? (ih0 + i2 * sH) * W + iw0 + (q - i2 * Wp) * sW : q;
            float* colp = slice + (long long)b * inC * HW + pix;
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int m = m0 + Map::row(i);
                if (m < inC) colp[(long long)m * HW] = acc[i][jg + j];
            }
        }
    }
}

#define CE_WGRAD_KERNEL(NAME, BM, MAIN) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit) \
{ \
    conv2d_wgrad_ce_body<BM, BM, MAIN>(input, gradOutput, out, batch, inC, H, W, outC, outH, outW, \
        kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit); \
}
CE_WGRAD_KERNEL(conv2d_wgrad_ce128, 128, 0)
CE_WGRAD_KERNEL(conv2d_wgrad_ce64, 64, 0)
CE_WGRAD_KERNEL(conv2d_wgrad_ce128_x3, 128, 1)
CE_WGRAD_KERNEL(conv2d_wgrad_ce64_x3, 64, 1)
CE_WGRAD_KERNEL(conv2d_wgrad_ce128_tf32, 128, 2)
CE_WGRAD_KERNEL(conv2d_wgrad_ce64_tf32, 64, 2)

#define CE_DGRAD_KERNEL(NAME, BM, PHASE, MAIN) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit, int tapMajor) \
{ \
    conv2d_dgrad_ce_body<BM, BM, PHASE, MAIN>(gradOutput, kernel, out, batch, inC, H, W, outC, outH, outW, \
        kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit, tapMajor); \
}
CE_DGRAD_KERNEL(conv2d_dgrad_ce128, 128, false, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64, 64, false, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_phase, 128, true, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_phase, 64, true, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_x3, 128, false, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_x3, 64, false, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_phase_x3, 128, true, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_phase_x3, 64, true, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_tf32, 128, false, 2)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_tf32, 64, false, 2)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_phase_tf32, 128, true, 2)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_phase_tf32, 64, true, 2)
