
// ===== #1325: implicit-GEMM conv engine (cuBLAS-class SIMT tiling, strict fp32 FMA) =====
// One main loop for every conv GEMM: a BM x BN tile of C per 256-thread block, (BM/16) x (BN/16) accumulators per
// thread (8 x 8 at 128 x 128, 4 x 4 at 64 x 64), CE_BK = 8 deep K tiles. The operand loaders of each conv kind fill
// registers for the next K tile while the current one is multiplied out of the other of two shared buffers (one
// barrier per tile). A thread's accumulator rows are ty*4 + {0..3} + 64*g and its columns tx*4 + {0..3} + 64*g, so
// every shared read is a float4.
#define CE_BK 8

template <int BM, int BN, typename Load, typename Store, typename Advance>
__device__ __forceinline__ void ce_mainloop(float* smem, int tiles, Load load, Store store, Advance advance,
    float (&acc)[BM / 16][BN / 16])
{
    constexpr int TM = BM / 16, TN = BN / 16;
    float (*As)[CE_BK][BM + 4] = reinterpret_cast<float (*)[CE_BK][BM + 4]>(smem);
    float (*Bs)[CE_BK][BN + 4] = reinterpret_cast<float (*)[CE_BK][BN + 4]>(smem + 2 * CE_BK * (BM + 4));
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
    if (tiles <= 0) return;
    int buf = 0;
    load();
    store(As[0], Bs[0]);
    __syncthreads();
    for (int t = 0; t < tiles; t++)
    {
        bool more = t + 1 < tiles;
        if (more) { advance(); load(); }
        #pragma unroll
        for (int kk = 0; kk < CE_BK; kk++)
        {
            float a[TM], b[TN];
            #pragma unroll
            for (int g = 0; g < TM / 4; g++)
            {
                float4 v = *reinterpret_cast<const float4*>(&As[buf][kk][g * 64 + ty * 4]);
                a[g * 4] = v.x; a[g * 4 + 1] = v.y; a[g * 4 + 2] = v.z; a[g * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int g = 0; g < TN / 4; g++)
            {
                float4 v = *reinterpret_cast<const float4*>(&Bs[buf][kk][g * 64 + tx * 4]);
                b[g * 4] = v.x; b[g * 4 + 1] = v.y; b[g * 4 + 2] = v.z; b[g * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        if (more)
        {
            // The other buffer was last read in the previous iteration, before its closing barrier.
            store(As[buf ^ 1], Bs[buf ^ 1]);
            __syncthreads();
            buf ^= 1;
        }
    }
}

// #1335: the tensor-core mainloop. MAIN selects the inner product of every engine body: 0 = FP32 FMA (ce_mainloop),
// 1 = 3xTF32 (each fp32 operand split into TF32 hi + lo, three mma.sync m16n8k8 per product: lo*hi + hi*lo + hi*hi,
// fp32-level accuracy), 2 = plain TF32 (hi*hi only; explicit reduced-precision opt-in). The tensor-core variants need
// sm_80+; the host dispatches them only there, and below that the helpers compile to a no-op.
template <int MAIN> struct CePad { static constexpr int value = MAIN == 0 ? 4 : 8; };

// Output coordinates of accumulator acc[i][j] within the block tile, and how many consecutive columns a thread owns
// (4 for the FMA layout, 2 for the mma C fragment). Both layouts are separable: the row depends on i only, the column
// on j only, and columns jg .. jg + VEC - 1 (jg a multiple of VEC) are consecutive.
template <int BM, int BN, int MAIN>
struct CeMap
{
    static constexpr int VEC = MAIN == 0 ? 4 : 2;
    __device__ __forceinline__ static int row(int i)
    {
        if (MAIN == 0) return (i >> 2) * 64 + (threadIdx.x >> 4) * 4 + (i & 3);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        return (warp >> 2) * (BM / 2) + (i >> 1) * 16 + (lane >> 2) + 8 * (i & 1);
    }
    __device__ __forceinline__ static int col(int j)
    {
        if (MAIN == 0) return (j >> 2) * 64 + (threadIdx.x & 15) * 4 + (j & 3);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        return (warp & 3) * (BN / 4) + (j >> 1) * 8 + 2 * (lane & 3) + (j & 1);
    }
};

template <int VEC> __device__ __forceinline__ void ce_store_vec(float* p, const float* v);
template <> __device__ __forceinline__ void ce_store_vec<4>(float* p, const float* v)
{
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
template <> __device__ __forceinline__ void ce_store_vec<2>(float* p, const float* v)
{
    *reinterpret_cast<float2*>(p) = make_float2(v[0], v[1]);
}

__device__ __forceinline__ unsigned int ce_tf32(float x)
{
    unsigned int r;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
#else
    r = __float_as_uint(x) & 0xffffe000u;
#endif
    return r;
}

// d (4 floats, the C fragment) += A (m16 x k8, 4 TF32 regs) * B (k8 x n8, 2 TF32 regs).
__device__ __forceinline__ void ce_mma_tf32(float* d, const unsigned int* a, const unsigned int* b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// Same pipeline as ce_mainloop (register prefetch, two shared buffers, one barrier per K tile; CE_BK = 8 is one mma k
// step). The 8 warps tile the block 2 x 4; a warp owns (BM/2) x (BN/4) of C as MI x NI m16n8 tiles, its accumulators
// laid out as CeMap<BM, BN, 1> describes. Rows are padded by 8 so the fragment reads (lane = 4 g + t reading
// [t][base + g]) hit 32 distinct banks.
template <int BM, int BN, bool X3, typename Load, typename Store, typename Advance>
__device__ __forceinline__ void ce_mainloop_tc(float* smem, int tiles, Load load, Store store, Advance advance,
    float (&acc)[BM / 16][BN / 16])
{
    constexpr int TM = BM / 16, TN = BN / 16, MI = BM / 32, NI = BN / 32;
    float (*As)[CE_BK][BM + 8] = reinterpret_cast<float (*)[CE_BK][BM + 8]>(smem);
    float (*Bs)[CE_BK][BN + 8] = reinterpret_cast<float (*)[CE_BK][BN + 8]>(smem + 2 * CE_BK * (BM + 8));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = (warp >> 2) * (BM / 2), wn = (warp & 3) * (BN / 4);
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
    if (tiles <= 0) return;
    int buf = 0;
    load();
    store(As[0], Bs[0]);
    __syncthreads();
    for (int tile = 0; tile < tiles; tile++)
    {
        bool more = tile + 1 < tiles;
        if (more) { advance(); load(); }
        unsigned int ahi[MI][4], alo[MI][4];
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
        {
            int r = wm + mi * 16 + g;
            float a[4] = { As[buf][t][r], As[buf][t][r + 8], As[buf][t + 4][r], As[buf][t + 4][r + 8] };
            #pragma unroll
            for (int q = 0; q < 4; q++)
            {
                ahi[mi][q] = ce_tf32(a[q]);
                alo[mi][q] = X3 ? ce_tf32(a[q] - __uint_as_float(ahi[mi][q])) : 0u;
            }
        }
        #pragma unroll
        for (int ni = 0; ni < NI; ni++)
        {
            int c = wn + ni * 8 + g;
            float bv[2] = { Bs[buf][t][c], Bs[buf][t + 4][c] };
            unsigned int bhi[2], blo[2];
            #pragma unroll
            for (int q = 0; q < 2; q++)
            {
                bhi[q] = ce_tf32(bv[q]);
                blo[q] = X3 ? ce_tf32(bv[q] - __uint_as_float(bhi[q])) : 0u;
            }
            #pragma unroll
            for (int mi = 0; mi < MI; mi++)
            {
                float d[4] = { acc[mi * 2][ni * 2], acc[mi * 2][ni * 2 + 1], acc[mi * 2 + 1][ni * 2], acc[mi * 2 + 1][ni * 2 + 1] };
                if (X3)
                {
                    // Small terms first, then the dominant hi * hi.
                    ce_mma_tf32(d, alo[mi], bhi);
                    ce_mma_tf32(d, ahi[mi], blo);
                }
                ce_mma_tf32(d, ahi[mi], bhi);
                acc[mi * 2][ni * 2] = d[0]; acc[mi * 2][ni * 2 + 1] = d[1];
                acc[mi * 2 + 1][ni * 2] = d[2]; acc[mi * 2 + 1][ni * 2 + 1] = d[3];
            }
        }
        if (more)
        {
            store(As[buf ^ 1], Bs[buf ^ 1]);
            __syncthreads();
            buf ^= 1;
        }
    }
}

template <int BM, int BN, int MAIN, typename Load, typename Store, typename Advance>
__device__ __forceinline__ void ce_run_mainloop(float* smem, int tiles, Load load, Store store, Advance advance,
    float (&acc)[BM / 16][BN / 16])
{
    if constexpr (MAIN == 0) ce_mainloop<BM, BN>(smem, tiles, load, store, advance, acc);
    else ce_mainloop_tc<BM, BN, MAIN == 1>(smem, tiles, load, store, advance, acc);
}

// Forward: Y[Cout, B*oH*oW] = W[Cout, K] * X_gathered[K, B*oH*oW], K = Cin*kH*kW. TAPU (Cin % 8 == 0) orders K as
// (ci / 8, kh, kw, ci % 8), so a K tile shares one tap: each lane's input position and bounds check are computed once per tile and
// the tap advances by addition. Otherwise K is W's natural (ci, kh, kw) order with per-element index math. Split-K
// slices go to out + z * B*Cout*oH*oW and are summed in order by conv2d_wgrad_splitk_reduce.
// Fusion hooks (no-ops here): Pro(b, ci, ih * W + iw, x) transforms each in-bounds input element as it is loaded (padding
// stays 0, so it composes with an activation applied before the convolution); Epi(b, co, l, y) transforms each output
// value before it is stored (with one split; a split-K launch stores partials, so Epi must be the identity there).
struct CeIdentityPro { __device__ __forceinline__ float operator()(int, int, int, float x) const { return x; } };
struct CeIdentityEpi { __device__ __forceinline__ float operator()(int, int, int, float y) const { return y; } };

template <int BM, int BN, bool TAPU, typename Pro = CeIdentityPro, typename Epi = CeIdentityEpi, int MAIN = 0>
__device__ __forceinline__ void conv2d_fprop_ig_body(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit,
    Pro pro = Pro(), Epi epi = Epi())
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + CePad<MAIN>::value) + 2 * CE_BK * (BN + CePad<MAIN>::value)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256, BSTEP = 256 / BN;
    const int L = outH * outW, N = batch * L, khw = kH * kW, K = inC * khw, HW = H * W;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kBegin = blockIdx.z * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + CE_BK - 1) / CE_BK : 0;
    const int tid = threadIdx.x;
    const int ak = tid & 7, am = tid >> 3;       // A: k column ak, rows am + 32 r
    const int bn = tid % BN, bk = tid / BN;      // B: pixel column bn, k rows bk + BSTEP r
    const int n = n0 + bn;
    const bool nIn = n < N;
    int ih0 = 0, iw0 = 0, nb = 0;
    if (nIn) { nb = n / L; int l = n - nb * L; int oh = l / outW; ih0 = oh * sH - pH; iw0 = (l - oh * outW) * sW - pW; }
    const float* xBatch = input + (long long)nb * inC * HW;

    int kc = kBegin;
    int tap = 0, ci0 = 0, kh = 0, kw = 0;
    // TAPU K order: (ci group of CE_BK, tap, ci within the group). A K tile is one (group, tap), and the kH*kW tiles of
    // a group read the weights w[co][ci0 .. ci0+7][0 .. kH*kW-1], one contiguous run per output channel, back to back,
    // so that run stays in L1 across its taps. In (tap, ci) order every tap re-fetched it from L2 or DRAM 32 bytes per
    // lane (measured: the weight loads were 18-19 us of the 256@8 and 512@4 layers).
    if (TAPU)
    {
        int g = kBegin / (CE_BK * khw);
        tap = (kBegin - g * CE_BK * khw) / CE_BK; ci0 = g * CE_BK; kh = tap / kW; kw = tap - kh * kW;
    }

    float ra[RA], rb[RB];
    auto load = [&]()
    {
        #pragma unroll
        for (int r = 0; r < RA; r++)
        {
            int m = m0 + am + 32 * r;
            if (TAPU)
                ra[r] = m < outC ? kernel[(long long)m * K + (ci0 + ak) * khw + tap] : 0.0f;
            else
            {
                int k = kc + ak;
                ra[r] = (m < outC && k < kEnd) ? kernel[(long long)m * K + k] : 0.0f;
            }
        }
        if (TAPU)
        {
            int ih = ih0 + kh * dH, iw = iw0 + kw * dW;
            bool valid = nIn && ih >= 0 && ih < H && iw >= 0 && iw < W;
            int pix = valid ? ih * W + iw : 0;
            const float* src = xBatch + (long long)(ci0 + bk) * HW + pix;
            #pragma unroll
            for (int r = 0; r < RB; r++)
                rb[r] = valid ? pro(nb, ci0 + bk + BSTEP * r, pix, src[(long long)(BSTEP * r) * HW]) : 0.0f;
        }
        else
        {
            #pragma unroll
            for (int r = 0; r < RB; r++)
            {
                int k = kc + bk + BSTEP * r;
                float v = 0.0f;
                if (nIn && k < kEnd)
                {
                    int ci = k / khw, t = k - ci * khw;
                    int th = t / kW, tw = t - th * kW;
                    int ih = ih0 + th * dH, iw = iw0 + tw * dW;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = pro(nb, ci, ih * W + iw, xBatch[(long long)ci * HW + ih * W + iw]);
                }
                rb[r] = v;
            }
        }
    };
    // Concrete parameter types, not (auto as, auto bs): NVRTC 12.x (the 12.6 runtime AiDotNet.Native.CUDA ships) treats
    // a generic lambda as a host function in JIT mode and rejects the whole module, leaving the CUDA backend
    // unavailable. The tiles are the mainloop's shared buffers, rows padded by CePad<MAIN>.
    auto store = [&](float (*as)[BM + CePad<MAIN>::value], float (*bs)[BN + CePad<MAIN>::value])
    {
        #pragma unroll
        for (int r = 0; r < RA; r++) as[ak][am + 32 * r] = ra[r];
        #pragma unroll
        for (int r = 0; r < RB; r++) bs[bk + BSTEP * r][bn] = rb[r];
    };
    auto advance = [&]()
    {
        kc += CE_BK;
        if (TAPU) { tap++; if (++kw == kW) { kw = 0; kh++; } if (tap == khw) { tap = 0; kh = 0; ci0 += CE_BK; } }
    };

    float acc[TM][TN];
    ce_run_mainloop<BM, BN, MAIN>(smem, tiles, load, store, advance, acc);

    using Map = CeMap<BM, BN, MAIN>;
    constexpr int VEC = Map::VEC;
    float* slice = out + (long long)blockIdx.z * batch * outC * L;
    const bool vec = (L % VEC) == 0;   // then a column group is VEC consecutive pixels of one image, aligned
    #pragma unroll
    for (int jg = 0; jg < TN; jg += VEC)
    {
        int nn = n0 + Map::col(jg);
        if (nn >= N) continue;
        int b = nn / L, l = nn - b * L;
        #pragma unroll
        for (int i = 0; i < TM; i++)
        {
            int m = m0 + Map::row(i);
            if (m >= outC) continue;
            float v[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) v[j] = acc[i][jg + j];
            if (vec)
            {
                #pragma unroll
                for (int j = 0; j < VEC; j++) v[j] = epi(b, m, l + j, v[j]);
                ce_store_vec<VEC>(slice + ((long long)b * outC + m) * L + l, v);
            }
            else
            {
                #pragma unroll
                for (int j = 0; j < VEC; j++)
                {
                    int nj = nn + j;
                    if (nj >= N) break;
                    int bj = nj / L, lj = nj - bj * L;
                    slice[((long long)bj * outC + m) * L + lj] = epi(bj, m, lj, v[j]);
                }
            }
        }
    }
}

