// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// The fused mean-squared-error loss and its gradient (<see cref="IFusedMseLossBackend"/>), and the same loss of a linear
/// head's output with every gradient of the head (<see cref="IFusedLinearMseLossBackend"/>).
/// </summary>
internal static class CudaMseLossKernels
{
    public static string GetSource() => @"
// One block of 1024 threads: each thread sums (p - y)^2 over a fixed stride of the elements, then the block reduces
// through warp shuffles and one shared-memory step (a fixed order: deterministic).
extern ""C"" __global__ __launch_bounds__(1024) void mse_loss_mean(
    const float* __restrict__ p, const float* __restrict__ y, float* __restrict__ loss, int n)
{
    __shared__ float warpSums[32];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = p[i] - y[i];
        sum = fmaf(d, d, sum);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < (blockDim.x >> 5) ? warpSums[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (threadIdx.x == 0) loss[0] = sum / (float)n;
    }
}

// dp = 2 (p - y) g / n with g the loss's gradient (gradLoss[0]); dy = -dp when requested.
extern ""C"" __global__ __launch_bounds__(256) void mse_loss_backward(
    const float* __restrict__ p, const float* __restrict__ y, const float* __restrict__ gradLoss,
    float* __restrict__ dp, float* __restrict__ dy, int n)
{
    float scale = 2.0f * gradLoss[0] / (float)n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float g = scale * (p[i] - y[i]);
        dp[i] = g;
        if (dy != 0) dy[i] = -g;
    }
}

// The linear head + loss kernels, one block of LM_THREADS threads with x, w, y and the bias staged in shared memory (one
// round of loads). Every phase maps lanes to independent outputs, so nothing is reduced across lanes:
//   predictions  lane = row m, warp = (4-output chunk c, part h of k): four sums over its quarter of k from x (row stride
//                K + 1, so a warp's 32 rows hit 32 banks) and w (rows padded to LM_WS floats, read as broadcast float4);
//                the four parts are combined per output, in a fixed order, with the bias and target.
//   dx           threads below LM_DX_THREADS: column k with w's row k in registers, looping over rows m (dp's row is a
//                broadcast float4 read).
//   dw, db       the other threads, at the same time: (k, c), a fixed-order sum over the rows of x[m, k] dp[m, 4c..4c+3];
//                column K is db.
// N <= 16. The parity LSTM's [32, 64] x [64, 10] head: 2.9 us forward, 4.2 backward in the captured step, against
// 1.0 us for an empty launch. Measured on the way: a warp per row reducing its sums across lanes was issue-bound on
// one SM (3.4 / 6.8 us); copy loops with a load before each store, and integer divisions by runtime sizes, each cost
// about a microsecond of dependent latency in kernels this short.
#define LM_THREADS 512
#define LM_KPARTS 4     // predictions: 16 warps = 4 output chunks x 4 parts of k
#define LM_DX_THREADS 256   // backward: dx on threads below, dw and db on the rest, concurrently
#define LM_NC 16
#define LM_WS 20   // padded w / partial / dp row: 16 outputs + 4, so a quarter-warp's float4 rows hit distinct banks

__device__ __forceinline__ int linear_mse_round4(int n) { return (n + 3) & ~3; }

// e = q d + r for 0 <= e < 2^24 and d > 0, from the float reciprocal and one exact correction: an integer division by a
// runtime divisor is a ~20-instruction dependent chain, and these kernels are short enough for that to show (each
// staging or output loop paid it per element).
__device__ __forceinline__ int linear_mse_divmod(int e, int d, float inv, int& r)
{
    int q = __float2int_rz(((float)e + 0.5f) * inv);
    r = e - q * d;
    if (r < 0) { q--; r += d; } else if (r >= d) { q++; r -= d; }
    return q;
}

struct LinearMseSmem
{
    float* xs;   // [M][K + 1]
    float* ws;   // [K][LM_WS], outputs past N zero
    float* ys;   // [M * N]
    float* pp;   // [LM_KPARTS][M][LM_WS] partial predictions per part of k
    float* dps;  // [M][LM_WS] backward: dp, outputs past N zero
    float* bs;   // [LM_NC] the bias, zero past N (or without one)
};

__device__ __forceinline__ LinearMseSmem linear_mse_layout(float* smem, int M, int K, int N)
{
    LinearMseSmem s;
    s.xs = smem;
    s.ws = s.xs + linear_mse_round4(M * (K + 1));
    s.ys = s.ws + K * LM_WS;
    s.pp = s.ys + linear_mse_round4(M * N);
    s.dps = s.pp + LM_KPARTS * M * LM_WS;
    s.bs = s.dps + M * LM_WS;
    return s;
}

// Stages x (padded rows), w (padded, zero past N), y and the bias; then the partial predictions of every (row, chunk,
// half). Every thread issues its loads of all four into registers before its first shared-memory store, so the staging
// costs one load latency: copy loops (store after each load, trip counts unknown to the compiler) took one latency per
// iteration, ~3,600 cycles on the parity head. Elements past one round (8 float4 of x, 4 of w and 2 of y per thread)
// take a plain loop.
#define LM_XV 8
#define LM_WV 4
#define LM_YV 2
__device__ __forceinline__ void linear_mse_predict(
    const LinearMseSmem& s, const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    const float* __restrict__ y, int M, int K, int N)
{
    const int KS = K + 1, tid = threadIdx.x;
    const float invK = __frcp_rn((float)K);
    const bool xVec = (K & 3) == 0 && (((size_t)x) & 15) == 0;
    const int xUnits = xVec ? (M * K) >> 2 : M * K, wUnits = K * LM_WS, yCount = M * N;
    float4 xr[LM_XV];
    float wr[LM_WV], yr[LM_YV];
    #pragma unroll
    for (int j = 0; j < LM_XV; j++) {
        int i = tid + j * LM_THREADS;
        if (i < xUnits) xr[j] = xVec ? __ldg(reinterpret_cast<const float4*>(x) + i) : make_float4(__ldg(x + i), 0.0f, 0.0f, 0.0f);
    }
    #pragma unroll
    for (int j = 0; j < LM_WV; j++) {
        int i = tid + j * LM_THREADS, k = i / LM_WS, n = i - k * LM_WS;
        if (i < wUnits) wr[j] = n < N ? __ldg(w + k * N + n) : 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < LM_YV; j++) { int i = tid + j * LM_THREADS; if (i < yCount) yr[j] = __ldg(y + i); }
    const float bv = tid < LM_NC && b != 0 && tid < N ? __ldg(b + tid) : 0.0f;
    #pragma unroll
    for (int j = 0; j < LM_XV; j++) {
        int i = tid + j * LM_THREADS;
        if (i < xUnits) {
            int k, m = linear_mse_divmod(xVec ? i << 2 : i, K, invK, k);
            float* d = s.xs + m * KS + k;
            if (xVec) { d[0] = xr[j].x; d[1] = xr[j].y; d[2] = xr[j].z; d[3] = xr[j].w; }
            else d[0] = xr[j].x;
        }
    }
    #pragma unroll
    for (int j = 0; j < LM_WV; j++) { int i = tid + j * LM_THREADS; if (i < wUnits) s.ws[i] = wr[j]; }
    #pragma unroll
    for (int j = 0; j < LM_YV; j++) { int i = tid + j * LM_THREADS; if (i < yCount) s.ys[i] = yr[j]; }
    if (tid < LM_NC) s.bs[tid] = bv;
    // Past one round.
    for (int i = tid + LM_XV * LM_THREADS; i < xUnits; i += LM_THREADS) {
        int k, m = linear_mse_divmod(xVec ? i << 2 : i, K, invK, k);
        float* d = s.xs + m * KS + k;
        if (xVec) {
            float4 v = __ldg(reinterpret_cast<const float4*>(x) + i);
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        } else {
            d[0] = __ldg(x + i);
        }
    }
    for (int i = tid + LM_WV * LM_THREADS; i < wUnits; i += LM_THREADS) {
        int k = i / LM_WS, n = i - k * LM_WS;
        s.ws[i] = n < N ? __ldg(w + k * N + n) : 0.0f;
    }
    for (int i = tid + LM_YV * LM_THREADS; i < yCount; i += LM_THREADS) s.ys[i] = __ldg(y + i);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int c = warp & 3, h = warp >> 2;            // 16 warps: 4 chunks x 4 parts of k
    const int part = (K + LM_KPARTS - 1) / LM_KPARTS, k0 = h * part, k1 = min(K, k0 + part);
    if (4 * c < N) {
        for (int m = lane; m < M; m += 32) {
            float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float* xr = s.xs + m * KS;
            #pragma unroll 8
            for (int k = k0; k < k1; k++) {
                float xv = xr[k];
                float4 wv = *reinterpret_cast<const float4*>(s.ws + k * LM_WS + 4 * c);
                a.x = fmaf(xv, wv.x, a.x); a.y = fmaf(xv, wv.y, a.y); a.z = fmaf(xv, wv.z, a.z); a.w = fmaf(xv, wv.w, a.w);
            }
            *reinterpret_cast<float4*>(s.pp + (h * M + m) * LM_WS + 4 * c) = a;
        }
    }
    __syncthreads();
}

// loss = mean((x w + b - y)^2): the head's predictions are never written. Fixed-order reduction.
extern ""C"" __global__ __launch_bounds__(LM_THREADS) void linear_mse_loss(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, const float* __restrict__ y,
    float* __restrict__ loss, int M, int K, int N)
{
    extern __shared__ __align__(16) float smem[];
    __shared__ float warpSums[LM_THREADS / 32];
    LinearMseSmem s = linear_mse_layout(smem, M, K, N);
    linear_mse_predict(s, x, w, b, y, M, K, N);
    const int MN = M * N;
    float sum = 0.0f;
    const float invN = __frcp_rn((float)N);
    for (int e = threadIdx.x; e < MN; e += LM_THREADS) {
        int n, m = linear_mse_divmod(e, N, invN, n);
        int r = m * LM_WS + n, P = M * LM_WS;
        float d = ((s.pp[r] + s.pp[P + r]) + (s.pp[2 * P + r] + s.pp[3 * P + r])) + s.bs[n] - s.ys[e];
        sum = fmaf(d, d, sum);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < LM_THREADS / 32; i++) total += warpSums[i];
        loss[0] = total / (float)MN;
    }
}

// The head's gradients from the loss's gradient g = gradLoss[0], with dp = 2 (p - y) g / (M N): dx = dp w^T [M, K],
// dw = x^T dp [K, N] and db = the column sums of dp [N] (when db != 0). Every element of each output is written.
extern ""C"" __global__ __launch_bounds__(LM_THREADS) void linear_mse_loss_backward(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, const float* __restrict__ y,
    const float* __restrict__ gradLoss, float* __restrict__ dx, float* __restrict__ dw, float* __restrict__ db,
    int M, int K, int N)
{
    extern __shared__ __align__(16) float smem[];
    LinearMseSmem s = linear_mse_layout(smem, M, K, N);
    const float scale = 2.0f * __ldg(gradLoss) / (float)(M * N);
    linear_mse_predict(s, x, w, b, y, M, K, N);
    for (int e = threadIdx.x; e < M * LM_WS; e += LM_THREADS) {
        int m = e / LM_WS, n = e - m * LM_WS;
        s.dps[e] = n < N
            ? scale * (((s.pp[e] + s.pp[M * LM_WS + e]) + (s.pp[2 * M * LM_WS + e] + s.pp[3 * M * LM_WS + e])) + s.bs[n]
                       - s.ys[m * N + n])
            : 0.0f;
    }
    __syncthreads();
    const int KS = K + 1;
    // dx on the first LM_DX_THREADS threads: column k's w row in registers; the threads past K * groups idle.
    const int groups = max(1, LM_DX_THREADS / K);
    const float invK = __frcp_rn((float)K);
    for (int i = threadIdx.x; i < K * groups && threadIdx.x < LM_DX_THREADS; i += LM_DX_THREADS) {
        int k, g = linear_mse_divmod(i, K, invK, k);
        const float4* wr = reinterpret_cast<const float4*>(s.ws + k * LM_WS);
        float4 w0 = wr[0], w1 = wr[1], w2 = wr[2], w3 = wr[3];
        #pragma unroll 4
        for (int m = g; m < M; m += groups) {
            const float4* dr = reinterpret_cast<const float4*>(s.dps + m * LM_WS);
            float4 d0 = dr[0], d1 = dr[1], d2 = dr[2], d3 = dr[3];
            float a0 = d0.x * w0.x, a1 = d1.x * w1.x;
            a0 = fmaf(d0.y, w0.y, a0); a0 = fmaf(d0.z, w0.z, a0); a0 = fmaf(d0.w, w0.w, a0);
            a1 = fmaf(d1.y, w1.y, a1); a1 = fmaf(d1.z, w1.z, a1); a1 = fmaf(d1.w, w1.w, a1);
            a0 = fmaf(d2.x, w2.x, a0); a0 = fmaf(d2.y, w2.y, a0); a0 = fmaf(d2.z, w2.z, a0); a0 = fmaf(d2.w, w2.w, a0);
            a1 = fmaf(d3.x, w3.x, a1); a1 = fmaf(d3.y, w3.y, a1); a1 = fmaf(d3.z, w3.z, a1); a1 = fmaf(d3.w, w3.w, a1);
            dx[(size_t)m * K + k] = a0 + a1;
        }
    }
    // dw on the other threads: (k, 4-output chunk c), rows summed in order; column k == K is db (x taken as ones).
    const int chunks = (N + 3) >> 2, columns = db != 0 ? K + 1 : K;
    const float invColumns = __frcp_rn((float)columns);
    for (int i = (int)threadIdx.x - LM_DX_THREADS; i >= 0 && i < columns * chunks; i += LM_THREADS - LM_DX_THREADS) {
        int k, c = linear_mse_divmod(i, columns, invColumns, k);
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll 8
        for (int m = 0; m < M; m++) {
            float xv = k < K ? s.xs[m * KS + k] : 1.0f;
            float4 d = *reinterpret_cast<const float4*>(s.dps + m * LM_WS + 4 * c);
            a.x = fmaf(xv, d.x, a.x); a.y = fmaf(xv, d.y, a.y); a.z = fmaf(xv, d.z, a.z); a.w = fmaf(xv, d.w, a.w);
        }
        float* out = k < K ? dw + (size_t)k * N + 4 * c : db + 4 * c;
        int left = N - 4 * c;
        out[0] = a.x;
        if (left > 1) out[1] = a.y;
        if (left > 2) out[2] = a.z;
        if (left > 3) out[3] = a.w;
    }
}
";

    public static string[] GetKernelNames() =>
        new[] { "mse_loss_mean", "mse_loss_backward", "linear_mse_loss", "linear_mse_loss_backward" };
}
