// gdn_step: one decode position of the gated delta rule (IGatedDeltaNetStepBackend). Warp per state row (r, i), with
// k' = keyScale k[r]: pred = d S[r,i,:] . k' (d = alpha[r] when decayFirst, the paper's rule; else 1), delta =
// beta[r] (v[r,i] - pred), S[r,i,:] = alpha[r] S[r,i,:] + delta k' (in place: the row is read and written by its own
// warp only), o[r,i] = S[r,i,:] . q[r].
extern "C" __global__ __launch_bounds__(256) void gdn_step(
    float* __restrict__ S, const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ alpha, const float* __restrict__ beta, float* __restrict__ o, int rows, int dv, int dk,
    float keyScale, int decayFirst)
{
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= rows * dv) return;
    int r = w / dv;
    float* row = S + (long long)w * dk;
    const float* kr = k + (long long)r * dk;
    const float* qr = q + (long long)r * dk;
    float pred = 0.0f;
    for (int j = lane; j < dk; j += 32) pred = fmaf(row[j], kr[j], pred);
    for (int off = 16; off > 0; off >>= 1) pred += __shfl_xor_sync(0xFFFFFFFF, pred, off);
    float a = alpha[r];
    pred *= keyScale * (decayFirst ? a : 1.0f);
    float delta = beta[r] * (v[w] - pred) * keyScale, acc = 0.0f;
    for (int j = lane; j < dk; j += 32)
    {
        float s = fmaf(a, row[j], delta * kr[j]);
        row[j] = s;
        acc = fmaf(s, qr[j], acc);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if (lane == 0) o[w] = acc;
}
