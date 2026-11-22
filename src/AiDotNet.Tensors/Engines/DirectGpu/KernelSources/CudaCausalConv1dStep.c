// conv1d_step: one decode position of a causal depthwise convolution (ICausalConv1dStepBackend). Thread per (b, c):
// y = x w[c, 0] + sum_m h[m-1, b, c] w[c, m] (+ bias[c]) (then SiLU); then h[m] = h[m-1] (m = taps-1 .. 1), h[0] = x, in
// place (the column (b, c) belongs to this thread alone). Taps of the history up to 15 (kernel size 16).
extern "C" __global__ __launch_bounds__(256) void conv1d_step(
    float* __restrict__ h, const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
    float* __restrict__ y, int taps, int batch, int channels, int silu)
{
    // 64-bit: batch * channels can pass int range.
    long long plane = (long long)batch * channels;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= plane) return;
    int c = (int)(i % channels), K = taps + 1;
    float xv = x[i], acc = xv * w[(long long)c * K];
    float past[15];
    for (int m = 1; m <= taps; m++)
    {
        float hv = h[(long long)(m - 1) * plane + i];
        past[m - 1] = hv;
        acc += hv * w[(long long)c * K + m];
    }
    if (bias) acc += bias[c];
    if (silu) acc = acc / (1.0f + expf(-acc));
    y[i] = acc;
    for (int m = taps - 1; m >= 1; m--) h[(long long)m * plane + i] = past[m - 1];
    if (taps > 0) h[i] = xv;
}
