// Copyright (c) AiDotNet. All rights reserved.
// CUDA audio primitive kernels — Issue #217 tail. Spectrogram /
// PitchShift / TimeStretch compose existing GPU STFT and stay in managed
// code; this file holds only the element-wise and small-window kernels
// that benefit from direct GPU execution.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    public static class CudaAudioKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "audio_amplitude_to_db",
            "audio_mulaw_encoding",
            "audio_mulaw_decoding",
            "audio_compute_deltas",
            "audio_resample",
        };

        public static string GetSource() => @"
#include <math.h>

extern ""C"" __global__ __launch_bounds__(256) void audio_amplitude_to_db(
    const float* __restrict__ input, float* __restrict__ output,
    int length, float minAmp, float topDbFloor, int clipTopDb)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= length) return;
    float v = fmaxf(input[gid], minAmp);
    float db = 20.0f * log10f(v);
    // topDbFloor is already peak-topDb for the batch; the launcher precomputes
    // it so we avoid a reduction here. If clipTopDb == 0, topDbFloor is ignored.
    if (clipTopDb != 0 && db < topDbFloor) db = topDbFloor;
    output[gid] = db;
}

extern ""C"" __global__ __launch_bounds__(256) void audio_mulaw_encoding(
    const float* __restrict__ input, float* __restrict__ output,
    int length, int quantizationChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= length) return;
    float mu = (float)(quantizationChannels - 1);
    float logMu = log1pf(mu);
    float x = input[gid];
    if (x > 1.0f) x = 1.0f;
    else if (x < -1.0f) x = -1.0f;
    float y = ((x > 0.0f) - (x < 0.0f)) * log1pf(mu * fabsf(x)) / logMu;
    float q = floorf((y + 1.0f) * 0.5f * mu + 0.5f);
    if (q < 0.0f) q = 0.0f;
    else if (q > mu) q = mu;
    output[gid] = q;
}

extern ""C"" __global__ __launch_bounds__(256) void audio_mulaw_decoding(
    const float* __restrict__ input, float* __restrict__ output,
    int length, int quantizationChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= length) return;
    float mu = (float)(quantizationChannels - 1);
    float q = input[gid];
    float y = (q / mu) * 2.0f - 1.0f;
    float x = ((y > 0.0f) - (y < 0.0f)) * (powf(1.0f + mu, fabsf(y)) - 1.0f) / mu;
    output[gid] = x;
}

// One thread per output (row, t). Savitzky-Golay: out[t] = Σ_{k=1..n} k · (in[t+k] − in[t−k]) / (2·Σk²).
extern ""C"" __global__ __launch_bounds__(256) void audio_compute_deltas(
    const float* __restrict__ input, float* __restrict__ output,
    int leading, int timeAxis, int winLength)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = leading * timeAxis;
    if (gid >= total) return;
    int t = gid % timeAxis;
    int row = gid / timeAxis;

    int n = winLength / 2;
    // Defence-in-depth: winLength ≤ 1 → n = 0 → denom = 0 → output is
    // NaN. The managed layer already rejects this, but kernels should
    // fail closed if someone calls the raw interface directly.
    if (n < 1) { output[gid] = 0.0f; return; }
    float denom = 0.0f;
    for (int i = 1; i <= n; i++) denom += 2.0f * i * i;
    float acc = 0.0f;
    int base_ = row * timeAxis;
    for (int k = 1; k <= n; k++) {
        int left = t - k < 0 ? 0 : t - k;
        int right = t + k >= timeAxis ? timeAxis - 1 : t + k;
        acc += k * (input[base_ + right] - input[base_ + left]);
    }
    output[gid] = acc / denom;
}

// Polyphase Hann-windowed sinc. srcIdx = ot · down / up; sum over
// [-halfWidth, halfWidth] taps.
extern ""C"" __global__ __launch_bounds__(256) void audio_resample(
    const float* __restrict__ input, float* __restrict__ output,
    int leading, int inLen, int outLen,
    int up, int down, int halfWidth)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = leading * outLen;
    if (gid >= total) return;
    if (halfWidth < 1 || up <= 0 || down <= 0) { output[gid] = 0.0f; return; }
    int ot = gid % outLen;
    int row = gid / outLen;
    int sBase = row * inLen;

    float cutoff = 1.0f / (float)(up > down ? up : down);
    float srcIdx = (float)ot * down / up;
    int centre = (int)floorf(srcIdx);

    float acc = 0.0f, wSum = 0.0f;
    const float PI = 3.14159265358979323846f;
    for (int k = -halfWidth; k <= halfWidth; k++) {
        int idx = centre + k;
        if (idx < 0 || idx >= inLen) continue;
        float t = (idx - srcIdx) * cutoff;
        float sinc = (fabsf(t) < 1e-12f) ? 1.0f : __sinf(PI * t) / (PI * t);
        float hann = 0.5f - 0.5f * __cosf(2.0f * PI * (k + halfWidth) / (2.0f * halfWidth));
        float w = sinc * hann;
        acc += w * input[sBase + idx];
        wSum += w;
    }
    output[gid] = wSum > 0.0f ? acc / wSum : 0.0f;
}
";
    }
}
