// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public static class OpenClAudioKernels
{
    public static string[] GetKernelNames() =>
    [
        "audio_amplitude_to_db", "audio_mulaw_encoding", "audio_mulaw_decoding",
        "audio_compute_deltas", "audio_resample",
        .. OpenClInstantNgpKernels.GetKernelNames(),
    ];

    public static string GetSource() => @"
__kernel void audio_amplitude_to_db(
    __global const float* input, __global float* output,
    const int length, const float minAmp, const float topDbFloor, const int clipTopDb)
{
    int gid = get_global_id(0);
    if (gid >= length) return;
    float v = fmax(input[gid], minAmp);
    float db = 20.0f * log10(v);
    if (clipTopDb != 0 && db < topDbFloor) db = topDbFloor;
    output[gid] = db;
}

__kernel void audio_mulaw_encoding(
    __global const float* input, __global float* output,
    const int length, const int quantizationChannels)
{
    int gid = get_global_id(0);
    if (gid >= length) return;
    float mu = (float)(quantizationChannels - 1);
    float logMu = log1p(mu);
    float x = input[gid];
    if (x > 1.0f) x = 1.0f; else if (x < -1.0f) x = -1.0f;
    float sgn = (x > 0.0f) - (x < 0.0f);
    float y = sgn * log1p(mu * fabs(x)) / logMu;
    float q = floor((y + 1.0f) * 0.5f * mu + 0.5f);
    if (q < 0.0f) q = 0.0f; else if (q > mu) q = mu;
    output[gid] = q;
}

__kernel void audio_mulaw_decoding(
    __global const int* input, __global float* output,
    const int length, const int quantizationChannels)
{
    int gid = get_global_id(0);
    if (gid >= length) return;
    float mu = (float)(quantizationChannels - 1);
    float q = (float)input[gid];
    float y = (q / mu) * 2.0f - 1.0f;
    float sgn = (y > 0.0f) - (y < 0.0f);
    float x = sgn * (pow(1.0f + mu, fabs(y)) - 1.0f) / mu;
    output[gid] = x;
}

__kernel void audio_compute_deltas(
    __global const float* input, __global float* output,
    const int leading, const int timeAxis, const int winLength)
{
    int gid = get_global_id(0);
    int total = leading * timeAxis;
    if (gid >= total) return;
    int t = gid % timeAxis;
    int row = gid / timeAxis;
    int n = winLength / 2;
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

__kernel void audio_resample(
    __global const float* input, __global float* output,
    const int leading, const int inLen, const int outLen,
    const int up, const int down, const int halfWidth)
{
    int gid = get_global_id(0);
    int total = leading * outLen;
    if (gid >= total) return;
    if (halfWidth < 1 || up <= 0 || down <= 0) { output[gid] = 0.0f; return; }
    int ot = gid % outLen;
    int row = gid / outLen;
    int sBase = row * inLen;

    float cutoff = 1.0f / (float)(up > down ? up : down);
    // Exact source position: ot*down/up as an integer centre plus a fraction, computed in 32-bit integers as
    // ot = q*up + r (the host guarantees up*down < 2^31). A float division here rounds positions just below an
    // integer up on relaxed-math devices, moving the whole 513-tap window by one sample.
    int q = ot / up, r = ot - q * up;
    int rd = r * down;
    int centre = q * down + rd / up;
    float frac = (float)(rd - (rd / up) * up) / (float)up;

    float acc = 0.0f, wSum = 0.0f;
    const float PI = 3.14159265358979323846f;
    for (int k = -halfWidth; k <= halfWidth; k++) {
        int idx = centre + k;
        if (idx < 0 || idx >= inLen) continue;
        float tt = ((float)k - frac) * cutoff;
        // Near zero, sin(x)/x by its Taylor series: a fast or relaxed sin has ~1e-7 absolute error, which turns
        // sin(x)/x for x ~ 1e-8 (a tap almost exactly on the source position) into garbage. Error below 1e-14.
        float px = PI * tt;
        float sinc = fabs(px) < 1e-3f ? 1.0f - px * px * (1.0f / 6.0f) : sin(px) / px;
        float hann = 0.5f - 0.5f * cos(2.0f * PI * (k + halfWidth) / (2.0f * halfWidth));
        float w = sinc * hann;
        acc += w * input[sBase + idx];
        wSum += w;
    }
    output[gid] = wSum > 0.0f ? acc / wSum : 0.0f;
}
" + OpenClInstantNgpKernels.GetSource();
}
