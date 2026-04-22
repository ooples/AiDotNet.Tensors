// Copyright (c) AiDotNet. All rights reserved.
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public static class OpenClAudioKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "audio_amplitude_to_db", "audio_mulaw_encoding", "audio_mulaw_decoding",
        "audio_compute_deltas", "audio_resample",
    };

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
    __global const float* input, __global float* output,
    const int length, const int quantizationChannels)
{
    int gid = get_global_id(0);
    if (gid >= length) return;
    float mu = (float)(quantizationChannels - 1);
    float q = input[gid];
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
    int ot = gid % outLen;
    int row = gid / outLen;
    int sBase = row * inLen;

    float cutoff = 1.0f / (float)(up > down ? up : down);
    float srcIdx = (float)ot * down / up;
    int centre = (int)floor(srcIdx);

    float acc = 0.0f, wSum = 0.0f;
    const float PI = 3.14159265358979323846f;
    for (int k = -halfWidth; k <= halfWidth; k++) {
        int idx = centre + k;
        if (idx < 0 || idx >= inLen) continue;
        float tt = (idx - srcIdx) * cutoff;
        float sinc = fabs(tt) < 1e-12f ? 1.0f : sin(PI * tt) / (PI * tt);
        float hann = 0.5f - 0.5f * cos(2.0f * PI * (k + halfWidth) / (2.0f * halfWidth));
        float w = sinc * hann;
        acc += w * input[sBase + idx];
        wSum += w;
    }
    output[gid] = wSum > 0.0f ? acc / wSum : 0.0f;
}
";
}
#endif
