// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    public static class MetalAudioKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "audio_amplitude_to_db", "audio_mulaw_encoding", "audio_mulaw_decoding",
            "audio_compute_deltas", "audio_resample",
        };

        public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

kernel void audio_amplitude_to_db(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& length [[buffer(2)]],
    constant float& minAmp [[buffer(3)]],
    constant float& topDbFloor [[buffer(4)]],
    constant int& clipTopDb [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= length) return;
    float v = max(input[gid], minAmp);
    float db = 20.0 * log10(v);
    if (clipTopDb != 0 && db < topDbFloor) db = topDbFloor;
    output[gid] = db;
}

kernel void audio_mulaw_encoding(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& length [[buffer(2)]],
    constant int& quantizationChannels [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= length) return;
    if (quantizationChannels < 2) { output[gid] = 0.0; return; }
    float mu = float(quantizationChannels - 1);
    float logMu = log(1.0 + mu);
    float x = input[gid];
    if (x > 1.0) x = 1.0; else if (x < -1.0) x = -1.0;
    float sgn = float(x > 0.0) - float(x < 0.0);
    float y = sgn * log(1.0 + mu * fabs(x)) / logMu;
    float q = floor((y + 1.0) * 0.5 * mu + 0.5);
    if (q < 0.0) q = 0.0; else if (q > mu) q = mu;
    output[gid] = q;
}

kernel void audio_mulaw_decoding(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& length [[buffer(2)]],
    constant int& quantizationChannels [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= length) return;
    if (quantizationChannels < 2) { output[gid] = 0.0; return; }
    float mu = float(quantizationChannels - 1);
    float q = input[gid];
    float y = (q / mu) * 2.0 - 1.0;
    float sgn = float(y > 0.0) - float(y < 0.0);
    output[gid] = sgn * (pow(1.0 + mu, fabs(y)) - 1.0) / mu;
}

kernel void audio_compute_deltas(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& leading [[buffer(2)]],
    constant int& timeAxis [[buffer(3)]],
    constant int& winLength [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = leading * timeAxis;
    if ((int)gid >= total) return;
    int t = (int)gid % timeAxis;
    int row = (int)gid / timeAxis;
    int n = winLength / 2;
    if (n < 1) { output[gid] = 0.0; return; }
    float denom = 0.0;
    for (int i = 1; i <= n; i++) denom += 2.0 * i * i;
    float acc = 0.0;
    int base_ = row * timeAxis;
    for (int k = 1; k <= n; k++) {
        int left = t - k < 0 ? 0 : t - k;
        int right = t + k >= timeAxis ? timeAxis - 1 : t + k;
        acc += k * (input[base_ + right] - input[base_ + left]);
    }
    output[gid] = acc / denom;
}

kernel void audio_resample(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& leading [[buffer(2)]],
    constant int& inLen [[buffer(3)]],
    constant int& outLen [[buffer(4)]],
    constant int& up [[buffer(5)]],
    constant int& down [[buffer(6)]],
    constant int& halfWidth [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int total = leading * outLen;
    if ((int)gid >= total) return;
    if (halfWidth < 1 || up <= 0 || down <= 0) { output[gid] = 0.0; return; }
    int ot = (int)gid % outLen;
    int row = (int)gid / outLen;
    int sBase = row * inLen;
    float cutoff = 1.0 / float(up > down ? up : down);
    float srcIdx = float(ot) * down / up;
    int centre = int(floor(srcIdx));
    float acc = 0.0, wSum = 0.0;
    const float PI = 3.14159265358979323846;
    for (int k = -halfWidth; k <= halfWidth; k++) {
        int idx = centre + k;
        if (idx < 0 || idx >= inLen) continue;
        float tt = (idx - srcIdx) * cutoff;
        float sinc = fabs(tt) < 1e-12f ? 1.0 : sin(PI * tt) / (PI * tt);
        float hann = 0.5 - 0.5 * cos(2.0 * PI * float(k + halfWidth) / float(2 * halfWidth));
        float w = sinc * hann;
        acc += w * input[sBase + idx];
        wSum += w;
    }
    output[gid] = wSum > 0.0 ? acc / wSum : 0.0;
}
";
    }
}
