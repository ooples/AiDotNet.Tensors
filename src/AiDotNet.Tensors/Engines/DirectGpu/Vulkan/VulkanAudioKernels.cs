namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public static class VulkanAudioKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    public static string AmplitudeToDB => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P { int length; float minAmp; float topDbFloor; int clipTopDb; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= length) return;
    float v = max(input_[gid], minAmp);
    float db = 20.0 * log(v) / log(10.0);
    if (clipTopDb != 0 && db < topDbFloor) db = topDbFloor;
    output_[gid] = db;
}
";

    public static string MuLawEncoding => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P { int length; int quantizationChannels; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= length) return;
    if (quantizationChannels < 2) { output_[gid] = 0.0; return; }
    float mu = float(quantizationChannels - 1);
    float logMu = log(1.0 + mu);
    float x = input_[gid];
    if (x > 1.0) x = 1.0; else if (x < -1.0) x = -1.0;
    float sgn = float(x > 0.0) - float(x < 0.0);
    float y = sgn * log(1.0 + mu * abs(x)) / logMu;
    float q = floor((y + 1.0) * 0.5 * mu + 0.5);
    if (q < 0.0) q = 0.0; else if (q > mu) q = mu;
    output_[gid] = q;
}
";

    public static string MuLawDecoding => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P { int length; int quantizationChannels; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= length) return;
    if (quantizationChannels < 2) { output_[gid] = 0.0; return; }
    float mu = float(quantizationChannels - 1);
    float q = input_[gid];
    float y = (q / mu) * 2.0 - 1.0;
    float sgn = float(y > 0.0) - float(y < 0.0);
    output_[gid] = sgn * (pow(1.0 + mu, abs(y)) - 1.0) / mu;
}
";

    public static string ComputeDeltas => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P { int leading; int timeAxis; int winLength; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = leading * timeAxis;
    if (gid >= total) return;
    int t = gid % timeAxis;
    int row = gid / timeAxis;
    int n = winLength / 2;
    if (n < 1) { output_[gid] = 0.0; return; }
    float denom = 0.0;
    for (int i = 1; i <= n; i++) denom += 2.0 * i * i;
    float acc = 0.0;
    int base_ = row * timeAxis;
    for (int k = 1; k <= n; k++) {
        int left = t - k < 0 ? 0 : t - k;
        int right = t + k >= timeAxis ? timeAxis - 1 : t + k;
        acc += k * (input_[base_ + right] - input_[base_ + left]);
    }
    output_[gid] = acc / denom;
}
";

    public static string Resample => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P {
    int leading; int inLen; int outLen; int up; int down; int halfWidth;
};
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = leading * outLen;
    if (gid >= total) return;
    if (halfWidth < 1 || up <= 0 || down <= 0) { output_[gid] = 0.0; return; }
    int ot = gid % outLen;
    int row = gid / outLen;
    int sBase = row * inLen;
    float cutoff = 1.0 / float(up > down ? up : down);
    float srcIdx = float(ot) * float(down) / float(up);
    int centre = int(floor(srcIdx));
    float acc = 0.0, wSum = 0.0;
    const float PI = 3.14159265358979323846;
    for (int k = -halfWidth; k <= halfWidth; k++) {
        int idx = centre + k;
        if (idx < 0 || idx >= inLen) continue;
        float tt = (float(idx) - srcIdx) * cutoff;
        float sinc = abs(tt) < 1e-12 ? 1.0 : sin(PI * tt) / (PI * tt);
        float hann = 0.5 - 0.5 * cos(2.0 * PI * float(k + halfWidth) / float(2 * halfWidth));
        float w = sinc * hann;
        acc += w * input_[sBase + idx];
        wSum += w;
    }
    output_[gid] = wSum > 0.0 ? acc / wSum : 0.0;
}
";
}
