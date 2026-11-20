// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static partial class VulkanBatchExecutionKernels
{
    internal static class WeightNormShaders
    {
        internal static readonly string Forward = Shader(new[] { "r:v", "r:g", "w:w" },
            "int rows; int columns; float epsilon;", @"
shared float normPart[256];
void main() {
    int row = int(gl_WorkGroupID.x); int lane = int(gl_LocalInvocationID.x);
    if (row >= rows) return;
    float sum = 0.0;
    for (int j = lane; j < columns; j += 256) { float x = v[row * columns + j]; sum += x * x; }
    normPart[lane] = sum; barrier();
    for (int s = 128; s > 0; s /= 2) { if (lane < s) normPart[lane] += normPart[lane + s]; barrier(); }
    float scale = g[row] * (1.0 / sqrt(normPart[0] + epsilon));
    for (int j = lane; j < columns; j += 256) w[row * columns + j] = v[row * columns + j] * scale;
}
");
        internal static readonly string Stats = Shader(new[] { "r:h", "r:v", "w:stats" },
            "int rows; int columns; float epsilon;", @"
shared float normPart[256]; shared float dotPart[256];
void main() {
    int row = int(gl_WorkGroupID.x); int lane = int(gl_LocalInvocationID.x);
    if (row >= rows) return;
    float sum = 0.0; float dot = 0.0;
    for (int j = lane; j < columns; j += 256) {
        float x = v[row * columns + j]; sum += x * x; dot += h[row * columns + j] * x;
    }
    normPart[lane] = sum; dotPart[lane] = dot; barrier();
    for (int s = 128; s > 0; s /= 2) {
        if (lane < s) { normPart[lane] += normPart[lane + s]; dotPart[lane] += dotPart[lane + s]; }
        barrier();
    }
    if (lane == 0) { stats[2 * row] = 1.0 / sqrt(normPart[0] + epsilon); stats[2 * row + 1] = dotPart[0]; }
}
");
        internal static readonly string DirectionGrad = Shader(new[] { "r:h", "r:v", "r:g", "r:stats", "rw:dv" },
            "int rows; int columns; int accumulate;", @"
void main() {
    int i = bx_gid(); if (i >= rows * columns) return;
    int row = i / columns; float inv = stats[2 * row];
    float value = (g[row] * inv) * (h[i] - v[i] * (stats[2 * row + 1] * inv * inv));
    dv[i] = accumulate != 0 ? dv[i] + value : value;
}
");
        internal static readonly string GainGrad = Shader(new[] { "r:stats", "rw:dg" },
            "int rows; int accumulate;", @"
void main() {
    int row = bx_gid(); if (row >= rows) return;
    float value = stats[2 * row + 1] * stats[2 * row];
    dg[row] = accumulate != 0 ? dg[row] + value : value;
}
");
    }
}

public sealed partial class VulkanBackend
{
    public void WeightNorm(IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer output, int rows, int columns, float epsilon)
        => GlslNaryOp(VulkanBatchExecutionKernels.WeightNormShaders.Forward, new[] { direction, gain, output },
            checked(rows * 256), new[] { Bits(rows), Bits(columns), FloatBits(epsilon) });
    public void WeightNormStats(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer stats, int rows, int columns, float epsilon)
        => GlslNaryOp(VulkanBatchExecutionKernels.WeightNormShaders.Stats, new[] { gradOutput, direction, stats },
            checked(rows * 256), new[] { Bits(rows), Bits(columns), FloatBits(epsilon) });
    public void WeightNormDirectionGradient(IGpuBuffer gradOutput, IGpuBuffer direction, IGpuBuffer gain, IGpuBuffer stats,
        IGpuBuffer gradDirection, int rows, int columns, bool accumulate)
        => GlslNaryOp(VulkanBatchExecutionKernels.WeightNormShaders.DirectionGrad,
            new[] { gradOutput, direction, gain, stats, gradDirection }, checked(rows * columns),
            new[] { Bits(rows), Bits(columns), Bits(accumulate ? 1 : 0) });
    public void WeightNormGainGradient(IGpuBuffer stats, IGpuBuffer gradGain, int rows, bool accumulate)
        => GlslNaryOp(VulkanBatchExecutionKernels.WeightNormShaders.GainGrad, new[] { stats, gradGain }, rows,
            new[] { Bits(rows), Bits(accumulate ? 1 : 0) });
}

