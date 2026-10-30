// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Generated row / column kernel variants for Vulkan (GLSL), served through the tuned-kernel registry (#1115).
using System.Globalization;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL instances of the registry's row-wise and column-wise variants (CUDA: CudaTunedRowKernels; OpenCL:
/// OpenClTunedRowKernels), one compute shader per (op, lane count). Vulkan's established kernels give each row ONE
/// invocation (softmax, softmax backward, the normalization phases) or each column one invocation looping over every
/// row (the affine gradients); these give a row <c>L</c> lanes (8, 16 or 32 of a 256-invocation group, or all 256) or a 32-column
/// tile <c>R</c> row lanes, and the registry measures which wins per shape class.
/// </summary>
/// <remarks>
/// Reductions go through shared memory with CUDA's shuffle-down tree order (lane i adds lane i + o); every invocation
/// reaches every barrier, inactive tail rows carrying neutral values. Fixed order, no atomics: deterministic.
/// </remarks>
internal static class VulkanTunedRowKernels
{
    internal const int GroupSize = 256;

    /// <summary>Lane counts each row-kernel family is instantiated for.</summary>
    internal static readonly int[] RowLanes = { 8, 16, 32, 256 };

    /// <summary>
    /// Whether a row of <paramref name="n"/> columns is worth offering to the <paramref name="lanes"/>-lane variant:
    /// a lane walks n / lanes columns, so a narrow lane count is not offered rows where each lane would still walk
    /// more than <see cref="MaxColumnsPerLane"/>. The widest count (a whole workgroup per row) has no such limit; it
    /// is the one for the widest rows.
    /// </summary>
    internal static bool RowLanesFit(int n, int lanes) => lanes == GroupSize || n <= MaxColumnsPerLane * lanes;

    /// <summary>Columns one lane may walk before a narrow lane count stops being offered for the row.</summary>
    internal const int MaxColumnsPerLane = 64;

    /// <summary>Row-lane counts the column-reduction family is instantiated for.</summary>
    internal static readonly int[] ColumnRowLanes = { 8, 32 };

    private const string Reductions = @"
shared float scratch[256];
float lane_sum(float v, int lid, int L)
{
    scratch[lid] = v;
    barrier();
    for (int o = L / 2; o > 0; o >>= 1)
    {
        if ((lid % L) < o) scratch[lid] += scratch[lid + o];
        barrier();
    }
    float r = scratch[lid - lid % L];
    barrier();
    return r;
}
float lane_max(float v, int lid, int L)
{
    scratch[lid] = v;
    barrier();
    for (int o = L / 2; o > 0; o >>= 1)
    {
        if ((lid % L) < o) scratch[lid] = max(scratch[lid], scratch[lid + o]);
        barrier();
    }
    float r = scratch[lid - lid % L];
    barrier();
    return r;
}
";

    private static string RowHeader(int lanes, string bindings, string push) => "#version 450\nlayout(local_size_x = 256) in;\n"
        + bindings + "layout(push_constant) uniform P { " + push + " } pc;\n" + Reductions
        + "const int L = " + lanes.ToString(CultureInfo.InvariantCulture) + ";\n";

    internal static string Softmax(int lanes) => RowHeader(lanes, @"
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) buffer Y { float y[]; };
", "int rows; int n;") + @"
void main()
{
    int t = int(gl_GlobalInvocationID.x), lid = int(gl_LocalInvocationID.x);
    int row = t / L, lane = t % L;
    bool live = row < pc.rows;
    int base = (live ? row : 0) * pc.n;
    float m = uintBitsToFloat(0xFF800000u);
    if (live) for (int f = lane; f < pc.n; f += L) m = max(m, x[base + f]);
    m = lane_max(m, lid, L);
    float s = 0.0;
    if (live) for (int f = lane; f < pc.n; f += L) { float e = exp(x[base + f] - m); y[base + f] = e; s += e; }
    s = lane_sum(s, lid, L);
    float inv = (s > 0.0) ? (1.0 / s) : 1.0;
    if (live) for (int f = lane; f < pc.n; f += L) y[base + f] *= inv;
}";

    internal static string SoftmaxBackward(int lanes) => RowHeader(lanes, @"
layout(std430, set = 0, binding = 0) readonly buffer Dy { float dy[]; };
layout(std430, set = 0, binding = 1) readonly buffer Yv { float yv[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Dx { float dx[]; };
", "int rows; int n;") + @"
void main()
{
    int t = int(gl_GlobalInvocationID.x), lid = int(gl_LocalInvocationID.x);
    int row = t / L, lane = t % L;
    bool live = row < pc.rows;
    int base = (live ? row : 0) * pc.n;
    float dot = 0.0;
    if (live) for (int f = lane; f < pc.n; f += L) dot += dy[base + f] * yv[base + f];
    dot = lane_sum(dot, lid, L);
    if (live) for (int f = lane; f < pc.n; f += L) dx[base + f] = yv[base + f] * (dy[base + f] - dot);
}";

    internal static string LayerNorm(int lanes) => RowHeader(lanes, @"
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Y { float y[]; };
layout(std430, set = 0, binding = 2) readonly buffer G { float gamma[]; };
layout(std430, set = 0, binding = 3) readonly buffer B { float beta[]; };
layout(std430, set = 0, binding = 4) writeonly buffer M { float saveMean[]; };
layout(std430, set = 0, binding = 5) writeonly buffer I { float saveInvVar[]; };
", "int rows; int n; float epsilon;") + @"
void main()
{
    int t = int(gl_GlobalInvocationID.x), lid = int(gl_LocalInvocationID.x);
    int row = t / L, lane = t % L;
    bool live = row < pc.rows;
    int base = (live ? row : 0) * pc.n;
    float s = 0.0;
    if (live) for (int i = lane; i < pc.n; i += L) s += x[base + i];
    float mean = lane_sum(s, lid, L) / float(pc.n);
    float v = 0.0;
    if (live) for (int i = lane; i < pc.n; i += L) { float d = x[base + i] - mean; v += d * d; }
    float invVar = inversesqrt(lane_sum(v, lid, L) / float(pc.n) + pc.epsilon);
    if (!live) return;
    if (lane == 0) { saveMean[row] = mean; saveInvVar[row] = invVar; }
    for (int i = lane; i < pc.n; i += L) y[base + i] = gamma[i] * ((x[base + i] - mean) * invVar) + beta[i];
}";

    internal static string LayerNormBackward(int lanes) => RowHeader(lanes, @"
layout(std430, set = 0, binding = 0) readonly buffer Dy { float dy[]; };
layout(std430, set = 0, binding = 1) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 2) readonly buffer G { float gamma[]; };
layout(std430, set = 0, binding = 3) readonly buffer M { float saveMean[]; };
layout(std430, set = 0, binding = 4) readonly buffer I { float saveInvVar[]; };
layout(std430, set = 0, binding = 5) writeonly buffer Dx { float dx[]; };
", "int rows; int n;") + @"
void main()
{
    int t = int(gl_GlobalInvocationID.x), lid = int(gl_LocalInvocationID.x);
    int row = t / L, lane = t % L;
    bool live = row < pc.rows;
    int r = live ? row : 0;
    int base = r * pc.n;
    float mean = saveMean[r], invVar = saveInvVar[r];
    float sumDy = 0.0, sumDyXmu = 0.0;
    if (live) for (int i = lane; i < pc.n; i += L) { float d = dy[base + i] * gamma[i]; sumDy += d; sumDyXmu += d * (x[base + i] - mean); }
    sumDy = lane_sum(sumDy, lid, L);
    sumDyXmu = lane_sum(sumDyXmu, lid, L);
    if (!live) return;
    for (int i = lane; i < pc.n; i += L)
    {
        float xmu = x[base + i] - mean;
        float dxhat = dy[base + i] * gamma[i];
        dx[base + i] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / float(pc.n));
    }
}";

    // A 32-column tile with R row lanes per workgroup (1-D groups of 32 * R: cx = lid % 32, ry = lid / 32).
    private static string ColumnGrad(int rowLanes, bool rms) => "#version 450\nlayout(local_size_x = "
        + (32 * rowLanes).ToString(CultureInfo.InvariantCulture) + ") in;\n"
        + (rms ? @"
layout(std430, set = 0, binding = 0) readonly buffer Dy { float dy[]; };
layout(std430, set = 0, binding = 1) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 2) readonly buffer S0 { float stat0[]; };
layout(std430, set = 0, binding = 3) writeonly buffer O0 { float out0[]; };
" : @"
layout(std430, set = 0, binding = 0) readonly buffer Dy { float dy[]; };
layout(std430, set = 0, binding = 1) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 2) readonly buffer S0 { float stat0[]; };
layout(std430, set = 0, binding = 3) readonly buffer S1 { float stat1[]; };
layout(std430, set = 0, binding = 4) writeonly buffer O0 { float out0[]; };
layout(std430, set = 0, binding = 5) writeonly buffer O1 { float out1[]; };
")
        + "layout(push_constant) uniform P { int rows; int n; } pc;\nconst int R = " + rowLanes.ToString(CultureInfo.InvariantCulture) + ";\n"
        + @"shared float s0[R * 33];
shared float s1[R * 33];
void main()
{
    int lid = int(gl_LocalInvocationID.x), cx = lid % 32, ry = lid / 32;
    int col = int(gl_WorkGroupID.x) * 32 + cx;
    float a0 = 0.0, a1 = 0.0;
    if (col < pc.n)
    {
        for (int b = ry; b < pc.rows; b += R)
        {
            int idx = b * pc.n + col;
            float g = dy[idx];
" + (rms ? "            a0 += g * x[idx] * (1.0 / stat0[b]);\n"
          : "            float normalized = (x[idx] - stat0[b]) * stat1[b];\n            a0 += g * normalized;\n            a1 += g;\n") + @"        }
    }
    s0[ry * 33 + cx] = a0; s1[ry * 33 + cx] = a1;
    barrier();
    for (int h = R / 2; h > 0; h >>= 1)
    {
        if (ry < h) { s0[ry * 33 + cx] += s0[(ry + h) * 33 + cx]; s1[ry * 33 + cx] += s1[(ry + h) * 33 + cx]; }
        barrier();
    }
    if (ry == 0 && col < pc.n)
    {
        out0[col] = s0[cx];
" + (rms ? string.Empty : "        out1[col] = s1[cx];\n") + @"    }
}";

    internal static string LayerNormGradParams(int rowLanes) => ColumnGrad(rowLanes, rms: false);

    internal static string RmsNormGradGamma(int rowLanes) => ColumnGrad(rowLanes, rms: true);
}