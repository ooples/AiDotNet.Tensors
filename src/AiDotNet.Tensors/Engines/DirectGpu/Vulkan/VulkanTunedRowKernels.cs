// Copyright (c) AiDotNet. All rights reserved.
// Generated row/column/convolution kernel variants served through the tuned-kernel registry.

using System.Collections.Generic;
using System.Globalization;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Generates the registry's Vulkan kernel variants from ONE GLSL template per op, instantiated over a lane-count
/// parameter. The established kernels map work to invocations in one fixed way whatever the shape: a single
/// invocation per row walking every column (softmax, softmax backward, LayerNorm forward and input gradient), a
/// single invocation per COLUMN walking every row (the affine-parameter gradients), or a single invocation per output
/// element (convolution). These variants instead give each row <c>L</c> lanes that reduce through workgroup shared
/// memory, give each workgroup a tile of columns with <c>R</c> row lanes, or compute <c>T</c> output channels of one
/// output pixel per invocation so the input window is loaded once; the registry measures which wins per shape class.
/// </summary>
/// <remarks>
/// <para>Every variant is deterministic: reductions run in a fixed order with no atomics.</para>
/// <para>A workgroup is 256 invocations in every variant, the size the rest of the backend's kernels use, and holds at
/// most two 256-float shared arrays.</para>
/// <para>Rows past the end keep executing every <c>barrier()</c> with neutral values: a barrier must be reached by the
/// whole workgroup.</para>
/// </remarks>
internal static class VulkanTunedRowKernels
{
    /// <summary>Invocations per workgroup of every generated variant.</summary>
    internal const int WorkgroupSize = 256;

    /// <summary>Lane counts each row-kernel family is instantiated for.</summary>
    internal static readonly int[] RowLanes = { 8, 32, 256 };

    /// <summary>Row-lane counts the column-reduction family is instantiated for (columns per tile = 256 / R).</summary>
    internal static readonly int[] ColumnRowLanes = { 8, 32 };

    /// <summary>Output channels per invocation the convolution family is instantiated for.</summary>
    internal static readonly int[] ConvChannelTiles = { 4, 8 };

    /// <summary>
    /// Whether an L-lane row variant suits rows of <paramref name="n"/> columns: at least a quarter of the lanes
    /// have work, and (below a whole workgroup per row) no lane walks more than 64 columns.
    /// </summary>
    internal static bool RowLanesFit(int lanes, int rows, int n) =>
        n >= lanes / 4 && (lanes == WorkgroupSize || n <= 64 * lanes) && (long)rows * lanes <= int.MaxValue;

    private static string S(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static string RowHeader(int lanes) => $@"#version 450
layout(local_size_x = {S(WorkgroupSize)}) in;
const uint L = {S(lanes)}u;
";

    // The preamble every row variant shares: which row and lane this invocation is. An inactive invocation (a row
    // past the end) reads row 0 so its loads stay in bounds, and contributes neutral values.
    private const string RowPreamble = @"
    uint t = gl_LocalInvocationID.x;
    uint lane = t % L;
    uint row = gl_GlobalInvocationID.x / L;
    bool live = row < rows;
    uint offset = (live ? row : 0u) * cols;
";

    // Folds the L per-lane partials of every row in `array` and leaves the row's total in `target`. The trailing
    // barrier lets the array be reused by the next reduction.
    private static string Fold(string array, string target, string combine) => $@"
    barrier();
    for (uint o = L / 2u; o > 0u; o >>= 1u) {{
        if (lane < o) {array}[t] = {string.Format(CultureInfo.InvariantCulture, combine, array + "[t]", array + "[t + o]")};
        barrier();
    }}
    {target} = {array}[t - lane];
    barrier();
";

    private const string Sum = "{0} + {1}";
    private const string Max = "max({0}, {1})";

    /// <summary>Row softmax, L lanes per row: max, sum of exp, exp / sum, as the established kernel.</summary>
    internal static string Softmax(int lanes) => RowHeader(lanes) + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint rows; uint cols; };
shared float s0[" + S(WorkgroupSize) + @"];
void main() {" + RowPreamble + @"
    float m = -3.402823466e+38;
    if (live) for (uint col = lane; col < cols; col += L) m = max(m, a[offset + col]);
    s0[t] = m;
    float rowMax;" + Fold("s0", "rowMax", Max) + @"
    float e = 0.0;
    if (live) for (uint col = lane; col < cols; col += L) e += exp(a[offset + col] - rowMax);
    s0[t] = e;
    float sum;" + Fold("s0", "sum", Sum) + @"
    if (live) for (uint col = lane; col < cols; col += L) b[offset + col] = exp(a[offset + col] - rowMax) / sum;
}";

    /// <summary>Row softmax backward, L lanes per row: dx = y * (dy - sum(dy * y)).</summary>
    internal static string SoftmaxBackward(int lanes) => RowHeader(lanes) + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) writeonly buffer C { float c[]; };
layout(push_constant) uniform Params { uint rows; uint cols; };
shared float s0[" + S(WorkgroupSize) + @"];
void main() {" + RowPreamble + @"
    float d = 0.0;
    if (live) for (uint col = lane; col < cols; col += L) d += a[offset + col] * bdata[offset + col];
    s0[t] = d;
    float rowDot;" + Fold("s0", "rowDot", Sum) + @"
    if (live) for (uint col = lane; col < cols; col += L) {
        uint idx = offset + col;
        c[idx] = bdata[idx] * (a[idx] - rowDot);
    }
}";

    /// <summary>LayerNorm forward, L lanes per row: two-pass mean / variance, as the established kernel.</summary>
    internal static string LayerNormForward(int lanes) => RowHeader(lanes) + @"
layout(set = 0, binding = 0) readonly buffer Input { float inputData[]; };
layout(set = 0, binding = 1) writeonly buffer Output { float outputData[]; };
layout(set = 0, binding = 2) readonly buffer Gamma { float gammaData[]; };
layout(set = 0, binding = 3) readonly buffer Beta { float betaData[]; };
layout(set = 0, binding = 4) writeonly buffer SavedMean { float savedMeanData[]; };
layout(set = 0, binding = 5) writeonly buffer SavedScale { float savedScaleData[]; };
layout(push_constant) uniform Params { uint rows; uint cols; float epsilon; };
shared float s0[" + S(WorkgroupSize) + @"];
void main() {" + RowPreamble + @"
    float p = 0.0;
    if (live) for (uint i = lane; i < cols; i += L) p += inputData[offset + i];
    s0[t] = p;
    float mean;" + Fold("s0", "mean", Sum) + @"
    mean /= float(cols);
    float v = 0.0;
    if (live) for (uint i = lane; i < cols; i += L) { float x = inputData[offset + i] - mean; v += x * x; }
    s0[t] = v;
    float variance;" + Fold("s0", "variance", Sum) + @"
    variance /= float(cols);
    float inv = 1.0 / sqrt(variance + epsilon);
    if (!live) return;
    if (lane == 0u) { savedMeanData[row] = mean; savedScaleData[row] = inv; }
    for (uint i = lane; i < cols; i += L)
        outputData[offset + i] = gammaData[i] * (inputData[offset + i] - mean) * inv + betaData[i];
}";

    /// <summary>LayerNorm backward (input gradient only), L lanes per row, same formula as the established kernel.</summary>
    internal static string LayerNormBackward(int lanes) => RowHeader(lanes) + @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float go[]; };
layout(set = 0, binding = 1) readonly buffer Input { float inputData[]; };
layout(set = 0, binding = 2) readonly buffer Gamma { float gammaData[]; };
layout(set = 0, binding = 3) readonly buffer SavedMean { float savedMeanData[]; };
layout(set = 0, binding = 4) readonly buffer SavedScale { float savedScaleData[]; };
layout(set = 0, binding = 5) writeonly buffer GradInput { float gi[]; };
layout(push_constant) uniform Params { uint rows; uint cols; };
shared float s0[" + S(WorkgroupSize) + @"];
shared float s1[" + S(WorkgroupSize) + @"];
void main() {" + RowPreamble + @"
    uint r = live ? row : 0u;
    float mean = savedMeanData[r], inv = savedScaleData[r];
    float pg = 0.0, pgx = 0.0;
    if (live) for (uint i = lane; i < cols; i += L) {
        float xhat = (inputData[offset + i] - mean) * inv;
        float g = go[offset + i] * gammaData[i];
        pg += g;
        pgx += g * xhat;
    }
    s0[t] = pg;
    s1[t] = pgx;
    barrier();
    for (uint o = L / 2u; o > 0u; o >>= 1u) {
        if (lane < o) { s0[t] = s0[t] + s0[t + o]; s1[t] = s1[t] + s1[t + o]; }
        barrier();
    }
    float sumGrad = s0[t - lane], sumGradX = s1[t - lane];
    if (!live) return;
    for (uint i = lane; i < cols; i += L) {
        float xhat = (inputData[offset + i] - mean) * inv;
        gi[offset + i] = inv / float(cols) * (float(cols) * go[offset + i] * gammaData[i] - sumGrad - xhat * sumGradX);
    }
}";

    // Column reduction over rows for the affine-parameter gradients. A workgroup is C columns x R row lanes
    // (C * R = 256); each row lane walks rows lane, lane + R, ... in order, then the R partials are folded in a
    // fixed tree. No atomics.
    private static string ColumnHeader(int rowLanes) => $@"#version 450
layout(local_size_x = {S(WorkgroupSize)}) in;
const uint R = {S(rowLanes)}u;
const uint C = {S(WorkgroupSize / rowLanes)}u;
";

    private const string ColumnPreamble = @"
    uint t = gl_LocalInvocationID.x;
    uint cx = t % C, ry = t / C;
    uint col = gl_WorkGroupID.x * C + cx;
    bool live = col < cols;
";

    /// <summary>LayerNorm gradGamma / gradBeta: a column-tile reduction with R row lanes.</summary>
    internal static string LayerNormGradParameters(int rowLanes) => ColumnHeader(rowLanes) + @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float go[]; };
layout(set = 0, binding = 1) readonly buffer Input { float inputData[]; };
layout(set = 0, binding = 2) readonly buffer SavedMean { float savedMeanData[]; };
layout(set = 0, binding = 3) readonly buffer SavedScale { float savedScaleData[]; };
layout(set = 0, binding = 4) writeonly buffer GradGamma { float gg[]; };
layout(set = 0, binding = 5) writeonly buffer GradBeta { float gb[]; };
layout(push_constant) uniform Params { uint rows; uint cols; };
shared float s0[" + S(WorkgroupSize) + @"];
shared float s1[" + S(WorkgroupSize) + @"];
void main() {" + ColumnPreamble + @"
    float a0 = 0.0, a1 = 0.0;
    if (live) for (uint b0 = ry; b0 < rows; b0 += R) {
        uint idx = b0 * cols + col;
        a0 += go[idx] * (inputData[idx] - savedMeanData[b0]) * savedScaleData[b0];
        a1 += go[idx];
    }
    s0[t] = a0;
    s1[t] = a1;
    barrier();
    for (uint h = R / 2u; h > 0u; h >>= 1u) {
        if (ry < h) { s0[t] = s0[t] + s0[t + h * C]; s1[t] = s1[t] + s1[t + h * C]; }
        barrier();
    }
    if (ry == 0u && live) { gg[col] = s0[t]; gb[col] = s1[t]; }
}";

    /// <summary>RMSNorm gradGamma: a column-tile reduction with R row lanes.</summary>
    internal static string RmsNormGradGamma(int rowLanes) => ColumnHeader(rowLanes) + @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float go[]; };
layout(set = 0, binding = 1) readonly buffer Input { float inputData[]; };
layout(set = 0, binding = 2) readonly buffer SavedScale { float savedScaleData[]; };
layout(set = 0, binding = 3) writeonly buffer GradGamma { float gg[]; };
layout(push_constant) uniform Params { uint rows; uint cols; };
shared float s0[" + S(WorkgroupSize) + @"];
void main() {" + ColumnPreamble + @"
    float a0 = 0.0;
    if (live) for (uint b0 = ry; b0 < rows; b0 += R) {
        uint idx = b0 * cols + col;
        a0 += go[idx] * inputData[idx] / savedScaleData[b0];
    }
    s0[t] = a0;
    barrier();
    for (uint h = R / 2u; h > 0u; h >>= 1u) {
        if (ry < h) s0[t] = s0[t] + s0[t + h * C];
        barrier();
    }
    if (ry == 0u && live) gg[col] = s0[t];
}";

    /// <summary>
    /// Convolution forward computing T consecutive output channels of one output pixel per invocation: each input
    /// tap is loaded and bounds-checked once and feeds T accumulators. Every output keeps the established kernel's
    /// (ic, kh, kw) summation order.
    /// </summary>
    internal static string Conv2DChannelTile(int tile) => $@"#version 450
layout(local_size_x = {S(WorkgroupSize)}) in;
const int T = {S(tile)};
" + @"
layout(set=0, binding=0) readonly buffer B0 { float inp[]; };
layout(set=0, binding=1) readonly buffer B1 { float wgt[]; };
layout(set=0, binding=2) writeonly buffer B2 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int tiles = (outChannels + T - 1) / T;
    int total = batch * tiles * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int q = idx / outWidth;
    int oh = q % outHeight; q = q / outHeight;
    int oc0 = (q % tiles) * T; int b = q / tiles;
    int count = min(T, outChannels - oc0);
    float acc[T];
    for (int j = 0; j < T; j++) acc[j] = 0.0;
    int taps = inChannels * kernelH * kernelW;
    for (int ic = 0; ic < inChannels; ic++)
        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH - padH + kh * dilationH;
            if (ih < 0 || ih >= inHeight) continue;
            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW - padW + kw * dilationW;
                if (iw < 0 || iw >= inWidth) continue;
                float x = inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                int w0 = ((oc0 * inChannels + ic) * kernelH + kh) * kernelW + kw;
                for (int j = 0; j < count; j++) acc[j] += x * wgt[w0 + j * taps];
            }
        }
    for (int j = 0; j < count; j++)
        outp[((b * outChannels + oc0 + j) * outHeight + oh) * outWidth + ow] = acc[j];
}";

    /// <summary>Every generated variant with its storage-buffer count and push-constant size in bytes.</summary>
    internal static IReadOnlyList<(string Name, string Source, int Bindings, uint PushSize)> All()
    {
        var list = new List<(string, string, int, uint)>();
        foreach (int l in RowLanes)
        {
            list.Add(("softmax.lanes" + S(l), Softmax(l), 2, 2 * sizeof(uint)));
            list.Add(("softmax_backward.lanes" + S(l), SoftmaxBackward(l), 3, 2 * sizeof(uint)));
            list.Add(("layernorm_forward.lanes" + S(l), LayerNormForward(l), 6, 3 * sizeof(uint)));
            list.Add(("layernorm_backward.lanes" + S(l), LayerNormBackward(l), 6, 2 * sizeof(uint)));
        }
        foreach (int r in ColumnRowLanes)
        {
            list.Add(("layernorm_grad_params.rows" + S(r), LayerNormGradParameters(r), 6, 2 * sizeof(uint)));
            list.Add(("rmsnorm_grad_gamma.rows" + S(r), RmsNormGradGamma(r), 4, 2 * sizeof(uint)));
        }
        foreach (int t in ConvChannelTiles)
            list.Add(("conv2d.channels" + S(t), Conv2DChannelTile(t), 3, 15 * sizeof(uint)));
        return list;
    }
}
