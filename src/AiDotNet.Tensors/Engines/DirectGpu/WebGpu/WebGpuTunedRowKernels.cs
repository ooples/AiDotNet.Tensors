#if NET7_0_OR_GREATER
using System.Globalization;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WGSL variants for the tuned-kernel registry on WebGPU (#1117): one template instantiated per lane count, as
/// <c>CudaTunedRowKernels</c> and <c>VulkanTunedRowKernels</c> are. A row kernel gives each row <c>L</c> lanes of a
/// 256-invocation workgroup (256 / L rows per workgroup) and reduces each row's partials in a fixed tree; the column
/// kernels give each 32-column tile <c>R</c> row lanes. Fixed order, no atomics: deterministic.
/// </summary>
/// <remarks>
/// Every storage buffer is bound in order from 0 and the parameter block follows at the next binding, which is the
/// layout <see cref="WebGpuBindGroup"/> builds. Barriers run in uniform control flow: an invocation past the last row
/// still joins every reduction (its <c>live</c> flag only gates its loads and stores) and returns after the last one.
/// </remarks>
internal static class WebGpuTunedRowKernels
{
    internal const int GroupSize = 256;

    /// <summary>The widest workgroup count one dispatch dimension may have.</summary>
    internal const int MaxWorkgroups = 65535;

    /// <summary>Lane counts each row-kernel family is instantiated for.</summary>
    internal static readonly int[] RowLanes = { 8, 16, 32, 256 };

    /// <summary>Row-lane counts the column-reduction family is instantiated for.</summary>
    internal static readonly int[] ColumnRowLanes = { 8, 32 };

    /// <summary>Columns one lane may walk before a narrow lane count stops being offered for the row.</summary>
    internal const int MaxColumnsPerLane = 64;

    /// <summary>
    /// Whether the <paramref name="lanes"/>-lane variant is offered a <paramref name="rows"/> x <paramref name="n"/>
    /// problem: the widest count has no column limit, a narrow one stops past <see cref="MaxColumnsPerLane"/> columns
    /// per lane, and the dispatch must fit one dimension's workgroup limit.
    /// </summary>
    internal static bool RowLanesFit(long rows, long n, int lanes) =>
        (lanes == GroupSize || n <= (long)MaxColumnsPerLane * lanes) && RowWorkgroups(rows, lanes) <= MaxWorkgroups;

    internal static long RowWorkgroups(long rows, int lanes) => (rows * lanes + GroupSize - 1) / GroupSize;

    internal static bool ColumnFit(long n) => (n + 31) / 32 <= MaxWorkgroups;

    private const string Reductions = @"
var<workgroup> scratch: array<f32, 256>;
fn lane_sum(v: f32, lid: i32) -> f32 {
    scratch[lid] = v;
    workgroupBarrier();
    for (var o: i32 = L / 2; o > 0; o = o >> 1u) {
        if ((lid % L) < o) { scratch[lid] = scratch[lid] + scratch[lid + o]; }
        workgroupBarrier();
    }
    let r = scratch[lid - lid % L];
    workgroupBarrier();
    return r;
}
fn lane_max(v: f32, lid: i32) -> f32 {
    scratch[lid] = v;
    workgroupBarrier();
    for (var o: i32 = L / 2; o > 0; o = o >> 1u) {
        if ((lid % L) < o) { scratch[lid] = max(scratch[lid], scratch[lid + o]); }
        workgroupBarrier();
    }
    let r = scratch[lid - lid % L];
    workgroupBarrier();
    return r;
}
";

    private static string Storage(int binding, string name, bool write) =>
        "@group(0) @binding(" + binding.ToString(CultureInfo.InvariantCulture) + ") var<storage, "
        + (write ? "read_write" : "read") + "> " + name + ": array<f32>;\n";

    private static string Params(int binding) =>
        "struct Params { rows: i32, n: i32, epsilon: f32, pad: f32, }\n@group(0) @binding("
        + binding.ToString(CultureInfo.InvariantCulture) + ") var<uniform> p: Params;\n";

    private static string RowHeader(int lanes, string bindings) => bindings
        + "const L: i32 = " + lanes.ToString(CultureInfo.InvariantCulture) + ";\n" + Reductions;

    private const string RowPrologue = @"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lidv: vec3<u32>) {
    let t = i32(gid.x);
    let lid = i32(lidv.x);
    let row = t / L;
    let lane = t % L;
    let live = row < p.rows;
    let r = select(0, row, live);
    let base = r * p.n;
";

    internal static string Softmax(int lanes) => RowHeader(lanes,
        Storage(0, "x", false) + Storage(1, "y", true) + Params(2)) + RowPrologue + @"
    var m: f32 = -3.4028234663852886e38;
    if (live) { for (var f = lane; f < p.n; f = f + L) { m = max(m, x[base + f]); } }
    m = lane_max(m, lid);
    var s: f32 = 0.0;
    if (live) { for (var f = lane; f < p.n; f = f + L) { let e = exp(x[base + f] - m); y[base + f] = e; s = s + e; } }
    s = lane_sum(s, lid);
    let inv = select(1.0, 1.0 / s, s > 0.0);
    if (live) { for (var f = lane; f < p.n; f = f + L) { y[base + f] = y[base + f] * inv; } }
}
";

    internal static string SoftmaxBackward(int lanes) => RowHeader(lanes,
        Storage(0, "dy", false) + Storage(1, "yv", false) + Storage(2, "dx", true) + Params(3)) + RowPrologue + @"
    var d: f32 = 0.0;
    if (live) { for (var f = lane; f < p.n; f = f + L) { d = d + dy[base + f] * yv[base + f]; } }
    d = lane_sum(d, lid);
    if (live) { for (var f = lane; f < p.n; f = f + L) { dx[base + f] = yv[base + f] * (dy[base + f] - d); } }
}
";

    internal static string LayerNorm(int lanes) => RowHeader(lanes,
        Storage(0, "x", false) + Storage(1, "y", true) + Storage(2, "gamma", false) + Storage(3, "beta", false)
        + Storage(4, "save_mean", true) + Storage(5, "save_inv_var", true) + Params(6)) + RowPrologue + @"
    var s: f32 = 0.0;
    if (live) { for (var i = lane; i < p.n; i = i + L) { s = s + x[base + i]; } }
    let mean = lane_sum(s, lid) / f32(p.n);
    var v: f32 = 0.0;
    if (live) { for (var i = lane; i < p.n; i = i + L) { let dd = x[base + i] - mean; v = v + dd * dd; } }
    let inv_var = inverseSqrt(lane_sum(v, lid) / f32(p.n) + p.epsilon);
    if (!live) { return; }
    if (lane == 0) { save_mean[row] = mean; save_inv_var[row] = inv_var; }
    for (var i = lane; i < p.n; i = i + L) { y[base + i] = gamma[i] * ((x[base + i] - mean) * inv_var) + beta[i]; }
}
";

    /// <summary>
    /// The row statistics of the established layer_norm kernel (one 256-lane workgroup per row, the same reduction
    /// tree), which it does not store: mean and 1 / sqrt(variance + epsilon) per row.
    /// </summary>
    internal static readonly string LayerNormRowStats = RowHeader(GroupSize,
        Storage(0, "x", false) + Storage(1, "save_mean", true) + Storage(2, "save_inv_var", true) + Params(3)) + RowPrologue + @"
    var s: f32 = 0.0;
    if (live) { for (var i = lane; i < p.n; i = i + L) { s = s + x[base + i]; } }
    let mean = lane_sum(s, lid) / f32(p.n);
    var v: f32 = 0.0;
    if (live) { for (var i = lane; i < p.n; i = i + L) { let dd = x[base + i] - mean; v = v + dd * dd; } }
    let inv_var = inverseSqrt(lane_sum(v, lid) / f32(p.n) + p.epsilon);
    if (live && lane == 0) { save_mean[row] = mean; save_inv_var[row] = inv_var; }
}
";
    internal static string LayerNormBackward(int lanes) => RowHeader(lanes,
        Storage(0, "dy", false) + Storage(1, "x", false) + Storage(2, "gamma", false) + Storage(3, "save_mean", false)
        + Storage(4, "save_inv_var", false) + Storage(5, "dx", true) + Params(6)) + RowPrologue + @"
    let mean = save_mean[r];
    let inv_var = save_inv_var[r];
    var sum_dy: f32 = 0.0;
    var sum_dy_xmu: f32 = 0.0;
    if (live) {
        for (var i = lane; i < p.n; i = i + L) {
            let d = dy[base + i] * gamma[i];
            sum_dy = sum_dy + d;
            sum_dy_xmu = sum_dy_xmu + d * (x[base + i] - mean);
        }
    }
    sum_dy = lane_sum(sum_dy, lid);
    sum_dy_xmu = lane_sum(sum_dy_xmu, lid);
    if (!live) { return; }
    for (var i = lane; i < p.n; i = i + L) {
        let xmu = x[base + i] - mean;
        let dxhat = dy[base + i] * gamma[i];
        dx[base + i] = inv_var * (dxhat - (sum_dy + xmu * inv_var * inv_var * sum_dy_xmu) / f32(p.n));
    }
}
";

    // A 32-column tile with R row lanes per workgroup (1-D groups of 32 * R: cx = lid % 32, ry = lid / 32).
    private static string ColumnGrad(int rowLanes, bool rms)
    {
        string bindings = rms
            ? Storage(0, "dy", false) + Storage(1, "x", false) + Storage(2, "stat0", false) + Storage(3, "out0", true) + Params(4)
            : Storage(0, "dy", false) + Storage(1, "x", false) + Storage(2, "stat0", false) + Storage(3, "stat1", false)
              + Storage(4, "out0", true) + Storage(5, "out1", true) + Params(6);
        string tile = (rowLanes * 33).ToString(CultureInfo.InvariantCulture);
        return bindings
            + "const R: i32 = " + rowLanes.ToString(CultureInfo.InvariantCulture) + ";\n"
            + "var<workgroup> s0: array<f32, " + tile + ">;\nvar<workgroup> s1: array<f32, " + tile + ">;\n"
            + "@compute @workgroup_size(" + (32 * rowLanes).ToString(CultureInfo.InvariantCulture) + ")\n" + @"
fn main(@builtin(local_invocation_id) lidv: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lid = i32(lidv.x);
    let cx = lid % 32;
    let ry = lid / 32;
    let col = i32(wid.x) * 32 + cx;
    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    if (col < p.n) {
        for (var b = ry; b < p.rows; b = b + R) {
            let idx = b * p.n + col;
            let g = dy[idx];
" + (rms ? "            a0 = a0 + g * x[idx] * (1.0 / stat0[b]);\n"
          : "            let normalized = (x[idx] - stat0[b]) * stat1[b];\n            a0 = a0 + g * normalized;\n            a1 = a1 + g;\n") + @"        }
    }
    s0[ry * 33 + cx] = a0;
    s1[ry * 33 + cx] = a1;
    workgroupBarrier();
    for (var h = R / 2; h > 0; h = h >> 1u) {
        if (ry < h) {
            s0[ry * 33 + cx] = s0[ry * 33 + cx] + s0[(ry + h) * 33 + cx];
            s1[ry * 33 + cx] = s1[ry * 33 + cx] + s1[(ry + h) * 33 + cx];
        }
        workgroupBarrier();
    }
    if (ry == 0 && col < p.n) {
        out0[col] = s0[cx];
" + (rms ? string.Empty : "        out1[col] = s1[cx];\n") + @"    }
}
";
    }

    internal static string LayerNormGradParams(int rowLanes) => ColumnGrad(rowLanes, rms: false);

    internal static string RmsNormGradGamma(int rowLanes) => ColumnGrad(rowLanes, rms: true);
}
#endif
