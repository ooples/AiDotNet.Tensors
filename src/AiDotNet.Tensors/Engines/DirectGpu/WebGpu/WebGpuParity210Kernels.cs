// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL compute shaders for the parity-210 op surface. Mirrors
// CudaParity210Kernels / HipParity210Kernels / MetalParity210Kernels /
// VulkanParity210Kernels / OpenClParity210Kernels function-for-function.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WGSL compute shader sources for the parity-210 kernels. Each shader uses
/// @workgroup_size(256) and a 1-D dispatch shape.  WGSL lacks native erf /
/// erfc / lgamma / bessel so we ship polynomial approximations in-shader —
/// accuracy targets single-precision parity with the CPU reference (~1e-6
/// for polynomials, ~1e-4 for erfinv where the Winitzki seed + two Newton
/// refinements bound the residual).
/// </summary>
public static class WebGpuParity210Kernels
{
    public static string[] GetKernelNames() => new[]
    {
        "parity210_roll_1d","parity210_flip_axis","parity210_triu","parity210_tril",
        "parity210_diag_embed",
        "parity210_cumsum_axis","parity210_cumprod_axis","parity210_cummax_axis",
        "parity210_cummin_axis","parity210_logcumsumexp_axis",
        "parity210_take_linear","parity210_take_along_dim",
        "parity210_index_copy","parity210_index_fill",
        // parity210_index_add is SKIPPED on WebGPU — no standard atomic<f32>.
        // See the Skip-with-reason convention in Parity210Kernels.cs.
        "parity210_hypot","parity210_copysign","parity210_fmod","parity210_remainder",
        "parity210_float_power","parity210_log_add_exp","parity210_log_add_exp2",
        "parity210_xlogy","parity210_xlog1py",
        "parity210_erfc","parity210_erfinv","parity210_lgamma_approx","parity210_digamma",
        "parity210_i0","parity210_i1","parity210_i0e","parity210_i1e",
        "parity210_is_finite","parity210_is_nan","parity210_is_inf","parity210_nan_to_num",
        "parity210_cosine_similarity_last","parity210_cdist_l2",
        "parity210_clamp_min_max",
    };

    private const string Helpers = @"
// erf via Abramowitz-Stegun 7.1.26 (6 digits).
fn p210_erf(x: f32) -> f32 {
    let sign_ : f32 = select(1.0, -1.0, x < 0.0);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((
            1.061405429 * t - 1.453152027) * t
            + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t) * exp(-ax * ax);
    return sign_ * y;
}

fn p210_i0(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 3.75) {
        var y = x / 3.75;
        y = y * y;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }
    let y = 3.75 / ax;
    let ans = 0.39894228 + y * (0.01328592 + y * (0.00225319
            + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706
            + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
    return (exp(ax) / sqrt(ax)) * ans;
}

fn p210_i1(x: f32) -> f32 {
    let ax = abs(x);
    var ans : f32;
    if (ax < 3.75) {
        var y = x / 3.75; y = y * y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
                + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
    } else {
        let y = 3.75 / ax;
        ans = 0.39894228 + y * (-0.03988024 + y * (-0.00362018
                + y * (0.00163801 + y * (-0.01031555 + y * (0.02282967
                + y * (-0.02895312 + y * (0.01787654 + y * -0.00420059)))))));
        ans = ans * (exp(ax) / sqrt(ax));
    }
    if (x < 0.0) { return -ans; } else { return ans; }
}

// Lanczos lgamma (reflection handled by caller to keep recursion-free).
fn p210_lgamma_positive(x: f32) -> f32 {
    // x >= 0.5 — valid for the Lanczos tail.
    let g : f32 = 7.0;
    let t = x - 1.0;
    let hi = t + g + 0.5;
    var sum : f32 = 0.99999999999980993;
    var lc : array<f32, 8>;
    lc[0] = 676.5203681218851;
    lc[1] = -1259.1392167224028;
    lc[2] = 771.32342877765313;
    lc[3] = -176.61502916214059;
    lc[4] = 12.507343278686905;
    lc[5] = -0.13857109526572012;
    lc[6] = 9.9843695780195716e-6;
    lc[7] = 1.5056327351493116e-7;
    for (var i : i32 = 0; i < 8; i = i + 1) {
        sum = sum + lc[i] / (t + f32(i) + 1.0);
    }
    return 0.5 * log(2.0 * 3.14159265358979) + (t + 0.5) * log(hi) - hi + log(sum);
}

fn p210_lgamma(x: f32) -> f32 {
    if (x < 0.5) {
        // reflection: lgamma(x) = log(pi / |sin(pi*x)|) - lgamma(1-x)
        return log(3.14159265358979 / abs(sin(3.14159265358979 * x))) - p210_lgamma_positive(1.0 - x);
    }
    return p210_lgamma_positive(x);
}
";

    // Note: WGSL `isFinite` / `isInf` / `isNan` are not available in core
    // WebGPU. We emulate via arithmetic: NaN != NaN; Inf - Inf = NaN;
    // finite iff (x == x) && (x - x == 0).

    public static string Roll1D => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { outerSize: i32, axisSize: i32, innerSize: i32, shift: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.axisSize * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let tmp = gid / p.innerSize;
    let ax = tmp % p.axisSize;
    let outer = tmp / p.axisSize;
    let src = ((ax - p.shift) % p.axisSize + p.axisSize) % p.axisSize;
    o[gid] = a[(outer * p.axisSize + src) * p.innerSize + inner];
}
";

    public static string FlipAxis => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { outerSize: i32, axisSize: i32, innerSize: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.axisSize * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let tmp = gid / p.innerSize;
    let ax = tmp % p.axisSize;
    let outer = tmp / p.axisSize;
    let src = p.axisSize - 1 - ax;
    o[gid] = a[(outer * p.axisSize + src) * p.innerSize + inner];
}
";

    public static string Triu => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { batchSize: i32, rows: i32, cols: i32, diagonal: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.batchSize * p.rows * p.cols;
    if (gid >= total) { return; }
    let col = gid % p.cols;
    let tmp = gid / p.cols;
    let row = tmp % p.rows;
    if ((col - row) >= p.diagonal) { o[gid] = a[gid]; } else { o[gid] = 0.0; }
}
";

    public static string Tril => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { batchSize: i32, rows: i32, cols: i32, diagonal: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.batchSize * p.rows * p.cols;
    if (gid >= total) { return; }
    let col = gid % p.cols;
    let tmp = gid / p.cols;
    let row = tmp % p.rows;
    if ((col - row) <= p.diagonal) { o[gid] = a[gid]; } else { o[gid] = 0.0; }
}
";

    public static string DiagEmbed => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { batchSize: i32, diagLen: i32, matSize: i32, offset: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.batchSize * p.matSize * p.matSize;
    if (gid >= total) { return; }
    let col = gid % p.matSize;
    let tmp = gid / p.matSize;
    let row = tmp % p.matSize;
    let b = tmp / p.matSize;
    let diagRow = select(row + p.offset, row, p.offset >= 0);
    let diagCol = select(col, col - p.offset, p.offset >= 0);
    if (diagRow == diagCol && diagRow >= 0 && diagRow < p.diagLen) {
        o[gid] = a[b * p.diagLen + diagRow];
    } else {
        o[gid] = 0.0;
    }
}
";

    private static string CumulativeScan(string opName, string init, string update, string output) => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { outerSize: i32, axisSize: i32, innerSize: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let outer = gid / p.innerSize;
    let base_ = outer * p.axisSize * p.innerSize + inner;
    var acc : f32 = " + init + @";
    for (var k : i32 = 0; k < p.axisSize; k = k + 1) {
        let v = a[base_ + k * p.innerSize];
        " + update + @"
        o[base_ + k * p.innerSize] = " + output + @";
    }
}
";

    public static string CumSumAxis => CumulativeScan("cumsum", "0.0", "acc = acc + v;", "acc");
    public static string CumProdAxis => CumulativeScan("cumprod", "1.0", "acc = acc * v;", "acc");
    public static string CumMaxAxis => CumulativeScan("cummax", "-3.402823e+38", "if (v > acc) { acc = v; }", "acc");
    public static string CumMinAxis => CumulativeScan("cummin", "3.402823e+38", "if (v < acc) { acc = v; }", "acc");

    public static string LogCumSumExpAxis => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { outerSize: i32, axisSize: i32, innerSize: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let outer = gid / p.innerSize;
    let base_ = outer * p.axisSize * p.innerSize + inner;
    if (p.axisSize <= 0) { return; }
    // Bootstrap from the first element so a leading -Inf doesn't yield NaN.
    var m : f32 = a[base_];
    var s : f32 = 1.0;
    o[base_] = m;
    for (var k : i32 = 1; k < p.axisSize; k = k + 1) {
        let x = a[base_ + k * p.innerSize];
        if (x > m) { s = s * exp(m - x) + 1.0; m = x; }
        else { s = s + exp(x - m); }
        o[base_ + k * p.innerSize] = m + log(s);
    }
}
";

    public static string TakeLinear => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> idx : array<i32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { outSize: i32, inputLinearLen: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.outSize) { return; }
    let pos = idx[gid];
    if (pos >= 0 && pos < p.inputLinearLen) { o[gid] = a[pos]; } else { o[gid] = 0.0; }
}
";

    public static string TakeAlongDim => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> idx : array<i32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { outerSize: i32, idxAxis: i32, innerSize: i32, srcAxis: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.idxAxis * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let tmp = gid / p.innerSize;
    let i = tmp % p.idxAxis;
    let outer = tmp / p.idxAxis;
    let target = idx[gid];
    let srcIdx = (outer * p.srcAxis + target) * p.innerSize + inner;
    if (target >= 0 && target < p.srcAxis) { o[gid] = a[srcIdx]; } else { o[gid] = 0.0; }
}
";

    public static string IndexCopy => @"
@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@group(0) @binding(1) var<storage, read> idx : array<i32>;
@group(0) @binding(2) var<storage, read> src : array<f32>;
struct P { outerSize: i32, dstAxis: i32, innerSize: i32, idxLen: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.idxLen * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let tmp = gid / p.innerSize;
    let i = tmp % p.idxLen;
    let outer = tmp / p.idxLen;
    let target = idx[i];
    if (target < 0 || target >= p.dstAxis) { return; }
    o[(outer * p.dstAxis + target) * p.innerSize + inner] =
        src[(outer * p.idxLen + i) * p.innerSize + inner];
}
";

    public static string IndexFill => @"
@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@group(0) @binding(1) var<storage, read> idx : array<i32>;
struct P { outerSize: i32, dstAxis: i32, innerSize: i32, idxLen: i32, fillValue: f32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.idxLen * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let tmp = gid / p.innerSize;
    let i = tmp % p.idxLen;
    let outer = tmp / p.idxLen;
    let target = idx[i];
    if (target < 0 || target >= p.dstAxis) { return; }
    o[(outer * p.dstAxis + target) * p.innerSize + inner] = p.fillValue;
}
";

    public static string Hypot => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    // Numerically stable hypot.
    let ax = abs(a[gid]); let bx = abs(b[gid]);
    let hi = max(ax, bx); let lo = min(ax, bx);
    if (hi == 0.0) { o[gid] = 0.0; return; }
    let r = lo / hi;
    o[gid] = hi * sqrt(1.0 + r * r);
}
";

    public static string Copysign => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    // Sign-bit transplant so copysign(1.0, -0.0) returns -1.0 (IEEE).
    let magBits = bitcast<u32>(abs(a[gid])) & 0x7FFFFFFFu;
    let signBit = bitcast<u32>(b[gid]) & 0x80000000u;
    o[gid] = bitcast<f32>(magBits | signBit);
}
";

    public static string Fmod => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let bv = b[gid];
    if (bv == 0.0) { o[gid] = 0.0; return; }
    let av = a[gid];
    let q = trunc(av / bv);
    o[gid] = av - q * bv;
}
";

    public static string Remainder => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let av = a[gid]; let bv = b[gid];
    if (bv == 0.0) { o[gid] = 0.0; return; }
    let q = floor(av / bv);
    o[gid] = av - q * bv;
}
";

    public static string FloatPower => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    o[gid] = pow(a[gid], b[gid]);
}
";

    public static string LogAddExp => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let av = a[gid]; let bv = b[gid];
    let m = max(av, bv);
    let s = min(av, bv);
    o[gid] = m + log(1.0 + exp(s - m));
}
";

    public static string LogAddExp2 => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let av = a[gid]; let bv = b[gid];
    let m = max(av, bv);
    let s = min(av, bv);
    o[gid] = m + log2(1.0 + exp2(s - m));
}
";

    public static string Xlogy => @"
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> y : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let xv = x[gid];
    if (xv == 0.0) { o[gid] = 0.0; } else { o[gid] = xv * log(y[gid]); }
}
";

    public static string Xlog1py => @"
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> y : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let xv = x[gid];
    if (xv == 0.0) { o[gid] = 0.0; } else { o[gid] = xv * log(1.0 + y[gid]); }
}
";

    public static string Erfc => Helpers + @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    o[gid] = 1.0 - p210_erf(a[gid]);
}
";

    public static string Erfinv => Helpers + @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let y = a[gid];
    // Signed infinities at the boundary — matches CUDA/HIP/Metal so downstream
    // isinf() checks stay honest.  WGSL has no INF literal so we use 1.0/0.0.
    if (y >= 1.0) { o[gid] = 1.0 / 0.0; return; }
    if (y <= -1.0) { o[gid] = -1.0 / 0.0; return; }
    let ln = log(1.0 - y * y);
    let ac = 0.147;
    let pi : f32 = 3.14159265358979;
    let t = 2.0 / (pi * ac) + ln * 0.5;
    var xs : f32 = sign(y) * sqrt(sqrt(t * t - ln / ac) - t);
    for (var k : i32 = 0; k < 2; k = k + 1) {
        let e = p210_erf(xs);
        let df = 2.0 / sqrt(pi) * exp(-xs * xs);
        xs = xs - (e - y) / df;
    }
    o[gid] = xs;
}
";

    public static string LgammaApprox => Helpers + @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    o[gid] = p210_lgamma(a[gid]);
}
";

    public static string Digamma => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    var x = a[gid];
    var result : f32 = 0.0;
    for (var k : i32 = 0; k < 64; k = k + 1) {
        if (x >= 6.0) { break; }
        result = result - 1.0 / x;
        x = x + 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result = result + log(x) - 0.5 * inv
            - inv2 * ((1.0/12.0)
              - inv2 * ((1.0/120.0)
                - inv2 * (1.0/252.0)));
    o[gid] = result;
}
";

    public static string I0 => Helpers + @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    o[gid] = p210_i0(a[gid]);
}
";

    public static string I1 => Helpers + @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    o[gid] = p210_i1(a[gid]);
}
";

    // Overflow-safe I0e — see CUDA/Metal counterparts for commentary.
    public static string I0e => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let x = a[gid];
    let ax = abs(x);
    var ans : f32;
    if (ax < 3.75) {
        var y = x / 3.75; y = y * y;
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
        ans = exp(-ax) * ans;
    } else {
        let y = 3.75 / ax;
        ans = 0.39894228 + y * (0.01328592 + y * (0.00225319
                + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706
                + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
        ans = ans / sqrt(ax);
    }
    o[gid] = ans;
}
";

    public static string I1e => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let x = a[gid];
    let ax = abs(x);
    var ans : f32;
    if (ax < 3.75) {
        var y = x / 3.75; y = y * y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
                + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
        ans = exp(-ax) * ans;
    } else {
        let y = 3.75 / ax;
        ans = 0.39894228 + y * (-0.03988024 + y * (-0.00362018
                + y * (0.00163801 + y * (-0.01031555 + y * (0.02282967
                + y * (-0.02895312 + y * (0.01787654 + y * -0.00420059)))))));
        ans = ans / sqrt(ax);
    }
    if (x < 0.0) { o[gid] = -ans; } else { o[gid] = ans; }
}
";

    // WGSL lacks isFinite/isInf/isNan; emulate via IEEE identities.
    public static string IsFinite => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let x = a[gid];
    let eqSelf = (x == x);        // false for NaN
    let sub = x - x;               // 0 for finite, NaN for ±Inf
    let subEq = (sub == 0.0);      // false for NaN / ±Inf
    if (eqSelf && subEq) { o[gid] = 1.0; } else { o[gid] = 0.0; }
}
";

    public static string IsNan => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let x = a[gid];
    if (x == x) { o[gid] = 0.0; } else { o[gid] = 1.0; }
}
";

    public static string IsInf => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let x = a[gid];
    let eqSelf = (x == x);         // true for finite and ±Inf; false for NaN
    let sub = x - x;                // NaN for ±Inf; 0 for finite
    if (eqSelf && !(sub == 0.0)) { o[gid] = 1.0; } else { o[gid] = 0.0; }
}
";

    public static string NanToNum => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> o : array<f32>;
struct P { size: i32, nanVal: f32, posInfVal: f32, negInfVal: f32 };
@group(0) @binding(2) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    let x = a[gid];
    let isNaN_ = !(x == x);
    let isInf_ = (x == x) && !((x - x) == 0.0);
    if (isNaN_) { o[gid] = p.nanVal; }
    else if (isInf_) {
        if (x > 0.0) { o[gid] = p.posInfVal; } else { o[gid] = p.negInfVal; }
    }
    else { o[gid] = x; }
}
";

    public static string CosineSimilarityLast => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { n: i32, d: i32, eps: f32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let row = i32(id.x);
    if (row >= p.n) { return; }
    var dotv : f32 = 0.0;
    var na : f32 = 0.0;
    var nb : f32 = 0.0;
    let base_ = row * p.d;
    for (var k : i32 = 0; k < p.d; k = k + 1) {
        let av = a[base_ + k]; let bv = b[base_ + k];
        dotv = dotv + av * bv;
        na = na + av * av;
        nb = nb + bv * bv;
    }
    let denom = max(sqrt(na * nb), p.eps);
    o[row] = dotv / denom;
}
";

    public static string CdistL2 => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
struct P { n: i32, m: i32, d: i32 };
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.n * p.m;
    if (gid >= total) { return; }
    let j = gid % p.m;
    let i = gid / p.m;
    var acc : f32 = 0.0;
    for (var k : i32 = 0; k < p.d; k = k + 1) {
        let v = a[i * p.d + k] - b[j * p.d + k];
        acc = acc + v * v;
    }
    o[gid] = sqrt(acc);
}
";

    public static string ClampMinMax => @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> lo : array<f32>;
@group(0) @binding(2) var<storage, read> hi : array<f32>;
@group(0) @binding(3) var<storage, read_write> o : array<f32>;
struct P { size: i32, hasLo: i32, hasHi: i32 };
@group(0) @binding(4) var<uniform> p : P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }
    var x = a[gid];
    if (p.hasLo != 0) { let l = lo[gid]; if (x < l) { x = l; } }
    if (p.hasHi != 0) { let h = hi[gid]; if (x > h) { x = h; } }
    o[gid] = x;
}
";
}
#endif
