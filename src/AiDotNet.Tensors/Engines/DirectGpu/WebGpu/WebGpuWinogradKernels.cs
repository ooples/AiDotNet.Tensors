namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1372: the WGSL instance of WinogradF23KernelTemplate's tile transforms. WGSL is not C-like enough to share the
// template's bodies (let/var, explicit casts, no ?: on buffer reads, uniform structs instead of push constants), so the
// arithmetic is restated here operation for operation; WinogradF23PortTests checks both against the same CPU reference.
// Each kernel is its own module: the read buffer at binding 0, the written one at 1, the int parameters in a uniform
// struct at 2, padded to a 16-byte multiple. Boundary reads use if, never select(), which would evaluate the
// out-of-bounds read too. Function-parameter arrays are copied into a var before they are indexed dynamically.
internal static class WebGpuWinogradKernels
{
    private const string FilterTail = @"
fn filter_tail(gIn: array<f32, 9>, idx: i32) {
    var g = gIn;
    var t: array<f32, 12>;
    for (var j = 0; j < 3; j = j + 1) {
        let g0 = g[j]; let g1 = g[3 + j]; let g2 = g[6 + j];
        t[j] = g0;
        t[3 + j] = 0.5 * (g0 + g1 + g2);
        t[6 + j] = 0.5 * (g0 - g1 + g2);
        t[9 + j] = g2;
    }
    let plane = pm.K * pm.C;
    for (var i = 0; i < 4; i = i + 1) {
        let a = t[i * 3]; let b = t[i * 3 + 1]; let e = t[i * 3 + 2];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = 0.5 * (a + b + e);
        u[(i * 4 + 2) * plane + idx] = 0.5 * (a - b + e);
        u[(i * 4 + 3) * plane + idx] = e;
    }
}
";

    private const string FilterHeader = @"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
struct Params { K: i32, C: i32, pad0: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
";

    public const string Filter = FilterHeader + FilterTail + @"
@compute @workgroup_size(256)
fn wino_f23_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.K * pm.C) { return; }
    var g: array<f32, 9>;
    for (var q = 0; q < 9; q = q + 1) { g[q] = w[idx * 9 + q]; }
    filter_tail(g, idx);
}
";

    public const string FilterFlip = FilterHeader + FilterTail + @"
@compute @workgroup_size(256)
fn wino_f23_filter_flip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.K * pm.C) { return; }
    let c = idx / pm.K;
    let k = idx - c * pm.K;
    let src = (k * pm.C + c) * 9;
    var g: array<f32, 9>;
    for (var i = 0; i < 3; i = i + 1) {
        for (var j = 0; j < 3; j = j + 1) { g[i * 3 + j] = w[src + (2 - i) * 3 + (2 - j)]; }
    }
    filter_tail(g, idx);
}
";

    private const string InputHeader = @"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> v: array<f32>;
struct Params { N: i32, C: i32, H: i32, W: i32, pad: i32, tilesH: i32, tilesW: i32, pad0: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
fn input_tail(c: i32, p: i32, idx: i32) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let n = p / (pm.tilesH * pm.tilesW);
    let tt = p - n * (pm.tilesH * pm.tilesW);
    let th = tt / pm.tilesW;
    let tw = tt - th * pm.tilesW;
    let h0 = th * 2 - pm.pad;
    let w0 = tw * 2 - pm.pad;
    let src = (n * pm.C + c) * pm.H * pm.W;
    var d: array<f32, 16>;
    for (var i = 0; i < 4; i = i + 1) {
        let hh = h0 + i;
        for (var j = 0; j < 4; j = j + 1) {
            let ww = w0 + j;
            var value = 0.0;
            if (hh >= 0 && hh < pm.H && ww >= 0 && ww < pm.W) { value = x[src + hh * pm.W + ww]; }
            d[i * 4 + j] = value;
        }
    }
    var s: array<f32, 16>;
    for (var j = 0; j < 4; j = j + 1) {
        s[j] = d[j] - d[8 + j];
        s[4 + j] = d[4 + j] + d[8 + j];
        s[8 + j] = d[8 + j] - d[4 + j];
        s[12 + j] = d[4 + j] - d[12 + j];
    }
    let plane = pm.C * P;
    for (var i = 0; i < 4; i = i + 1) {
        v[(i * 4 + 0) * plane + idx] = s[i * 4] - s[i * 4 + 2];
        v[(i * 4 + 1) * plane + idx] = s[i * 4 + 1] + s[i * 4 + 2];
        v[(i * 4 + 2) * plane + idx] = s[i * 4 + 2] - s[i * 4 + 1];
        v[(i * 4 + 3) * plane + idx] = s[i * 4 + 1] - s[i * 4 + 3];
    }
}
";

    // V[x][c][p]; idx = c*P + p.
    public const string Input = InputHeader + @"
@compute @workgroup_size(256)
fn wino_f23_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.C * P) { return; }
    let c = idx / P;
    input_tail(c, idx - c * P, idx);
}
";

    // Weight gradient: V[x][p][c], tile-major; idx = p*C + c.
    public const string InputTilesTileMajor = InputHeader + @"
@compute @workgroup_size(256)
fn wino_f32w_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.C * P) { return; }
    let p = idx / pm.C;
    input_tail(idx - p * pm.C, p, idx);
}
";

    private const string TileHeader = @"
struct Params { N: i32, K: i32, outH: i32, outW: i32, tilesH: i32, tilesW: i32, pad0: i32, pad1: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
";

    public const string Output = @"
@group(0) @binding(0) var<storage, read> mm: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
" + TileHeader + @"
@compute @workgroup_size(256)
fn wino_f23_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.K * P) { return; }
    let k = idx / P;
    let p = idx - k * P;
    let n = p / (pm.tilesH * pm.tilesW);
    let tt = p - n * (pm.tilesH * pm.tilesW);
    let th = tt / pm.tilesW;
    let tw = tt - th * pm.tilesW;
    let plane = pm.K * P;
    var m: array<f32, 16>;
    for (var q = 0; q < 16; q = q + 1) { m[q] = mm[q * plane + idx]; }
    var s: array<f32, 8>;
    for (var j = 0; j < 4; j = j + 1) {
        s[j] = m[j] + m[4 + j] + m[8 + j];
        s[4 + j] = m[4 + j] - m[8 + j] - m[12 + j];
    }
    let dst = (n * pm.K + k) * pm.outH * pm.outW;
    let h = th * 2;
    let w2 = tw * 2;
    for (var i = 0; i < 2; i = i + 1) {
        if (h + i >= pm.outH) { break; }
        y[dst + (h + i) * pm.outW + w2] = s[i * 4] + s[i * 4 + 1] + s[i * 4 + 2];
        if (w2 + 1 < pm.outW) { y[dst + (h + i) * pm.outW + w2 + 1] = s[i * 4 + 1] - s[i * 4 + 2] - s[i * 4 + 3]; }
    }
}
";

    public const string GradOutputTiles = @"
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
" + TileHeader + @"
@compute @workgroup_size(256)
fn wino_f32w_dy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let P = pm.N * pm.tilesH * pm.tilesW;
    let idx = i32(gid.x);
    if (idx >= pm.K * P) { return; }
    let k = idx / P;
    let p = idx - k * P;
    let n = p / (pm.tilesH * pm.tilesW);
    let tt = p - n * (pm.tilesH * pm.tilesW);
    let th = tt / pm.tilesW;
    let tw = tt - th * pm.tilesW;
    let src = (n * pm.K + k) * pm.outH * pm.outW;
    let h = th * 2;
    let w2 = tw * 2;
    let g00 = dy[src + h * pm.outW + w2];
    var g01 = 0.0;
    var g10 = 0.0;
    var g11 = 0.0;
    if (w2 + 1 < pm.outW) { g01 = dy[src + h * pm.outW + w2 + 1]; }
    if (h + 1 < pm.outH) { g10 = dy[src + (h + 1) * pm.outW + w2]; }
    if (h + 1 < pm.outH && w2 + 1 < pm.outW) { g11 = dy[src + (h + 1) * pm.outW + w2 + 1]; }
    // t = A g (4x2): rows g0, g0+g1, g0-g1, -g1 over the two columns.
    var r: array<f32, 8>;
    r[0] = g00; r[1] = g01;
    r[2] = g00 + g10; r[3] = g01 + g11;
    r[4] = g00 - g10; r[5] = g01 - g11;
    r[6] = -g10; r[7] = -g11;
    let plane = pm.K * P;
    for (var i = 0; i < 4; i = i + 1) {
        let a = r[i * 2];
        let b = r[i * 2 + 1];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = a + b;
        u[(i * 4 + 2) * plane + idx] = a - b;
        u[(i * 4 + 3) * plane + idx] = -b;
    }
}
";

    public const string FilterGradient = @"
@group(0) @binding(0) var<storage, read> mm: array<f32>;
@group(0) @binding(1) var<storage, read_write> dw: array<f32>;
struct Params { K: i32, C: i32, splits: i32, pad0: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn wino_f32w_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= pm.K * pm.C) { return; }
    let plane = pm.K * pm.C;
    var m: array<f32, 16>;
    for (var q = 0; q < 16; q = q + 1) {
        var acc = 0.0;
        for (var sp = 0; sp < pm.splits; sp = sp + 1) { acc = acc + mm[(q * pm.splits + sp) * plane + idx]; }
        m[q] = acc;
    }
    var t: array<f32, 12>;
    for (var j = 0; j < 4; j = j + 1) {
        let m0 = m[j]; let m1 = m[4 + j]; let m2 = m[8 + j]; let m3 = m[12 + j];
        t[j] = m0 + 0.5 * (m1 + m2);
        t[4 + j] = 0.5 * (m1 - m2);
        t[8 + j] = 0.5 * (m1 + m2) + m3;
    }
    let dst = idx * 9;
    for (var i = 0; i < 3; i = i + 1) {
        let a = t[i * 4]; let b = t[i * 4 + 1]; let e = t[i * 4 + 2]; let f = t[i * 4 + 3];
        dw[dst + i * 3 + 0] = a + 0.5 * (b + e);
        dw[dst + i * 3 + 1] = 0.5 * (b - e);
        dw[dst + i * 3 + 2] = 0.5 * (b + e) + f;
    }
}
";

    /// <summary>The module source for a <see cref="WinogradF23KernelTemplate"/> kernel name.</summary>
    internal static string SourceFor(string name) => name switch
    {
        WinogradF23KernelTemplate.Filter => Filter,
        WinogradF23KernelTemplate.FilterFlip => FilterFlip,
        WinogradF23KernelTemplate.Input => Input,
        WinogradF23KernelTemplate.InputTilesTileMajor => InputTilesTileMajor,
        WinogradF23KernelTemplate.Output => Output,
        WinogradF23KernelTemplate.GradOutputTiles => GradOutputTiles,
        WinogradF23KernelTemplate.FilterGradient => FilterGradient,
        _ => throw new System.ArgumentException($"No WGSL Winograd kernel named '{name}'.", nameof(name)),
    };
}