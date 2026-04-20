// Copyright (c) AiDotNet. All rights reserved.
// WGSL compute kernels for the torch.linalg decomposition surface (#211 moat #2).
// Matches the pattern in WebGpuParity210Kernels: one static string per op, one
// workgroup per batch slice, workgroup_size = 256. WGSL's subgroup + barrier
// primitives mirror GLSL's so the algorithms port 1:1 from VulkanLinalgKernels.

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu
{
    public static class WebGpuLinalgKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "parity211_cholesky",
            "parity211_lu_factor",
            "parity211_qr_reduced",
            "parity211_eigh",
        };

        // ═══════════════════════════════════════════════════════════════════
        // CHOLESKY
        // ═══════════════════════════════════════════════════════════════════
        public static string Cholesky => @"
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read_write> L : array<f32>;
@group(0) @binding(2) var<storage, read_write> info : array<i32>;
struct P { batchCount: i32, n: i32, upper: i32 };
@group(0) @binding(3) var<uniform> p : P;

var<workgroup> localInfo : i32;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) li : vec3<u32>)
{
    let b = i32(wg.x);
    if (b >= p.batchCount) { return; }
    let tid = i32(li.x);
    let bs = 256;
    let n = p.n;
    let off = b * n * n;

    for (var idx : i32 = tid; idx < n * n; idx = idx + bs) { L[off + idx] = A[off + idx]; }
    if (tid == 0) { localInfo = 0; }
    workgroupBarrier();

    for (var j : i32 = 0; j < n; j = j + 1) {
        if (tid == 0) {
            var diag : f32 = L[off + j * n + j];
            for (var k : i32 = 0; k < j; k = k + 1) {
                let l_jk : f32 = select(L[off + j * n + k], L[off + k * n + j], p.upper != 0);
                diag = diag - l_jk * l_jk;
            }
            if (diag <= 0.0) { localInfo = j + 1; L[off + j * n + j] = 0.0; }
            else { L[off + j * n + j] = sqrt(diag); }
        }
        workgroupBarrier();
        if (localInfo != 0) { break; }

        let djj : f32 = L[off + j * n + j];
        if (djj == 0.0) { break; }

        var i : i32 = j + 1 + tid;
        loop {
            if (i >= n) { break; }
            var sum : f32 = select(L[off + i * n + j], L[off + j * n + i], p.upper != 0);
            for (var k : i32 = 0; k < j; k = k + 1) {
                let l_ik : f32 = select(L[off + i * n + k], L[off + k * n + i], p.upper != 0);
                let l_jk : f32 = select(L[off + j * n + k], L[off + k * n + j], p.upper != 0);
                sum = sum - l_ik * l_jk;
            }
            if (p.upper != 0) { L[off + j * n + i] = sum / djj; }
            else              { L[off + i * n + j] = sum / djj; }
            i = i + bs;
        }
        workgroupBarrier();
    }

    var idx : i32 = tid;
    loop {
        if (idx >= n * n) { break; }
        let i2 = idx / n;
        let j2 = idx % n;
        let shouldZero : bool = select(i2 < j2, i2 > j2, p.upper != 0);
        if (shouldZero) { L[off + idx] = 0.0; }
        idx = idx + bs;
    }
    if (tid == 0) { info[b] = localInfo; }
}
";

        // ═══════════════════════════════════════════════════════════════════
        // LU FACTOR
        // ═══════════════════════════════════════════════════════════════════
        public static string LuFactor => @"
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read_write> LU : array<f32>;
@group(0) @binding(2) var<storage, read_write> pivots : array<i32>;
struct P { batchCount: i32, m: i32, n: i32 };
@group(0) @binding(3) var<uniform> p : P;

var<workgroup> pivRow : i32;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) li : vec3<u32>)
{
    let b = i32(wg.x);
    if (b >= p.batchCount) { return; }
    let tid = i32(li.x);
    let bs = 256;
    let m = p.m; let n = p.n;
    let k = select(n, m, m < n);
    let matOff = b * m * n;
    let pivOff = b * k;

    var idx0 : i32 = tid;
    loop { if (idx0 >= m * n) { break; } LU[matOff + idx0] = A[matOff + idx0]; idx0 = idx0 + bs; }
    workgroupBarrier();

    for (var j : i32 = 0; j < k; j = j + 1) {
        if (tid == 0) {
            var best : i32 = j;
            var bestVal : f32 = abs(LU[matOff + j * n + j]);
            for (var i : i32 = j + 1; i < m; i = i + 1) {
                let v : f32 = abs(LU[matOff + i * n + j]);
                if (v > bestVal) { bestVal = v; best = i; }
            }
            pivRow = best;
            pivots[pivOff + j] = best;
        }
        workgroupBarrier();

        let pp : i32 = pivRow;
        if (pp != j) {
            var c : i32 = tid;
            loop {
                if (c >= n) { break; }
                let t : f32 = LU[matOff + pp * n + c];
                LU[matOff + pp * n + c] = LU[matOff + j * n + c];
                LU[matOff + j * n + c] = t;
                c = c + bs;
            }
            workgroupBarrier();
        }
        let pivVal : f32 = LU[matOff + j * n + j];
        if (pivVal == 0.0) { continue; }
        var i1 : i32 = j + 1 + tid;
        loop { if (i1 >= m) { break; } LU[matOff + i1 * n + j] = LU[matOff + i1 * n + j] / pivVal; i1 = i1 + bs; }
        workgroupBarrier();
        let trailRows = m - j - 1;
        let trailCols = n - j - 1;
        var idx2 : i32 = tid;
        loop {
            if (idx2 >= trailRows * trailCols) { break; }
            let di = idx2 / trailCols;
            let dc = idx2 % trailCols;
            let i = j + 1 + di;
            let c = j + 1 + dc;
            LU[matOff + i * n + c] = LU[matOff + i * n + c] - LU[matOff + i * n + j] * LU[matOff + j * n + c];
            idx2 = idx2 + bs;
        }
        workgroupBarrier();
    }
}
";

        // ═══════════════════════════════════════════════════════════════════
        // QR REDUCED
        // ═══════════════════════════════════════════════════════════════════
        public static string QrReduced => @"
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read_write> Q : array<f32>;
@group(0) @binding(2) var<storage, read_write> R : array<f32>;
struct P { batchCount: i32, m: i32, n: i32 };
@group(0) @binding(3) var<uniform> p : P;

var<workgroup> sNorm : f32;
var<workgroup> sAlpha : f32;
var<workgroup> sBeta : f32;
var<workgroup> sV0 : f32;
var<workgroup> partials : array<f32, 256>;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) li : vec3<u32>)
{
    let b = i32(wg.x);
    if (b >= p.batchCount) { return; }
    let tid = i32(li.x);
    let bs = 256;
    let m = p.m; let n = p.n;
    let k = select(n, m, m < n);
    let aOff = b * m * n;
    let qOff = b * m * k;
    let rOff = b * k * n;

    var idx0 : i32 = tid;
    loop {
        if (idx0 >= m * n) { break; }
        let i = idx0 / n;
        if (i < k) { R[rOff + idx0] = A[aOff + idx0]; }
        idx0 = idx0 + bs;
    }
    workgroupBarrier();
    var idx1 : i32 = tid;
    loop {
        if (idx1 >= m * k) { break; }
        let i = idx1 / k;
        let j = idx1 % k;
        Q[qOff + i * k + j] = select(0.0, 1.0, i == j);
        idx1 = idx1 + bs;
    }
    workgroupBarrier();

    for (var j : i32 = 0; j < k; j = j + 1) {
        var partial : f32 = 0.0;
        var ii : i32 = j + tid;
        loop {
            if (ii >= m) { break; }
            let v : f32 = select(A[aOff + ii * n + j], R[rOff + ii * n + j], ii < k);
            partial = partial + v * v;
            ii = ii + bs;
        }
        partials[tid] = partial;
        workgroupBarrier();
        var s : i32 = bs / 2;
        loop {
            if (s <= 0) { break; }
            if (tid < s) { partials[tid] = partials[tid] + partials[tid + s]; }
            workgroupBarrier();
            s = s / 2;
        }
        if (tid == 0) { sNorm = sqrt(partials[0]); }
        workgroupBarrier();
        if (sNorm == 0.0) { continue; }

        if (tid == 0) {
            let x0 : f32 = select(A[aOff + j * n + j], R[rOff + j * n + j], j < k);
            sAlpha = select(sNorm, -sNorm, x0 >= 0.0);
            sV0 = x0 - sAlpha;
            let vNorm2 : f32 = sV0 * sV0 + (sNorm * sNorm - x0 * x0);
            sBeta = select(0.0, 2.0 / vNorm2, vNorm2 > 0.0);
        }
        workgroupBarrier();
        if (sBeta == 0.0) { continue; }

        var c : i32 = j + tid;
        loop {
            if (c >= n) { break; }
            var dotp : f32 = sV0 * select(A[aOff + j * n + c], R[rOff + j * n + c], j < k);
            for (var i : i32 = 1; i + j < m; i = i + 1) {
                let v_i : f32 = select(A[aOff + (j + i) * n + j], R[rOff + (j + i) * n + j], j + i < k);
                let a_ic : f32 = select(A[aOff + (j + i) * n + c], R[rOff + (j + i) * n + c], j + i < k);
                dotp = dotp + v_i * a_ic;
            }
            let ss : f32 = sBeta * dotp;
            if (j < k) { R[rOff + j * n + c] = R[rOff + j * n + c] - ss * sV0; }
            for (var i : i32 = 1; i + j < m; i = i + 1) {
                let v_i : f32 = select(A[aOff + (j + i) * n + j], R[rOff + (j + i) * n + j], j + i < k);
                if (j + i < k) { R[rOff + (j + i) * n + c] = R[rOff + (j + i) * n + c] - ss * v_i; }
            }
            c = c + bs;
        }
        workgroupBarrier();
        var ri : i32 = tid;
        loop {
            if (ri >= m) { break; }
            let dotp : f32 = sV0 * Q[qOff + ri * k + j];
            let ss : f32 = sBeta * dotp;
            Q[qOff + ri * k + j] = Q[qOff + ri * k + j] - ss * sV0;
            ri = ri + bs;
        }
        workgroupBarrier();
    }
    var idx3 : i32 = tid;
    loop {
        if (idx3 >= k * n) { break; }
        let i = idx3 / n;
        let j = idx3 % n;
        if (i > j && i < k) { R[rOff + idx3] = 0.0; }
        idx3 = idx3 + bs;
    }
}
";

        // ═══════════════════════════════════════════════════════════════════
        // EIGH — WGSL requires static-size workgroup arrays. Cap shared-memory
        // usage to n ≤ 32 for the GPU path (32*32*2 f32 = 8 KB ≈ typical
        // workgroup memory budget). Larger matrices fall back to the CPU.
        // ═══════════════════════════════════════════════════════════════════
        public static string Eigh => @"
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read_write> W : array<f32>;
@group(0) @binding(2) var<storage, read_write> V : array<f32>;
struct P { batchCount: i32, n: i32 };
@group(0) @binding(3) var<uniform> p : P;

var<workgroup> Ash : array<f32, 1024>;
var<workgroup> Vsh : array<f32, 1024>;
var<workgroup> c_sh : f32;
var<workgroup> s_sh : f32;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) li : vec3<u32>)
{
    let b = i32(wg.x);
    if (b >= p.batchCount) { return; }
    let tid = i32(li.x);
    let bs = 256;
    let n = p.n;
    let eps : f32 = 1e-7;
    let maxSweeps : i32 = 64;
    let matOff = b * n * n;
    let wOff = b * n;

    var idx0 : i32 = tid;
    loop {
        if (idx0 >= n * n) { break; }
        let i = idx0 / n;
        let j = idx0 % n;
        Ash[idx0] = select(A[matOff + j * n + i], A[matOff + i * n + j], i <= j);
        Vsh[idx0] = select(0.0, 1.0, i == j);
        idx0 = idx0 + bs;
    }
    workgroupBarrier();

    for (var sweep : i32 = 0; sweep < maxSweeps; sweep = sweep + 1) {
        var offSum : f32 = 0.0;
        for (var pp : i32 = 0; pp < n; pp = pp + 1) {
            for (var qq : i32 = pp + 1; qq < n; qq = qq + 1) {
                offSum = offSum + Ash[pp * n + qq] * Ash[pp * n + qq];
            }
        }
        if (sqrt(offSum) < eps) { break; }

        for (var pp : i32 = 0; pp < n - 1; pp = pp + 1) {
            for (var qq : i32 = pp + 1; qq < n; qq = qq + 1) {
                if (tid == 0) {
                    let apq : f32 = Ash[pp * n + qq];
                    if (abs(apq) < eps) { c_sh = 1.0; s_sh = 0.0; }
                    else {
                        let app : f32 = Ash[pp * n + pp];
                        let aqq : f32 = Ash[qq * n + qq];
                        let theta : f32 = (aqq - app) / (2.0 * apq);
                        var t : f32;
                        if (theta == 0.0) { t = 1.0; }
                        else { t = sign(theta) / (abs(theta) + sqrt(1.0 + theta * theta)); }
                        c_sh = 1.0 / sqrt(1.0 + t * t);
                        s_sh = t * c_sh;
                    }
                }
                workgroupBarrier();
                if (s_sh == 0.0) { continue; }
                let cc : f32 = c_sh;
                let ss : f32 = s_sh;

                var i1 : i32 = tid;
                loop {
                    if (i1 >= n) { break; }
                    let aip : f32 = Ash[i1 * n + pp];
                    let aiq : f32 = Ash[i1 * n + qq];
                    Ash[i1 * n + pp] = cc * aip - ss * aiq;
                    Ash[i1 * n + qq] = ss * aip + cc * aiq;
                    i1 = i1 + bs;
                }
                workgroupBarrier();
                var jj : i32 = tid;
                loop {
                    if (jj >= n) { break; }
                    let apj : f32 = Ash[pp * n + jj];
                    let aqj : f32 = Ash[qq * n + jj];
                    Ash[pp * n + jj] = cc * apj - ss * aqj;
                    Ash[qq * n + jj] = ss * apj + cc * aqj;
                    jj = jj + bs;
                }
                workgroupBarrier();
                var i2 : i32 = tid;
                loop {
                    if (i2 >= n) { break; }
                    let vip : f32 = Vsh[i2 * n + pp];
                    let viq : f32 = Vsh[i2 * n + qq];
                    Vsh[i2 * n + pp] = cc * vip - ss * viq;
                    Vsh[i2 * n + qq] = ss * vip + cc * viq;
                    i2 = i2 + bs;
                }
                workgroupBarrier();
            }
        }
    }

    if (tid == 0) {
        for (var i : i32 = 0; i < n; i = i + 1) { W[wOff + i] = Ash[i * n + i]; }
        for (var i : i32 = 0; i < n - 1; i = i + 1) {
            var minIdx : i32 = i;
            for (var j : i32 = i + 1; j < n; j = j + 1) {
                if (W[wOff + j] < W[wOff + minIdx]) { minIdx = j; }
            }
            if (minIdx != i) {
                let tmp : f32 = W[wOff + i];
                W[wOff + i] = W[wOff + minIdx];
                W[wOff + minIdx] = tmp;
                for (var r : i32 = 0; r < n; r = r + 1) {
                    let vv : f32 = Vsh[r * n + i];
                    Vsh[r * n + i] = Vsh[r * n + minIdx];
                    Vsh[r * n + minIdx] = vv;
                }
            }
        }
    }
    workgroupBarrier();
    var idx4 : i32 = tid;
    loop {
        if (idx4 >= n * n) { break; }
        V[matOff + idx4] = Vsh[idx4];
        idx4 = idx4 + bs;
    }
}
";

        public static (string Name, string Source)[] All => new[]
        {
            ("parity211_cholesky", Cholesky),
            ("parity211_lu_factor", LuFactor),
            ("parity211_qr_reduced", QrReduced),
            ("parity211_eigh", Eigh),
        };
    }
}
