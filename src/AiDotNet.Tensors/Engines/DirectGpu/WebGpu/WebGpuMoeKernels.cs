// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of MoeKernelTemplate, restated operation for operation: each kernel its own module, the storage
// buffers in the template's binding order and the ints in a uniform struct after them. Values read from storage that
// bound a barrier loop pass through workgroupUniformLoad (WGSL's uniformity analysis treats storage reads as varying).
internal static class WebGpuMoeKernels
{
    internal const int UniformWords = 8;

    private static string Header(string[] buffers, int[] written, string[] ints, int sharedFloats)
    {
        var sb = new StringBuilder("struct Params { ");
        for (int i = 0; i < UniformWords; i++) sb.Append(i < ints.Length ? ints[i] : "pad" + i).Append(": i32, ");
        sb.Append("}\n");
        for (int i = 0; i < buffers.Length; i++)
            sb.Append("@group(0) @binding(").Append(i).Append(") var<storage, ")
                .Append(System.Array.IndexOf(written, i) >= 0 ? "read_write" : "read").Append("> ")
                .Append(buffers[i]).Append(": array<f32>;\n");
        sb.Append("@group(0) @binding(").Append(buffers.Length).Append(") var<uniform> pm: Params;\n");
        if (sharedFloats > 0) sb.Append("var<workgroup> sm: array<f32, ").Append(sharedFloats).Append(">;\n");
        return sb.ToString();
    }

    private const string Entry = @"
@compute @workgroup_size(256)
fn NAME(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(global_invocation_id) gidv: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = i32(lidx);
    let grp = i32(wid.x);
    let gid = i32(gidv.x);
";

    private static readonly string[] GemmInts = { "M", "N", "Kd", "G", "tilesN", "tileBase", "TOPK", "F" };
    private static readonly string[] WgradInts = { "P", "Nn", "G", "tilesP", "tilesN", "tileBase", "TOPK", "F" };

    // loadA/loadB/store use arow/acol, brow/bcol, g, row/col/v and pm.*; F2 is 2 * pm.F.
    private static string GroupedRows(string loadA, string loadB, string store, bool transposedB) => @"
var<workgroup> wgStart: i32;
var<workgroup> wgEnd: i32;
fn first_group_past(row: i32) -> i32 {
    var lo = 0;
    var hi = pm.G;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2;
        if (i32(ends[mid]) > row) { hi = mid; } else { lo = mid + 1; }
    }
    return lo;
}
" + Entry + @"
    let F2 = 2 * pm.F;
    let t = grp + pm.tileBase;
    let rt = t / pm.tilesN;
    let ct = t - rt * pm.tilesN;
    let row0 = rt * 64;
    let col0 = ct * 64;
    let rowHi = min(row0 + 64, pm.M);
    let tr = tid / 16;
    let tc = tid - tr * 16;
    var acc: array<f32, 16>;
    if (tid == 0) { wgStart = first_group_past(row0); }
    var g = workgroupUniformLoad(&wgStart);
    loop {
        if (g >= pm.G) { break; }
        if (tid == 0) {
            var s0 = 0;
            if (g > 0) { s0 = i32(ends[g - 1]); }
            wgStart = s0;
            wgEnd = i32(ends[g]);
        }
        let gs = workgroupUniformLoad(&wgStart);
        let ge = workgroupUniformLoad(&wgEnd);
        if (gs >= rowHi) { break; }
        let r0 = max(gs, row0);
        let r1 = min(ge, rowHi);
        if (r1 > r0) {
            for (var k0 = 0; k0 < pm.Kd; k0 = k0 + 16) {
                for (var q = 0; q < 4; q = q + 1) {
                    let idx = tid + q * 256;
                    let ar = idx / 16;
                    let ak = idx - ar * 16;
                    let arow = row0 + ar;
                    let acol = k0 + ak;
                    var av = 0.0;
                    if (arow >= r0 && arow < r1 && acol < pm.Kd) { av = " + loadA + @"; }
                    sm[idx] = av;
                    " + (transposedB ? "let bc = idx / 16; let bk = idx - bc * 16;" : "let bk = idx / 64; let bc = idx - bk * 64;") + @"
                    let bcol = col0 + bc;
                    let brow = k0 + bk;
                    var bv = 0.0;
                    if (bcol < pm.N && brow < pm.Kd) { bv = " + loadB + @"; }
                    sm[1024 + bk * 64 + bc] = bv;
                }
                workgroupBarrier();
                for (var kk = 0; kk < 16; kk = kk + 1) {
                    for (var i = 0; i < 4; i = i + 1) {
                        let orow = row0 + tr * 4 + i;
                        if (orow >= r0 && orow < r1) {
                            let a = sm[(tr * 4 + i) * 16 + kk];
                            for (var j = 0; j < 4; j = j + 1) {
                                acc[i * 4 + j] = fma(a, sm[1024 + kk * 64 + tc * 4 + j], acc[i * 4 + j]);
                            }
                        }
                    }
                }
                workgroupBarrier();
            }
        }
        g = g + 1;
    }
    for (var i = 0; i < 4; i = i + 1) {
        let row = row0 + tr * 4 + i;
        for (var j = 0; j < 4; j = j + 1) {
            let col = col0 + tc * 4 + j;
            if (row < pm.M && col < pm.N) {
                let v = acc[i * 4 + j];
                " + store + @"
            }
        }
    }
}
";

    private static string Wgrad(string loadA, string loadC, string store) => @"
var<workgroup> wgStart: i32;
var<workgroup> wgEnd: i32;
" + Entry + @"
    let F2 = 2 * pm.F;
    let t = grp + pm.tileBase;
    let per = pm.tilesP * pm.tilesN;
    let g = t / per;
    let rem = t - g * per;
    let pt = rem / pm.tilesN;
    let nt = rem - pt * pm.tilesN;
    let p0 = pt * 64;
    let n0 = nt * 64;
    if (tid == 0) {
        var s0 = 0;
        if (g > 0) { s0 = i32(ends[g - 1]); }
        wgStart = s0;
        wgEnd = i32(ends[g]);
    }
    let gs = workgroupUniformLoad(&wgStart);
    let ge = workgroupUniformLoad(&wgEnd);
    let tr = tid / 16;
    let tc = tid - tr * 16;
    var acc: array<f32, 16>;
    for (var r0 = gs; r0 < ge; r0 = r0 + 16) {
        for (var q = 0; q < 4; q = q + 1) {
            let idx = tid + q * 256;
            let rr = idx / 64;
            let cc = idx - rr * 64;
            let row = r0 + rr;
            let pa = p0 + cc;
            let nc = n0 + cc;
            var av = 0.0;
            if (row < ge && pa < pm.P) { av = " + loadA + @"; }
            sm[idx] = av;
            var cv = 0.0;
            if (row < ge && nc < pm.Nn) { cv = " + loadC + @"; }
            sm[1024 + idx] = cv;
        }
        workgroupBarrier();
        for (var rr = 0; rr < 16; rr = rr + 1) {
            for (var i = 0; i < 4; i = i + 1) {
                let a = sm[rr * 64 + tr * 4 + i];
                for (var j = 0; j < 4; j = j + 1) {
                    acc[i * 4 + j] = fma(a, sm[1024 + rr * 64 + tc * 4 + j], acc[i * 4 + j]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0; i < 4; i = i + 1) {
        let p = p0 + tr * 4 + i;
        for (var j = 0; j < 4; j = j + 1) {
            let n = n0 + tc * 4 + j;
            if (p < pm.P && n < pm.Nn) {
                let v = acc[i * 4 + j];
                " + store + @"
            }
        }
    }
}
";

    private const string RouteBody = Entry + @"
    if (gid < pm.N - pm.baseOffset) {
        let t = pm.baseOffset + gid;
        var bv: array<f32, 16>;
        var bi: array<i32, 16>;
        for (var j = 0; j < 16; j = j + 1) { bv[j] = -3.0e38; bi[j] = 0; }
        for (var e = 0; e < pm.G; e = e + 1) {
            var v = scores[t * pm.G + e];
            if (pm.biasMode != 0) { v = v + bias[e]; }
            if (v > bv[pm.TOPK - 1]) {
                var p = pm.TOPK - 1;
                loop {
                    if (p <= 0 || bv[p - 1] >= v) { break; }
                    bv[p] = bv[p - 1];
                    bi[p] = bi[p - 1];
                    p = p - 1;
                }
                bv[p] = v;
                bi[p] = e;
            }
        }
        var s = 0.0;
        for (var j = 0; j < pm.TOPK; j = j + 1) { s = s + scores[t * pm.G + bi[j]]; }
        for (var j = 0; j < pm.TOPK; j = j + 1) {
            ids[t * pm.TOPK + j] = f32(bi[j]);
            gates[t * pm.TOPK + j] = scores[t * pm.G + bi[j]] / s;
        }
        ssum[t] = s;
    }
}
";

    private const string SortBody = Entry + @"
    let e = grp + pm.groupBase;
    if (pm.phase != 2) {
        var less = 0.0;
        var mine = 0.0;
        for (var b0 = pm.slotStart; b0 < pm.slotEnd; b0 = b0 + 256) {
            let s = b0 + tid;
            if (s < pm.slotEnd) {
                let v = i32(ids[s]);
                if (v < e) { less = less + 1.0; }
                if (v == e) { mine = mine + 1.0; }
            }
        }
        sm[tid] = less;
        sm[256 + tid] = mine;
        workgroupBarrier();
        for (var o = 128; o > 0; o = o / 2) {
            if (tid < o) { sm[tid] = sm[tid] + sm[tid + o]; sm[256 + tid] = sm[256 + tid] + sm[256 + tid + o]; }
            workgroupBarrier();
        }
        if (pm.phase == 3) {
            let start = sm[0];
            let count = sm[256];
            workgroupBarrier();
            if (tid == 0) { ends[e] = start + count; }
            var running = start;
            for (var b0 = pm.slotStart; b0 < pm.slotEnd; b0 = b0 + 256) {
                let s = b0 + tid;
                var flag = 0.0;
                if (s < pm.slotEnd) { if (i32(ids[s]) == e) { flag = 1.0; } }
                sm[tid] = flag;
                workgroupBarrier();
                for (var off = 1; off < 256; off = off * 2) {
                    var add = 0.0;
                    if (tid >= off) { add = sm[tid - off]; }
                    workgroupBarrier();
                    sm[tid] = sm[tid] + add;
                    workgroupBarrier();
                }
                if (flag != 0.0) {
                    let p = running + sm[tid] - 1.0;
                    perm[i32(p)] = f32(s);
                    pos[s] = p;
                }
                running = running + sm[255];
                workgroupBarrier();
            }
        } else {
            if (tid == 0) {
                if (pm.phase == 0) { ends[e] = sm[0]; } else { ends[e] = ends[e] + sm[0]; }
            }
            workgroupBarrier();
        }
        if (pm.SPS <= 256) {
            let bFirst = pm.slotStart / pm.SPS;
            let bEnd = pm.slotEnd / pm.SPS;
            for (var b = bFirst + tid; b < bEnd; b = b + 256) {
                var c = 0.0;
                let bs = b * pm.SPS;
                let be = bs + pm.SPS;
                for (var s = bs; s < be; s = s + 1) {
                    if (i32(ids[s]) == e) { c = c + 1.0; }
                }
                loads[b * pm.G + e] = c;
            }
        } else {
            let bFirst = pm.slotStart / pm.SPS;
            let bLast = (pm.slotEnd - 1) / pm.SPS;
            for (var b = bFirst; b <= bLast; b = b + 1) {
                let bs = b * pm.SPS;
                let be = (b + 1) * pm.SPS;
                let segStart = max(bs, pm.slotStart);
                let segEnd = min(be, pm.slotEnd);
                var c = 0.0;
                for (var s = segStart + tid; s < segEnd; s = s + 256) {
                    if (i32(ids[s]) == e) { c = c + 1.0; }
                }
                sm[tid] = c;
                workgroupBarrier();
                for (var o = 128; o > 0; o = o / 2) {
                    if (tid < o) { sm[tid] = sm[tid] + sm[tid + o]; }
                    workgroupBarrier();
                }
                if (tid == 0) {
                    if (segStart == bs) { loads[b * pm.G + e] = sm[0]; } else { loads[b * pm.G + e] = loads[b * pm.G + e] + sm[0]; }
                }
                workgroupBarrier();
            }
        }
    } else {
        if (tid == 0) { sm[0] = ends[e]; }
        workgroupBarrier();
        var running = sm[0];
        workgroupBarrier();
        for (var b0 = pm.slotStart; b0 < pm.slotEnd; b0 = b0 + 256) {
            let s = b0 + tid;
            var flag = 0.0;
            if (s < pm.slotEnd) { if (i32(ids[s]) == e) { flag = 1.0; } }
            sm[tid] = flag;
            workgroupBarrier();
            for (var off = 1; off < 256; off = off * 2) {
                var add = 0.0;
                if (tid >= off) { add = sm[tid - off]; }
                workgroupBarrier();
                sm[tid] = sm[tid] + add;
                workgroupBarrier();
            }
            if (flag != 0.0) {
                let p = running + sm[tid] - 1.0;
                perm[i32(p)] = f32(s);
                pos[s] = p;
            }
            running = running + sm[255];
            workgroupBarrier();
        }
        if (tid == 0) { ends[e] = running; }
    }
}
";

    private const string CombineBody = Entry + @"
    let total = pm.N * pm.D;
    if (gid < total - pm.baseOffset) {
        let i = pm.baseOffset + gid;
        let t = i / pm.D;
        let c = i - t * pm.D;
        var acc = 0.0;
        for (var j = 0; j < pm.TOPK; j = j + 1) {
            let sl = t * pm.TOPK + j;
            var w = 1.0;
            if (pm.useGates != 0) { w = gates[sl]; }
            acc = fma(w, src[i32(pos[sl]) * pm.D + c], acc);
        }
        y[i] = acc;
    }
}
";

    private const string DScoresBody = Entry + @"
    if (gid < pm.N - pm.baseOffset) {
        let t = pm.baseOffset + gid;
        var dg: array<f32, 16>;
        var dot = 0.0;
        for (var j = 0; j < pm.TOPK; j = j + 1) {
            let sl = t * pm.TOPK + j;
            let p = i32(pos[sl]);
            var a = 0.0;
            for (var c = 0; c < pm.D; c = c + 1) { a = fma(o[p * pm.D + c], dy[t * pm.D + c], a); }
            dg[j] = a;
            dot = fma(a, gates[sl], dot);
        }
        for (var e = 0; e < pm.G; e = e + 1) { ds[t * pm.G + e] = 0.0; }
        for (var j = 0; j < pm.TOPK; j = j + 1) {
            ds[t * pm.G + i32(ids[t * pm.TOPK + j])] = (dg[j] - dot) / ssum[t];
        }
    }
}
";

    private const string ActA = "(h13[arow * F2 + acol] / (1.0 + exp(-h13[arow * F2 + acol]))) * h13[arow * F2 + pm.F + acol]";

    private static readonly Dictionary<string, string> Sources = Build();

    private static Dictionary<string, string> Build()
    {
        var d = new Dictionary<string, string>(System.StringComparer.Ordinal);
        void Add(string name, string body)
        {
            foreach (var spec in MoeKernelTemplate.Kernels)
                if (spec.Name == name)
                {
                    int shared = spec.SharedFloats;
                    d[name] = Header(spec.Buffers, spec.Written, spec.Ints, shared) + body.Replace("fn NAME", "fn " + name);
                    return;
                }
            throw new System.ArgumentException(name);
        }
        Add(MoeKernelTemplate.Route, RouteBody);
        Add(MoeKernelTemplate.Sort, SortBody);
        Add(MoeKernelTemplate.Up, GroupedRows(
            "xin[(i32(perm[arow]) / pm.TOPK) * pm.Kd + acol]",
            "select(w3[(g * pm.Kd + brow) * pm.F + max(bcol - pm.F, 0)], w1[(g * pm.Kd + brow) * pm.F + min(bcol, pm.F - 1)], bcol < pm.F)",
            "h13[row * F2 + col] = v;", false));
        Add(MoeKernelTemplate.Down, GroupedRows(ActA, "w2[(g * pm.Kd + brow) * pm.N + bcol]", "o[row * pm.N + col] = v;", false));
        Add(MoeKernelTemplate.Combine, CombineBody);
        Add(MoeKernelTemplate.DScores, DScoresBody);
        Add(MoeKernelTemplate.DAct, GroupedRows(
            "gates[i32(perm[arow])] * dy[(i32(perm[arow]) / pm.TOPK) * pm.Kd + acol]",
            "w2[(g * pm.N + bcol) * pm.Kd + brow]",
            "let h = h13[row * F2 + col]; let u = h13[row * F2 + pm.F + col]; let sg = 1.0 / (1.0 + exp(-h)); "
            + "dh13[row * F2 + col] = v * u * sg * (1.0 + h * (1.0 - sg)); dh13[row * F2 + pm.F + col] = v * h * sg;", true));
        Add(MoeKernelTemplate.DX, GroupedRows(
            "dh13[arow * pm.Kd + acol]",
            "select(w3[(g * pm.N + bcol) * pm.F + max(brow - pm.F, 0)], w1[(g * pm.N + bcol) * pm.F + min(brow, pm.F - 1)], brow < pm.F)",
            "dxs[row * pm.N + col] = v;", true));
        Add(MoeKernelTemplate.WUp, Wgrad(
            "xin[(i32(perm[row]) / pm.TOPK) * pm.P + pa]",
            "dh13[row * pm.Nn + nc]",
            "if (n < pm.F) { dw1[(g * pm.P + p) * pm.F + n] = v; } else { dw3[(g * pm.P + p) * pm.F + n - pm.F] = v; }"));
        Add(MoeKernelTemplate.WDown, Wgrad(
            "(h13[row * F2 + pa] / (1.0 + exp(-h13[row * F2 + pa]))) * h13[row * F2 + pm.F + pa]",
            "gates[i32(perm[row])] * dy[(i32(perm[row]) / pm.TOPK) * pm.Nn + nc]",
            "dw2[(g * pm.P + p) * pm.Nn + n] = v;"));
        return d;
    }

    internal static string SourceFor(string name) => Sources[name];
}
