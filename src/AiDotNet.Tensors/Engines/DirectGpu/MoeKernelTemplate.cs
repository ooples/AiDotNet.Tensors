// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The fused SwiGLU mixture-of-experts layer (MixtureOfExperts.SwiGluExperts), written once in WinogradF23KernelTemplate's
// body conventions and emitted as OpenCL C, HIP/CUDA C++, Vulkan GLSL and Metal Shading Language. WGSL keeps its own
// restatement (WebGpu/WebGpuMoeKernels.cs).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The kernels of the fused mixture-of-experts layer. Every index rides in a float buffer (exact below 2^24); a slot is
/// <c>token * topK + j</c>, the j-th expert a token picked; rows of the expert-sorted layout are positions
/// <c>0 .. N*topK-1</c>, grouped by expert through <c>ends</c> (inclusive prefix counts, written by <see cref="Sort"/>).
/// <list type="bullet">
/// <item><see cref="Route"/> (a thread per token): top-k of <c>scores + bias</c>, gates = the picked scores over their sum.</item>
/// <item><see cref="Sort"/> (a group per expert): a stable counting sort of the slots by expert, every expert's offsets
/// and its per-sequence load; deterministic, no atomics.</item>
/// <item><see cref="Up"/>: <c>h13[r] = x[token(r)] [w1 | w3][expert(r)]</c>, the input rows read through the sort (no
/// gathered copy of x).</item>
/// <item><see cref="Down"/>: <c>o[r] = (silu(h1) * h3)[r] w2[expert(r)]</c>, the activation formed as the operand
/// loads (never stored).</item>
/// <item><see cref="Combine"/> (a thread per output): <c>y[t] = sum_j gate[t, j] o[pos(t, j)]</c> in slot order:
/// deterministic, no scatter-add.</item>
/// <item><see cref="DScores"/>, <see cref="DAct"/>, <see cref="DX"/>, <see cref="WUp"/>, <see cref="WDown"/>: the
/// backward (the gate gradient and the scores' gradient through the gate normalization; the SwiGLU derivative in the
/// down-projection's input-gradient epilogue; the input gradient, then <see cref="Combine"/> without gates; the weight
/// gradients reading their operands through the sort).</item>
/// </list>
/// The GEMM kernels own a 64 x 64 output tile per 256-thread group (4 x 4 values per thread, operands staged 16 deep);
/// a row tile that straddles experts runs once per expert it meets, each thread accumulating only its own expert's rows.
/// </summary>
internal static class MoeKernelTemplate
{
    internal const int GroupSize = 256;
    internal const int Tile = 64;

    /// <summary>The largest top-k the route and score-gradient kernels hold in registers.</summary>
    internal const int MaxTopK = 16;

    internal const string Route = "moe_route";
    internal const string Sort = "moe_sort";
    internal const string Up = "moe_up";
    internal const string Down = "moe_down";
    internal const string Combine = "moe_combine";
    internal const string DScores = "moe_dscores";
    internal const string DAct = "moe_dact";
    internal const string DX = "moe_dx";
    internal const string WUp = "moe_wup";
    internal const string WDown = "moe_wdown";

    // GEMM over expert-grouped rows: rows [0, M), columns [0, N), reduction [0, Kd). loadA(arow, acol), loadB(brow, bcol)
    // (b's expert is g) and store(row, col, v) are expressions/statements in the body conventions; transposedB stages b
    // so that consecutive threads read consecutive reduction indices (b stored [.., n, k]).
    private static string GroupedRowsBody(string loadA, string loadB, string store, bool transposedB) => @"
    int tid = lid;
    int t = grp + tileBase;
    int rt = t / tilesN;
    int ct = t - rt * tilesN;
    int row0 = rt * 64, col0 = ct * 64;
    int rowHi = row0 + 64 < M ? row0 + 64 : M;
    int tr = tid / 16, tc = tid - (tid / 16) * 16;
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    int lo = 0, hi = G;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (TOINT(ends[mid]) > row0) hi = mid; else lo = mid + 1;
    }
    int g = lo;
    while (g < G)
    {
        int gs = g == 0 ? 0 : TOINT(ends[g - 1]);
        if (gs >= rowHi) break;
        int ge = TOINT(ends[g]);
        int r0 = gs > row0 ? gs : row0;
        int r1 = ge < rowHi ? ge : rowHi;
        if (r1 > r0)
        {
            for (int k0 = 0; k0 < Kd; k0 += 16)
            {
                for (int q = 0; q < 4; q++)
                {
                    int idx = tid + q * 256;
                    int ar = idx / 16;
                    int ak = idx - ar * 16;
                    int arow = row0 + ar, acol = k0 + ak;
                    sm[idx] = (arow >= r0 && arow < r1 && acol < Kd) ? (" + loadA + @") : 0.0f;
                    int bc = 0, bk = 0;
                    " + (transposedB ? "bc = idx / 16; bk = idx - bc * 16;" : "bk = idx / 64; bc = idx - bk * 64;") + @"
                    int bcol = col0 + bc, brow = k0 + bk;
                    sm[1024 + bk * 64 + bc] = (bcol < N && brow < Kd) ? (" + loadB + @") : 0.0f;
                }
                BARRIER();
                for (int kk = 0; kk < 16; kk++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        int orow = row0 + tr * 4 + i;
                        if (orow < r0 || orow >= r1) continue;
                        float av = sm[(tr * 4 + i) * 16 + kk];
                        for (int j = 0; j < 4; j++)
                            acc[i * 4 + j] = FMA(av, sm[1024 + kk * 64 + tc * 4 + j], acc[i * 4 + j]);
                    }
                }
                BARRIER();
            }
        }
        g++;
    }
    for (int i = 0; i < 4; i++)
    {
        int row = row0 + tr * 4 + i;
        for (int j = 0; j < 4; j++)
        {
            int col = col0 + tc * 4 + j;
            if (row < M && col < N)
            {
                float v = acc[i * 4 + j];
                " + store + @"
            }
        }
    }
";

    // Per-expert weight gradient W[g][p][n] = sum over g's rows of A(row, p) C(row, n): P x Nn per expert, tiles of 64.
    private static string WeightGradBody(string loadA, string loadC, string store) => @"
    int tid = lid;
    int t = grp + tileBase;
    int per = tilesP * tilesN;
    int g = t / per;
    int rem = t - g * per;
    int pt = rem / tilesN;
    int nt = rem - pt * tilesN;
    int p0 = pt * 64, n0 = nt * 64;
    int gs = g == 0 ? 0 : TOINT(ends[g - 1]);
    int ge = TOINT(ends[g]);
    int tr = tid / 16, tc = tid - (tid / 16) * 16;
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (int r0 = gs; r0 < ge; r0 += 16)
    {
        for (int q = 0; q < 4; q++)
        {
            int idx = tid + q * 256;
            int rr = idx / 64;
            int cc = idx - rr * 64;
            int row = r0 + rr;
            int pa = p0 + cc, nc = n0 + cc;
            sm[idx] = (row < ge && pa < P) ? (" + loadA + @") : 0.0f;
            sm[1024 + idx] = (row < ge && nc < Nn) ? (" + loadC + @") : 0.0f;
        }
        BARRIER();
        for (int rr = 0; rr < 16; rr++)
        {
            for (int i = 0; i < 4; i++)
            {
                float av = sm[rr * 64 + tr * 4 + i];
                for (int j = 0; j < 4; j++)
                    acc[i * 4 + j] = FMA(av, sm[1024 + rr * 64 + tc * 4 + j], acc[i * 4 + j]);
            }
        }
        BARRIER();
    }
    for (int i = 0; i < 4; i++)
    {
        int p = p0 + tr * 4 + i;
        for (int j = 0; j < 4; j++)
        {
            int n = n0 + tc * 4 + j;
            if (p < P && n < Nn)
            {
                float v = acc[i * 4 + j];
                " + store + @"
            }
        }
    }
";

    // The SwiGLU activation of row r, column c, from the stored pre-activations h13 = [h1 | h3] ([rows, 2F]).
    private const string Act = "(h13[arow * F2 + acol] / (1.0f + EXP(-h13[arow * F2 + acol]))) * h13[arow * F2 + F + acol]";

    private const string RouteBody = @"
    if (gid < N - baseOffset)
    {
        int t = baseOffset + gid;
        float bv[16];
        int bi[16];
        for (int j = 0; j < 16; j++) { bv[j] = -3.0e38f; bi[j] = 0; }
        for (int e = 0; e < G; e++)
        {
            float v = scores[t * G + e] + (biasMode != 0 ? bias[e] : 0.0f);
            if (v > bv[TOPK - 1])
            {
                int p = TOPK - 1;
                while (p > 0 && bv[p - 1] < v) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; p--; }
                bv[p] = v;
                bi[p] = e;
            }
        }
        float s = 0.0f;
        for (int j = 0; j < TOPK; j++) s += scores[t * G + bi[j]];
        for (int j = 0; j < TOPK; j++)
        {
            ids[t * TOPK + j] = TOFLOAT(bi[j]);
            gates[t * TOPK + j] = scores[t * G + bi[j]] / s;
        }
        ssum[t] = s;
    }
";

    // One group per expert e: bounded slices over [slotStart, slotEnd) so no launch exceeds WDDM/GPU timeout budgets.
    // phase == 3 runs single-slice count + scatter + sequence loads (NK <= 65,536).
    // Multi-slice runs Pass 1 (phase == 0 initializes ends[e] = start_slice, phase == 1 accumulates start_slice, and
    // both compute overlapping per-sequence loads) followed by Pass 2 (phase == 2 scatters e's slots starting at ends[e]
    // and advances ends[e] to start + totalCount after the final slice).
    private const string SortBody = @"
    int e = grp + groupBase;
    int tid = lid;
    if (phase != 2)
    {
        float less = 0.0f, mine = 0.0f;
        for (int b0 = slotStart; b0 < slotEnd; b0 += 256)
        {
            int s = b0 + tid;
            if (s < slotEnd)
            {
                int v = TOINT(ids[s]);
                if (v < e) less += 1.0f;
                if (v == e) mine += 1.0f;
            }
        }
        sm[tid] = less;
        sm[256 + tid] = mine;
        BARRIER();
        for (int o = 128; o > 0; o = o / 2)
        {
            if (tid < o) { sm[tid] += sm[tid + o]; sm[256 + tid] += sm[256 + tid + o]; }
            BARRIER();
        }
        if (phase == 3)
        {
            float start = sm[0];
            float count = sm[256];
            BARRIER();
            if (tid == 0) ends[e] = start + count;
            float running = start;
            for (int b0 = slotStart; b0 < slotEnd; b0 += 256)
            {
                int s = b0 + tid;
                float flag = (s < slotEnd && TOINT(ids[s]) == e) ? 1.0f : 0.0f;
                sm[tid] = flag;
                BARRIER();
                for (int off = 1; off < 256; off = off * 2)
                {
                    float add = tid >= off ? sm[tid - off] : 0.0f;
                    BARRIER();
                    sm[tid] += add;
                    BARRIER();
                }
                if (flag != 0.0f)
                {
                    float p = running + sm[tid] - 1.0f;
                    perm[TOINT(p)] = TOFLOAT(s);
                    pos[s] = p;
                }
                running += sm[255];
                BARRIER();
            }
        }
        else
        {
            if (tid == 0) ends[e] = (phase == 0) ? sm[0] : (ends[e] + sm[0]);
            BARRIER();
        }
        if (SPS <= 256)
        {
            int bFirst = slotStart / SPS;
            int bEnd = slotEnd / SPS;
            for (int b = bFirst + tid; b < bEnd; b += 256)
            {
                float c = 0.0f;
                int bs = b * SPS;
                int be = bs + SPS;
                for (int s = bs; s < be; s++)
                    if (TOINT(ids[s]) == e) c += 1.0f;
                loads[b * G + e] = c;
            }
        }
        else
        {
            int bFirst = slotStart / SPS;
            int bLast = (slotEnd - 1) / SPS;
            for (int b = bFirst; b <= bLast; b++)
            {
                int bs = b * SPS;
                int be = (b + 1) * SPS;
                int segStart = bs > slotStart ? bs : slotStart;
                int segEnd = be < slotEnd ? be : slotEnd;
                float c = 0.0f;
                for (int s = segStart + tid; s < segEnd; s += 256)
                    if (TOINT(ids[s]) == e) c += 1.0f;
                sm[tid] = c;
                BARRIER();
                for (int o = 128; o > 0; o = o / 2)
                {
                    if (tid < o) sm[tid] += sm[tid + o];
                    BARRIER();
                }
                if (tid == 0) loads[b * G + e] = (segStart == bs) ? sm[0] : (loads[b * G + e] + sm[0]);
                BARRIER();
            }
        }
    }
    else
    {
        if (tid == 0) sm[0] = ends[e];
        BARRIER();
        float running = sm[0];
        BARRIER();
        for (int b0 = slotStart; b0 < slotEnd; b0 += 256)
        {
            int s = b0 + tid;
            float flag = (s < slotEnd && TOINT(ids[s]) == e) ? 1.0f : 0.0f;
            sm[tid] = flag;
            BARRIER();
            for (int off = 1; off < 256; off = off * 2)
            {
                float add = tid >= off ? sm[tid - off] : 0.0f;
                BARRIER();
                sm[tid] += add;
                BARRIER();
            }
            if (flag != 0.0f)
            {
                float p = running + sm[tid] - 1.0f;
                perm[TOINT(p)] = TOFLOAT(s);
                pos[s] = p;
            }
            running += sm[255];
            BARRIER();
        }
        if (tid == 0) ends[e] = running;
    }
";

    private const string CombineBody = @"
    int total = N * D;
    if (gid < total - baseOffset)
    {
        int i = baseOffset + gid;
        int t = i / D;
        int c = i - t * D;
        float acc = 0.0f;
        for (int j = 0; j < TOPK; j++)
        {
            int sl = t * TOPK + j;
            float w = useGates != 0 ? gates[sl] : 1.0f;
            acc = FMA(w, src[TOINT(pos[sl]) * D + c], acc);
        }
        y[i] = acc;
    }
";

    // d gates[t, j] = o[pos(t, j)] . dy[t]; gates = sel / sum(sel), so d sel_j = (d gate_j - sum_i d gate_i gate_i) / sum.
    private const string DScoresBody = @"
    if (gid < N - baseOffset)
    {
        int t = baseOffset + gid;
        float dg[16];
        float dot = 0.0f;
        for (int j = 0; j < TOPK; j++)
        {
            int sl = t * TOPK + j;
            int p = TOINT(pos[sl]);
            float a = 0.0f;
            for (int c = 0; c < D; c++) a = FMA(o[p * D + c], dy[t * D + c], a);
            dg[j] = a;
            dot = FMA(a, gates[sl], dot);
        }
        for (int e = 0; e < G; e++) ds[t * G + e] = 0.0f;
        for (int j = 0; j < TOPK; j++)
            ds[t * G + TOINT(ids[t * TOPK + j])] = (dg[j] - dot) / ssum[t];
    }
";

    private static readonly string[] GemmInts = { "M", "N", "Kd", "G", "tilesN", "tileBase", "TOPK", "F" };
    private static readonly string[] WgradInts = { "P", "Nn", "G", "tilesP", "tilesN", "tileBase", "TOPK", "F" };

    // h13 holds 2F floats per row; F2 is spelled out so every kernel keeps the same int list.
    private static string WithF2(string body) => body.Replace("F2", "(2 * F)");

    internal static readonly WinogradF23KernelTemplate.Spec[] Kernels =
    {
        new(Route, new[] { "scores", "bias", "ids", "gates", "ssum" }, new[] { "N", "G", "TOPK", "biasMode", "baseOffset" },
            RouteBody, 0, new[] { 2, 3, 4 }),
        new(Sort, new[] { "ids", "perm", "pos", "ends", "loads" },
            new[] { "NK", "G", "B", "SPS", "slotStart", "slotEnd", "phase", "groupBase" },
            SortBody, 512, new[] { 1, 2, 3, 4 }),
        new(Up, new[] { "xin", "perm", "w1", "w3", "ends", "h13" }, GemmInts, WithF2(GroupedRowsBody(
            "xin[(TOINT(perm[arow]) / TOPK) * Kd + acol]",
            "bcol < F ? w1[(g * Kd + brow) * F + bcol] : w3[(g * Kd + brow) * F + bcol - F]",
            "h13[row * F2 + col] = v;", transposedB: false)), 2048, new[] { 5 }),
        new(Down, new[] { "h13", "w2", "ends", "o" }, GemmInts, WithF2(GroupedRowsBody(
            Act, "w2[(g * Kd + brow) * N + bcol]", "o[row * N + col] = v;", transposedB: false)), 2048, new[] { 3 }),
        new(Combine, new[] { "src", "pos", "gates", "y" }, new[] { "N", "D", "TOPK", "useGates", "baseOffset" },
            CombineBody, 0, new[] { 3 }),
        new(DScores, new[] { "o", "dy", "pos", "gates", "ssum", "ids", "ds" }, new[] { "N", "D", "G", "TOPK", "baseOffset" },
            DScoresBody, 0, new[] { 6 }),
        new(DAct, new[] { "dy", "perm", "gates", "w2", "ends", "h13", "dh13" }, GemmInts, WithF2(GroupedRowsBody(
            "gates[TOINT(perm[arow])] * dy[(TOINT(perm[arow]) / TOPK) * Kd + acol]",
            "w2[(g * N + bcol) * Kd + brow]",
            "float h = h13[row * F2 + col]; float u = h13[row * F2 + F + col]; float sg = 1.0f / (1.0f + EXP(-h)); "
            + "dh13[row * F2 + col] = v * u * sg * (1.0f + h * (1.0f - sg)); dh13[row * F2 + F + col] = v * h * sg;",
            transposedB: true)), 2048, new[] { 6 }),
        new(DX, new[] { "dh13", "w1", "w3", "ends", "dxs" }, GemmInts, WithF2(GroupedRowsBody(
            "dh13[arow * Kd + acol]",
            "brow < F ? w1[(g * N + bcol) * F + brow] : w3[(g * N + bcol) * F + brow - F]",
            "dxs[row * N + col] = v;", transposedB: true)), 2048, new[] { 4 }),
        new(WUp, new[] { "xin", "perm", "dh13", "ends", "dw1", "dw3" }, WgradInts, WithF2(WeightGradBody(
            "xin[(TOINT(perm[row]) / TOPK) * P + pa]",
            "dh13[row * Nn + nc]",
            "if (n < F) dw1[(g * P + p) * F + n] = v; else dw3[(g * P + p) * F + n - F] = v;")), 2048, new[] { 4, 5 }),
        new(WDown, new[] { "h13", "perm", "gates", "dy", "ends", "dw2" }, WgradInts, WithF2(WeightGradBody(
            "(h13[row * F2 + pa] / (1.0f + EXP(-h13[row * F2 + pa]))) * h13[row * F2 + F + pa]",
            "gates[TOINT(perm[row])] * dy[(TOINT(perm[row]) / TOPK) * Nn + nc]",
            "dw2[(g * P + p) * Nn + n] = v;")), 2048, new[] { 5 }),
    };

    internal static string[] GetKernelNames()
    {
        var names = new List<string>();
        foreach (var spec in Kernels) names.Add(spec.Name);
        return names.ToArray();
    }

    internal static string GetModuleSource(WinogradKernelDialect dialect) => WinogradF23KernelTemplate.EmitModule(Kernels, dialect);

    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Kernels)
            if (spec.Name == name) return WinogradF23KernelTemplate.EmitGlsl(spec);
        throw new System.ArgumentException($"No MoE kernel named '{name}'.", nameof(name));
    }
}
