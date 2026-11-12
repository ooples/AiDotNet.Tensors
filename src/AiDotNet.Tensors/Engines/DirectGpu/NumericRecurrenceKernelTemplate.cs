// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The batched numeric recurrences of INumericRecurrenceBackend (tridiagonal solve, Fritsch-Carlson limiter, CTC
// lattice, linear sum assignment), written once in WinogradF23KernelTemplate's body conventions and emitted as OpenCL C,
// HIP C++ (which CUDA's NVRTC compiles too), Vulkan GLSL and Metal Shading Language. WGSL keeps its own copy
// (WebGpu/WebGpuNumericRecurrenceKernels.cs).
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The kernels behind <see cref="INumericRecurrenceBackend"/>. Each recurrence stays inside one kernel, so its data never
/// round-trips to the host:
/// <list type="bullet">
/// <item><see cref="Tridiagonal"/>: one thread per system runs the Thomas algorithm (forward elimination, then back
/// substitution), the steps of <c>CpuEngine.TridiagonalSolveHost</c> in its order.</item>
/// <item><see cref="FritschCarlson"/>: one thread per series runs the sequential Fritsch-Carlson step 2.</item>
/// <item><see cref="CtcForwardBackward"/>: one 256-wide group per batch item walks time; its threads split the lattice
/// positions, reading the previous time step's row back from the log-alpha / log-beta outputs in global memory behind a
/// global-memory barrier, so the lattice has no length cap. Writes log-alpha, log-beta and each item's
/// log-likelihood.</item>
/// <item><see cref="CtcPosteriors"/>: one thread per (t, n) sums <c>exp(alpha + beta - logP)</c> over its lattice into
/// that row of gamma, in ascending lattice order (no two threads share an output row, so nothing is atomic).</item>
/// <item><see cref="LinearSumAssignment"/>: one 256-wide group per batch item runs the shortest-augmenting-path solver
/// with its state (duals, path costs, matchings) in a global-memory workspace, so the size has no shared-memory cap;
/// the column scan and its argmin are split across the threads, the augmentation is one thread's.</item>
/// </list>
/// Integers (targets, lengths, matchings) ride in the float buffers exactly (below 2^24). Log-domain sentinels are finite
/// (-1e30) inside the kernels and written out as negative infinity, so no kernel computes with an infinity. Every
/// kernel's last int, <c>first</c>, offsets the group index, so a grid larger than one dispatch's 65535 groups runs as
/// several launches.
/// </summary>
internal static class NumericRecurrenceKernelTemplate
{
    /// <summary>Threads per group; every kernel launches 256-wide groups.</summary>
    internal const int GroupSize = 256;

    /// <summary>The largest assignment size S: the column keys (S + j) ride in floats, exact below 2^24.</summary>
    internal const int AssignmentMaxSize = 1 << 22;

    internal const string Tridiagonal = "nrec_tridiagonal";
    internal const string FritschCarlson = "nrec_fritsch_carlson";
    internal const string CtcForwardBackward = "nrec_ctc_forward_backward";
    internal const string CtcPosteriors = "nrec_ctc_posteriors";
    internal const string LinearSumAssignment = "nrec_linear_sum_assignment";

    // x = A^-1 rhs per system; cw holds the scaled super-diagonal c' between the sweeps.
    private const string TridiagonalBody = @"
    int b = gid + first * 256;
    if (b >= B) return;
    int o = b * N;
    int ob = b * (N - 1);
    float beta = dg[o];
    xs[o] = rh[o] / beta;
    for (int i = 1; i < N; i++)
    {
        float c = up[ob + i - 1] / beta;
        cw[ob + i - 1] = c;
        float a = lo[ob + i - 1];
        beta = dg[o + i] - a * c;
        xs[o + i] = (rh[o + i] - a * xs[o + i - 1]) / beta;
    }
    for (int i = N - 2; i >= 0; i--) xs[o + i] = xs[o + i] - cw[ob + i] * xs[o + i + 1];
";

    private const string FritschCarlsonBody = @"
    int b = gid + first * 256;
    if (b >= B) return;
    int o = b * N;
    int od = b * (N - 1);
    for (int i = 0; i < N; i++) mo[o + i] = ms[o + i];
    for (int i = 0; i < N - 1; i++)
    {
        float d = ds[od + i];
        if (d == 0.0f)
        {
            mo[o + i] = 0.0f;
            mo[o + i + 1] = 0.0f;
        }
        else
        {
            float a = mo[o + i] / d;
            float c = mo[o + i + 1] / d;
            float sum = a * a + c * c;
            if (sum > 9.0f)
            {
                float tau = 3.0f / SQRTF(sum);
                mo[o + i] = tau * (a * d);
                mo[o + i + 1] = tau * (c * d);
            }
        }
    }
";

    // The rows live in logAlpha / logBeta themselves (global memory, so the lattice has no length cap): each time step's
    // row is written by the group's threads, then a global-memory barrier (GBARRIER) publishes it to the next step, as
    // the LSTM (OpenCL) and FFT (WGSL, Metal) kernels do. Reads clamp the stored negative infinity back to the finite
    // sentinel. lens holds the input lengths, the target lengths and the target offsets, N each; sm[0] is unused (a
    // shared array makes every dialect declare lid and grp).
    private const string CtcForwardBackwardBody = @"
    int n = grp + first;
    if (n >= N) return;
    int tn = TOINT(lens[n]);
    int un = TOINT(lens[N + n]);
    int off = TOINT(lens[2 * N + n]);
    int S = 2 * un + 1;
    int obase = n * T * SM;
    for (int t = 0; t < T; t++)
    {
        int prv = obase + (t - 1) * SM;
        for (int s = lid; s < SM; s += 256)
        {
            float v = -1.0e30f;
            if (t < tn && s < S)
            {
                int lab = blank;
                if ((s & 1) == 1) lab = TOINT(tg[off + (s >> 1)]);
                float e = FMAX(lp[(t * N + n) * C + lab], -1.0e30f);
                if (t == 0)
                {
                    if (s < 2) v = e;
                }
                else
                {
                    float a0 = FMAX(la[prv + s], -1.0e30f);
                    float a1 = -1.0e30f;
                    if (s >= 1) a1 = FMAX(la[prv + s - 1], -1.0e30f);
                    float a2 = -1.0e30f;
                    if (s >= 2 && lab != blank)
                    {
                        int lab2 = TOINT(tg[off + (s >> 1) - 1]);
                        if (lab2 != lab) a2 = FMAX(la[prv + s - 2], -1.0e30f);
                    }
                    float m = FMAX(FMAX(a0, a1), a2);
                    if (m > -1.0e29f) v = FMAX(m + LOGF(EXPF(a0 - m) + EXPF(a1 - m) + EXPF(a2 - m)) + e, -1.0e30f);
                }
            }
            float w = v;
            if (v <= -1.0e29f) w = NEGINF;
            la[obase + t * SM + s] = w;
        }
        GBARRIER();
    }
    if (lid == 0)
    {
        float z = -1.0e30f;
        if (S == 1) z = 0.0f;
        if (tn > 0)
        {
            int last = obase + (tn - 1) * SM;
            float a0 = FMAX(la[last + S - 1], -1.0e30f);
            float a1 = -1.0e30f;
            if (S >= 2) a1 = FMAX(la[last + S - 2], -1.0e30f);
            float m = FMAX(a0, a1);
            z = -1.0e30f;
            if (m > -1.0e29f) z = m + LOGF(EXPF(a0 - m) + EXPF(a1 - m));
        }
        lz[n] = z;
    }
    for (int t = T - 1; t >= 0; t--)
    {
        int nxt = obase + (t + 1) * SM;
        for (int s = lid; s < SM; s += 256)
        {
            float v = -1.0e30f;
            if (t < tn && s < S)
            {
                if (t == tn - 1)
                {
                    if (s >= S - 2) v = 0.0f;
                }
                else
                {
                    int lab = blank;
                    if ((s & 1) == 1) lab = TOINT(tg[off + (s >> 1)]);
                    int row = ((t + 1) * N + n) * C;
                    float b0 = FMAX(FMAX(lb[nxt + s], -1.0e30f) + FMAX(lp[row + lab], -1.0e30f), -1.0e30f);
                    float b1 = -1.0e30f;
                    if (s + 1 < S)
                    {
                        int l1 = blank;
                        if ((s & 1) == 0) l1 = TOINT(tg[off + (s >> 1)]);
                        b1 = FMAX(FMAX(lb[nxt + s + 1], -1.0e30f) + FMAX(lp[row + l1], -1.0e30f), -1.0e30f);
                    }
                    float b2 = -1.0e30f;
                    if (s + 2 < S && (s & 1) == 1)
                    {
                        int l2 = TOINT(tg[off + (s >> 1) + 1]);
                        if (l2 != blank && l2 != lab)
                            b2 = FMAX(FMAX(lb[nxt + s + 2], -1.0e30f) + FMAX(lp[row + l2], -1.0e30f), -1.0e30f);
                    }
                    float m = FMAX(FMAX(b0, b1), b2);
                    if (m > -1.0e29f) v = FMAX(m + LOGF(EXPF(b0 - m) + EXPF(b1 - m) + EXPF(b2 - m)), -1.0e30f);
                }
            }
            float w = v;
            if (v <= -1.0e29f) w = NEGINF;
            lb[obase + t * SM + s] = w;
        }
        GBARRIER();
    }
";

    // gamma row (t, n): zero, NaN when the item has no alignment, else the occupancies of its lattice.
    private const string CtcPosteriorsBody = @"
    int idx = gid + first * 256;
    if (idx >= T * N) return;
    int t = idx / N;
    int n = idx - t * N;
    int o = idx * C;
    for (int c = 0; c < C; c++) gm[o + c] = 0.0f;
    int tn = TOINT(lens[n]);
    if (t >= tn) return;
    float z = lz[n];
    if (z <= -1.0e29f)
    {
        for (int c = 0; c < C; c++) gm[o + c] = QNAN;
        return;
    }
    int un = TOINT(lens[N + n]);
    int off = TOINT(lens[2 * N + n]);
    int S = 2 * un + 1;
    int row = (n * T + t) * SM;
    for (int s = 0; s < S; s++)
    {
        float a = la[row + s];
        float bt = lb[row + s];
        if (a > -1.0e29f && bt > -1.0e29f)
        {
            int lab = blank;
            if ((s & 1) == 1) lab = TOINT(tg[off + (s >> 1)]);
            gm[o + lab] = gm[o + lab] + EXPF(a + bt - z);
        }
    }
";

    // The solver state lives in the workspace ws [B, 8S] (global memory, so the size has no shared-memory cap): u at 0,
    // v at S, the path costs at 2S, path at 3S, row4col at 4S, col4row at 5S, the scanned-row flags at 6S and the
    // scanned-column flags at 7S of the item's slice, published between phases by a global-memory barrier (GBARRIER).
    // Shared memory holds only the argmin's values at 0 and keys at 256, then minVal, sink and the current row at 512.
    // A key is j for an unassigned column and S + j for an assigned one, so the smallest (value, key) is
    // CpuEngine.SolveAssignment's choice.
    private const string LinearSumAssignmentBody = @"
    int b = grp + first;
    if (b >= B) return;
    int obase = b * S * S;
    int wu = b * 8 * S;
    int wv = wu + S;
    int wd = wu + 2 * S;
    int wp = wu + 3 * S;
    int wrc = wu + 4 * S;
    int wcr = wu + 5 * S;
    int wsr = wu + 6 * S;
    int wsc = wu + 7 * S;
    float mx = 0.0f;
    for (int k = lid; k < S * S; k += 256)
    {
        float c = cost[obase + k];
        if ((FBITS(c) & 2139095040) != 2139095040) mx = FMAX(mx, FMAX(c, -c));
    }
    sm[lid] = mx;
    BARRIER();
    for (int st = 128; st > 0; st = st / 2)
    {
        if (lid < st) sm[lid] = FMAX(sm[lid], sm[lid + st]);
        BARRIER();
    }
    float sz = S;
    // Split exact power-of-two scaling: a Vulkan reciprocal of 2^127 is subnormal and may flush
    // to zero. Both divisors below are at most 2^64, so their reciprocals remain normal floats.
    int exponent = (FBITS(sm[0]) >> 23) - 127;
    if (exponent < 0) exponent = 0;
    int halfExponent = exponent / 2;
    float scaleA = FVALUE((halfExponent + 127) << 23);
    float scaleB = FVALUE((exponent - halfExponent + 127) << 23);
    float big = (sz + 1.0f) * (2.0f * ((sm[0] / scaleA) / scaleB) + 1.0f);
    for (int j = lid; j < S; j += 256)
    {
        ws[wu + j] = 0.0f;
        ws[wv + j] = 0.0f;
        ws[wp + j] = -1.0f;
        ws[wrc + j] = -1.0f;
        ws[wcr + j] = -1.0f;
    }
    GBARRIER();
    for (int cur = 0; cur < S; cur++)
    {
        for (int j = lid; j < S; j += 256)
        {
            ws[wd + j] = 3.0e38f;
            ws[wsr + j] = 0.0f;
            ws[wsc + j] = 0.0f;
        }
        if (lid == 0)
        {
            sm[512] = 0.0f;
            sm[513] = -1.0f;
            sm[514] = cur;
        }
        GBARRIER();
        while (sm[513] < 0.0f)
        {
            float minv = sm[512];
            int i = TOINT(sm[514]);
            float ui = ws[wu + i];
            int crow = obase + i * S;
            float best = 3.0e38f;
            float bkey = 1.0e9f;
            for (int j = lid; j < S; j += 256)
            {
                if (ws[wsc + j] == 0.0f)
                {
                    float c = cost[crow + j];
                    int bits = FBITS(c);
                    if ((bits & 2139095040) == 2139095040)
                    {
                        c = big;
                        if (bits == -8388608) c = -big;
                    }
                    else c = (c / scaleA) / scaleB;
                    float r = minv + c - ui - ws[wv + j];
                    if (r < ws[wd + j])
                    {
                        ws[wp + j] = i;
                        ws[wd + j] = r;
                    }
                    float sj = ws[wd + j];
                    float key = j;
                    if (ws[wrc + j] >= 0.0f) key = key + sz;
                    if (sj < best || (sj == best && key < bkey))
                    {
                        best = sj;
                        bkey = key;
                    }
                }
            }
            sm[lid] = best;
            sm[256 + lid] = bkey;
            BARRIER();
            for (int st = 128; st > 0; st = st / 2)
            {
                if (lid < st)
                {
                    float v1 = sm[lid];
                    float v2 = sm[lid + st];
                    float k2 = sm[256 + lid + st];
                    if (v2 < v1 || (v2 == v1 && k2 < sm[256 + lid]))
                    {
                        sm[lid] = v2;
                        sm[256 + lid] = k2;
                    }
                }
                BARRIER();
            }
            if (lid == 0)
            {
                int kk = TOINT(sm[256]);
                int j = kk;
                if (kk >= S) j = kk - S;
                ws[wsr + i] = 1.0f;
                sm[512] = sm[0];
                ws[wsc + j] = 1.0f;
                float r4 = ws[wrc + j];
                if (r4 < 0.0f) sm[513] = j;
                else sm[514] = r4;
            }
            GBARRIER();
        }
        float minv = sm[512];
        for (int r = lid; r < S; r += 256)
        {
            if (ws[wsr + r] != 0.0f && r != cur) ws[wu + r] = ws[wu + r] + (minv - ws[wd + TOINT(ws[wcr + r])]);
        }
        for (int j = lid; j < S; j += 256)
        {
            if (ws[wsc + j] != 0.0f) ws[wv + j] = ws[wv + j] - (minv - ws[wd + j]);
        }
        GBARRIER();
        if (lid == 0)
        {
            ws[wu + cur] = ws[wu + cur] + minv;
            int j = TOINT(sm[513]);
            while (true)
            {
                int row = TOINT(ws[wp + j]);
                ws[wrc + j] = row;
                int prev = TOINT(ws[wcr + row]);
                ws[wcr + row] = j;
                j = prev;
                if (row == cur) break;
            }
        }
        GBARRIER();
    }
    for (int r = lid; r < S; r += 256) asg[b * S + r] = ws[wcr + r];
";

    /// <summary>
    /// Buffers in binding order (the written ones listed in each spec) and int parameters:
    /// tridiagonal lower [B, N-1], x [B, N] (written), diagonal, upper, rhs, workspace [B, N-1] (written);
    /// Fritsch-Carlson slopes [B, N], limited [B, N] (written), secants [B, N-1];
    /// CTC forward-backward logProbs [T, N, C], logAlpha [N, T, SM] (written), targets, lens [3N], logBeta (written),
    /// logLikelihood [N] (written);
    /// CTC posteriors logAlpha, gamma [T, N, C] (written), logBeta, targets, lens, logLikelihood;
    /// assignment cost [B, S, S], assignment [B, S] (written), workspace [B, 8S] (written).
    /// </summary>
    private static readonly WinogradF23KernelTemplate.Spec[] Generic =
    {
        new(Tridiagonal, new[] { "lo", "xs", "dg", "up", "rh", "cw" }, new[] { "B", "N", "first" }, TridiagonalBody, 0, new[] { 1, 5 }),
        new(FritschCarlson, new[] { "ms", "mo", "ds" }, new[] { "B", "N", "first" }, FritschCarlsonBody, 0, new[] { 1 }),
        new(CtcForwardBackward, new[] { "lp", "la", "tg", "lens", "lb", "lz" }, new[] { "T", "N", "C", "SM", "blank", "first" },
            CtcForwardBackwardBody, 1, new[] { 1, 4, 5 }),
        new(CtcPosteriors, new[] { "la", "gm", "lb", "tg", "lens", "lz" }, new[] { "T", "N", "C", "SM", "blank", "first" },
            CtcPosteriorsBody, 0, new[] { 1 }),
        new(LinearSumAssignment, new[] { "cost", "asg", "ws" }, new[] { "B", "S", "first" }, LinearSumAssignmentBody, 515,
            new[] { 1, 2 }),
    };

    /// <summary>The kernels' specs, in binding conventions shared by every dialect.</summary>
    internal static IReadOnlyList<WinogradF23KernelTemplate.Spec> Kernels => Generic;

    /// <summary>Every kernel name.</summary>
    internal static string[] GetKernelNames() => new[] { Tridiagonal, FritschCarlson, CtcForwardBackward, CtcPosteriors, LinearSumAssignment };

    // This family's spellings on top of the template's: float to int, exp / log / sqrt (HIP and CUDA take the float
    // functions by name), a float's bits as an int, and the negative infinity and NaN written to the outputs.
    private static string Spell(string body, WinogradKernelDialect dialect)
    {
        string bitsToFloat = dialect switch
        {
            WinogradKernelDialect.OpenCl => "as_float",
            WinogradKernelDialect.Glsl => "intBitsToFloat",
            WinogradKernelDialect.Metal => "as_type<float>",
            _ => "__int_as_float",
        };
        return body
            .Replace("GBARRIER();", dialect switch
            {
                WinogradKernelDialect.OpenCl => "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);",
                WinogradKernelDialect.Metal => "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);",
                WinogradKernelDialect.Glsl => "memoryBarrierBuffer(); memoryBarrierShared(); barrier();",
                _ => "__syncthreads();",
            })
            .Replace("TOINT(", dialect == WinogradKernelDialect.Glsl ? "int(" : "(int)(")
            .Replace("EXPF(", dialect == WinogradKernelDialect.Hip ? "expf(" : "exp(")
            .Replace("LOGF(", dialect == WinogradKernelDialect.Hip ? "logf(" : "log(")
            .Replace("SQRTF(", dialect == WinogradKernelDialect.Hip ? "sqrtf(" : "sqrt(")
            .Replace("FBITS(", dialect switch
            {
                WinogradKernelDialect.OpenCl => "as_int(",
                WinogradKernelDialect.Glsl => "floatBitsToInt(",
                WinogradKernelDialect.Metal => "as_type<int>(",
                _ => "__float_as_int(",
            })
            .Replace("NEGINF", bitsToFloat + "(-8388608)")
            .Replace("FVALUE(", bitsToFloat + "(")
            .Replace("QNAN", bitsToFloat + "(2143289344)");
    }

    private static WinogradF23KernelTemplate.Spec ForDialect(WinogradF23KernelTemplate.Spec spec, WinogradKernelDialect dialect)
        => new(spec.Name, spec.Buffers, spec.Ints, Spell(spec.Body, dialect), spec.SharedFloats, spec.Written);

    /// <summary>Every kernel in one OpenCL C, HIP C++ (also compiled by CUDA's NVRTC) or Metal module.</summary>
    internal static string GetModuleSource(WinogradKernelDialect dialect)
    {
        var specs = new List<WinogradF23KernelTemplate.Spec>();
        foreach (var spec in Generic) specs.Add(ForDialect(spec, dialect));
        return WinogradF23KernelTemplate.EmitModule(specs, dialect);
    }

    /// <summary>The standalone GLSL compute shader for one kernel (Vulkan compiles one pipeline per kernel).</summary>
    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Generic)
            if (spec.Name == name)
                // coherent: the GBARRIER kernels read other invocations' global writes back (as ResidentProgram's rows).
                return WinogradF23KernelTemplate.EmitGlsl(ForDialect(spec, WinogradKernelDialect.Glsl))
                    .Replace(") buffer B", ") coherent buffer B");
        throw new ArgumentException($"No numeric recurrence kernel named '{name}'.", nameof(name));
    }
}
