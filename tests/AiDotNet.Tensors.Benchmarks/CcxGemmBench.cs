using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TorchSharp;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// CCX-aware hierarchical GEMM PROTOTYPE (the path to MKL on the 3990X). The flat per-tile scheme caps
/// at ~16-41% of MKL because it re-reads B from DRAM per tile and, when B is shared, the reads cross
/// CCXs over Infinity Fabric. This prototype partitions the N-columns across the 16 CCX domains and pins
/// a thread-group per CCX, so each CCX's B-panel is pulled into ITS OWN 16 MB L3 ONCE and re-used by that
/// CCX's threads with no cross-CCX traffic. Each (ic,jc) tile uses the proven GotoGemmFp32.RunTile.
/// Run: --ab-ccx. Once it beats the per-tile RunParallel toward MKL, it graduates into src as the
/// production parallel path (a persistent pinned pool + the same partition).
/// </summary>
internal static unsafe class CcxGemmBench
{
    [StructLayout(LayoutKind.Sequential)]
    private struct GROUP_AFFINITY { public ulong Mask; public ushort Group; public ushort R0, R1, R2; }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetThreadGroupAffinity(IntPtr hThread, ref GROUP_AFFINITY ga, IntPtr prev);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(int rel, IntPtr buffer, ref uint len);

    private static (ulong mask, ushort group)[] _ccx = Array.Empty<(ulong, ushort)>();
    private static int _numCcx, _threadsPerCcx;
    private static ManualResetEventSlim[] _go = Array.Empty<ManualResetEventSlim>();
    private static CountdownEvent? _done;
    private static Barrier[] _ccxBarrier = Array.Empty<Barrier>();
    private static IntPtr[] _ccxB = Array.Empty<IntPtr>();   // per-CCX whole-K packed-B buffer (pack-once)
    private static IntPtr[] _ccxA = Array.Empty<IntPtr>();   // per-CCX whole-K packed-A buffer (2D grid)
    private static long _ccxBLen, _ccxALen;                  // current capacities (floats)
    private static bool _use2D;
    private static int _gr, _gc;                             // 2D CCX grid (gr·gc = numCcx)
    private static volatile bool _shutdown;

    // GEMM params set per Run.
    private static float* _a, _b, _c;
    private static int _m, _n, _k, _mc, _nc, _kc, _lda, _ldb, _ldc;

    private static (ulong mask, ushort group)[] DetectCcx()
    {
        const int RelationCache = 2;
        uint len = 0;
        GetLogicalProcessorInformationEx(RelationCache, IntPtr.Zero, ref len);
        if (len == 0) return Array.Empty<(ulong, ushort)>();
        IntPtr buf = Marshal.AllocHGlobal((int)len);
        try
        {
            if (!GetLogicalProcessorInformationEx(RelationCache, buf, ref len)) return Array.Empty<(ulong, ushort)>();
            var list = new System.Collections.Generic.List<(ulong, ushort)>();
            long ptr = (long)buf, end = ptr + len;
            while (ptr < end)
            {
                int rel = Marshal.ReadInt32((IntPtr)ptr);
                int size = Marshal.ReadInt32((IntPtr)(ptr + 4));
                if (size <= 0) break;
                if (rel == RelationCache && Marshal.ReadByte((IntPtr)(ptr + 8)) == 3) // Level 3
                {
                    ulong mask = (ulong)Marshal.ReadInt64((IntPtr)(ptr + 40));
                    ushort group = (ushort)Marshal.ReadInt16((IntPtr)(ptr + 48));
                    list.Add((mask, group));
                }
                ptr += size;
            }
            return list.ToArray();
        }
        finally { Marshal.FreeHGlobal(buf); }
    }

    public static void Init(int threadsPerCcx)
    {
        _ccx = DetectCcx();
        _numCcx = _ccx.Length;
        _threadsPerCcx = threadsPerCcx;
        int total = _numCcx * threadsPerCcx;
        _go = new ManualResetEventSlim[total];
        _done = new CountdownEvent(total);
        _ccxBarrier = new Barrier[_numCcx];
        for (int c = 0; c < _numCcx; c++) _ccxBarrier[c] = new Barrier(threadsPerCcx);
        _ccxB = new IntPtr[_numCcx];
        _ccxA = new IntPtr[_numCcx];
        for (int i = 0; i < total; i++)
        {
            _go[i] = new ManualResetEventSlim(false);
            int id = i;
            var t = new Thread(() => WorkerLoop(id)) { IsBackground = true, Name = $"ccx-{id}" };
            t.Start();
        }
        Thread.Sleep(50); // let workers pin before first dispatch
    }

    private static void WorkerLoop(int id)
    {
        int ccx = id / _threadsPerCcx;
        var ga = new GROUP_AFFINITY { Mask = _ccx[ccx].mask, Group = _ccx[ccx].group };
        SetThreadGroupAffinity(GetCurrentThread(), ref ga, IntPtr.Zero);
        int lane = id % _threadsPerCcx;
        while (true)
        {
            _go[id].Wait();
            _go[id].Reset();
            if (_shutdown) return;
            try { if (_use2D) DoWork2D(ccx, lane); else DoWork(ccx, lane); }
            finally { _done!.Signal(); }
        }
    }

    // CCX owns a contiguous jc-block range → its B lives in its own L3. PACK-ONCE: lane 0 packs the
    // jc-block's whole-K B into the CCX-shared buffer, a per-CCX barrier releases the lanes, then the
    // lanes split the ic-blocks (each only packs its own A) via RunTilePackedB — no redundant B-pack and
    // B is DRAM-read once per CCX. A second barrier guards the shared buffer before the next jc-block.
    private static void DoWork(int ccx, int lane)
    {
        int numJc = (_n + _nc - 1) / _nc;
        int numIc = (_m + _mc - 1) / _mc;
        int jcStart = (int)((long)ccx * numJc / _numCcx);
        int jcEnd = (int)((long)(ccx + 1) * numJc / _numCcx);
        var bar = _ccxBarrier[ccx];
        float* pkb = (float*)_ccxB[ccx];
        for (int jb = jcStart; jb < jcEnd; jb++)
        {
            int jc = jb * _nc; int effNc = Math.Min(_nc, _n - jc);
            if (effNc <= 0) { bar.SignalAndWait(); bar.SignalAndWait(); continue; }
            // Parallel pack: each lane packs its stride of the Nr-tiles → no serial-pack stall.
            GotoGemmFp32.PackBPanel(_b, _ldb, jc, effNc, _k, _kc, pkb, lane, _threadsPerCcx);
            bar.SignalAndWait();
            for (int ib = lane; ib < numIc; ib += _threadsPerCcx)
            {
                int ic = ib * _mc; int effMc = Math.Min(_mc, _m - ic); if (effMc <= 0) continue;
                GotoGemmFp32.RunTilePackedB(_a, _lda, _b, _ldb, _c, _ldc, ic, jc, effMc, effNc, _k, _mc, _nc, _kc, pkb);
            }
            bar.SignalAndWait();
        }
    }

    // 2D CCX grid: CCX (r,c) owns M-block r × N-block c. Lanes parallel-pack the block's A AND B into the
    // CCX's two L3-resident buffers, barrier, then split the block's Mr×Nr micro-tiles (RunBlockPackedAB).
    // Cuts cross-CCX A re-read from numCcx× (1D-N) to ~gr× and keeps both operands DRAM-read once per CCX.
    private static void DoWork2D(int ccx, int lane)
    {
        int r = ccx / _gc, cc = ccx % _gc;
        var bar = _ccxBarrier[ccx];
        // Mr/Nr-aligned block boundaries so only the matrix-edge blocks carry sub-tile remainders.
        int mBlk = RoundUp((_m + _gr - 1) / _gr, 6);
        int nBlk = RoundUp((_n + _gc - 1) / _gc, 16);
        int ic0 = r * mBlk, jc0 = cc * nBlk;
        int effMc = Math.Min(mBlk, _m - ic0), effNc = Math.Min(nBlk, _n - jc0);
        if (effMc <= 0 || effNc <= 0) { return; } // this CCX has no block (grid > matrix); no barrier participation issue: all its lanes hit this
        float* pkA = (float*)_ccxA[ccx]; float* pkB = (float*)_ccxB[ccx];
        GotoGemmFp32.PackAPanel(_a, _lda, ic0, effMc, _k, _kc, pkA, lane, _threadsPerCcx);
        GotoGemmFp32.PackBPanel(_b, _ldb, jc0, effNc, _k, _kc, pkB, lane, _threadsPerCcx);
        bar.SignalAndWait();
        GotoGemmFp32.RunBlockPackedAB(_a, _lda, _b, _ldb, _c, _ldc, ic0, jc0, effMc, effNc, _k, _kc, pkA, pkB, lane, _threadsPerCcx);
        bar.SignalAndWait(); // guard buffers until all lanes done (reused next Run)
    }

    private static int RoundUp(int x, int mult) { int r = ((x + mult - 1) / mult) * mult; return r < mult ? mult : r; }

    // Pick grid gr·gc = numCcx minimizing block aspect mismatch (balance M/gr vs N/gc).
    private static (int gr, int gc) ChooseGrid(int m, int n)
    {
        int best = 0; double bestScore = double.MaxValue;
        for (int gr = 1; gr <= _numCcx; gr++)
        {
            if (_numCcx % gr != 0) continue;
            int gc = _numCcx / gr;
            double score = Math.Abs((double)m / gr - (double)n / gc);
            if (score < bestScore) { bestScore = score; best = gr; }
        }
        return (best, _numCcx / best);
    }

    public static void Run(float* a, int lda, float* b, int ldb, float* c, int ldc,
        int m, int n, int k, int mc, int nc, int kc)
    {
        if (_use2D)
        {
            (_gr, _gc) = ChooseGrid(m, n);
            int mBlk = RoundUp((m + _gr - 1) / _gr, 6), nBlk = RoundUp((n + _gc - 1) / _gc, 16);
            long needA = GotoGemmFp32.PackedALen(mBlk, k, kc), needB = GotoGemmFp32.PackedBLen(nBlk, k, kc);
            if (needA > _ccxALen)
            {
                for (int c2 = 0; c2 < _numCcx; c2++) { if (_ccxA[c2] != IntPtr.Zero) Marshal.FreeHGlobal(_ccxA[c2]); _ccxA[c2] = Marshal.AllocHGlobal(checked((int)(needA * sizeof(float)))); }
                _ccxALen = needA;
            }
            if (needB > _ccxBLen)
            {
                for (int c2 = 0; c2 < _numCcx; c2++) { if (_ccxB[c2] != IntPtr.Zero) Marshal.FreeHGlobal(_ccxB[c2]); _ccxB[c2] = Marshal.AllocHGlobal(checked((int)(needB * sizeof(float)))); }
                _ccxBLen = needB;
            }
            _a = a; _b = b; _c = c; _lda = lda; _ldb = ldb; _ldc = ldc; _m = m; _n = n; _k = k; _mc = mc; _nc = nc; _kc = kc;
            _done!.Reset();
            for (int i = 0; i < _go.Length; i++) _go[i].Set();
            _done.Wait();
            return;
        }
        // Ensure each CCX's whole-K packed-B buffer is large enough for this shape's jc-block (amortized:
        // reallocs only on a size increase, so it's outside the steady-state timing loop).
        long need = GotoGemmFp32.PackedBLen(nc, k, kc);
        if (need > _ccxBLen)
        {
            for (int c2 = 0; c2 < _numCcx; c2++)
            {
                if (_ccxB[c2] != IntPtr.Zero) Marshal.FreeHGlobal(_ccxB[c2]);
                _ccxB[c2] = Marshal.AllocHGlobal(checked((int)(need * sizeof(float))));
            }
            _ccxBLen = need;
        }
        _a = a; _b = b; _c = c; _lda = lda; _ldb = ldb; _ldc = ldc;
        _m = m; _n = n; _k = k; _mc = mc; _nc = nc; _kc = kc;
        _done!.Reset();
        for (int i = 0; i < _go.Length; i++) _go[i].Set();
        _done.Wait();
    }

    public static void Bench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        if (!GotoGemmFp32.IsAvailable) { Console.WriteLine("GotoGemmFp32 not available"); return; }
        int tpc = int.TryParse(Environment.GetEnvironmentVariable("CCX_TPC"), out var v) ? v : 4;
        _use2D = Environment.GetEnvironmentVariable("CCX_2D") == "1";
        Init(tpc);
        if (_numCcx <= 0) { Console.WriteLine("no L3/CCX domains detected (non-Windows or unsupported) — skipping CCX bench"); return; }
        Console.WriteLine($"=== CCX-aware GEMM ({(_use2D ? "2D grid" : "1D-N")}) vs MKL vs per-tile (cores={P}, CCX={_numCcx}, threads/CCX={_threadsPerCcx}) ===");
        int kc = GotoGemmFp32.DefaultKc;
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        var rng = new Random(13);
        var shapes = new (int M, int N, int K, string tag)[]
        {
            (256, 1152, 1152, "attn-proj"), (256, 1152, 4608, "mlp-fc"),
            (1024, 1024, 1024, "square1024"), (2048, 2048, 2048, "square2048"),
        };
        try
        {
            // Warm to steady thermal state.
            { var wa = torch.rand(2048, 2048); var wb = torch.rand(2048, 2048); torch.set_num_threads(P);
              var ww = Stopwatch.StartNew(); while (ww.ElapsedMilliseconds < 1500) { using var _ = torch.matmul(wa, wb); } }

            foreach (var (M, N, K, tag) in shapes)
            {
                var A = new float[(long)M * K]; var B = new float[(long)K * N]; var C = new float[(long)M * N];
                for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
                double gf = 2.0 * M * N * K / 1e9;
                // 1D-N: sweep mc/nc (nc ~ N/numCcx so each CCX ~1 jc-block; mc ~ M/tpc). 2D: mc/nc ignored
                // (the grid + block sizes are derived from the shape in Run), so a single dummy config.
                var cands = _use2D
                    ? new (int mc, int nc)[] { (120, 128) }
                    : new (int mc, int nc)[]
                {
                    (RoundMr(Math.Max(48, M / _threadsPerCcx)), RoundNr(Math.Max(64, N / _numCcx))),
                    (RoundMr(Math.Max(48, M / (_threadsPerCcx * 2))), RoundNr(Math.Max(64, N / _numCcx))),
                    (120, RoundNr(Math.Max(64, N / _numCcx))),
                    (RoundMr(Math.Max(48, M / _threadsPerCcx)), RoundNr(Math.Max(64, N / (_numCcx * 2)))),
                };
                double bestGf = 0; int bMc = 0, bNc = 0; string corr = "?";
                fixed (float* pa = A, pb = B, pc = C)
                {
                    foreach (var (mc, nc) in cands)
                    {
                        Run(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        double maxRel = 0; var rv = new Random(5);
                        for (int t = 0; t < 60; t++)
                        {
                            int r = rv.Next(M), col = rv.Next(N);
                            double s = 0; for (int kk = 0; kk < K; kk++) s += (double)A[(long)r * K + kk] * B[(long)kk * N + col];
                            double err = Math.Abs(C[(long)r * N + col] - s);
                            maxRel = Math.Max(maxRel, Math.Abs(s) > 1e-3 ? err / Math.Abs(s) : err);
                        }
                        double gbest = double.PositiveInfinity;
                        var w = Stopwatch.StartNew(); while (w.ElapsedMilliseconds < 40) Run(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        for (int r = 0; r < 5; r++)
                        {
                            var sw = Stopwatch.StartNew(); int reps = 0;
                            do { Run(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 40);
                            sw.Stop(); gbest = Math.Min(gbest, sw.Elapsed.TotalMilliseconds / reps);
                        }
                        double g = gf / (gbest / 1000);
                        Console.WriteLine($"    ccx cand mc={mc,4} nc={nc,4}: {g,7:F0} GF  {(maxRel < 1e-3 ? "OK" : "WRONG " + maxRel.ToString("E1"))}");
                        if (g > bestGf) { bestGf = g; bMc = mc; bNc = nc; corr = maxRel < 1e-3 ? "OK" : "WRONG"; }
                    }
                }
                // Thermally-fair three-way: CCX(best) / per-tile RunParallel / MKL, min-of-10 interleaved.
                var ta = torch.rand(M, K); var tb = torch.rand(K, N);
                var (pmc, pnc, pkc) = GotoGemmFp32.ChooseParallelBlocks(M, N);
                double ccxMs = double.PositiveInfinity, ptMs = double.PositiveInfinity, mklMs = double.PositiveInfinity;
                fixed (float* pa = A, pb = B, pc = C)
                {
                    for (int r = 0; r < 10; r++)
                    {
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { Run(pa, K, pb, N, pc, N, M, N, K, bMc, bNc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); ccxMs = Math.Min(ccxMs, sw.Elapsed.TotalMilliseconds / reps); }
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, pmc, pnc, pkc); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); ptMs = Math.Min(ptMs, sw.Elapsed.TotalMilliseconds / reps); }
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { using var _ = torch.matmul(ta, tb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); mklMs = Math.Min(mklMs, sw.Elapsed.TotalMilliseconds / reps); }
                    }
                }
                double ccxGf = gf / (ccxMs / 1000), ptGf = gf / (ptMs / 1000), mklGf = gf / (mklMs / 1000);
                Console.WriteLine($"{tag,-11} CCX={ccxGf,7:F0} (mc={bMc} nc={bNc})  per-tile={ptGf,7:F0}  MKL={mklGf,7:F0}   CCX/per-tile={(ptGf>0?ccxGf/ptGf:0),4:F2}x  CCX/MKL={(mklGf>0?ccxGf/mklGf*100:0),4:F0}%  {corr}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    private static int RoundMr(int x) { int r = (x / 6) * 6; return r < 6 ? 6 : r; }
    private static int RoundNr(int x) { int r = (x / 16) * 16; return r < 16 ? 16 : r; }
}
