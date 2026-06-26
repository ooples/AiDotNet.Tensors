#if NET5_0_OR_GREATER
using System;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// CCX-aware parallel FP32 GEMM for LARGE BALANCED shapes (the bandwidth lever the flat per-tile path
/// can't reach). A persistent pool of worker threads pinned per L3 domain (CCX): N is partitioned into one
/// thin strip per CCX, each CCX packs its B-strip ONCE (lane-parallel) into ITS OWN L3 and its threads
/// reuse it across that strip's M (ic) blocks via <see cref="GotoGemmFp32.RunTilePackedB"/> — so B is
/// DRAM-read once per CCX (not once per (ic,jc) tile) and never streamed cross-CCX. Measured (--ab-ccx):
/// square1024 ~49% of MKL vs ~37% for per-tile; square2048 ~49% vs ~47%.
///
/// <para>Bit-identical to <see cref="GotoGemmFp32.RunParallel"/> (same kc, kernel, K-order ⇒ deterministic
/// across thread counts). Reentrancy/oversubscription safe: <see cref="TryRun"/> returns false (caller
/// falls back to per-tile) when the pool is busy, the thread budget is restricted, or the shape isn't in
/// the proven win regime. Windows-only pinning (Linux ⇒ no L3 domains ⇒ never selected, per-tile is used).</para>
/// </summary>
internal static unsafe class CcxGemmPool
{
    private static CpuTopology.Domain[] _domains = Array.Empty<CpuTopology.Domain>();
    private static int _numCcx, _tpc, _total;
    private static ManualResetEventSlim[] _go = Array.Empty<ManualResetEventSlim>();
    private static CountdownEvent? _done;
    private static Barrier[] _bar = Array.Empty<Barrier>();
    private static IntPtr[] _bbuf = Array.Empty<IntPtr>();
    private static long _bbufLen;
    private static int _initState; // 0 unstarted, 1 ready, 2 unavailable
    private static readonly object _initGate = new();
    private static readonly object _runGate = new();

    private static float* _a, _b, _c;
    private static int _m, _n, _k, _mc, _nc, _kc, _lda, _ldb, _ldc;
    private static bool _use2D;       // 2D-NUMA grid (huge squares) vs 1D-N thin strips
    private static volatile bool _faulted; // a lane's compute threw → TryRun returns false (caller falls back)
    private static int _gr, _gc;      // 2D CCX grid (gr·gc = numCcx)

    /// <summary>Min M·N·K for the CCX path (below this the pinned-pool + barrier overhead isn't worth it).</summary>
    private const long CcxMinWork = 200L * 1024 * 1024;

    /// <summary>Test/diagnostic toggle (env AIDOTNET_DISABLE_CCX=1): force per-tile, for in-process A/B.</summary>
    internal static bool s_disable =
        System.Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_CCX") == "1";

    private static void EnsureInit()
    {
        if (Volatile.Read(ref _initState) != 0) return;
        lock (_initGate)
        {
            if (_initState != 0) return;
            try
            {
                _domains = CpuTopology.DetectL3Domains();
                _numCcx = _domains.Length;
                if (_numCcx < 2 || !GotoGemmFp32.IsAvailable) { Volatile.Write(ref _initState, 2); return; }
                int tpc = _domains[0].Cores;
                for (int i = 1; i < _numCcx; i++) tpc = Math.Min(tpc, _domains[i].Cores);
                _tpc = Math.Max(1, tpc);
                _total = _numCcx * _tpc;
                _go = new ManualResetEventSlim[_total];
                _done = new CountdownEvent(_total);
                _bar = new Barrier[_numCcx];
                for (int c = 0; c < _numCcx; c++) _bar[c] = new Barrier(_tpc);
                _bbuf = new IntPtr[_numCcx];
                for (int i = 0; i < _total; i++)
                {
                    _go[i] = new ManualResetEventSlim(false);
                    int id = i;
                    new Thread(() => WorkerLoop(id)) { IsBackground = true, Name = $"aidotnet-ccx-{id}" }.Start();
                }
                Volatile.Write(ref _initState, 1);
            }
            catch { Volatile.Write(ref _initState, 2); }
        }
    }

    private static void WorkerLoop(int id)
    {
        int ccx = id / _tpc;
        CpuTopology.TryPinCurrentThread(_domains[ccx]); // best-effort; correctness independent of pinning
        int lane = id % _tpc;
        while (true)
        {
            _go[id].Wait();
            _go[id].Reset();
            // Backstop: compute faults are caught INSIDE DoWork/DoWork2D (around the compute, not the
            // barriers) so a faulting lane still completes its barrier sequence and never deadlocks siblings.
            // This catches any non-compute fault. Either way _faulted ⇒ TryRun returns false ⇒ caller re-runs.
            try { if (_use2D) DoWork2D(ccx, lane); else DoWork(ccx, lane); }
            catch { _faulted = true; }
            finally { _done!.Signal(); }
        }
    }

    // CCX owns a contiguous thin N-strip → its B lives in its own L3. Lane-parallel pack B ONCE, barrier,
    // then lanes split the ic-blocks (each only packs its own A, SIMD) via RunTilePackedB; barrier guards
    // the shared buffer across the (rare, >1) strips a CCX may own.
    private static void DoWork(int ccx, int lane)
    {
        int numJc = (_n + _nc - 1) / _nc;
        int numIc = (_m + _mc - 1) / _mc;
        int jcStart = (int)((long)ccx * numJc / _numCcx);
        int jcEnd = (int)((long)(ccx + 1) * numJc / _numCcx);
        var bar = _bar[ccx];
        float* pkb = (float*)_bbuf[ccx];
        for (int jb = jcStart; jb < jcEnd; jb++)
        {
            int jc = jb * _nc; int effNc = Math.Min(_nc, _n - jc);
            if (effNc <= 0) { bar.SignalAndWait(); bar.SignalAndWait(); continue; }
            // Compute wrapped (NOT the barriers): a fault sets _faulted but the lane still hits every barrier.
            try { GotoGemmFp32.PackBPanel(_b, _ldb, jc, effNc, _k, _kc, pkb, lane, _tpc); } catch { _faulted = true; }
            bar.SignalAndWait();
            for (int ib = lane; ib < numIc; ib += _tpc)
            {
                int ic = ib * _mc; int effMc = Math.Min(_mc, _m - ic); if (effMc <= 0) continue;
                try { GotoGemmFp32.RunTilePackedB(_a, _lda, _b, _ldb, _c, _ldc, ic, jc, effMc, effNc, _k, _mc, _nc, _kc, pkb); } catch { _faulted = true; }
            }
            bar.SignalAndWait();
        }
    }

    // 2D-NUMA GotoBLAS: CCX (r,c) owns M-block r × N-block c. For each K-panel, lane-parallel pack that
    // pc-panel of the N-block's B into the CCX's L3 (kc×nBlk ⇒ stays cache-friendly), barrier, then the
    // CCX's lanes split the M-block's ic-blocks (each packs its A-panel into L2) and accumulate C across pc
    // via RunMacroPanelStep; barrier guards the shared B-panel before the next pc. Cuts A DRAM re-reads to
    // gc× (vs numCcx× for 1D) — the lever for huge squares whose 1D full-K strip would spill L3.
    private static void DoWork2D(int ccx, int lane)
    {
        int r = ccx / _gc, cc = ccx % _gc;
        int mBlk = RoundUp((_m + _gr - 1) / _gr, 6);
        int nBlk = RoundUp((_n + _gc - 1) / _gc, Nr32);
        int m0 = r * mBlk, n0 = cc * nBlk;
        int effMblk = Math.Min(mBlk, _m - m0), effNblk = Math.Min(nBlk, _n - n0);
        if (effMblk <= 0 || effNblk <= 0) return; // empty CCX (grid > matrix): all its lanes skip; no barrier use
        var bar = _bar[ccx];
        float* pkB = (float*)_bbuf[ccx];
        for (int pc = 0; pc < _k; pc += _kc)
        {
            int effKc = Math.Min(_kc, _k - pc);
            // Compute wrapped (NOT the barriers): a fault sets _faulted but the lane still hits every barrier.
            try { GotoGemmFp32.PackBPanelPc(_b, _ldb, n0, effNblk, pc, effKc, _kc, pkB, lane, _tpc); } catch { _faulted = true; }
            bar.SignalAndWait();
            int ii = 0;
            for (int ic = m0; ic < m0 + effMblk; ic += _mc)
            {
                if (ii++ % _tpc != lane) continue;
                int effMc = Math.Min(_mc, m0 + effMblk - ic);
                try { GotoGemmFp32.RunMacroPanelStep(_a, _lda, _b, _ldb, _c, _ldc, ic, n0, effMc, effNblk, pc, effKc, _mc, _kc, pkB, pc == 0); } catch { _faulted = true; }
            }
            bar.SignalAndWait();
        }
    }

    private static int RoundUp(int x, int mult) { int r = ((x + mult - 1) / mult) * mult; return r < mult ? mult : r; }

    // Pick grid gr·gc = numCcx minimizing block aspect mismatch (balance M/gr vs N/gc).
    private static (int gr, int gc) ChooseGrid(int m, int n)
    {
        int best = 1; double bestScore = double.MaxValue;
        for (int gr = 1; gr <= _numCcx; gr++)
        {
            if (_numCcx % gr != 0) continue;
            int gc = _numCcx / gr;
            double score = Math.Abs((double)m / gr - (double)n / gc);
            if (score < bestScore) { bestScore = score; best = gr; }
        }
        return (best, _numCcx / best);
    }

    /// <summary>True when the CCX pool exists (≥2 L3 domains, FP32 machine kernel available).</summary>
    internal static bool IsAvailable { get { EnsureInit(); return _initState == 1; } }

    /// <summary>Run C := A·B through the CCX pool, or return false if the caller should use the per-tile
    /// path (pool busy / nested / thread-budget restricted / shape outside the proven win regime).</summary>
    internal static bool TryRun(float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k)
    {
        if (s_disable || !IsAvailable) return false;
        // Win regime (measured): large + BALANCED squares (CCX 1.12-1.19x on sq2048 = ~46% MKL). Below
        // ~1536 the win is noise-level (±5%) so gate it out to GUARANTEE no regression; skewed/wide-N
        // (DiT) regress and are excluded by the balance check below.
        // Gate to the PROVEN balanced-square regime: m,n>=512 (DiT m=256 excluded — its deep-K MLP shapes
        // are barrier-heavy in 2D and the diffusion hot path must not regress; they stay on per-tile/PackBoth).
        if (m < s_minM || n < 512 || k < 256) return false;
        if ((long)m * n * k < CcxMinWork) return false;
        if (CpuParallelSettings.MaxDegreeOfParallelism < _total) return false; // restricted budget → per-tile

        // Deep-K-aware kc for the 2D path: more K per panel ⇒ fewer (pc-panels × 2) per-CCX barriers, which
        // dominate when K is large. Capped so the per-C-tile working set still ~fits L1. s_kc2dOverride (env
        // AIDOTNET_CCX_KC) for the sweep.
        int kc = s_kc2dOverride > 0 ? s_kc2dOverride : (k >= 4096 ? 384 : 256);
        (_gr, _gc) = ChooseGrid(m, n);
        // Require a GENUINE 2D grid (both dims partitioned): gr==1 (very wide-N) or gc==1 (very tall-M) is
        // a degenerate 1D split that re-reads the big operand numCcx× and regresses — those stay on
        // per-tile/PackBoth. A genuine grid gives the gc×-A / gr×-B DRAM split that wins on balanced squares.
        // The 1D-N path is reachable only via s_force1D (A/B).
        int nc1d = ChooseNc(n);
        bool oneDFits = (long)k * nc1d <= 512L * 1024;
        bool use2D = !(s_force1D && oneDFits);
        int mc, nc; long need;
        if (use2D)
        {
            if (_gr < 2 || _gc < 2) return false;            // degenerate grid → per-tile/PackBoth
            int nBlk = RoundUp((n + _gc - 1) / _gc, Nr32);
            int mBlk = RoundUp((m + _gr - 1) / _gr, 6);
            if ((long)kc * nBlk > 1024L * 1024) return false; // 2D B-panel (kc×nBlk) > 4MB → per-tile
            mc = RoundMr(mBlk / _tpc); if (mc < 48) mc = 48; if (mc > 240) mc = 240;
            nc = nBlk;
            need = GotoGemmFp32.PackedBPanelLen(nBlk, kc);
        }
        else
        {
            if ((n / _numCcx) < Nr32) return false;          // 1D strips too thin
            mc = ChooseMc(m); nc = nc1d;
            need = GotoGemmFp32.PackedBLen(nc, k, kc);
        }
        if (!Monitor.TryEnter(_runGate)) return false;       // concurrent GEMM owns the pool → per-tile
        try
        {
            _use2D = use2D;
            if (need > _bbufLen)
            {
                // Allocate ALL replacements first; only free the old buffers once every new alloc succeeded.
                // If AllocHGlobal throws (OOM), free any partial new allocs, keep the (still-valid) old
                // buffers + _bbufLen, and fall back to per-tile — never leave a dangling _bbuf[cc].
                var fresh = new IntPtr[_numCcx];
                try { for (int cc = 0; cc < _numCcx; cc++) fresh[cc] = Marshal.AllocHGlobal(checked((int)(need * sizeof(float)))); }
                catch
                {
                    for (int cc = 0; cc < _numCcx; cc++) if (fresh[cc] != IntPtr.Zero) Marshal.FreeHGlobal(fresh[cc]);
                    return false;
                }
                for (int cc = 0; cc < _numCcx; cc++)
                {
                    if (_bbuf[cc] != IntPtr.Zero) Marshal.FreeHGlobal(_bbuf[cc]);
                    _bbuf[cc] = fresh[cc];
                }
                _bbufLen = need;
            }
            _a = a; _b = b; _c = c; _lda = lda; _ldb = ldb; _ldc = ldc;
            _m = m; _n = n; _k = k; _mc = mc; _nc = nc; _kc = kc;
            _faulted = false;
            _done!.Reset();
            for (int i = 0; i < _total; i++) _go[i].Set();
            _done.Wait();
            // A lane's compute threw → C may be partially written; report failure so the caller recomputes
            // on the correct per-tile/PackBoth path (which fully overwrites C).
            return !_faulted;
        }
        finally { Monitor.Exit(_runGate); }
    }

    private static int RoundMr(int x) { int r = (x / 6) * 6; return r < 6 ? 6 : r; }

    /// <summary>A/B knob (env AIDOTNET_CCX_1D=1): force the 1D-N path where its strip fits (huge shapes
    /// still take 2D), to compare against the production 2D default.</summary>
    internal static bool s_force1D =
        System.Environment.GetEnvironmentVariable("AIDOTNET_CCX_1D") == "1";

    /// <summary>A/B knob (env AIDOTNET_CCX_KC): override the 2D-path kc to sweep the barrier-vs-L1-fit tradeoff.</summary>
    internal static int s_kc2dOverride =
        int.TryParse(System.Environment.GetEnvironmentVariable("AIDOTNET_CCX_KC"), out var v) ? v : 0;

    /// <summary>A/B knob (env AIDOTNET_CCX_MINM): override the min-M gate (default 512) to test CCX on the
    /// wide-N small-M DiT shapes (m=256). Lets the 1D-N path (pack B once per CCX) be measured there.</summary>
    internal static int s_minM =
        int.TryParse(System.Environment.GetEnvironmentVariable("AIDOTNET_CCX_MINM"), out var mm) && mm > 0 ? mm : 512;

    private const int Nr32 = 16;
    // mc: enough ic-blocks for the CCX's lanes (≈2·tpc) while keeping the A-panel cache-friendly.
    private static int ChooseMc(int m)
    {
        int mc = (m / (_tpc * 2) / 6) * 6;
        if (mc < 48) mc = 48;
        if (mc > 240) mc = 240;
        return mc;
    }
    // nc: one thin strip per CCX (≈ N/numCcx), Nr-aligned → B-strip fits the CCX's L3 and is reused there.
    private static int ChooseNc(int n)
    {
        int nc = (n / _numCcx / Nr32) * Nr32;
        if (nc < Nr32) nc = Nr32;
        return nc;
    }
}
#endif
