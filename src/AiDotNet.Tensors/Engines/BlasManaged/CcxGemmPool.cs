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
            try { DoWork(ccx, lane); }
            catch { /* a lane fault must not deadlock the join; correctness is gated by tests */ }
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
            GotoGemmFp32.PackBPanel(_b, _ldb, jc, effNc, _k, _kc, pkb, lane, _tpc);
            bar.SignalAndWait();
            for (int ib = lane; ib < numIc; ib += _tpc)
            {
                int ic = ib * _mc; int effMc = Math.Min(_mc, _m - ic); if (effMc <= 0) continue;
                GotoGemmFp32.RunTilePackedB(_a, _lda, _b, _ldb, _c, _ldc, ic, jc, effMc, effNc, _k, _mc, _nc, _kc, pkb);
            }
            bar.SignalAndWait();
        }
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
        if (m < 1536 || n < 1536 || k < 256) return false;
        if ((long)m * n * k < CcxMinWork) return false;
        if (n > 2 * m || m > 2 * n) return false;            // not balanced → per-tile/PackBoth
        if ((n / _numCcx) < Nr32) return false;              // strips too thin to tile
        int mc = ChooseMc(m), nc = ChooseNc(n), kc = 256;
        // The CCX win requires the per-CCX B-strip (K·nc) to stay cache-friendly. Above ~2MB the microkernel
        // reads it from L3 and per-tile's L2-resident per-panel B wins (MEASURED: sq2048 K·nc=1MB CCX 1.14x;
        // sq4096 K·nc=4MB CCX 0.86x). So cap the strip → huge squares fall back to per-tile.
        if ((long)k * nc > 512L * 1024) return false;
        if (CpuParallelSettings.MaxDegreeOfParallelism < _total) return false; // restricted budget → per-tile
        if (!Monitor.TryEnter(_runGate)) return false;       // concurrent GEMM owns the pool → per-tile
        try
        {
            long need = GotoGemmFp32.PackedBLen(nc, k, kc);
            if (need > _bbufLen)
            {
                for (int cc = 0; cc < _numCcx; cc++)
                {
                    if (_bbuf[cc] != IntPtr.Zero) Marshal.FreeHGlobal(_bbuf[cc]);
                    _bbuf[cc] = Marshal.AllocHGlobal(checked((int)(need * sizeof(float))));
                }
                _bbufLen = need;
            }
            _a = a; _b = b; _c = c; _lda = lda; _ldb = ldb; _ldc = ldc;
            _m = m; _n = n; _k = k; _mc = mc; _nc = nc; _kc = kc;
            _done!.Reset();
            for (int i = 0; i < _total; i++) _go[i].Set();
            _done.Wait();
            return true;
        }
        finally { Monitor.Exit(_runGate); }
    }

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
