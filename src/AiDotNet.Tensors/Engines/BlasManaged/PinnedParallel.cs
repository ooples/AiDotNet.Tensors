#if NET5_0_OR_GREATER
using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// #85 EXPERIMENT (gated <see cref="s_enabled"/>, default OFF): run a parallel-for on PERSISTENT worker
/// threads PINNED per L3 domain. OpenBLAS hits ~59 GF/s/core on pinned physical cores; our threadpool
/// N-axis gets ~42 (thread migration + SMT-sibling contention, per the PerfView/busy-core diagnosis).
/// This is a GENERIC dispatch: it runs the caller's UNCHANGED <c>body(i)</c> work-stealing across the
/// pinned threads, so any GEMM routed through it is BIT-EXACT (identical compute, only the threads
/// differ). Windows-only pinning (via <see cref="CpuTopology.TryPinCurrentThread"/>); on other OSes or
/// when unavailable, <see cref="For"/> returns false and the caller uses the threadpool path.
/// </summary>
internal static class PinnedParallel
{
    private static int _n, _initState, _domainCount; // _initState: 0 unstarted, 1 ready, 2 unavailable
    private static readonly object _initGate = new(), _runGate = new();
    private static ManualResetEventSlim[] _go = Array.Empty<ManualResetEventSlim>();
    private static CountdownEvent? _done;
    private static Action<int>? _body;
    private static int _from, _to, _next;
    private static volatile bool _faulted;
    [ThreadStatic] private static int _curDomain; // this worker's L3-domain (CCX) index, 0..DomainCount-1

    /// <summary>Route the FP32 N-axis parallel-for through the L3-domain-pinned pool. Default ON — measured
    /// 1.08-1.16x bit-exact on the diffusion thin-M wide-N shapes (--ab-pinned). Opt out with
    /// AIDOTNET_PIN_NAXIS=0. Self-limiting: only the N-axis path consults it, and <see cref="For"/> falls
    /// back to the threadpool when the pool is busy/nested/unavailable.</summary>
    internal static bool s_enabled =
        System.Environment.GetEnvironmentVariable("AIDOTNET_PIN_NAXIS") != "0";

    /// <summary>Per-CCX A-replication (env AIDOTNET_PIN_REPLICATE=1): the N-axis packs one packed-A copy
    /// per L3 domain and each pinned worker reads its CCX-local copy via <see cref="CurrentDomain"/>,
    /// killing cross-CCX shared-A reads. Correctness is independent (copies are identical; copy 0 is the
    /// fallback on any non-pinned thread).</summary>
    internal static bool s_replicate =
        System.Environment.GetEnvironmentVariable("AIDOTNET_PIN_REPLICATE") == "1";

    internal static bool IsAvailable { get { EnsureInit(); return _initState == 1; } }

    /// <summary>Distinct L3 domains (CCX) the pool spans (≥1).</summary>
    internal static int DomainCount { get { EnsureInit(); return _domainCount < 1 ? 1 : _domainCount; } }

    /// <summary>The running worker's L3-domain index; 0 on threadpool/non-pool threads (so reading copy 0
    /// is always valid).</summary>
    internal static int CurrentDomain => _curDomain;

    private static void EnsureInit()
    {
        if (Volatile.Read(ref _initState) != 0) return;
        lock (_initGate)
        {
            if (_initState != 0) return;
            try
            {
                // Build (pin target, L3-domain index) per thread. The domain index groups threads by CCX
                // so per-CCX A-replication can hand each worker its local copy. Per-core mode
                // (AIDOTNET_PIN_PERCORE=1): one thread per physical core (no SMT), mapped back to its L3
                // domain. Default: Domain.Cores threads per CCX, each pinned to the CCX mask.
                var l3 = CpuTopology.DetectL3Domains();
                _domainCount = l3.Length;
                var pins = new System.Collections.Generic.List<CpuTopology.Domain>();
                var domIdx = new System.Collections.Generic.List<int>();
                if (System.Environment.GetEnvironmentVariable("AIDOTNET_PIN_PERCORE") == "1")
                {
                    foreach (var core in CpuTopology.DetectPhysicalCores())
                    {
                        int di = 0;
                        for (int j = 0; j < l3.Length; j++) if ((l3[j].Mask & core.Mask) == core.Mask) { di = j; break; }
                        pins.Add(core); domIdx.Add(di);
                    }
                }
                if (pins.Count < 2)
                {
                    pins.Clear(); domIdx.Clear();
                    for (int di = 0; di < l3.Length; di++)
                        for (int c = 0; c < l3[di].Cores; c++) { pins.Add(l3[di]); domIdx.Add(di); }
                }
                if (pins.Count < 2) { Volatile.Write(ref _initState, 2); return; }
                if (_domainCount < 1) _domainCount = 1;
                _n = pins.Count;
                _go = new ManualResetEventSlim[_n];
                _done = new CountdownEvent(_n);
                for (int id = 0; id < _n; id++)
                {
                    int myId = id;
                    var pin = pins[id];
                    int myDom = domIdx[id];
                    _go[myId] = new ManualResetEventSlim(false);
                    new Thread(() => Worker(myId, pin, myDom)) { IsBackground = true, Name = $"aidotnet-pin-{myId}" }.Start();
                }
                Volatile.Write(ref _initState, 1);
            }
            catch { Volatile.Write(ref _initState, 2); }
        }
    }

    private static void Worker(int id, CpuTopology.Domain dom, int domainIndex)
    {
        CpuTopology.TryPinCurrentThread(dom); // pin to L3 domain (best-effort; correctness is pin-independent)
        _curDomain = domainIndex;             // expose the worker's CCX index for per-CCX A-replication
        while (true)
        {
            _go[id].Wait();
            _go[id].Reset();
            try
            {
                var body = _body;
                if (body != null)
                {
                    int i;
                    // Work-stealing: each thread grabs the next index until the range is drained.
                    while ((i = Interlocked.Increment(ref _next) - 1 + _from) < _to) body(i);
                }
            }
            catch { _faulted = true; }
            finally { _done!.Signal(); }
        }
    }

    /// <summary>
    /// Run <paramref name="body"/> for every index in [from, to) on the pinned pool, work-stealing.
    /// Returns false (C UNTOUCHED — safe to fall back to the threadpool) when the pool is unavailable or
    /// busy. THROWS if a body invocation faulted mid-run (C may be partial; caller must not silently
    /// fall back — a deterministic GEMM body never throws, so this only fires on a real defect).
    /// </summary>
    internal static bool For(int from, int to, Action<int> body)
    {
        if (!IsAvailable) return false;
        if (to <= from) return true;
        if (!Monitor.TryEnter(_runGate)) return false; // nested/concurrent use → threadpool
        try
        {
            _body = body; _from = from; _to = to; _next = 0; _faulted = false;
            _done!.Reset();
            for (int i = 0; i < _n; i++) _go[i].Set();
            _done.Wait();
            _body = null;
            if (_faulted) throw new InvalidOperationException("PinnedParallel body faulted.");
            return true;
        }
        finally { Monitor.Exit(_runGate); }
    }
}
#else
namespace AiDotNet.Tensors.Engines.BlasManaged;

// net471 has no function-pointer/JIT GEMM path and no affinity pinning here; the N-axis stays on the
// threadpool. Stub so callers compile; For always returns false (→ threadpool).
internal static class PinnedParallel
{
    internal static bool s_enabled;
    internal static bool s_replicate;
    internal static bool IsAvailable => false;
    internal static int DomainCount => 1;
    internal static int CurrentDomain => 0;
    internal static bool For(int from, int to, System.Action<int> body) => false;
}
#endif
