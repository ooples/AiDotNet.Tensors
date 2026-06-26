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
    private static int _n, _initState; // _initState: 0 unstarted, 1 ready, 2 unavailable
    private static readonly object _initGate = new(), _runGate = new();
    private static ManualResetEventSlim[] _go = Array.Empty<ManualResetEventSlim>();
    private static CountdownEvent? _done;
    private static Action<int>? _body;
    private static int _from, _to, _next;
    private static volatile bool _faulted;

    /// <summary>Route the FP32 N-axis parallel-for through the L3-domain-pinned pool. Default ON — measured
    /// 1.08-1.16x bit-exact on the diffusion thin-M wide-N shapes (--ab-pinned). Opt out with
    /// AIDOTNET_PIN_NAXIS=0. Self-limiting: only the N-axis path consults it, and <see cref="For"/> falls
    /// back to the threadpool when the pool is busy/nested/unavailable.</summary>
    internal static bool s_enabled =
        System.Environment.GetEnvironmentVariable("AIDOTNET_PIN_NAXIS") != "0";

    internal static bool IsAvailable { get { EnsureInit(); return _initState == 1; } }

    private static void EnsureInit()
    {
        if (Volatile.Read(ref _initState) != 0) return;
        lock (_initGate)
        {
            if (_initState != 0) return;
            try
            {
                var domains = CpuTopology.DetectL3Domains();
                int total = 0;
                foreach (var d in domains) total += d.Cores;
                if (domains.Length < 1 || total < 2) { Volatile.Write(ref _initState, 2); return; }
                _n = total;
                _go = new ManualResetEventSlim[_n];
                _done = new CountdownEvent(_n);
                int id = 0;
                for (int di = 0; di < domains.Length; di++)
                {
                    for (int c = 0; c < domains[di].Cores; c++)
                    {
                        int myId = id;
                        var dom = domains[di];
                        _go[myId] = new ManualResetEventSlim(false);
                        new Thread(() => Worker(myId, dom)) { IsBackground = true, Name = $"aidotnet-pin-{myId}" }.Start();
                        id++;
                    }
                }
                Volatile.Write(ref _initState, 1);
            }
            catch { Volatile.Write(ref _initState, 2); }
        }
    }

    private static void Worker(int id, CpuTopology.Domain dom)
    {
        CpuTopology.TryPinCurrentThread(dom); // pin to L3 domain (best-effort; correctness is pin-independent)
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
    internal static bool IsAvailable => false;
    internal static bool For(int from, int to, System.Action<int> body) => false;
}
#endif
