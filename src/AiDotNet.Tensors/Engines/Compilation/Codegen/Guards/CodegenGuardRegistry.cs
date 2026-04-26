// Copyright (c) AiDotNet. All rights reserved.
// Thread-safe registry for compiled codegen kernels keyed by
// compilation guard. Enforces a recompile budget so pathological
// shape sweeps don't blow up the cache.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Guards;

/// <summary>
/// Central registry for shape-bucket policy + recompile-log
/// destination + compiled-kernel cache keyed by
/// <see cref="CompilationGuard"/>. Matches the role of PyTorch's
/// guard system — cheap at call time, decisive on recompile vs.
/// reuse.
/// </summary>
/// <remarks>
/// <para><b>Why a process-wide singleton:</b></para>
/// <para>
/// Compiled kernels are expensive (JIT in Phase B, external
/// compilation for Phase C emitters). Reusing across tapes / model
/// instances is essential for perf. The registry is thread-safe via
/// <see cref="ConcurrentDictionary{TKey, TValue}"/>; per-thread
/// policy overrides are available through
/// <see cref="SetPolicyForThread"/>.
/// </para>
/// <para><b>Recompile budget:</b></para>
/// <para>
/// A recompile budget bounds how many times a given <i>graph
/// content hash</i> (ignoring shape) may be recompiled. If a hot
/// loop sweeps through many shapes, we recompile up to the budget
/// then stop — further mismatched calls fall back to the composed-
/// ops path. Prevents the pathological "compile cache blows out
/// memory and CPU" scenario PyTorch calls out in its own docs.
/// </para>
/// </remarks>
public static class CodegenGuardRegistry
{
    // ─── Shape-bucket policy ─────────────────────────────────────────

    private static IShapeBucketPolicy _globalPolicy = new PowerOfTwoBucketPolicy();

    [ThreadStatic]
    private static IShapeBucketPolicy? _threadPolicy;

    /// <summary>
    /// The active shape-bucket policy — thread override takes
    /// precedence over the process-wide default.
    /// </summary>
    public static IShapeBucketPolicy Policy => _threadPolicy ?? _globalPolicy;

    /// <summary>Sets the process-wide default policy.</summary>
    public static void SetPolicy(IShapeBucketPolicy policy)
        => _globalPolicy = policy ?? throw new ArgumentNullException(nameof(policy));

    /// <summary>
    /// Overrides the policy for the calling thread only — useful for
    /// tests and for ad-hoc configuration within a single training
    /// loop without touching global state.
    /// </summary>
    public static IDisposable SetPolicyForThread(IShapeBucketPolicy policy)
    {
        if (policy is null) throw new ArgumentNullException(nameof(policy));
        var prev = _threadPolicy;
        _threadPolicy = policy;
        return new Restorer(() => _threadPolicy = prev);
    }

    // ─── Compiled kernel cache ───────────────────────────────────────

    private static readonly ConcurrentDictionary<CompilationGuard, CodegenKernel> _cache
        = new();

    /// <summary>
    /// Number of entries currently cached — exposed for Phase G
    /// telemetry.
    /// </summary>
    public static int CacheEntryCount => _cache.Count;

    /// <summary>
    /// Returns the cached kernel for the given guard, or null if
    /// no match exists.
    /// </summary>
    public static CodegenKernel? Lookup(CompilationGuard guard)
        => _cache.TryGetValue(guard, out var k) ? k : null;

    /// <summary>
    /// Stores a compiled kernel in the cache under the given guard.
    /// Returns true on successful insert; false if an entry already
    /// existed (in which case the existing kernel is preserved — the
    /// caller should use <see cref="Lookup"/> result).
    /// </summary>
    public static bool Insert(CompilationGuard guard, CodegenKernel kernel)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        return _cache.TryAdd(guard, kernel);
    }

    /// <summary>
    /// Clears the cache. Useful for tests and for unloading a
    /// workload whose compiled kernels won't be reused.
    /// </summary>
    public static void Clear()
    {
        _cache.Clear();
        _recompileCount.Clear();
        // ConcurrentQueue<T>.Clear is unavailable on net471 — drain explicitly.
        while (_recompileLog.TryDequeue(out _)) { }
    }

    // ─── Recompile budget + log ──────────────────────────────────────

    private static int _globalBudget = 32;

    /// <summary>
    /// Maximum number of recompiles allowed per unique graph content
    /// hash across the life of the process. Defaults to 32 — matches
    /// PyTorch's default cache size limit. Set via
    /// <see cref="SetRecompileBudget"/>.
    /// </summary>
    public static int RecompileBudget => _globalBudget;

    /// <summary>Sets the process-wide recompile budget.</summary>
    public static void SetRecompileBudget(int budget)
    {
        if (budget < 0) throw new ArgumentOutOfRangeException(nameof(budget));
        _globalBudget = budget;
    }

    private static readonly ConcurrentDictionary<long, int> _recompileCount = new();
    private static readonly ConcurrentQueue<RecompileLogEntry> _recompileLog = new();
    private const int MaxLogEntries = 1024;

    /// <summary>
    /// Records a recompile event and returns whether the budget
    /// permits another recompile for this graph content hash. Call
    /// this once per compilation attempt; budget-exceeded calls
    /// should fall back to the composed-ops path instead of
    /// compiling.
    /// </summary>
    public static bool TryReserveRecompile(long graphContentHash, string reason)
    {
        int next = _recompileCount.AddOrUpdate(graphContentHash, 1, (_, c) => c + 1);
        var entry = new RecompileLogEntry(
            Timestamp: DateTimeOffset.UtcNow,
            GraphHash: graphContentHash,
            AttemptIndex: next,
            Reason: reason ?? "(unspecified)",
            Allowed: next <= _globalBudget);
        _recompileLog.Enqueue(entry);
        while (_recompileLog.Count > MaxLogEntries)
            _recompileLog.TryDequeue(out _);
        return entry.Allowed;
    }

    /// <summary>
    /// Returns a snapshot of the recompile log (oldest first). Used
    /// by Phase G to surface <c>TORCH_LOGS=recompiles</c>-equivalent
    /// diagnostics to operators.
    /// </summary>
    public static IReadOnlyList<RecompileLogEntry> DumpRecompileLog()
    {
        var arr = _recompileLog.ToArray();
        return arr;
    }

    private sealed class Restorer : IDisposable
    {
        private readonly Action _restore;
        private bool _disposed;
        public Restorer(Action restore) => _restore = restore;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _restore();
        }
    }
}

/// <summary>
/// A single recompile log entry — used by Phase G observability to
/// surface why a compiled kernel was re-emitted.
/// </summary>
public readonly record struct RecompileLogEntry(
    DateTimeOffset Timestamp,
    long GraphHash,
    int AttemptIndex,
    string Reason,
    bool Allowed);
