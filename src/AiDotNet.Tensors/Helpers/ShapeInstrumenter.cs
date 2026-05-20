// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Per-shape GEMM-call profiler (issue #403 Phase A.1).
///
/// <para>Subscribes to <see cref="BlasProvider.ShapeLogHook"/> for the
/// duration of a scope, accumulates a unique-shape catalog keyed by
/// (M, N, K, transA, transB), and dumps it in descending call-count order
/// on dispose. Designed to wrap a single training step so the per-substep
/// shape mix is visible without instrumenting every GEMM call site.</para>
///
/// <para>Single-consumer: BlasProvider exposes one delegate slot. Concurrent
/// instances throw on construction to guarantee a clean snapshot. The hook
/// is fast-path null-checked, so per-call cost is one volatile read when
/// no instance is active.</para>
///
/// <para>Usage:</para>
/// <code>
/// using (var probe = new ShapeInstrumenter())
/// {
///     RunOneTrainingStep();
///     probe.PrintCatalog(); // or probe.GetCatalog() / probe.GetSnapshot()
/// }
/// </code>
/// </summary>
internal sealed class ShapeInstrumenter : IDisposable
{
    private static int _activeCount;
    private readonly Action<int, int, int, bool, bool>? _previousHook;
    private readonly ConcurrentDictionary<ShapeKey, long> _counts = new();
    private bool _disposed;

    public ShapeInstrumenter()
    {
        if (System.Threading.Interlocked.Increment(ref _activeCount) != 1)
        {
            System.Threading.Interlocked.Decrement(ref _activeCount);
            throw new InvalidOperationException(
                "Another ShapeInstrumenter is already active. BlasProvider.ShapeLogHook is a single-consumer slot.");
        }

        _previousHook = BlasProvider.ShapeLogHook;
        BlasProvider.ShapeLogHook = OnGemm;
    }

    private void OnGemm(int m, int n, int k, bool transA, bool transB)
    {
        var key = new ShapeKey(m, n, k, transA, transB);
        _counts.AddOrUpdate(key, 1L, static (_, v) => v + 1L);
    }

    /// <summary>Total GEMM calls observed since construction.</summary>
    public long TotalCalls => _counts.Values.Sum();

    /// <summary>Number of unique (M,N,K,transA,transB) tuples observed.</summary>
    public int UniqueShapes => _counts.Count;

    /// <summary>
    /// Snapshot of the catalog ordered by descending call count. Safe to call
    /// while the probe is still active.
    /// </summary>
    public IReadOnlyList<ShapeEntry> GetCatalog()
    {
        return _counts
            .Select(kvp => new ShapeEntry(kvp.Key.M, kvp.Key.N, kvp.Key.K, kvp.Key.TransA, kvp.Key.TransB, kvp.Value))
            .OrderByDescending(e => e.Calls)
            .ThenByDescending(e => (long)e.M * e.N * e.K)
            .ToList();
    }

    /// <summary>
    /// Renders the catalog as a fixed-width markdown table suitable for
    /// pasting into PR descriptions or the Phase A investigation report.
    /// </summary>
    public string FormatCatalog()
    {
        var entries = GetCatalog();
        var sb = new StringBuilder();
        sb.AppendLine($"Unique shapes: {entries.Count} | Total calls: {TotalCalls}");
        sb.AppendLine();
        sb.AppendLine("| Calls |     M |     N |     K | tA | tB |       FLOPs |");
        sb.AppendLine("|------:|------:|------:|------:|:--:|:--:|------------:|");
        foreach (var e in entries)
        {
            double flops = 2.0 * e.M * e.N * e.K * e.Calls;
            sb.AppendLine($"| {e.Calls,5} | {e.M,5} | {e.N,5} | {e.K,5} |  {(e.TransA ? "T" : "N")} |  {(e.TransB ? "T" : "N")} | {flops,11:N0} |");
        }
        return sb.ToString();
    }

    /// <summary>Convenience: prints <see cref="FormatCatalog"/> to console.</summary>
    public void PrintCatalog() => Console.WriteLine(FormatCatalog());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        BlasProvider.ShapeLogHook = _previousHook;
        System.Threading.Interlocked.Decrement(ref _activeCount);
    }

    private readonly record struct ShapeKey(int M, int N, int K, bool TransA, bool TransB);

    /// <summary>Public-facing catalog entry.</summary>
    internal readonly record struct ShapeEntry(int M, int N, int K, bool TransA, bool TransB, long Calls);
}
