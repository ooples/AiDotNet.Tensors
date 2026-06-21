using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Diagnostic-only GEMM shape histogram. Records (m, n, k, transA, transB, dtype)
/// call counts + cumulative wall time so a profiling harness can see exactly which
/// shapes a workload issues and their achieved GFLOP/s. Off unless the env var
/// <c>AIDOTNET_GEMM_HISTOGRAM=1</c> is set (read once at startup) or
/// <see cref="Enabled"/> is toggled directly. The record path is intentionally cheap
/// (a dictionary upsert under a striped lock) but is NOT meant for production builds —
/// it exists to drive kernel-routing decisions, not to ship.
/// </summary>
public static class GemmShapeHistogram
{
    public readonly struct Key : IEquatable<Key>
    {
        public readonly int M, N, K;
        public readonly bool TransA, TransB, IsFloat;
        public Key(int m, int n, int k, bool transA, bool transB, bool isFloat)
        { M = m; N = n; K = k; TransA = transA; TransB = transB; IsFloat = isFloat; }
        public bool Equals(Key o) => M == o.M && N == o.N && K == o.K
            && TransA == o.TransA && TransB == o.TransB && IsFloat == o.IsFloat;
        public override bool Equals(object? o) => o is Key k && Equals(k);
        public override int GetHashCode()
        {
            unchecked
            {
                int h = M;
                h = h * 397 ^ N; h = h * 397 ^ K;
                h = h * 397 ^ (TransA ? 1 : 0);
                h = h * 397 ^ (TransB ? 2 : 0);
                h = h * 397 ^ (IsFloat ? 4 : 0);
                return h;
            }
        }
    }

    public sealed class Stat { public long Count; public long TicksTotal; }

    /// <summary>Toggle recording. Initialized from AIDOTNET_GEMM_HISTOGRAM=1.</summary>
    public static bool Enabled = Environment.GetEnvironmentVariable("AIDOTNET_GEMM_HISTOGRAM") == "1";

    private static readonly ConcurrentDictionary<Key, Stat> Stats = new();

    /// <summary>Record a single GEMM call's shape and elapsed ticks (Stopwatch ticks).</summary>
    public static void Record(int m, int n, int k, bool transA, bool transB, bool isFloat, long elapsedTicks)
    {
        var stat = Stats.GetOrAdd(new Key(m, n, k, transA, transB, isFloat), static _ => new Stat());
        System.Threading.Interlocked.Increment(ref stat.Count);
        System.Threading.Interlocked.Add(ref stat.TicksTotal, elapsedTicks);
    }

    public static void Clear() => Stats.Clear();

    /// <summary>
    /// Render the top <paramref name="topN"/> shapes by cumulative time, with achieved
    /// GFLOP/s (2·m·n·k·count / total-seconds). Highlights shapes that run well below peak.
    /// </summary>
    public static string Report(int topN = 40)
    {
        var rows = new List<(Key key, long count, double secs, double gflops)>();
        double tickToSec = 1.0 / Stopwatch.Frequency;
        foreach (var kv in Stats)
        {
            double secs = kv.Value.TicksTotal * tickToSec;
            double flops = 2.0 * kv.Key.M * kv.Key.N * kv.Key.K * kv.Value.Count;
            double gflops = secs > 0 ? flops / secs / 1e9 : 0;
            rows.Add((kv.Key, kv.Value.Count, secs, gflops));
        }
        rows.Sort((x, y) => y.secs.CompareTo(x.secs));
        var sb = new StringBuilder();
        double totalSecs = 0; long totalCalls = 0;
        foreach (var r in rows) { totalSecs += r.secs; totalCalls += r.count; }
        sb.AppendLine($"=== GEMM shape histogram: {rows.Count} distinct shapes, {totalCalls} calls, {totalSecs:F2}s total ===");
        sb.AppendLine($"{"m",6} {"n",6} {"k",6} tA tB dt {"calls",7} {"sec",8} {"%",5} {"GFLOP/s",9}");
        int shown = 0;
        foreach (var r in rows)
        {
            if (shown++ >= topN) break;
            string dt = r.key.IsFloat ? "f" : "d";
            double pct = totalSecs > 0 ? 100 * r.secs / totalSecs : 0;
            sb.AppendLine($"{r.key.M,6} {r.key.N,6} {r.key.K,6} " +
                $"{(r.key.TransA ? 'T' : '.'),2} {(r.key.TransB ? 'T' : '.'),2} {dt,2} " +
                $"{r.count,7} {r.secs,8:F3} {pct,5:F1} {r.gflops,9:F1}");
        }
        return sb.ToString();
    }
}
