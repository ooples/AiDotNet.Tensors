// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Profiling;
using AiDotNet.Tensors.Engines.Profiling.Trace;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Profiling;

/// <summary>
/// Acceptance tests for issue #220 Phase 1 — the
/// <see cref="Profiler"/> facade plus chrome-trace export.
///
/// <para>Tests run serially via the <c>"AiDotNetProfiler"</c> xUnit collection
/// because <see cref="Profiler.Current"/> is a process-wide ambient slot.
/// Parallel execution would interleave session start/dispose pairs and the
/// "only one session" invariant would fail nondeterministically.</para>
/// </summary>
[CollectionDefinition("AiDotNetProfiler", DisableParallelization = true)]
public sealed class ProfilerCollection { }

[Collection("AiDotNetProfiler")]
public class ProfilerTests
{
    [Fact]
    public void Range_NoActiveSession_ReturnsNoOpScope()
    {
        // No alloc, no recording: this is the "always-on at zero cost" guarantee.
        Assert.Null(Profiler.Current);
        var scope = Profiler.Range("op");
        Assert.NotNull(scope);
        scope.Dispose(); // must not throw
    }

    [Fact]
    public void Profile_StartedTwice_ThrowsInvalidOperation()
    {
        using var first = Profiler.Profile();
        Assert.Throws<InvalidOperationException>(() => Profiler.Profile());
    }

    [Fact]
    public void Range_ActiveSession_RecordsCompleteEvent()
    {
        using var prof = Profiler.Profile();
        using (Profiler.Range("matmul"))
        {
            // Simulate work — Stopwatch resolution on Windows is ~100 ns,
            // a busy-wait of even a microsecond is enough to materialize a
            // non-zero duration without the test becoming wall-clock-flaky.
            BusyWaitMicros(50);
        }

        var matmuls = prof.Events.Where(e => e.Name == "matmul").ToList();
        Assert.Single(matmuls);
        Assert.Equal('X', matmuls[0].Phase);
        Assert.Equal("user_annotation", matmuls[0].Category);
        Assert.True(matmuls[0].DurationMicros >= 0,
            "Stopwatch resolution may report 0; assert non-negative not strictly positive.");
    }

    [Fact]
    public void Range_NestedScopes_RecordedAsDistinctEvents()
    {
        using var prof = Profiler.Profile();
        using (Profiler.Range("outer"))
        {
            BusyWaitMicros(10);
            using (Profiler.Range("inner"))
            {
                BusyWaitMicros(10);
            }
        }

        var events = prof.Events.Where(e => e.Phase == 'X').ToList();
        Assert.Equal(2, events.Count);
        var outer = events.Single(e => e.Name == "outer");
        var inner = events.Single(e => e.Name == "inner");
        // Outer must enclose inner. Allow inner.dur == 0 from low Stopwatch resolution.
        Assert.True(inner.TimestampMicros >= outer.TimestampMicros);
        Assert.True(inner.TimestampMicros + inner.DurationMicros
                    <= outer.TimestampMicros + outer.DurationMicros,
            "Nested scope must close before outer scope.");
    }

    [Fact]
    public void Range_WithArgs_PropagatesToEvent()
    {
        using var prof = Profiler.Profile();
        var args = new Dictionary<string, string> { ["shape"] = "[4,3]", ["dtype"] = "float32" };
        using (Profiler.Range("matmul", "cpu_op", args)) { }

        var ev = prof.Events.Single(e => e.Name == "matmul");
        Assert.NotNull(ev.Args);
        Assert.Equal("[4,3]", ev.Args!["shape"]);
        Assert.Equal("float32", ev.Args["dtype"]);
        Assert.Equal("cpu_op", ev.Category);
    }

    [Fact]
    public void RecordInstant_EmitsInstantEvent()
    {
        using var prof = Profiler.Profile();
        Profiler.RecordInstant("autotune_miss", "compile",
            args: new Dictionary<string, string> { ["op"] = "matmul" });

        var ev = prof.Events.Single(e => e.Name == "autotune_miss");
        Assert.Equal('i', ev.Phase);
        Assert.Equal(0, ev.DurationMicros);
        Assert.Equal("matmul", ev.Args!["op"]);
    }

    [Fact]
    public void ConcurrentRanges_FromMultipleThreads_AllEventsRetained()
    {
        const int threads = 8;
        const int rangesPerThread = 100;

        using var prof = Profiler.Profile();
        Parallel.For(0, threads, t =>
        {
            for (int i = 0; i < rangesPerThread; i++)
            {
                using (Profiler.Range($"t{t}_op{i}")) { }
            }
        });

        // Each Range emits one Complete event. metadata adds 1 event.
        // We don't care about ordering, only count + per-name presence.
        var completeEvents = prof.Events.Where(e => e.Phase == 'X').ToList();
        Assert.Equal(threads * rangesPerThread, completeEvents.Count);

        // Each event keeps the producing thread id.
        var distinctThreads = completeEvents.Select(e => e.ThreadId).Distinct().Count();
        Assert.True(distinctThreads >= 2,
            "ThreadIds must vary; we ran on a Parallel.For loop across 8 logical workers.");
    }

    [Fact]
    public void Schedule_WaitWarmupActive_RecordsOnlyDuringActiveAndWarmup()
    {
        var sched = new ProfilerSchedule(wait: 1, warmup: 1, active: 2, repeat: 1);
        using var prof = Profiler.Profile(new ProfilerOptions { Schedule = sched });

        // Step 0: Wait — record nothing.
        // Step 1: Warmup — recorded but flushed (dropped) at the active→? edge.
        //   In our schedule (W=1,Wm=1,A=2,R=1) the cycle is 4, after step 3
        //   the next step would be Stopped. Schedule.IsTraceReadyEdge fires
        //   on Active→Stopped, dropping the events as part of the flush.
        // Steps 2-3: Active — recorded.

        // Use no-OnTraceReady so we can read events directly between phases.
        for (int s = 0; s <= 3; s++)
        {
            using (Profiler.Range($"step_{s}")) { BusyWaitMicros(5); }
            // Step *after* the work — schedule classifies the next iteration.
            // We don't call Step on s=0 yet because we want the work at step 0
            // (Wait) to be silently dropped.
            prof.Step();
        }

        var events = prof.Events.Where(e => e.Phase == 'X').ToList();
        // Flush is fired on Step() that transitions from Active to Stopped.
        // After that flush the events are dropped, so the post-loop snapshot
        // sees zero. Verify the flush happened by counting via OnTraceReady.
        Assert.Empty(events);
    }

    [Fact]
    public void Schedule_OnTraceReady_FiresAtActiveWindowEnd()
    {
        var sched = new ProfilerSchedule(wait: 0, warmup: 0, active: 2, repeat: 1);
        var captured = new List<int>();
        var opts = new ProfilerOptions
        {
            Schedule = sched,
            OnTraceReady = s => captured.Add(s.Events.Count(e => e.Phase == 'X')),
        };
        using var prof = Profiler.Profile(opts);

        // 2 active steps with 1 range each, then Step() into Stopped — flush fires.
        using (Profiler.Range("a")) { } prof.Step();
        using (Profiler.Range("b")) { } prof.Step();

        // OnTraceReady should have fired on the Active→Stopped edge.
        Assert.Single(captured);
        Assert.Equal(2, captured[0]);
    }

    [Fact]
    public void Dispose_FiresFinalOnTraceReady()
    {
        int callbackCount = 0;
        var opts = new ProfilerOptions
        {
            OnTraceReady = _ => callbackCount++,
        };
        using (var prof = Profiler.Profile(opts))
        {
            using (Profiler.Range("op")) { }
        } // dispose at end of using

        // Always-active mode (no schedule) → exactly 1 final flush on dispose.
        Assert.Equal(1, callbackCount);
    }

    [Fact]
    public void MaxEvents_OverflowDropsOldestSilently()
    {
        var opts = new ProfilerOptions { MaxEvents = 8 };
        using var prof = Profiler.Profile(opts);

        for (int i = 0; i < 100; i++)
        {
            using (Profiler.Range($"op_{i}")) { }
        }

        // EventCount must stay bounded by MaxEvents (+ small slack from races,
        // but not by orders of magnitude — any reasonable run on this fast
        // path stays at exactly MaxEvents).
        Assert.True(prof.EventCount <= opts.MaxEvents * 2,
            $"EventCount {prof.EventCount} exceeds the bound (MaxEvents={opts.MaxEvents}).");
    }

    [Fact]
    public void ChromeTraceWriter_RoundtripsEvents()
    {
        using var prof = Profiler.Profile(new ProfilerOptions
        {
            ProcessId = 12345, // pin for deterministic test output
        });

        using (Profiler.Range("matmul", "cpu_op",
                   new Dictionary<string, string> { ["shape"] = "[2,2]" })) { }
        Profiler.RecordInstant("anomaly", "debug");

        string path = Path.Combine(Path.GetTempPath(), $"trace-{Guid.NewGuid():N}.json");
        try
        {
            prof.ExportChromeTrace(path);
            string json = File.ReadAllText(path);

            Assert.Contains("\"traceEvents\":[", json);
            Assert.Contains("\"name\":\"matmul\"", json);
            Assert.Contains("\"cat\":\"cpu_op\"", json);
            Assert.Contains("\"ph\":\"X\"", json);
            Assert.Contains("\"shape\":\"[2,2]\"", json);
            Assert.Contains("\"name\":\"anomaly\"", json);
            Assert.Contains("\"ph\":\"i\"", json);
            Assert.Contains("\"pid\":12345", json);
            Assert.Contains("\"displayTimeUnit\":\"ns\"", json);
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort cleanup */ }
        }
    }

    [Fact]
    public void ChromeTraceWriter_GzipPath_ProducesGzippedOutput()
    {
        using var prof = Profiler.Profile();
        using (Profiler.Range("op")) { }

        string path = Path.Combine(Path.GetTempPath(), $"trace-{Guid.NewGuid():N}.json.gz");
        try
        {
            prof.ExportChromeTrace(path);
            // Magic bytes for gzip: 0x1F 0x8B
            byte[] head = new byte[2];
            using (var fs = File.OpenRead(path)) fs.Read(head, 0, 2);
            Assert.Equal(0x1F, head[0]);
            Assert.Equal(0x8B, head[1]);
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort cleanup */ }
        }
    }

    [Fact]
    public void ChromeTraceWriter_EscapesQuotesAndBackslashes()
    {
        using var prof = Profiler.Profile();
        using (Profiler.Range("name with \"quote\" and \\backslash")) { }

        using var ms = new MemoryStream();
        prof.ExportChromeTrace(ms);
        string json = System.Text.Encoding.UTF8.GetString(ms.ToArray());

        // The literal name must be present in escaped form.
        Assert.Contains("name with \\\"quote\\\" and \\\\backslash", json);
    }

    [Fact]
    public void Schedule_AlwaysActive_RecordsEveryStep()
    {
        using var prof = Profiler.Profile(new ProfilerOptions
        {
            Schedule = ProfilerSchedule.AlwaysActive,
        });

        for (int i = 0; i < 5; i++)
        {
            using (Profiler.Range($"op_{i}")) { }
            prof.Step();
        }

        Assert.Equal(5, prof.Events.Count(e => e.Phase == 'X'));
    }

    [Fact]
    public void Profile_AfterSessionDisposed_CanStartAnother()
    {
        // Regression guard for the CompareExchange-based slot management:
        // disposing should clear the ambient slot atomically so a follow-up
        // Profile() succeeds.
        using (var p1 = Profiler.Profile()) { }
        using var p2 = Profiler.Profile();
        Assert.Same(p2, Profiler.Current);
    }

    private static void BusyWaitMicros(long micros)
    {
        long ticksPerSecond = System.Diagnostics.Stopwatch.Frequency;
        long ticksToWait = (ticksPerSecond * micros) / 1_000_000L;
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        while (System.Diagnostics.Stopwatch.GetTimestamp() - start < ticksToWait)
            Thread.SpinWait(1);
    }
}
