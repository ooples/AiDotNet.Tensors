using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Regression tests for the nested-parallelism ThreadPool-starvation fix in
/// <see cref="CpuParallelSettings"/>.
///
/// <para>
/// The bug: float <c>ScaledDotProductAttention</c> runs an outer
/// <see cref="CpuParallelSettings.ParallelForOrSerial(int,int,long,Action{int})"/>
/// over batch·heads, and each iteration calls a (multi-threaded) BlasManaged GEMM.
/// Without a guard the outer workers occupy every ThreadPool thread and then block
/// waiting on the inner loop's tasks, which cannot get a thread — the pool only
/// recovers by injecting threads ~1/500ms, so a single forward at
/// <c>[1, 8, 1024, 64]</c> crawls for minutes (and balloons GC). The fix marks each
/// parallel worker as "in a parallel region" so nested parallel loops collapse to
/// serial automatically.
/// </para>
/// </summary>
public class NestedParallelismTests
{
    private readonly ITestOutputHelper _output;
    public NestedParallelismTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void IsInParallelRegion_FalseAtTopLevel()
    {
        Assert.False(CpuParallelSettings.IsInParallelRegion);
    }

    [Fact]
    public void NestedParallelLoop_RunsSerially_WhenInsideParallelRegion()
    {
        // Force the OUTER loop to dispatch in parallel: huge totalWork + many
        // iterations. On a multi-core box this runs the body on several workers,
        // each of which sets IsInParallelRegion. A NESTED parallel loop must then
        // run on exactly one thread (serial) — proving nesting collapsed. On a
        // single-core box the outer loop is serial (no region) and the nested loop
        // is also serial, so the assertion (one thread) holds either way.
        Assert.False(CpuParallelSettings.IsInParallelRegion);

        bool everyNestedRanSerial = true;
        bool outerSawRegion = false;
        int outerIterations = Math.Max(4, Environment.ProcessorCount * 4);

        CpuParallelSettings.ParallelForOrSerial(0, outerIterations, long.MaxValue / 4, _ =>
        {
            if (CpuParallelSettings.IsInParallelRegion) outerSawRegion = true;

            // A nested loop that, unguarded, would fan out across many threads.
            var threadIds = new ConcurrentDictionary<int, byte>();
            CpuParallelSettings.ParallelForRegion(Math.Max(8, Environment.ProcessorCount * 2),
                _ => threadIds.TryAdd(Environment.CurrentManagedThreadId, 0));

            // Serialized → exactly one distinct thread serviced the nested loop.
            if (threadIds.Count != 1) everyNestedRanSerial = false;
        });

        Assert.True(everyNestedRanSerial,
            "a nested ParallelForRegion fanned out across multiple threads — nesting did not collapse to serial.");
        // The flag must not leak past the loop on the calling thread.
        Assert.False(CpuParallelSettings.IsInParallelRegion);

        if (Environment.ProcessorCount > 1)
            Assert.True(outerSawRegion, "outer parallel loop never marked its workers in-region.");
    }

    [Fact]
    public void NestedParallelForOrSerial_RunsSerially_WhenInsideParallelRegion()
    {
        // Same guarantee for the grain-size-gated ParallelForOrSerial overload:
        // a nested call with above-threshold work still serializes when nested.
        int outerIterations = Math.Max(4, Environment.ProcessorCount * 4);
        bool everyNestedRanSerial = true;

        CpuParallelSettings.ParallelForOrSerial(0, outerIterations, long.MaxValue / 4, _ =>
        {
            var threadIds = new ConcurrentDictionary<int, byte>();
            CpuParallelSettings.ParallelForOrSerial(0, Environment.ProcessorCount * 2, long.MaxValue / 4,
                _ => threadIds.TryAdd(Environment.CurrentManagedThreadId, 0));
            if (threadIds.Count != 1) everyNestedRanSerial = false;
        });

        Assert.True(everyNestedRanSerial,
            "a nested ParallelForOrSerial fanned out across multiple threads — nesting did not collapse to serial.");
    }

    [Theory]
    // The shape from #467 that triggered the deadlock, plus a couple of nearby
    // shapes that route through the multi-threaded GEMM (D large enough that the
    // per-head Q·K^T / P·V cross the GPU/parallel work threshold).
    [InlineData(1, 8, 1024, 64)]
    [InlineData(2, 8, 512, 64)]
    [InlineData(1, 16, 256, 72)]
    public void FloatSdpa_LargeShape_CompletesWithoutStarvation(int b, int h, int s, int d)
    {
        var engine = new CpuEngine();
        var q = RandomFloat(new[] { b, h, s, d }, 1);
        var k = RandomFloat(new[] { b, h, s, d }, 2);
        var v = RandomFloat(new[] { b, h, s, d }, 3);

        // Before the fix, this nested-parallel path starves the ThreadPool and runs
        // for minutes. After, it's well under a second. A generous 60s ceiling makes
        // the test robust to a loaded machine while still failing hard on a regress.
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var task = Task.Run(() =>
            engine.ScaledDotProductAttention(q, k, v, mask: null, scale: null, out _));
        bool finished = task.Wait(TimeSpan.FromSeconds(60));
        sw.Stop();

        Assert.True(finished,
            $"float SDPA [{b},{h},{s},{d}] did not finish within 60s — nested-parallelism starvation regressed.");
        _output.WriteLine($"[{b},{h},{s},{d}] float SDPA completed in {sw.Elapsed.TotalMilliseconds:F0} ms");

        var outSpan = task.Result.AsSpan();
        for (int i = 0; i < outSpan.Length; i++)
            // float.IsFinite is unavailable on net471 — use the NaN/Inf primitives.
            Assert.True(!float.IsNaN(outSpan[i]) && !float.IsInfinity(outSpan[i]),
                $"output[{i}] is not finite ({outSpan[i]}).");
    }

    [Fact]
    public void FloatSdpa_MatchesDoubleReference_AtLargeShape()
    {
        // Correctness alongside the no-deadlock guarantee: the (now serialized)
        // float per-head GEMM must still produce the same attention as the double
        // reference path (within FP32 tolerance).
        var engine = new CpuEngine();
        const int B = 1, H = 4, S = 256, D = 64;
        var qf = RandomFloat(new[] { B, H, S, D }, 11);
        var kf = RandomFloat(new[] { B, H, S, D }, 12);
        var vf = RandomFloat(new[] { B, H, S, D }, 13);
        var qd = ToDouble(qf); var kd = ToDouble(kf); var vd = ToDouble(vf);

        var fOut = engine.ScaledDotProductAttention(qf, kf, vf, mask: null, scale: null, out _);
        var dOut = engine.ScaledDotProductAttention(qd, kd, vd, mask: null, scale: null, out _);

        var fs = fOut.AsSpan();
        var ds = dOut.AsSpan();
        Assert.Equal(ds.Length, fs.Length);
        double maxDiff = 0;
        for (int i = 0; i < fs.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(fs[i] - ds[i]));
        _output.WriteLine($"float-vs-double SDPA maxAbsDiff = {maxDiff:E3}");
        Assert.True(maxDiff < 2e-3, $"float SDPA drifted {maxDiff:E3} from double reference (> 2e-3).");
    }

    // ----------------- Helpers -----------------

    private static Tensor<float> RandomFloat(int[] shape, int seed)
    {
        var t = new Tensor<float>(shape);
        var rng = new Random(seed);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor<double> ToDouble(Tensor<float> t)
    {
        var src = t.AsSpan();
        var result = new Tensor<double>((int[])t._shape.Clone());
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dst[i] = src[i];
        return result;
    }
}
