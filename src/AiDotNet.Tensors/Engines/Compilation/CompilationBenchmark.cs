using System.Diagnostics;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Phase 3: A/B testing infrastructure for compilation performance.
///
/// Measures steps/sec, throughput, latency, and live-heap delta for
/// eager vs compiled execution paths. Memory tracking uses GC.GetTotalMemory
/// for managed heap deltas and Gen0 collection counts as an allocation proxy.
///
/// Each optimization pass can be toggled independently via TensorCodecOptions
/// to measure its individual contribution to overall performance.
/// </summary>
internal sealed class CompilationBenchmark
{
    /// <summary>
    /// Measures execution performance of a compiled inference plan.
    /// </summary>
    public static BenchmarkResult MeasureInference<T>(
        CompiledInferencePlan<T> plan, int warmupIterations = 50, int measureIterations = 500)
    {
        // Warmup
        for (int i = 0; i < warmupIterations; i++)
            plan.Execute();

        // Force GC before measurement
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        long memBefore = GC.GetTotalMemory(true);
        int gen0Before = GC.CollectionCount(0);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < measureIterations; i++)
            plan.Execute();
        sw.Stop();

        long memAfter = GC.GetTotalMemory(false);
        int gen0After = GC.CollectionCount(0);

        return new BenchmarkResult(
            totalMs: sw.Elapsed.TotalMilliseconds,
            iterations: measureIterations,
            allocatedBytes: Math.Max(0, memAfter - memBefore),
            gen0Collections: gen0After - gen0Before);
    }

    /// <summary>
    /// Measures execution performance of a compiled training plan.
    /// </summary>
    public static BenchmarkResult MeasureTraining<T>(
        CompiledTrainingPlan<T> plan, int warmupIterations = 10, int measureIterations = 100)
    {
        for (int i = 0; i < warmupIterations; i++)
            plan.Step();

        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        long memBefore = GC.GetTotalMemory(true);
        int gen0Before = GC.CollectionCount(0);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < measureIterations; i++)
            plan.Step();
        sw.Stop();

        long memAfter = GC.GetTotalMemory(false);
        int gen0After = GC.CollectionCount(0);

        return new BenchmarkResult(
            totalMs: sw.Elapsed.TotalMilliseconds,
            iterations: measureIterations,
            allocatedBytes: Math.Max(0, memAfter - memBefore),
            gen0Collections: gen0After - gen0Before);
    }

    /// <summary>
    /// Measures execution performance of an arbitrary eager action.
    /// </summary>
    public static BenchmarkResult MeasureEager(
        Action action, int warmupIterations = 50, int measureIterations = 500)
    {
        for (int i = 0; i < warmupIterations; i++)
            action();

        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        long memBefore = GC.GetTotalMemory(true);
        int gen0Before = GC.CollectionCount(0);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < measureIterations; i++)
            action();
        sw.Stop();

        long memAfter = GC.GetTotalMemory(false);
        int gen0After = GC.CollectionCount(0);

        return new BenchmarkResult(
            totalMs: sw.Elapsed.TotalMilliseconds,
            iterations: measureIterations,
            allocatedBytes: Math.Max(0, memAfter - memBefore),
            gen0Collections: gen0After - gen0Before);
    }

    /// <summary>
    /// Compares eager vs compiled execution and returns the speedup factor.
    /// </summary>
    public static ComparisonResult Compare<T>(
        Action eagerAction, CompiledInferencePlan<T> compiledPlan,
        int warmup = 50, int measure = 500)
    {
        var eager = MeasureEager(eagerAction, warmup, measure);
        var compiled = MeasureInference(compiledPlan, warmup, measure);

        return new ComparisonResult(eager, compiled);
    }
}

/// <summary>Benchmark measurement result.</summary>
internal sealed class BenchmarkResult
{
    public double TotalMs { get; }
    public int Iterations { get; }
    public double MsPerIteration => TotalMs / Iterations;
    public double IterationsPerSecond => Iterations / (TotalMs / 1000.0);
    public long AllocatedBytes { get; }
    public int Gen0Collections { get; }
    public bool IsZeroAlloc => Gen0Collections == 0 && AllocatedBytes < 1024;

    internal BenchmarkResult(double totalMs, int iterations, long allocatedBytes, int gen0Collections)
    {
        TotalMs = totalMs;
        Iterations = iterations;
        AllocatedBytes = allocatedBytes;
        Gen0Collections = gen0Collections;
    }

    public override string ToString() =>
        $"{MsPerIteration:F3}ms/iter, {IterationsPerSecond:F0} iter/s, " +
        $"{AllocatedBytes / 1024.0:F1}KB alloc, {Gen0Collections} GC0" +
        (IsZeroAlloc ? " [ZERO-ALLOC]" : "");
}

/// <summary>Comparison between eager and compiled execution.</summary>
internal sealed class ComparisonResult
{
    public BenchmarkResult Eager { get; }
    public BenchmarkResult Compiled { get; }
    public double Speedup => Eager.MsPerIteration / Compiled.MsPerIteration;
    public bool CompiledIsZeroAlloc => Compiled.IsZeroAlloc;

    internal ComparisonResult(BenchmarkResult eager, BenchmarkResult compiled)
    {
        Eager = eager;
        Compiled = compiled;
    }

    public override string ToString() =>
        $"Eager: {Eager}\n" +
        $"Compiled: {Compiled}\n" +
        $"Speedup: {Speedup:F2}x" +
        (CompiledIsZeroAlloc ? " [ZERO-ALLOC]" : "");
}
