// Copyright (c) AiDotNet. All rights reserved.

#if NET8_0_OR_GREATER

using System;
using System.IO;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.TensorBoard;
using BenchmarkDotNet.Attributes;

namespace AiDotNet.Tensors.Benchmarks.Serialization;

/// <summary>
/// Issue #218 acceptance benchmark: TensorBoard scalar-flood
/// throughput. Target: ≥ <c>tensorboardX</c>. Writes
/// <see cref="ScalarsPerRun"/> scalars to a fresh log dir per
/// invocation so each run measures cold-write throughput rather than
/// repeatedly appending to the same file.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(5)]
[MaxIterationCount(20)]
public class TensorBoardWriteBenchmarks
{
    private string _logDir = "";
    private IDisposable? _trialOverride;

    [Params(1_000, 10_000, 100_000)]
    public int ScalarsPerRun;

    [GlobalSetup]
    public void Setup()
    {
        _trialOverride = PersistenceGuard.SetTestTrialFilePathOverride(
            Path.Combine(Path.GetTempPath(), "aidn-tb-bench-trial-" + Guid.NewGuid().ToString("N") + ".json"));
    }

    [IterationSetup]
    public void IterSetup()
    {
        // Fresh log dir per iteration so we measure cold-start
        // throughput, not append-to-existing.
        _logDir = Path.Combine(Path.GetTempPath(), "aidn-tb-bench-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_logDir);
    }

    [IterationCleanup]
    public void IterCleanup()
    {
        try { Directory.Delete(_logDir, recursive: true); } catch { /* best-effort */ }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _trialOverride?.Dispose();
    }

    [Benchmark(Baseline = true, Description = "AddScalar repeated — single tag, monotonic step")]
    public int FloodOneTag()
    {
        using var w = TensorBoardSummaryWriter.OpenLogDir(_logDir);
        for (int i = 0; i < ScalarsPerRun; i++)
            w.AddScalar("loss", i * 0.001, i);
        return ScalarsPerRun;
    }

    [Benchmark(Description = "AddScalar across 8 tags — typical multi-metric loop")]
    public int FloodEightTags()
    {
        using var w = TensorBoardSummaryWriter.OpenLogDir(_logDir);
        string[] tags = { "loss", "lr", "acc/train", "acc/val", "grad_norm", "weight_norm", "ppl", "throughput" };
        for (int i = 0; i < ScalarsPerRun; i++)
        {
            w.AddScalar(tags[i & 7], (i & 7) * 0.01 + i * 1e-5, i);
        }
        return ScalarsPerRun;
    }
}

#endif
