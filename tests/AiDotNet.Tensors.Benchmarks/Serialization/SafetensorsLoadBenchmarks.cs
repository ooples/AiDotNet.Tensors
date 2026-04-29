// Copyright (c) AiDotNet. All rights reserved.

#if NET8_0_OR_GREATER

using System;
using System.IO;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Serialization.Safetensors;
using BenchmarkDotNet.Attributes;

namespace AiDotNet.Tensors.Benchmarks.Serialization;

/// <summary>
/// Issue #218 acceptance benchmark: safetensors load throughput.
/// Targets: within 10% of the Rust <c>safetensors</c> crate on Linux,
/// clear win on Windows (no mmap perf cliff). Three load strategies
/// are compared on a representative payload (transformer block ≈ 25
/// tensors / ~50 MB at FP32) so the user sees the cost of each path
/// — heap-buffered reader, mmap, and per-tensor zero-copy slice.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(5)]
[MaxIterationCount(20)]
public class SafetensorsLoadBenchmarks
{
    private string _path = "";
    private IDisposable? _trialOverride;

    [Params(64, 256, 1024)]
    public int HiddenDim;

    [GlobalSetup]
    public void Setup()
    {
        // Test-trial override so the benchmark machine doesn't tick
        // the developer's real ~/.aidotnet/tensors-trial.json budget.
        _trialOverride = PersistenceGuard.SetTestTrialFilePathOverride(
            Path.Combine(Path.GetTempPath(), "aidn-bench-trial-" + Guid.NewGuid().ToString("N") + ".json"));

        _path = Path.Combine(Path.GetTempPath(), $"aidn-bench-{HiddenDim}-{Guid.NewGuid():N}.safetensors");
        var rng = new Random(0);
        using var w = SafetensorsWriter.Create(_path);
        // 25 tensors mirrors a single transformer block (q/k/v/o + 2
        // norms + 3 mlp + biases) at the chosen hidden width.
        for (int i = 0; i < 25; i++)
        {
            int rows = i % 4 == 0 ? 4 * HiddenDim : HiddenDim;
            float[] data = new float[rows * HiddenDim];
            for (int k = 0; k < data.Length; k++) data[k] = (float)(rng.NextDouble() * 2 - 1);
            w.Add($"layer.{i}.weight", new Tensor<float>(data, new[] { rows, HiddenDim }));
        }
        w.Save();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        try { File.Delete(_path); } catch { /* best-effort */ }
        _trialOverride?.Dispose();
    }

    [Benchmark(Baseline = true, Description = "SafetensorsReader.Open + ReadTensor (heap copy)")]
    public long HeapBufferedRead()
    {
        long total = 0;
        using var r = SafetensorsReader.Open(_path);
        foreach (var name in r.Names)
        {
            var t = r.ReadTensor<float>(name);
            total += t.Length;
        }
        return total;
    }

    [Benchmark(Description = "SafetensorsMmapReader.Open + ReadTensor (mmap-backed)")]
    public long MmapBackedRead()
    {
        long total = 0;
        using var r = SafetensorsMmapReader.Open(_path);
        foreach (var name in r.Entries.Keys)
        {
            var t = r.ReadTensor<float>(name);
            total += t.Length;
        }
        return total;
    }

    [Benchmark(Description = "SafetensorsReader.Open metadata-only (no tensor read)")]
    public int MetadataOnly()
    {
        using var r = SafetensorsReader.Open(_path);
        return r.Entries.Count;
    }
}

#endif
