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
/// Sharded safetensors save/load throughput. The HF tooling shards
/// any model whose total weight bytes exceed a per-file budget
/// (5 GiB by default); these benchmarks measure the cost of writing
/// and re-reading such a checkpoint at synthetic sizes that fit in
/// CI memory.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(3)]
[MaxIterationCount(15)]
public class ShardedCheckpointBenchmarks
{
    private string _dir = "";
    private string _indexPath = "";
    private IDisposable? _trialOverride;
    private float[][] _data = System.Array.Empty<float[]>();

    [Params(8, 32)]
    public int TensorCount;

    [Params(64 * 1024, 1024 * 1024)]
    public long ShardSizeBytes;

    [GlobalSetup]
    public void Setup()
    {
        _trialOverride = PersistenceGuard.SetTestTrialFilePathOverride(
            Path.Combine(Path.GetTempPath(), "aidn-shard-bench-trial-" + Guid.NewGuid().ToString("N") + ".json"));
        var rng = new Random(0);
        _data = new float[TensorCount][];
        for (int i = 0; i < TensorCount; i++)
        {
            // ~1 MB per tensor (256 × 1024 floats), so the param
            // matrix lets us land in single-shard / multi-shard
            // territory deterministically.
            _data[i] = new float[256 * 1024];
            for (int k = 0; k < _data[i].Length; k++) _data[i][k] = (float)(rng.NextDouble() - 0.5);
        }
    }

    [IterationSetup(Target = nameof(LoadSharded))]
    public void PrepareLoadIter()
    {
        _dir = Path.Combine(Path.GetTempPath(), "aidn-shard-load-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        var w = new ShardedSafetensorsWriter(_dir, "model", ShardSizeBytes);
        for (int i = 0; i < _data.Length; i++)
            w.Add($"tensor.{i}", new Tensor<float>(_data[i], new[] { 256, 1024 }));
        w.Save();
        _indexPath = Path.Combine(_dir, "model.safetensors.index.json");
    }

    [IterationCleanup(Target = nameof(LoadSharded))]
    public void CleanupLoadIter()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    [IterationSetup(Target = nameof(SaveSharded))]
    public void PrepareSaveIter()
    {
        _dir = Path.Combine(Path.GetTempPath(), "aidn-shard-save-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    [IterationCleanup(Target = nameof(SaveSharded))]
    public void CleanupSaveIter()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
    }

    [GlobalCleanup]
    public void Cleanup() => _trialOverride?.Dispose();

    [Benchmark(Baseline = true, Description = "ShardedSafetensorsWriter.Save — write all shards + index")]
    public int SaveSharded()
    {
        var w = new ShardedSafetensorsWriter(_dir, "model", ShardSizeBytes);
        for (int i = 0; i < _data.Length; i++)
            w.Add($"tensor.{i}", new Tensor<float>(_data[i], new[] { 256, 1024 }));
        return w.Save();
    }

    [Benchmark(Description = "ShardedSafetensorsReader.Open + read every tensor")]
    public long LoadSharded()
    {
        long total = 0;
        using var r = ShardedSafetensorsReader.Open(_indexPath);
        foreach (var name in r.Entries.Keys)
        {
            var t = r.ReadTensor<float>(name);
            total += t.Length;
        }
        return total;
    }
}

#endif
