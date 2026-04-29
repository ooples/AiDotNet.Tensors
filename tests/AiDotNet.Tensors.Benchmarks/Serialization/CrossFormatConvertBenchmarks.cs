// Copyright (c) AiDotNet. All rights reserved.

#if NET8_0_OR_GREATER

using System;
using System.IO;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using AiDotNet.Tensors.Serialization.Safetensors;
using BenchmarkDotNet.Attributes;

namespace AiDotNet.Tensors.Benchmarks.Serialization;

/// <summary>
/// Cross-format conversion throughput — the same payload is
/// converted through each path in turn so the user can compare
/// "what does it cost to ship this checkpoint as <c>X</c>?".
/// PyTorch ships no equivalent — its converters all roundtrip
/// through Python pickling and require the source format's loader to
/// instantiate a real <c>torch.Tensor</c> first.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(3)]
[MaxIterationCount(15)]
public class CrossFormatConvertBenchmarks
{
    private string _srcSafetensors = "";
    private string _outDir = "";
    private string _outFile = "";
    private IDisposable? _trialOverride;

    [Params(8, 64)]
    public int TensorCount;

    [GlobalSetup]
    public void Setup()
    {
        _trialOverride = PersistenceGuard.SetTestTrialFilePathOverride(
            Path.Combine(Path.GetTempPath(), "aidn-conv-bench-trial-" + Guid.NewGuid().ToString("N") + ".json"));

        _srcSafetensors = Path.Combine(Path.GetTempPath(), $"aidn-conv-src-{TensorCount}-{Guid.NewGuid():N}.safetensors");
        var rng = new Random(0);
        using var w = SafetensorsWriter.Create(_srcSafetensors);
        for (int i = 0; i < TensorCount; i++)
        {
            // 256-divisible to allow Q4_K quant in the safetensors→gguf
            // path without skipping rows.
            int rows = 256;
            int cols = 256;
            float[] data = new float[rows * cols];
            for (int k = 0; k < data.Length; k++) data[k] = (float)(rng.NextDouble() - 0.5);
            w.Add($"layer.{i}.weight", new Tensor<float>(data, new[] { rows, cols }));
        }
        w.Save();
    }

    [IterationSetup]
    public void IterSetup()
    {
        _outDir = Path.Combine(Path.GetTempPath(), "aidn-conv-out-" + Guid.NewGuid().ToString("N"));
        _outFile = Path.Combine(Path.GetTempPath(), "aidn-conv-out-" + Guid.NewGuid().ToString("N") + ".gguf");
        Directory.CreateDirectory(_outDir);
    }

    [IterationCleanup]
    public void IterCleanup()
    {
        try { Directory.Delete(_outDir, recursive: true); } catch { /* best-effort */ }
        try { File.Delete(_outFile); } catch { /* best-effort */ }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        try { File.Delete(_srcSafetensors); } catch { /* best-effort */ }
        _trialOverride?.Dispose();
    }

    [Benchmark(Baseline = true, Description = "safetensors → safetensors-sharded (raw byte passthrough)")]
    public int ConvertToSharded()
    {
        using var r = SafetensorsReader.Open(_srcSafetensors);
        var w = new ShardedSafetensorsWriter(_outDir, "model", shardSizeBytes: 1024 * 1024);
        foreach (var kv in r.Entries)
        {
            var bytes = r.ReadRawBytes(kv.Key);
            w.AddRaw(kv.Key, kv.Value.Dtype, kv.Value.Shape, bytes);
        }
        return w.Save();
    }

    [Benchmark(Description = "safetensors → gguf F32 (raw byte passthrough)")]
    public int ConvertToGgufF32()
    {
        using var r = SafetensorsReader.Open(_srcSafetensors);
        using var w = GgufWriter.Create(_outFile);
        int count = 0;
        foreach (var kv in r.Entries)
        {
            var bytes = r.ReadRawBytes(kv.Key);
            w.AddRaw(kv.Key, GgufType.F32, kv.Value.Shape, bytes);
            count++;
        }
        w.Save();
        return count;
    }

    [Benchmark(Description = "safetensors → gguf Q4_K (256-block quant)")]
    public int ConvertToGgufQ4K()
    {
        using var r = SafetensorsReader.Open(_srcSafetensors);
        using var w = GgufWriter.Create(_outFile);
        int count = 0;
        foreach (var kv in r.Entries)
        {
            var floats = r.ReadTensor<float>(kv.Key).AsSpan();
            w.AddQ4_K(kv.Key, kv.Value.Shape, floats);
            count++;
        }
        w.Save();
        return count;
    }
}

#endif
