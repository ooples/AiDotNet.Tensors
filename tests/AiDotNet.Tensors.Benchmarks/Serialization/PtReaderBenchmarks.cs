// Copyright (c) AiDotNet. All rights reserved.

#if NET8_0_OR_GREATER

using System;
using System.IO;
using System.IO.Compression;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.Pickle;
using BenchmarkDotNet.Attributes;

namespace AiDotNet.Tensors.Benchmarks.Serialization;

/// <summary>
/// PyTorch <c>.pt</c> reader open-cost benchmark. Reports the time
/// to open a zip-format <c>.pt</c>, parse the pickle stream's
/// metadata-only opcode prelude, and surface the (empty) Tensors
/// collection. Real-tensor materialization throughput needs a
/// pickle+storage emitter that doesn't yet exist on the test side
/// — that's tracked separately under the larger PtWriter follow-up;
/// this benchmark deliberately scopes itself to the open + parse
/// hot path so a regression in PtReader's pickle decoder shows up.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(5)]
[MaxIterationCount(20)]
public class PtReaderBenchmarks
{
    private string _path = "";
    private IDisposable? _trialOverride;

    [GlobalSetup]
    public void Setup()
    {
        _trialOverride = PersistenceGuard.SetTestTrialFilePathOverride(
            Path.Combine(Path.GetTempPath(), "aidn-pt-bench-trial-" + Guid.NewGuid().ToString("N") + ".json"));
        _path = Path.Combine(Path.GetTempPath(), $"aidn-pt-bench-{Guid.NewGuid():N}.pt");

        // Minimal zip-format .pt with an empty top-level dict — exactly
        // what PtReader needs to exercise its open + pickle-prelude
        // path. This benchmark is "metadata-only open cost" by design;
        // a full materialization benchmark needs a tensor-emitting
        // pickle+storage harness (deferred to the PtWriter task).
        using var fs = File.Create(_path);
        using var zip = new ZipArchive(fs, ZipArchiveMode.Create, leaveOpen: false);
        var entry = zip.CreateEntry("model/data.pkl");
        using var es = entry.Open();
        es.WriteByte(0x80); es.WriteByte(0x02);  // PROTO 2
        es.WriteByte(0x7D);                       // EMPTY_DICT '}'
        es.WriteByte(0x2E);                       // STOP '.'
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        try { File.Delete(_path); } catch { /* best-effort */ }
        _trialOverride?.Dispose();
    }

    [Benchmark(Baseline = true, Description = "PtReader.Open: parse zip + pickle prelude (metadata-only)")]
    public int OpenAndCount()
    {
        var r = PtReader.Open(_path);
        return r.Tensors.Count;
    }
}

#endif
