// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System;
using System.IO;
using System.Text.Json;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.Safetensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.Safetensors;

[Collection("PersistenceGuard")]
public class ShardedSafetensorsReaderTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    [Fact]
    public void OpensTwoShards_DispatchesEachTensorToCorrectShard()
    {
        using var _ = IsolatedTrial();
        var dir = Path.Combine(Path.GetTempPath(), "shtest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var aT = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
            var bT = new Tensor<float>(new[] { 10f, 20f }, new[] { 2 });
            using (var w = SafetensorsWriter.Create(Path.Combine(dir, "model-00001-of-00002.safetensors")))
            {
                w.Add("a", aT);
                w.Save();
            }
            using (var w = SafetensorsWriter.Create(Path.Combine(dir, "model-00002-of-00002.safetensors")))
            {
                w.Add("b", bT);
                w.Save();
            }

            // Hand-write a minimal index file pointing each tensor at
            // the right shard.
            File.WriteAllText(Path.Combine(dir, "model.safetensors.index.json"),
                """
                {
                  "metadata": { "total_size": 999 },
                  "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors"
                  }
                }
                """);

            using var r = ShardedSafetensorsReader.Open(Path.Combine(dir, "model.safetensors.index.json"));
            var aRt = r.ReadTensor<float>("a");
            var bRt = r.ReadTensor<float>("b");
            Assert.Equal(new[] { 1f, 2f, 3f }, aRt.AsSpan().ToArray());
            Assert.Equal(new[] { 10f, 20f }, bRt.AsSpan().ToArray());

            // Index metadata propagated.
            Assert.True(r.Metadata.ContainsKey("total_size"));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void RejectsMissingIndex()
    {
        using var _ = IsolatedTrial();
        Assert.Throws<FileNotFoundException>(
            () => ShardedSafetensorsReader.Open("/nonexistent/path/model.safetensors.index.json"));
    }

    [Fact]
    public void RejectsIndexMissingWeightMap()
    {
        using var _ = IsolatedTrial();
        var path = Path.Combine(Path.GetTempPath(), "bad-index-" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(path, "{ \"metadata\": {} }");
        try
        {
            Assert.Throws<InvalidDataException>(() => ShardedSafetensorsReader.Open(path));
        }
        finally { try { File.Delete(path); } catch { } }
    }

    [Fact]
    public void UnknownTensorName_Throws()
    {
        using var _ = IsolatedTrial();
        var dir = Path.Combine(Path.GetTempPath(), "shtest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using (var w = SafetensorsWriter.Create(Path.Combine(dir, "shard.safetensors")))
            {
                w.Add("x", new Tensor<float>(new[] { 1f }, new[] { 1 }));
                w.Save();
            }
            File.WriteAllText(Path.Combine(dir, "model.safetensors.index.json"),
                """{"weight_map":{"x":"shard.safetensors"}}""");

            using var r = ShardedSafetensorsReader.Open(Path.Combine(dir, "model.safetensors.index.json"));
            Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() => r.ReadTensor<float>("y"));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
        }
    }
}
