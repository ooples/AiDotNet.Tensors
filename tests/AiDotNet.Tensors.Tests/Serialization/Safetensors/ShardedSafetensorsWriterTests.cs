// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.Safetensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.Safetensors;

[Collection("PersistenceGuard")]
public class ShardedSafetensorsWriterTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; }
        public TempDir()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "shtest-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }
        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Write_ThreeTensorsUnderSmallBudget_ProducesMultipleShards()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();
        // 3 tensors of 1024 floats = 4096 bytes each. With shard
        // budget = 5000 bytes, one tensor fits per shard.
        var w = new ShardedSafetensorsWriter(dir.Path, "model", shardSizeBytes: 5000);
        w.Add("a", new Tensor<float>(new float[1024], new[] { 1024 }));
        w.Add("b", new Tensor<float>(new float[1024], new[] { 1024 }));
        w.Add("c", new Tensor<float>(new float[1024], new[] { 1024 }));
        int shards = w.Save();
        Assert.Equal(3, shards);

        // Index file present + shape correct on disk.
        var indexPath = Path.Combine(dir.Path, "model.safetensors.index.json");
        Assert.True(File.Exists(indexPath));
        Assert.True(File.Exists(Path.Combine(dir.Path, "model-00001-of-00003.safetensors")));
        Assert.True(File.Exists(Path.Combine(dir.Path, "model-00003-of-00003.safetensors")));

        // Round-trip: open as sharded reader and confirm 3 tensors.
        using var r = ShardedSafetensorsReader.Open(indexPath);
        Assert.Equal(new[] { "a", "b", "c" }, r.Names.OrderBy(n => n));
    }

    [Fact]
    public void Write_BigBudget_ProducesSingleShard()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();
        var w = new ShardedSafetensorsWriter(dir.Path, "model", shardSizeBytes: long.MaxValue);
        w.Add("x", new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 }));
        int shards = w.Save();
        Assert.Equal(1, shards);
        Assert.True(File.Exists(Path.Combine(dir.Path, "model-00001-of-00001.safetensors")));
    }

    [Fact]
    public void Write_EmptyChechpoint_StillProducesIndex()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();
        var w = new ShardedSafetensorsWriter(dir.Path, "model");
        w.Metadata["framework"] = "AiDotNet.Tensors";
        int shards = w.Save();
        Assert.Equal(1, shards);
        Assert.True(File.Exists(Path.Combine(dir.Path, "model.safetensors.index.json")));
    }

    [Fact]
    public void Save_TwiceThrows()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();
        var w = new ShardedSafetensorsWriter(dir.Path, "model");
        w.Add("x", new Tensor<float>(new[] { 1f }, new[] { 1 }));
        w.Save();
        Assert.Throws<InvalidOperationException>(() => w.Save());
    }

    [Fact]
    public void DuplicateName_Throws()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();
        var w = new ShardedSafetensorsWriter(dir.Path, "model");
        w.Add("x", new Tensor<float>(new[] { 1f }, new[] { 1 }));
        Assert.Throws<ArgumentException>(
            () => w.Add("x", new Tensor<float>(new[] { 2f }, new[] { 1 })));
    }
}
