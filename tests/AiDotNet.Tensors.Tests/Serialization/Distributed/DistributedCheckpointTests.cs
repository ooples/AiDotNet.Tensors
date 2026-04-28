// Copyright (c) AiDotNet. All rights reserved.

#nullable disable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.Distributed;
using AiDotNet.Tensors.Serialization.Safetensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Serialization.Distributed;

[Collection("PersistenceGuard")]
public class DistributedCheckpointTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-test-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; }
        public TempDir(string prefix = "dctest-")
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), prefix + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }
        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Save_LoadRoundTrip()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();

        var stateDict = new Dictionary<string, IPersistableTensor>
        {
            ["embed.weight"] = new PersistableTensor<float>(new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 })),
            ["fc.weight"] = new PersistableTensor<float>(new Tensor<float>(new[] { 5f, 6f }, new[] { 2 })),
        };
        int shards = DistributedCheckpoint.Save(dir.Path, stateDict, numShards: 2);
        Assert.True(shards >= 1);

        var indexPath = Path.Combine(dir.Path, "model.safetensors.index.json");
        Assert.True(File.Exists(indexPath));

        var loaded = DistributedCheckpoint.Load(indexPath);
        Assert.Equal(2, loaded.Count);
        Assert.True(loaded.ContainsKey("embed.weight"));
        Assert.True(loaded.ContainsKey("fc.weight"));
    }

    [Fact]
    public void Reshard_4to2_PreservesAllTensors()
    {
        using var _ = IsolatedTrial();
        using var src = new TempDir("reshard-src-");
        using var dst = new TempDir("reshard-dst-");

        // Write 4 tensors, request 4 shards (one each).
        var stateDict = new Dictionary<string, IPersistableTensor>();
        for (int i = 0; i < 4; i++)
            stateDict[$"t{i}"] = new PersistableTensor<float>(
                new Tensor<float>(new float[] { i, i + 0.1f, i + 0.2f }, new[] { 3 }));
        DistributedCheckpoint.Save(src.Path, stateDict, numShards: 4);
        var srcIndex = Path.Combine(src.Path, "model.safetensors.index.json");

        // Reshard to 2.
        int newShardCount = DistributedCheckpoint.Reshard(srcIndex, dst.Path, newShardCount: 2);
        Assert.True(newShardCount >= 1 && newShardCount <= 2);

        var dstIndex = Path.Combine(dst.Path, "model.safetensors.index.json");
        var loaded = DistributedCheckpoint.Load(dstIndex);
        Assert.Equal(4, loaded.Count);
        for (int i = 0; i < 4; i++) Assert.True(loaded.ContainsKey($"t{i}"));
    }

    [Fact]
    public void Reshard_To1Shard_Consolidates()
    {
        using var _ = IsolatedTrial();
        using var src = new TempDir("reshard-src-");
        using var dst = new TempDir("reshard-dst-");
        var stateDict = new Dictionary<string, IPersistableTensor>
        {
            ["a"] = new PersistableTensor<float>(new Tensor<float>(new[] { 1f }, new[] { 1 })),
            ["b"] = new PersistableTensor<float>(new Tensor<float>(new[] { 2f }, new[] { 1 })),
            ["c"] = new PersistableTensor<float>(new Tensor<float>(new[] { 3f }, new[] { 1 })),
        };
        DistributedCheckpoint.Save(src.Path, stateDict, numShards: 3);

        int n = DistributedCheckpoint.Reshard(
            Path.Combine(src.Path, "model.safetensors.index.json"),
            dst.Path,
            newShardCount: 1);
        Assert.Equal(1, n);
    }

    [Fact]
    public void Save_ZeroShards_Throws()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();
        Assert.Throws<ArgumentOutOfRangeException>(
            () => DistributedCheckpoint.Save(dir.Path, new Dictionary<string, IPersistableTensor>(), numShards: 0));
    }

    [Fact]
    public void RawPersistableTensor_RoundTripsSubByteDtype()
    {
        using var _ = IsolatedTrial();
        using var dir = new TempDir();
        var stateDict = new Dictionary<string, IPersistableTensor>
        {
            ["q.weight"] = new RawPersistableTensor(SafetensorsDtype.AIDN_INT4, new long[] { 8 }, new byte[] { 0x12, 0x34, 0x56, 0x78 }),
        };
        DistributedCheckpoint.Save(dir.Path, stateDict, numShards: 1);

        var loaded = DistributedCheckpoint.Load(Path.Combine(dir.Path, "model.safetensors.index.json"));
        Assert.True(loaded.ContainsKey("q.weight"));
        Assert.Equal(new byte[] { 0x12, 0x34, 0x56, 0x78 }, loaded["q.weight"]);
    }
}
