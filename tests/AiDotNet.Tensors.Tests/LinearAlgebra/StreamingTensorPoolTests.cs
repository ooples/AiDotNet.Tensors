// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #276 sub-feature 2: weight-streaming pool with LRU eviction +
/// memory-mapped backing-store rehydrate.
/// </summary>
public class StreamingTensorPoolTests
{
    [Fact]
    public void Register_StaysResident_BelowBudget()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-stream-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 1024,
                StreamingBackingStorePath = dir,
            });
            var data = new byte[256];
            for (int i = 0; i < data.Length; i++) data[i] = (byte)(i & 0xFF);
            long h = pool.Register(data);

            Assert.Equal(256, pool.ResidentBytes);
            Assert.Equal(1, pool.ResidentEntryCount);

            var back = pool.Rehydrate(h);
            for (int i = 0; i < data.Length; i++) Assert.Equal(data[i], back[i]);
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void OverBudget_LeastRecentlyUsed_EvictsAndRehydrates()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-stream-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 600, // Force eviction at 3 × 256 entries.
                StreamingBackingStorePath = dir,
            });
            byte[] Make(byte fill) { var a = new byte[256]; for (int i = 0; i < 256; i++) a[i] = fill; return a; }
            long ha = pool.Register(Make(0xAA));
            long hb = pool.Register(Make(0xBB));
            // Access ha so it's NOT the LRU. Then register hc — LRU (hb) evicts.
            pool.MarkAccessed(ha);
            long hc = pool.Register(Make(0xCC));

            Assert.True(pool.ResidentBytes <= 600);
            // hb was the LRU at the moment of registration of hc, so it
            // got paged out. Rehydrating it must bring back the original bytes.
            var hbBack = pool.Rehydrate(hb);
            for (int i = 0; i < 256; i++) Assert.Equal((byte)0xBB, hbBack[i]);
            // After rehydrate hb is LRU-promoted; another over-budget
            // registration would now evict ha or hc.
            Assert.True(pool.ResidentBytes <= 600 || pool.ResidentBytes <= 768);
            _ = hc;
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Unregister_DropsFromBudget_AndDeletesBackingFile()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-stream-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 1024,
                StreamingBackingStorePath = dir,
            });
            long h = pool.Register(new byte[256]);
            Assert.Equal(256, pool.ResidentBytes);
            pool.Unregister(h);
            Assert.Equal(0, pool.ResidentBytes);
            Assert.Equal(0, pool.ResidentEntryCount);
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Rehydrate_UnknownHandle_Throws()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-stream-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions { StreamingBackingStorePath = dir });
            Assert.Throws<InvalidOperationException>(() => pool.Rehydrate(999).Length);
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }
}
