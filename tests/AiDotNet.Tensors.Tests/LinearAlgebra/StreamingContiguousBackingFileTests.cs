// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The streaming pool pages out to ONE contiguous append-only backing file (entries
/// land in first-eviction order → sequential reads + a single persistent handle,
/// instead of a file open/mmap/close per rehydrate). Verifies data integrity across
/// the shared file under heavy eviction churn, and that there is exactly one backing
/// file rather than one per handle.
/// </summary>
public class StreamingContiguousBackingFileTests
{
    [Fact]
    public void ManyHandles_OneBackingFile_AllRehydrateCorrectly()
    {
        const int entryBytes = 2048, n = 40, budget = 5; // 8x over budget → constant churn
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-contig-" + Guid.NewGuid().ToString("N"));
        try
        {
            string poolDir;
            using (var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = (long)budget * entryBytes,
                StreamingBackingStorePath = dir,
            }))
            {
                var handles = new long[n];
                for (int i = 0; i < n; i++)
                {
                    var data = new byte[entryBytes];
                    for (int j = 0; j < entryBytes; j++) data[j] = (byte)((i * 31 + j) & 0xFF);
                    handles[i] = pool.Register(data);
                }

                // Rehydrate every handle several times in shuffled order — each read
                // pulls its slice out of the single shared file at its own offset.
                for (int pass = 0; pass < 3; pass++)
                {
                    for (int s = 0; s < n; s++)
                    {
                        int i = (s * 17 + pass) % n;
                        var got = pool.Rehydrate(handles[i]);
                        Assert.Equal(entryBytes, got.Length);
                        for (int j = 0; j < entryBytes; j++)
                            Assert.Equal((byte)((i * 31 + j) & 0xFF), got[j]);
                    }
                }

                // Exactly one backing file (not n per-handle files).
                poolDir = Directory.GetDirectories(dir)[0];
                var files = Directory.GetFiles(poolDir);
                Assert.Single(files);
                Assert.Equal("backing.bin", Path.GetFileName(files[0]));
            }
            // Backing store is cleaned up on dispose.
            Assert.False(Directory.Exists(poolDir));
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }
}
