// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// End-to-end zero-copy mmap residency (lever 5): when a paged-out, natively-stored,
/// read-only weight is materialized with <see cref="GpuOffloadOptions.EnableZeroCopyMmapResidency"/>
/// on, the registry aliases the weight's bytes directly from a read-only memory-mapping of
/// the backing file instead of paging them into a fresh array. These tests prove the alias
/// (1) returns byte-correct values, (2) skips the pool's copy/rehydrate path (no disk read,
/// resident set doesn't grow), (3) is gated to native encoding + inference, and (4) tears
/// down cleanly. Serialized against other registry tests (global state).
/// </summary>
[Collection("WeightRegistry")]
public class StreamingZeroCopyMmapAliasTests
{
    // Two equal-sized weights with a budget that holds exactly one → registering the second
    // evicts the first to disk, leaving it paged out (the precondition for zero-copy). The
    // budget is 1.5× the per-weight STORED bytes (which depend on the encoding), so eviction
    // happens regardless of whether the store is native (4 B/elem) or bf16 (2 B/elem).
    private const int N = 512;

    private static float Val(int i) => (float)Math.Sin(i * 0.013) * 0.05f;

    private static (string dir, Tensor<float> t1, Tensor<float> t2) SetupEvictedFirst(
        StreamingStoreDtype dtype, bool zeroCopy, bool? training, int storedBytesPerWeight)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-zc-alias-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = storedBytesPerWeight * 3L / 2L, // holds one, not two
            StreamingStoreDtype = dtype,
            EnableZeroCopyMmapResidency = zeroCopy,
        });
        WeightRegistry.SetStreamingExecutionTraining(training);

        var t1 = new Tensor<float>(new[] { N });
        for (int i = 0; i < N; i++) t1[i] = Val(i);
        t1.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t1);

        var t2 = new Tensor<float>(new[] { N });
        for (int i = 0; i < N; i++) t2[i] = Val(i) + 1f;
        t2.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t2); // pushes t1 over budget → t1 evicted to disk

        return (dir, t1, t2);
    }

    [Fact]
    public void ZeroCopyEnabled_NativeInference_AliasesWithoutDiskRead_AndIsByteCorrect()
    {
        var (dir, t1, _) = SetupEvictedFirst(StreamingStoreDtype.FullPrecision, zeroCopy: true, training: false, storedBytesPerWeight: N * 4);
        try
        {
            // t1 is paged out to disk; native (encoding 0); inference mode → zero-copy eligible.
            Assert.Equal((byte)0, t1.StreamingStoreEncoding);
            long diskReadsBefore = WeightRegistry.StreamingPool.GetReport().DiskReadCount;
            long residentBefore = WeightRegistry.StreamingPool.ResidentBytes;

            WeightRegistry.Materialize(t1);

            // (1) Byte-correct values read straight through the aliased mapping.
            for (int i = 0; i < N; i++)
                Assert.Equal(BitExactHelpers.SingleBits(Val(i)), BitExactHelpers.SingleBits(t1[i]));

            // (2) The pool's copy path was skipped: no rehydrate disk read, and the weight's
            // bytes were NOT paged back into the pool's resident set (the OS page cache holds
            // them now). The copy path would have incremented DiskReadCount and grown resident.
            var rpt = WeightRegistry.StreamingPool.GetReport();
            Assert.Equal(diskReadsBefore, rpt.DiskReadCount);
            Assert.Equal(residentBefore, WeightRegistry.StreamingPool.ResidentBytes);

            // (4) A second materialize is the resident fast path (no fault); dispose is clean.
            WeightRegistry.Materialize(t1);
            for (int i = 0; i < N; i++) Assert.Equal(Val(i), t1[i], 6);
            t1.Dispose(); // releases the mmap mapping via DisposeStreamingMmapOwner — no throw
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void ZeroCopyDisabled_UsesCopyPath_ReadsFromDisk()
    {
        // Default (flag off): the proven copy path rehydrates from disk.
        var (dir, t1, _) = SetupEvictedFirst(StreamingStoreDtype.FullPrecision, zeroCopy: false, training: false, storedBytesPerWeight: N * 4);
        try
        {
            long diskReadsBefore = WeightRegistry.StreamingPool.GetReport().DiskReadCount;

            WeightRegistry.Materialize(t1);

            for (int i = 0; i < N; i++) Assert.Equal(Val(i), t1[i], 6);
            // The copy path read the slice back from the backing file.
            Assert.True(WeightRegistry.StreamingPool.GetReport().DiskReadCount > diskReadsBefore,
                "copy-path materialize should have read the evicted slice from disk");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void ZeroCopyEnabled_ButBf16Encoded_FallsBackToCopy()
    {
        // bf16 store needs a widen (decode) on restore → cannot be aliased; even with the
        // flag on, the encoding gate routes it through the copy path.
        var (dir, t1, _) = SetupEvictedFirst(StreamingStoreDtype.Bf16, zeroCopy: true, training: false, storedBytesPerWeight: N * 2);
        try
        {
            Assert.Equal((byte)1, t1.StreamingStoreEncoding); // bf16
            long diskReadsBefore = WeightRegistry.StreamingPool.GetReport().DiskReadCount;

            WeightRegistry.Materialize(t1);

            // Copy path ran (disk read happened); values are within bf16 precision.
            Assert.True(WeightRegistry.StreamingPool.GetReport().DiskReadCount > diskReadsBefore,
                "bf16 weight must use the decode/copy path, not zero-copy");
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < N; i++) { double e = t1[i] - Val(i); sum2 += e * e; ref2 += (double)Val(i) * Val(i); }
            Assert.True(Math.Sqrt(sum2 / ref2) < 0.01, "bf16 round-trip RMS error should be <1%");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void ZeroCopyEnabled_TrainingMode_FallsBackToCopy()
    {
        // Training mode → weights may be written; the read-only alias is unsafe, so the gate
        // forces the copy path even with the flag on. (Native encoding, but training=true.)
        var (dir, t1, _) = SetupEvictedFirst(StreamingStoreDtype.FullPrecision, zeroCopy: true, training: true, storedBytesPerWeight: N * 4);
        try
        {
            Assert.Equal((byte)0, t1.StreamingStoreEncoding);
            long diskReadsBefore = WeightRegistry.StreamingPool.GetReport().DiskReadCount;

            WeightRegistry.Materialize(t1);

            Assert.True(WeightRegistry.StreamingPool.GetReport().DiskReadCount > diskReadsBefore,
                "training-mode materialize must use the writable copy path, not the read-only alias");
            for (int i = 0; i < N; i++) Assert.Equal(Val(i), t1[i], 6);
        }
        finally { Cleanup(dir); }
    }

    private static void Cleanup(string dir)
    {
        WeightRegistry.SetStreamingExecutionTraining(null);
        WeightRegistry.Reset();
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }
}
