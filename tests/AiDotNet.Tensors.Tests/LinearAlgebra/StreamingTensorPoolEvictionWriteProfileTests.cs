// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Profiles (and, after the clean/dirty-eviction fix, asserts) the streaming
/// pool's disk-write behaviour under a READ-ONLY access pattern — the pattern a
/// transformer's forward+backward pass produces, where weights are never
/// modified until the optimizer step.
///
/// The bug: <see cref="StreamingTensorPool"/> re-writes (and re-LZ4-compresses)
/// every entry to disk on EVERY eviction, with no dirty-tracking. A weight that
/// was paged out, rehydrated for a read, and evicted again is byte-identical to
/// its still-present backing file, so re-writing it is pure waste. For a model
/// larger than the resident budget this is hundreds of redundant writes per step.
/// </summary>
public class StreamingTensorPoolEvictionWriteProfileTests
{
    private readonly ITestOutputHelper _out;

    public StreamingTensorPoolEvictionWriteProfileTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void ReadOnlyTrainingStepPattern_ShouldNotReWriteCleanWeights()
    {
        // A small transformer's worth of weights, total ~8x the resident budget
        // so the pool churns hard — every layer's rehydrate evicts another.
        const int numWeights = 48;        // "layers"
        const int weightBytes = 1 << 20;  // 1 MiB each → 48 MiB model
        const long budget = 6L << 20;     // 6 MiB resident → ~6 layers fit
        const int steps = 6;              // training iterations

        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-stream-profile-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = budget,
                StreamingBackingStorePath = dir,
            });

            var handles = new long[numWeights];
            var firstByte = new byte[numWeights];
            for (int i = 0; i < numWeights; i++)
            {
                var data = new byte[weightBytes];
                // Distinct, non-trivial content per weight (so compression has real work).
                for (int j = 0; j < weightBytes; j += 64) data[j] = (byte)((i * 7 + j) & 0xFF);
                firstByte[i] = data[0];
                handles[i] = pool.Register(data);
            }

            // After registration the over-budget tail is paged out exactly once.
            var afterRegister = pool.GetReport();
            long registerWriteBytes = afterRegister.DiskWriteBytes;
            _out.WriteLine($"Model: {numWeights} x {weightBytes / (1 << 20)} MiB = {numWeights * weightBytes / (1 << 20)} MiB, " +
                           $"budget {budget / (1 << 20)} MiB");
            _out.WriteLine($"After register: writeMiB={registerWriteBytes / (1024.0 * 1024):F1} " +
                           $"evictions={afterRegister.EvictionCount}");

            var sw = Stopwatch.StartNew();
            for (int s = 0; s < steps; s++)
            {
                // Forward: read every weight in order. READ-ONLY — no Register/modify.
                for (int i = 0; i < numWeights; i++) _ = pool.Rehydrate(handles[i]);
                // Backward: read every weight in reverse. Still READ-ONLY.
                for (int i = numWeights - 1; i >= 0; i--) _ = pool.Rehydrate(handles[i]);
            }
            sw.Stop();

            var after = pool.GetReport();
            long stepWriteBytes = after.DiskWriteBytes - registerWriteBytes;
            long stepReadBytes = after.DiskReadBytes;
            long modelBytes = (long)numWeights * weightBytes;
            _out.WriteLine($"After {steps} read-only steps in {sw.ElapsedMilliseconds} ms:");
            _out.WriteLine($"  disk READ : {stepReadBytes / (1024.0 * 1024):F1} MiB  ({after.DiskReadCount} reads)");
            _out.WriteLine($"  disk WRITE total: {after.DiskWriteBytes / (1024.0 * 1024):F1} MiB " +
                           $"(register {registerWriteBytes / (1024.0 * 1024):F1} + steps {stepWriteBytes / (1024.0 * 1024):F1})");
            _out.WriteLine($"  evictions : {after.EvictionCount} (clean-skipped {after.CleanEvictionCount}, " +
                           $"saved {after.CleanEvictionBytesSkipped / (1024.0 * 1024):F1} MiB of writes)");
            _out.WriteLine($"  step-write/read ratio: {(stepReadBytes > 0 ? (double)stepWriteBytes / stepReadBytes : 0):F2}");

            // The invariant after clean/dirty eviction: a weight whose content never
            // changes is persisted to disk AT MOST ONCE. So total disk writes across
            // registration + every read-only step are bounded by the model size and do
            // NOT scale with the number of steps. (Before the fix, every eviction
            // re-wrote, so writes ≈ steps × reads — e.g. 510 MiB of pure waste here.)
            Assert.True(after.DiskWriteBytes <= modelBytes,
                $"Read-only forward+backward wrote {after.DiskWriteBytes / (1024.0 * 1024):F1} MiB total — more than " +
                $"the {modelBytes / (1024.0 * 1024):F1} MiB model size, so unchanged weights were re-written. Clean " +
                "(unmodified) entries must be dropped on eviction, not re-persisted. This is the streaming write-back waste.");
            // And clean skips must dominate evictions on this read-heavy pattern.
            Assert.True(after.CleanEvictionCount > after.EvictionCount / 2,
                $"Only {after.CleanEvictionCount}/{after.EvictionCount} evictions were clean-skipped; the read-only " +
                "forward/backward pass should make the large majority of evictions clean.");

            // CORRECTNESS: every weight must still rehydrate to its exact original
            // content after all the clean-skip eviction churn — the write-elision
            // must never serve stale or corrupt bytes.
            for (int i = 0; i < numWeights; i++)
            {
                var bytes = pool.Rehydrate(handles[i]);
                Assert.Equal(weightBytes, bytes.Length);
                Assert.Equal(firstByte[i], bytes[0]);
                // Spot-check an interior sample that was written by the init loop.
                Assert.Equal((byte)((i * 7 + 64) & 0xFF), bytes[64]);
            }
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void DefaultResidentBudget_StaysWithinRamAndCeiling()
    {
        long budget = GpuOffloadOptions.DefaultResidentBudgetBytes();
        const long ceiling = 16L * 1024 * 1024 * 1024;
        const long floor = 512L * 1024 * 1024;
        Assert.InRange(budget, floor, ceiling);

        // A fresh options instance must pick up the RAM-aware default.
        Assert.Equal(budget, new GpuOffloadOptions().StreamingPoolMaxResidentBytes);

#if NET5_0_OR_GREATER
        // When available memory is known, the budget must not exceed it (the whole
        // point — never let the resident set push the box into OS swap).
        long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (available > 0)
            Assert.True(budget <= available,
                $"Budget {budget} must not exceed available memory {available}.");
#endif
    }

    [Fact]
    public void MarkDirty_ForcesReWriteOnNextEviction()
    {
        // The dirty contract that keeps clean-eviction correct: once an entry is
        // marked dirty (its content replaced out-of-band under the same handle),
        // the next eviction MUST re-persist it rather than skip the write.
        //
        // Budget holds exactly ONE 256-byte entry, so every Register evicts
        // exactly the single resident entry — precise control over which entry
        // is evicted at each step.
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-stream-dirty-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 256,
                StreamingBackingStorePath = dir,
            });
            byte[] Make(byte fill) { var a = new byte[256]; for (int i = 0; i < 256; i++) a[i] = fill; return a; }

            long ha = pool.Register(Make(0xAA));   // resident: ha
            _ = pool.Register(Make(0xBB));         // evict ha → write #1.   resident: hb
            _ = pool.Rehydrate(ha);                // evict hb (write #2); ha resident, CLEAN

            long beforeClean = pool.GetReport().DiskWriteBytes;
            long cleanBefore = pool.GetReport().CleanEvictionCount;
            _ = pool.Register(Make(0xCC));         // evict ha → CLEAN (written + rehydrated) → SKIP
            var afterClean = pool.GetReport();
            Assert.Equal(beforeClean, afterClean.DiskWriteBytes);          // no write for a clean entry
            Assert.Equal(cleanBefore + 1, afterClean.CleanEvictionCount);  // counted as a clean skip

            _ = pool.Rehydrate(ha);                // evict hc (write); ha resident, CLEAN again
            pool.MarkDirty(ha);                    // ← content now considered replaced under same handle

            long beforeDirty = pool.GetReport().DiskWriteBytes;
            long cleanBeforeDirty = pool.GetReport().CleanEvictionCount;
            _ = pool.Register(Make(0xDD));         // evict ha → DIRTY → MUST write
            var afterDirty = pool.GetReport();
            Assert.True(afterDirty.DiskWriteBytes > beforeDirty,
                "A dirty entry must be re-persisted on eviction, not skipped.");
            Assert.Equal(cleanBeforeDirty, afterDirty.CleanEvictionCount); // not counted as clean
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }
}
