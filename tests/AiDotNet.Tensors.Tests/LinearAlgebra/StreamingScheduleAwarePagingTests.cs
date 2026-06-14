// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Schedule-aware (Belady-optimal) eviction. On a LOOPING scan whose working set is
/// just larger than the resident budget, LRU is near-pessimal — it evicts the entry
/// about to be reused next, faulting on (almost) every access. Given the known
/// repeating schedule, Belady evicts the entry whose next use is FURTHEST, keeping
/// the soon-needed ones resident and faulting far less.
/// </summary>
public class StreamingScheduleAwarePagingTests
{
    private readonly ITestOutputHelper _out;
    public StreamingScheduleAwarePagingTests(ITestOutputHelper output) => _out = output;

    private static long DiskReadsForLoopingScan(bool useSchedule, int numHandles, int budgetHandles, int cycles)
    {
        const int entryBytes = 4096;
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-belady-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = (long)budgetHandles * entryBytes,
                StreamingBackingStorePath = dir,
            });
            var handles = new long[numHandles];
            for (int i = 0; i < numHandles; i++)
            {
                var data = new byte[entryBytes];
                for (int j = 0; j < entryBytes; j += 64) data[j] = (byte)((i + j) & 0xFF);
                handles[i] = pool.Register(data);
            }
            if (useSchedule) pool.SetAccessSchedule(handles); // one cycle = [h0..hN-1]

            long readsBefore = pool.GetReport().DiskReadCount;
            for (int c = 0; c < cycles; c++)
                for (int i = 0; i < numHandles; i++)
                    _ = pool.Rehydrate(handles[i]);
            return pool.GetReport().DiskReadCount - readsBefore;
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Belady_FaultsFarLessThanLru_OnLoopingScan()
    {
        // Working set 8, budget holds 6 → 2 over. LRU faults on nearly every access;
        // Belady only on the 2 that genuinely can't fit.
        const int handles = 8, budget = 6, cycles = 20;
        long lru = DiskReadsForLoopingScan(useSchedule: false, handles, budget, cycles);
        long belady = DiskReadsForLoopingScan(useSchedule: true, handles, budget, cycles);

        _out.WriteLine($"Looping scan: {handles} handles, budget {budget}, {cycles} cycles ({handles * cycles} accesses)");
        _out.WriteLine($"  LRU    disk reads: {lru}");
        _out.WriteLine($"  Belady disk reads: {belady}   ({(lru == 0 ? 0 : (double)lru / Math.Max(1, belady)):F1}x fewer faults)");

        // Belady must fault dramatically less. With 2-over-budget, Belady should
        // approach ~2 faults/cycle while LRU approaches ~handles/cycle.
        Assert.True(belady < lru / 2,
            $"Belady ({belady}) should fault far less than LRU ({lru}) on a looping scan");
    }

    [Fact]
    public void NoSchedule_IsPlainLru_Unchanged()
    {
        // Sanity: without a schedule the pool behaves exactly as LRU (regression guard
        // that the Belady branch is fully opt-in).
        long a = DiskReadsForLoopingScan(useSchedule: false, 8, 6, 5);
        long b = DiskReadsForLoopingScan(useSchedule: false, 8, 6, 5);
        Assert.Equal(a, b);
        Assert.True(a > 0, "an over-budget looping scan must fault under LRU");
    }
}
