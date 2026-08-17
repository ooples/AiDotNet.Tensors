// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// A pool whose process is killed must not strand its backing file forever.
///
/// <para><see cref="StreamingTensorPool.Dispose"/> deletes the backing directory,
/// and a process that OOMs or is killed never reaches it. For this pool that is
/// the NORMAL case, not the exotic one: model-family test hosts die under memory
/// pressure routinely. Measured on a developer machine, 430 stranded pool
/// directories accumulated in eight days totalling 257 GB, the largest single
/// backing.bin 117 GB, until the disk hit zero and every build on the box
/// started failing.</para>
///
/// <para>Two mechanisms, tested here. DeleteOnClose makes the kernel drop the
/// bytes when the last handle closes, including handles closed on our behalf
/// when a process dies. The sweep reclaims the directories — and is the only
/// half that can clean up what earlier runs already stranded.</para>
/// </summary>
public class StreamingPoolOrphanReclaimTests
{
    /// <summary>An isolated base directory, so a sweep here cannot see real pools.</summary>
    private static string NewBaseDir() =>
        Path.Combine(Path.GetTempPath(), "aidotnet-sweep-test-" + Guid.NewGuid().ToString("N"));

    /// <summary>A stranded pool directory, aged past the sweep's minimum.</summary>
    private static string GivenOrphan(string baseDir, long bytes = 4096)
    {
        var dir = Path.Combine(baseDir, StreamingTensorPool.BackingDirPrefix + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "backing.bin"), new byte[bytes]);
        Age(dir);
        return dir;
    }

    /// <summary>Backdates a directory past <c>OrphanMinimumAge</c>.</summary>
    private static void Age(string dir) =>
        Directory.SetCreationTimeUtc(dir, DateTime.UtcNow - TimeSpan.FromDays(1));

    [Fact]
    public void Sweep_ReclaimsAStrandedBackingStore()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            var orphan = GivenOrphan(baseDir);

            var reclaimed = StreamingTensorPool.SweepOrphanedBackingStores(baseDir);

            Assert.Equal(1, reclaimed);
            Assert.False(Directory.Exists(orphan));
        }
        finally { Cleanup(baseDir); }
    }

    [Fact]
    public void Sweep_ReclaimsAnEmptyDirectoryLeftByDeleteOnClose()
    {
        // The Windows shape after an abnormal exit: the kernel dropped
        // backing.bin, the directory outlived it.
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            var orphan = Path.Combine(baseDir, StreamingTensorPool.BackingDirPrefix + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(orphan);
            Age(orphan);

            Assert.Equal(1, StreamingTensorPool.SweepOrphanedBackingStores(baseDir));
            Assert.False(Directory.Exists(orphan));
        }
        finally { Cleanup(baseDir); }
    }

    /// <summary>
    /// THE ONE THAT MATTERS. A sweep that can delete a live pool's backing store
    /// would turn a disk-space fix into data loss under the running test.
    /// </summary>
    [Fact]
    public void Sweep_LeavesALivePoolAlone_EvenWhenItsDirectoryLooksOld()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                // Tiny budget so Register pages out immediately and the backing
                // file is genuinely open and held.
                StreamingPoolMaxResidentBytes = 1,
                StreamingBackingStorePath = baseDir,
            });
            var payload = new byte[4096];
            for (int i = 0; i < payload.Length; i++) payload[i] = (byte)(i & 0xFF);
            var handle = pool.Register(payload);

            var live = Assert.Single(Directory.GetDirectories(baseDir));
            Assert.True(File.Exists(Path.Combine(live, "backing.bin")),
                "the pool should have paged out, so its backing file is open");

            // Defeat the age gate deliberately: the file lock, not the
            // timestamp, is what has to protect a running pool.
            Age(live);

            var reclaimed = StreamingTensorPool.SweepOrphanedBackingStores(baseDir);

            Assert.Equal(0, reclaimed);
            Assert.True(Directory.Exists(live));

            // And the weight the sweep ran alongside is still readable, byte for
            // byte. "The directory survived" would not catch a sweep that
            // truncated the file it could not delete.
            var got = pool.Rehydrate(handle);
            Assert.Equal(payload.Length, got.Length);
            for (int i = 0; i < payload.Length; i++) Assert.Equal(payload[i], got[i]);
        }
        finally { Cleanup(baseDir); }
    }

    /// <summary>
    /// A pool constructed moments ago has a directory and NO backing file yet —
    /// paging out is lazy. There is no lock to protect it, so the age gate must.
    /// </summary>
    [Fact]
    public void Sweep_LeavesAFreshlyCreatedDirectoryAlone()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            var fresh = Path.Combine(baseDir, StreamingTensorPool.BackingDirPrefix + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(fresh); // not aged

            Assert.Equal(0, StreamingTensorPool.SweepOrphanedBackingStores(baseDir));
            Assert.True(Directory.Exists(fresh));
        }
        finally { Cleanup(baseDir); }
    }

    [Fact]
    public void Sweep_IgnoresDirectoriesThatAreNotOurs()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            var stranger = Path.Combine(baseDir, "somebody-elses-cache");
            Directory.CreateDirectory(stranger);
            Age(stranger);

            Assert.Equal(0, StreamingTensorPool.SweepOrphanedBackingStores(baseDir));
            Assert.True(Directory.Exists(stranger));
        }
        finally { Cleanup(baseDir); }
    }

    [Fact]
    public void Sweep_ReclaimsManyAtOnce_AndReportsTheCount()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            for (int i = 0; i < 12; i++) GivenOrphan(baseDir);

            Assert.Equal(12, StreamingTensorPool.SweepOrphanedBackingStores(baseDir));
            Assert.Empty(Directory.GetDirectories(baseDir));
        }
        finally { Cleanup(baseDir); }
    }

    [Fact]
    public void Sweep_OnAMissingDirectory_IsNotAnError()
    {
        // Best-effort by contract: a machine that cannot sweep a temp directory
        // has a problem this method is not entitled to escalate.
        Assert.Equal(0, StreamingTensorPool.SweepOrphanedBackingStores(NewBaseDir()));
    }

    /// <summary>
    /// The bytes must be gone the moment the pool's handle closes — not merely
    /// when Dispose gets as far as deleting the directory. That is the property
    /// that survives a kill.
    /// </summary>
    [Fact]
    public void BackingFile_IsDeletedAssoonAsTheHandleCloses()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            string backing;
            using (var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 1,
                StreamingBackingStorePath = baseDir,
            }))
            {
                pool.Register(new byte[4096]);
                backing = Path.Combine(Directory.GetDirectories(baseDir)[0], "backing.bin");
                Assert.True(File.Exists(backing));
            }

            Assert.False(File.Exists(backing));
        }
        finally { Cleanup(baseDir); }
    }

    /// <summary>Paging out and back in still works with the delete-on-close handle.</summary>
    [Fact]
    public void RoundTrip_StillWorks_WithDeleteOnClose()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 2048,
                StreamingBackingStorePath = baseDir,
            });

            var payload = new byte[4096];
            for (int i = 0; i < payload.Length; i++) payload[i] = (byte)(i & 0xFF);
            var handle = pool.Register(payload);

            var got = pool.Rehydrate(handle);

            Assert.Equal(payload.Length, got.Length);
            for (int i = 0; i < payload.Length; i++) Assert.Equal(payload[i], got[i]);
        }
        finally { Cleanup(baseDir); }
    }

    private static void Cleanup(string dir)
    {
        try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
        catch { /* the OS reclaims it */ }
    }
}
