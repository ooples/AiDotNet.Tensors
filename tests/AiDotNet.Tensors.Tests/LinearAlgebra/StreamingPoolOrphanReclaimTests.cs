// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.Threading;
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

    /// <summary>
    /// A prefix match is not ownership, and this sweep deletes RECURSIVELY.
    ///
    /// <para><c>GetDirectories(baseDir, prefix + "*")</c> returns anything
    /// starting with the prefix, so somebody's own
    /// <c>aidotnet-streaming-pool-backup</c> sitting next to the real pools
    /// would have been destroyed along with whatever they put in it. Only the
    /// exact shape we mint — prefix plus a 32-digit "N" Guid — is ours.</para>
    /// </summary>
    [Theory]
    [InlineData("backup")]
    [InlineData("old")]
    [InlineData("2026-08-17")]
    [InlineData("deadbeef")]                              // hex, but too short
    [InlineData("0123456789abcdef0123456789abcdefx")]     // right length, not hex
    [InlineData("0123456789abcdef0123456789abcdef0")]     // one digit too long
    public void Sweep_LeavesPrefixMatchingDirectoriesThatWeDidNotCreate(string suffix)
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            var impostor = Path.Combine(baseDir, StreamingTensorPool.BackingDirPrefix + suffix);
            Directory.CreateDirectory(impostor);
            var treasure = Path.Combine(impostor, "do-not-delete.txt");
            File.WriteAllText(treasure, "someone's data");
            Age(impostor);

            Assert.Equal(0, StreamingTensorPool.SweepOrphanedBackingStores(baseDir));
            Assert.True(Directory.Exists(impostor));
            Assert.True(File.Exists(treasure));
        }
        finally { Cleanup(baseDir); }
    }

    [Fact]
    public void PoolDirectoryName_AcceptsExactlyWhatThePoolMints()
    {
        var minted = StreamingTensorPool.BackingDirPrefix + Guid.NewGuid().ToString("N");
        Assert.True(StreamingTensorPool.IsPoolDirectoryName(minted));

        Assert.False(StreamingTensorPool.IsPoolDirectoryName(null));
        Assert.False(StreamingTensorPool.IsPoolDirectoryName(string.Empty));
        Assert.False(StreamingTensorPool.IsPoolDirectoryName(StreamingTensorPool.BackingDirPrefix));
        // A Guid in "D" format has hyphens, so it is not a name we produce.
        Assert.False(StreamingTensorPool.IsPoolDirectoryName(
            StreamingTensorPool.BackingDirPrefix + Guid.NewGuid().ToString("D")));
    }

    /// <summary>
    /// THE DANGEROUS ONE. A pool that stays under budget never pages out, so it
    /// has a directory and NO backing.bin to lock. Once that directory ages past
    /// OrphanMinimumAge, the age gate stops protecting it — and deleting it out
    /// from under its living owner breaks the owner's next page-out, because
    /// FileMode.Create cannot create a file under a directory that is gone.
    ///
    /// <para>Live-pool registration, not age, is what has to save it.</para>
    /// </summary>
    [Fact]
    public void Sweep_LeavesALiveButIdlePoolAlone_EvenWhenItsDirectoryIsOld()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            // Budget high enough that Register never evicts: no backing.bin.
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 1024 * 1024,
                StreamingBackingStorePath = baseDir,
            });

            var live = Assert.Single(Directory.GetDirectories(baseDir));
            Assert.False(File.Exists(Path.Combine(live, "backing.bin")),
                "this test is only meaningful while the pool has NOT paged out");
            Age(live);

            Assert.Equal(0, StreamingTensorPool.SweepOrphanedBackingStores(baseDir));
            Assert.True(Directory.Exists(live));

            // And the pool still works afterwards -- the point of not deleting
            // it. This forces the first page-out, which would throw if its
            // parent directory had been swept away.
            var payload = new byte[4096];
            for (int i = 0; i < payload.Length; i++) payload[i] = (byte)(i & 0xFF);
            var handle = pool.Register(payload);
            for (int i = 0; i < 40; i++) pool.Register(new byte[4096]); // force eviction
            var got = pool.Rehydrate(handle);
            Assert.Equal(payload.Length, got.Length);
            for (int i = 0; i < payload.Length; i++) Assert.Equal(payload[i], got[i]);
        }
        finally { Cleanup(baseDir); }
    }

    /// <summary>
    /// Sweep scheduling is per base directory, not per process.
    ///
    /// <para>It used to be one static flag, so the first pool constructed swept
    /// its own base directory and every pool afterwards skipped. A process that
    /// used the temp directory and then a custom StreamingBackingStorePath
    /// swept only the first, and orphans under the second accumulated forever.</para>
    /// </summary>
    [Fact]
    public void Sweep_ReclaimsOrphansUnderASecondBaseDirectory()
    {
        var first = NewBaseDir();
        var second = NewBaseDir();
        Directory.CreateDirectory(first);
        Directory.CreateDirectory(second);
        try
        {
            // A pool in the first directory: schedules that directory's sweep and,
            // under the old single-flag code, permanently latched the process.
            using (new StreamingTensorPool(new GpuOffloadOptions { StreamingBackingStorePath = first })) { }

            // An orphan under a DIFFERENT base directory must still be reclaimed
            // when a pool is created there.
            var orphan = GivenOrphan(second);
            using (new StreamingTensorPool(new GpuOffloadOptions { StreamingBackingStorePath = second })) { }

            // Wait for the SCHEDULED sweep specifically -- calling
            // SweepOrphanedBackingStores here instead would delete the orphan by
            // itself and pass even when nothing was ever scheduled, which is the
            // regression under test.
            Assert.True(WaitUntilGone(orphan),
                "constructing a pool under a second base directory must schedule a " +
                "sweep for THAT directory; with one process-wide flag it never did");
        }
        finally { Cleanup(first); Cleanup(second); }
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
    /// The bytes must be gone the moment the pool's HANDLE closes — not merely
    /// when Dispose gets as far as deleting the directory. That is the property
    /// that survives a kill, and it is the only part of this change that
    /// reclaims anything on an abnormal exit.
    ///
    /// <para>Asserting it after Dispose would prove nothing: Dispose closes the
    /// handle and then deletes the whole directory, so backing.bin vanishes
    /// either way and the test would keep passing with
    /// <c>FileOptions.DeleteOnClose</c> removed. Closing the handle ALONE, and
    /// checking that the directory is still standing while the file inside it
    /// is gone, is what actually pins the flag down.</para>
    /// </summary>
    [Fact]
    public void BackingFile_IsDeletedWhenTheHandleCloses_WithTheDirectoryStillStanding()
    {
        var baseDir = NewBaseDir();
        Directory.CreateDirectory(baseDir);
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 1,
                StreamingBackingStorePath = baseDir,
            });
            pool.Register(new byte[4096]);

            var poolDir = Assert.Single(Directory.GetDirectories(baseDir));
            var backing = Path.Combine(poolDir, "backing.bin");
            Assert.True(File.Exists(backing), "the pool should have paged out");

            pool.CloseBackingFileForTests();

            Assert.False(File.Exists(backing));
            Assert.True(Directory.Exists(poolDir),
                "only the handle was closed, so the directory must be untouched -- " +
                "which is what makes the vanished file attributable to DeleteOnClose");
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

    /// <summary>
    /// Waits for a background sweep to remove a directory. Polls rather than
    /// sleeping a fixed span so the test is neither flaky nor slow.
    /// </summary>
    private static bool WaitUntilGone(string dir, int timeoutMs = 15_000)
    {
        var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (DateTime.UtcNow < deadline)
        {
            if (!Directory.Exists(dir)) return true;
            Thread.Sleep(25);
        }
        return !Directory.Exists(dir);
    }

    private static void Cleanup(string dir)
    {
        try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
        catch { /* the OS reclaims it */ }
    }
}
