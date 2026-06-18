using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Locks the contract of the deterministic per-step activation-cache release added with
/// the managed-byte cap (PR #552). The outermost <see cref="Autodiff.GradientTape{T}"/>
/// snapshots <see cref="DirectGpuTensorEngine.ActivationCacheTimestampSnapshot"/> at
/// construction and on Dispose calls <see cref="DirectGpuTensorEngine.EvictActivationsCreatedAfter"/>
/// to release EXACTLY that forward+backward's intermediate activations (PyTorch/JAX
/// per-step memory semantics). Without this contract, prior cross-step activations
/// strong-rooted by the cache float[] keys grew the managed heap to 36GB+ before the OOM.
///
/// These tests pin three invariants on any CI host without requiring a GPU runtime:
///   1. Pre-snapshot entries are preserved (cross-tape intermediates survive).
///   2. Post-snapshot entries are released — buffer disposed, both byte/managed counters drop.
///   3. The #226 materialize-then-free contract still holds — a pending deferred download
///      is materialized BEFORE the buffer is freed, so a later CPU read never touches
///      a freed buffer.
/// </summary>
[Collection("DirectGpuSerial")]
public class EvictActivationsCreatedAfterLifetimeTests
{
    /// <summary>Fake GPU buffer — tracks Dispose so the test can assert on lifetime.</summary>
    private sealed class TrackingGpuBuffer : IGpuBuffer
    {
        public int DisposeCount;
        public int Size { get; }
        public long SizeInBytes => Size * sizeof(float);
        public IntPtr Handle => IntPtr.Zero;

        public TrackingGpuBuffer(int size) { Size = size; }
        public void Dispose() => DisposeCount++;
    }

    private static ConstructorInfo GetActivationCacheEntryCtor(Type activationCacheEntryType)
    {
        var ctor = activationCacheEntryType.GetConstructor(new[]
        {
            typeof(IGpuBuffer),
            typeof(int[]),
            typeof(long),
            typeof(IDirectGpuBackend),
            typeof(long)
        });
        Assert.NotNull(ctor);
        return ctor!;
    }

    [Fact]
    public void EvictActivationsCreatedAfter_KeepsPreSnapshot_ReleasesPostSnapshot()
    {
        using var engine = new DirectGpuTensorEngine();

        var engineType = typeof(DirectGpuTensorEngine);
        var activationCacheField = engineType.GetField(
            "_activationCache", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var timestampField = engineType.GetField(
            "_activationCacheTimestamp", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var snapshotMethod = engineType.GetMethod(
            "ActivationCacheTimestampSnapshot", BindingFlags.NonPublic | BindingFlags.Instance)!;
        // Resolve the single-argument EvictActivationsCreatedAfter(long) overload by its exact parameter types:
        // #633 added (long, HashSet) and (long, HashSet, bool) overloads, so a name-only GetMethod now throws
        // AmbiguousMatchException. This test exercises the 1-arg snapshot overload (see the single-arg Invoke
        // calls below), so bind that signature explicitly — same pattern as GetActivationCacheEntryCtor above.
        var evictMethod = engineType.GetMethod(
            "EvictActivationsCreatedAfter", BindingFlags.NonPublic | BindingFlags.Instance,
            null, new[] { typeof(long) }, null)!;
        var activationCacheEntryType = engineType.Assembly.GetType(
            "AiDotNet.Tensors.Engines.ActivationCacheEntry")!;
        var entryCtor = GetActivationCacheEntryCtor(activationCacheEntryType);
        object activationCache = activationCacheField.GetValue(engine)!;
        var tryAdd = activationCache.GetType().GetMethod("TryAdd")!;

        // Seed one entry at timestamp 1 (pre-snapshot survivor), snapshot, then seed two
        // more at timestamps 2 and 3 (post-snapshot, must be released).
        var keyPre = new float[4];
        var keyPost1 = new float[4];
        var keyPost2 = new float[4];
        var bufPre = new TrackingGpuBuffer(size: 4);
        var bufPost1 = new TrackingGpuBuffer(size: 4);
        var bufPost2 = new TrackingGpuBuffer(size: 4);

        // ManagedBytes argument matches what CacheActivation computes: product(Shape) * 4.
        object entryPre = entryCtor.Invoke(new object?[] { bufPre, new[] { 4 }, 1L, null, 16L });
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { (object)keyPre, entryPre })!);
        timestampField.SetValue(engine, 1L);

        long snapshot = (long)snapshotMethod.Invoke(engine, null)!;
        Assert.Equal(1L, snapshot);

        object entryPost1 = entryCtor.Invoke(new object?[] { bufPost1, new[] { 4 }, 2L, null, 16L });
        object entryPost2 = entryCtor.Invoke(new object?[] { bufPost2, new[] { 4 }, 3L, null, 16L });
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { (object)keyPost1, entryPost1 })!);
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { (object)keyPost2, entryPost2 })!);

        evictMethod.Invoke(engine, new object[] { snapshot });

        // Pre-snapshot entry survives — buffer not disposed, still in cache.
        Assert.Equal(0, bufPre.DisposeCount);
        Assert.True(((System.Collections.IDictionary)activationCache).Contains(keyPre));

        // Post-snapshot entries are released — buffers disposed, removed from cache.
        Assert.Equal(1, bufPost1.DisposeCount);
        Assert.Equal(1, bufPost2.DisposeCount);
        Assert.False(((System.Collections.IDictionary)activationCache).Contains(keyPost1));
        Assert.False(((System.Collections.IDictionary)activationCache).Contains(keyPost2));
    }

    [Fact]
    public void EvictActivationsCreatedAfter_MaterializesPendingDownloadBeforeFreeing()
    {
        // Same #226 contract as InvalidateActivationCacheEntryLifetimeTests, on the
        // per-step release path: if an entry being released has a registered
        // materializer, the download must complete BEFORE the buffer is freed.
        // Without this, a later CPU read of the activation array fires the
        // materializer against a freed GPU buffer (CL_INVALID_MEM_OBJECT, the
        // original gfx1012 crash).
        using var engine = new DirectGpuTensorEngine();
        var engineType = typeof(DirectGpuTensorEngine);
        var activationCacheField = engineType.GetField(
            "_activationCache", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var timestampField = engineType.GetField(
            "_activationCacheTimestamp", BindingFlags.NonPublic | BindingFlags.Instance)!;
        // Resolve the single-argument EvictActivationsCreatedAfter(long) overload by its exact parameter types:
        // #633 added (long, HashSet) and (long, HashSet, bool) overloads, so a name-only GetMethod now throws
        // AmbiguousMatchException. This test exercises the 1-arg snapshot overload (see the single-arg Invoke
        // calls below), so bind that signature explicitly — same pattern as GetActivationCacheEntryCtor above.
        var evictMethod = engineType.GetMethod(
            "EvictActivationsCreatedAfter", BindingFlags.NonPublic | BindingFlags.Instance,
            null, new[] { typeof(long) }, null)!;
        var activationCacheEntryType = engineType.Assembly.GetType(
            "AiDotNet.Tensors.Engines.ActivationCacheEntry")!;
        var entryCtor = GetActivationCacheEntryCtor(activationCacheEntryType);
        object activationCache = activationCacheField.GetValue(engine)!;
        var tryAdd = activationCache.GetType().GetMethod("TryAdd")!;

        var key = new float[4];
        var buffer = new TrackingGpuBuffer(size: 4);
        object entry = entryCtor.Invoke(new object?[] { buffer, new[] { 4 }, 5L, null, 16L });
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { (object)key, entry })!);
        timestampField.SetValue(engine, 5L);

        int disposeCountWhenMaterialized = -1;
        bool materializerRan = false;
        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.Register(key, _ =>
        {
            materializerRan = true;
            disposeCountWhenMaterialized = buffer.DisposeCount;
        });

        try
        {
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(key));

            // Snapshot of 0 means "all entries are post-snapshot" — releases everything.
            evictMethod.Invoke(engine, new object[] { 0L });

            Assert.True(materializerRan,
                "EvictActivationsCreatedAfter must materialize a pending deferred download " +
                "before disposing its buffer (#226 contract on the per-step release path).");
            Assert.Equal(0, disposeCountWhenMaterialized);
            Assert.False(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(key));
            Assert.Equal(1, buffer.DisposeCount);
        }
        finally
        {
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.Remove(key);
        }
    }

    [Fact]
    public void EvictActivationsCreatedAfter_EmptyCache_IsSafeNoOp()
    {
        // Construction-time path: a GradientTape may capture a snapshot before any
        // activation lands in the cache. The release call must be tolerant of an
        // empty cache — early-return without locking-and-iterating thrash, and
        // without throwing. This is the "nested tape under a no-op forward" shape.
        using var engine = new DirectGpuTensorEngine();
        var engineType = typeof(DirectGpuTensorEngine);
        // Resolve the single-argument EvictActivationsCreatedAfter(long) overload by its exact parameter types:
        // #633 added (long, HashSet) and (long, HashSet, bool) overloads, so a name-only GetMethod now throws
        // AmbiguousMatchException. This test exercises the 1-arg snapshot overload (see the single-arg Invoke
        // calls below), so bind that signature explicitly — same pattern as GetActivationCacheEntryCtor above.
        var evictMethod = engineType.GetMethod(
            "EvictActivationsCreatedAfter", BindingFlags.NonPublic | BindingFlags.Instance,
            null, new[] { typeof(long) }, null)!;
        evictMethod.Invoke(engine, new object[] { 0L });
        evictMethod.Invoke(engine, new object[] { long.MaxValue });
    }

    [Fact]
    public void ActivationCacheTimestampSnapshot_IsMonotonicAndMatchesField()
    {
        using var engine = new DirectGpuTensorEngine();
        var engineType = typeof(DirectGpuTensorEngine);
        var timestampField = engineType.GetField(
            "_activationCacheTimestamp", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var snapshotMethod = engineType.GetMethod(
            "ActivationCacheTimestampSnapshot", BindingFlags.NonPublic | BindingFlags.Instance)!;

        Assert.Equal(0L, (long)snapshotMethod.Invoke(engine, null)!);

        timestampField.SetValue(engine, 42L);
        Assert.Equal(42L, (long)snapshotMethod.Invoke(engine, null)!);

        // Monotonic — snapshot reflects later writes.
        timestampField.SetValue(engine, 1000L);
        Assert.Equal(1000L, (long)snapshotMethod.Invoke(engine, null)!);
    }
}
