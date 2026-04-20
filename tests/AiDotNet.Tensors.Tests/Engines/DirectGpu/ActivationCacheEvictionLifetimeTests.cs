using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Regression guard for issue #226. <see cref="DirectGpuTensorEngine"/> caches GPU
/// buffers in an LRU-like activation cache and evicts oldest entries when the cache
/// is full. Before the fix, eviction only skipped entries present in an engine-local
/// <c>_deferredDownloads</c> map that <see cref="DirectGpuTensorEngine.DeferTensorResult"/>
/// never populated — so GPU-resident tensors returned to the caller could have their
/// underlying buffer released while a <see cref="Helpers.DeferredArrayMaterializer"/>
/// callback was still pending. The next CPU access triggered a download against a
/// freed OpenCL buffer, producing <c>CL_INVALID_MEM_OBJECT</c> (OpenCL error -38)
/// one epoch into Transformer training on AMD <c>gfx1012</c>.
///
/// The fix replaces the old containsKey eviction guard with the unified
/// <see cref="Helpers.DeferredArrayMaterializer.IsPending(object)"/> check and
/// makes the materializer registry the single source of truth for pending
/// downloads. These tests reach through reflection into the engine's private
/// eviction method to assert the new invariant on any CI host — without
/// requiring a real OpenCL runtime.
/// </summary>
public class ActivationCacheEvictionLifetimeTests
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

    /// <summary>
    /// Looks up the <c>ActivationCacheEntry</c> constructor by its exact parameter
    /// signature rather than relying on <c>GetConstructors()[0]</c>, which silently
    /// breaks the test if a new overload is ever added.
    /// </summary>
    private static ConstructorInfo GetActivationCacheEntryCtor(Type activationCacheEntryType)
    {
        var ctor = activationCacheEntryType.GetConstructor(new[]
        {
            typeof(IGpuBuffer),
            typeof(int[]),
            typeof(long),
            typeof(IDirectGpuBackend)
        });
        Assert.NotNull(ctor);
        return ctor!;
    }

    [Fact]
    public void EvictOldest_SkipsEntriesWithPendingMaterializer()
    {
        // The fix: EvictOldestActivationsUnsafe must consult the global
        // DeferredArrayMaterializer registry. This test reaches past CacheActivation
        // and directly populates _activationCache so the assertion focuses purely
        // on the eviction predicate's skip behaviour.
        using var engine = new DirectGpuTensorEngine();

        var engineType = typeof(DirectGpuTensorEngine);
        var activationCacheField = engineType.GetField(
            "_activationCache", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var activationCacheLockField = engineType.GetField(
            "_activationCacheLock", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var timestampField = engineType.GetField(
            "_activationCacheTimestamp", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var evictMethod = engineType.GetMethod(
            "EvictOldestActivationsUnsafe", BindingFlags.NonPublic | BindingFlags.Instance)!;

        var activationCacheEntryType = engineType.Assembly.GetType(
            "AiDotNet.Tensors.Engines.ActivationCacheEntry")!;
        var entryCtor = GetActivationCacheEntryCtor(activationCacheEntryType);

        object activationCache = activationCacheField.GetValue(engine)!;
        object cacheLock = activationCacheLockField.GetValue(engine)!;

        // Four cache entries: the oldest is protected by a pending materializer;
        // the other three are unprotected and eligible for eviction. EvictOldest
        // removes entries.Length / 2 = 2 entries, scanning by timestamp from
        // oldest up to the median. Without the skip-check, the protected entry
        // would be disposed — that is the #226 regression this test pins.
        var protectedKey = new object();
        var victimKey1 = new object();
        var victimKey2 = new object();
        var victimKey3 = new object();

        var protectedBuffer = new TrackingGpuBuffer(size: 4);
        var victimBuffer1 = new TrackingGpuBuffer(size: 4);
        var victimBuffer2 = new TrackingGpuBuffer(size: 4);
        var victimBuffer3 = new TrackingGpuBuffer(size: 4);

        // ActivationCacheEntry stores the backend for later reads from cached
        // entries; eviction itself only calls Buffer.Dispose(), so a null backend
        // is safe for this test.
        object protectedEntry = entryCtor.Invoke(
            new object?[] { protectedBuffer, new[] { 4 }, 1L, null });
        object victimEntry1 = entryCtor.Invoke(
            new object?[] { victimBuffer1, new[] { 4 }, 2L, null });
        object victimEntry2 = entryCtor.Invoke(
            new object?[] { victimBuffer2, new[] { 4 }, 3L, null });
        object victimEntry3 = entryCtor.Invoke(
            new object?[] { victimBuffer3, new[] { 4 }, 4L, null });

        // ConcurrentDictionary<object, ActivationCacheEntry>.TryAdd via runtime dispatch.
        var tryAdd = activationCache.GetType().GetMethod("TryAdd")!;
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { protectedKey, protectedEntry })!);
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { victimKey1, victimEntry1 })!);
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { victimKey2, victimEntry2 })!);
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { victimKey3, victimEntry3 })!);
        timestampField.SetValue(engine, 4L);

        // Register a materializer for the protected key. IsPending(protectedKey) now true.
        // DeferredArrayMaterializer is process-global state, so any test body that
        // registers into it must guarantee cleanup even on assertion failure —
        // hence the try/finally.
        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.Register(protectedKey, _ => { });
        try
        {
            Assert.True(AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(protectedKey));

            object[] evictedList;
            lock (cacheLock)
            {
                evictedList = ((System.Collections.IEnumerable)evictMethod.Invoke(engine, null)!)
                    .Cast<object>().ToArray();
            }

            // Dispose like the real code does (outside the cache lock).
            var disposeMethod = activationCacheEntryType.GetMethod("Dispose")!;
            foreach (var e in evictedList) disposeMethod.Invoke(e, null);

            // The protected entry's buffer MUST NOT have been disposed; at least one of the
            // unprotected victims should have been swept (eviction removes entries.Length/2).
            Assert.Equal(0, protectedBuffer.DisposeCount);
            int totalVictimDisposes =
                victimBuffer1.DisposeCount + victimBuffer2.DisposeCount + victimBuffer3.DisposeCount;
            Assert.True(totalVictimDisposes >= 1,
                $"Expected eviction to dispose at least one unprotected buffer when the cache is over capacity. " +
                $"v1={victimBuffer1.DisposeCount}, v2={victimBuffer2.DisposeCount}, v3={victimBuffer3.DisposeCount}");
        }
        finally
        {
            // Always clear the process-global registry so later tests start clean.
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.Remove(protectedKey);
        }
    }

    [Fact]
    public void EvictOldest_EvictsHalfWhenNoMaterializerPending()
    {
        // Baseline negative: without any pending materializer, the old eviction path and
        // the new one agree — EvictOldest removes entries.Length / 2 entries (floor),
        // so 2 entries in the cache results in 1 disposal.
        using var engine = new DirectGpuTensorEngine();

        var engineType = typeof(DirectGpuTensorEngine);
        var activationCacheField = engineType.GetField(
            "_activationCache", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var activationCacheLockField = engineType.GetField(
            "_activationCacheLock", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var evictMethod = engineType.GetMethod(
            "EvictOldestActivationsUnsafe", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var activationCacheEntryType = engineType.Assembly.GetType(
            "AiDotNet.Tensors.Engines.ActivationCacheEntry")!;
        var entryCtor = GetActivationCacheEntryCtor(activationCacheEntryType);

        object activationCache = activationCacheField.GetValue(engine)!;
        object cacheLock = activationCacheLockField.GetValue(engine)!;

        var key1 = new object();
        var key2 = new object();
        var buffer1 = new TrackingGpuBuffer(size: 4);
        var buffer2 = new TrackingGpuBuffer(size: 4);

        object entry1 = entryCtor.Invoke(new object?[] { buffer1, new[] { 4 }, 1L, null });
        object entry2 = entryCtor.Invoke(new object?[] { buffer2, new[] { 4 }, 2L, null });

        var tryAdd = activationCache.GetType().GetMethod("TryAdd")!;
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { key1, entry1 })!);
        Assert.True((bool)tryAdd.Invoke(activationCache, new[] { key2, entry2 })!);

        object[] evictedList;
        lock (cacheLock)
        {
            evictedList = ((System.Collections.IEnumerable)evictMethod.Invoke(engine, null)!)
                .Cast<object>().ToArray();
        }

        var disposeMethod = activationCacheEntryType.GetMethod("Dispose")!;
        foreach (var e in evictedList) disposeMethod.Invoke(e, null);

        // EvictOldestActivationsUnsafe removes entries.Length/2 = 1 of 2 when nothing is pinned.
        Assert.Equal(1, buffer1.DisposeCount + buffer2.DisposeCount);
    }
}
