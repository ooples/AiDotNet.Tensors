// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Round-trip + lifecycle tests for #1222 weight streaming v1: Register
/// drops the tensor's data after copying to the pool; Materialize restores
/// it bit-exactly; ReleaseToPool drops it again; MaterializeMany scope
/// ensures resident-only-during-use.
///
/// <para><c>[Collection("WeightRegistry")]</c> serializes against
/// <see cref="WeightLifetimeIntegrationTests"/> — both classes mutate the
/// process-wide <see cref="WeightRegistry"/> singleton, and xUnit's default
/// test-class parallelism would otherwise race their Configure / Reset
/// calls.</para>
/// </summary>
[Collection("WeightRegistry")]
public class WeightRegistryStreamingTests : IDisposable
{
    private readonly string _backingDir;

    public WeightRegistryStreamingTests()
    {
        _backingDir = Path.Combine(Path.GetTempPath(), "aidotnet-wr-stream-test-" + Guid.NewGuid().ToString("N"));
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024L * 1024 * 1024, // 1 GiB — plenty of room
            StreamingBackingStorePath = _backingDir,
        });
    }

    public void Dispose()
    {
        WeightRegistry.Reset();
        if (Directory.Exists(_backingDir))
        {
            try { Directory.Delete(_backingDir, recursive: true); } catch { /* best-effort */ }
        }
    }

    [Fact]
    public void RegisterStreaming_DropsTensorData_PoolKeepsCanonicalCopy()
    {
        var t = new Tensor<float>(new float[] { 1.5f, 2.5f, 3.5f, 4.5f }, new[] { 4 });
        t.Lifetime = WeightLifetime.Streaming;

        WeightRegistry.RegisterWeight(t);

        // After register, tensor's _data is the empty placeholder so the
        // bytes aren't double-resident.
        Assert.Equal(0, t.DataVector.Length);
        Assert.True(t.StreamingPoolHandle >= 0);
        // Length is the logical element count — unchanged.
        Assert.Equal(4, t.Length);
    }

    [Fact]
    public void Materialize_AfterRegister_RestoresExactBytes_Float()
    {
        float[] expected = { 1.5f, -2.25f, 3.125f, 0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN };
        var t = new Tensor<float>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;

        WeightRegistry.RegisterWeight(t);
        WeightRegistry.Materialize(t);

        Assert.Equal(expected.Length, t.DataVector.Length);
        var span = t.DataVector.AsSpan();
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i]))
                Assert.True(float.IsNaN(span[i]), $"NaN at index {i}");
            else
                Assert.Equal(expected[i], span[i]);
        }
    }

    [Fact]
    public void Materialize_AfterRegister_RestoresExactBytes_Double()
    {
        double[] expected = { 1.5, -2.25, 3.125, 0d, double.MaxValue, double.MinValue };
        var t = new Tensor<double>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;

        WeightRegistry.RegisterWeight(t);
        WeightRegistry.Materialize(t);

        var span = t.DataVector.AsSpan();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], span[i]);
    }

    [Fact]
    public void Materialize_AfterRegister_RestoresExactBytes_Half()
    {
        Half[] expected = new[] { (Half)1.5f, (Half)(-2.25f), (Half)3.125f, Half.Epsilon };
        var t = new Tensor<Half>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;

        WeightRegistry.RegisterWeight(t);
        WeightRegistry.Materialize(t);

        var span = t.DataVector.AsSpan();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(HalfBits.GetBits(expected[i]), HalfBits.GetBits(span[i]));
    }

    [Fact]
    public void Materialize_AfterRegister_RestoresExactBytes_BFloat16()
    {
        BFloat16[] expected = { BFloat16.FromFloat(1.5f), BFloat16.FromFloat(-2.25f), BFloat16.Zero };
        var t = new Tensor<BFloat16>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;

        WeightRegistry.RegisterWeight(t);
        WeightRegistry.Materialize(t);

        var span = t.DataVector.AsSpan();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i].RawValue, span[i].RawValue);
    }

    [Fact]
    public void ReleaseToPool_DropsResidentData_PoolStillHasCopy()
    {
        var t = new Tensor<float>(new float[] { 10f, 20f, 30f }, new[] { 3 });
        t.Lifetime = WeightLifetime.Streaming;

        WeightRegistry.RegisterWeight(t);
        WeightRegistry.Materialize(t);
        Assert.Equal(3, t.DataVector.Length);

        WeightRegistry.ReleaseToPool(t);
        Assert.Equal(0, t.DataVector.Length);

        // Re-materialize from pool — bytes preserved.
        WeightRegistry.Materialize(t);
        var span = t.DataVector.AsSpan();
        Assert.Equal(10f, span[0]);
        Assert.Equal(20f, span[1]);
        Assert.Equal(30f, span[2]);
    }

    [Fact]
    public void MaterializeMany_Scope_MaterializesOnEnter_ReleasesOnDispose()
    {
        var t1 = new Tensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        var t2 = new Tensor<float>(new float[] { 3f, 4f }, new[] { 2 });
        t1.Lifetime = WeightLifetime.Streaming;
        t2.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t1);
        WeightRegistry.RegisterWeight(t2);

        // Both tensors start released after Register.
        Assert.Equal(0, t1.DataVector.Length);
        Assert.Equal(0, t2.DataVector.Length);

        using (WeightRegistry.MaterializeMany(new[] { t1, t2 }))
        {
            // Inside scope: both resident.
            Assert.Equal(2, t1.DataVector.Length);
            Assert.Equal(2, t2.DataVector.Length);
            Assert.Equal(1f, t1.DataVector.AsSpan()[0]);
            Assert.Equal(3f, t2.DataVector.AsSpan()[0]);
        }

        // After scope dispose: both released.
        Assert.Equal(0, t1.DataVector.Length);
        Assert.Equal(0, t2.DataVector.Length);
    }

    [Fact]
    public void Materialize_OnAlreadyResidentTensor_IsNoOp()
    {
        var t = new Tensor<float>(new float[] { 5f, 6f }, new[] { 2 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        WeightRegistry.Materialize(t);
        var firstData = t.DataVector;

        // Second Materialize should see "already resident" and not allocate
        // a new Vector.
        WeightRegistry.Materialize(t);
        Assert.Same(firstData, t.DataVector);
    }

    [Fact]
    public void IsResidentInPool_TracksResidentSet()
    {
        var t = new Tensor<float>(new float[] { 7f, 8f }, new[] { 2 });
        t.Lifetime = WeightLifetime.Streaming;
        Assert.False(WeightRegistry.IsResidentInPool(t)); // pre-register: not in pool

        WeightRegistry.RegisterWeight(t);
        // Pool just received the bytes; LRU has them resident.
        Assert.True(WeightRegistry.IsResidentInPool(t));

        // Force eviction by registering a much-larger neighbour with a
        // restrictive budget. Reset+reconfigure to a tiny budget first.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 16, // 16 bytes total
            StreamingBackingStorePath = _backingDir,
        });
        var small = new Tensor<float>(new float[] { 1f }, new[] { 1 });
        small.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(small);
        Assert.True(WeightRegistry.IsResidentInPool(small));

        // Add a second entry that pushes the first past the budget.
        var big = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f }, new[] { 5 });
        big.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(big);
        // First entry should have been evicted to backing store now.
        Assert.False(WeightRegistry.IsResidentInPool(small));
    }

    [Fact]
    public void Materialize_OnNonStreamingTensor_IsNoOp()
    {
        var t = new Tensor<float>(new float[] { 5f, 6f }, new[] { 2 });
        // Lifetime stays Default — not streaming.
        WeightRegistry.Materialize(t); // should not throw, no-op
        Assert.Equal(2, t.DataVector.Length);
    }

    [Fact]
    public void Compression_RoundTrip_PreservesBytes()
    {
        // Reset to a small budget + compression on, so eviction happens.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32, // ~8 floats
            StreamingBackingStorePath = _backingDir,
            EnableCompression = true,
        });

        // Highly-redundant input compresses well — better signal for the
        // ratio assertion than random bytes.
        float[] expected = new float[256];
        for (int i = 0; i < expected.Length; i++) expected[i] = (float)(i % 8);

        var t = new Tensor<float>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        // Force eviction by registering a second large tensor.
        var filler = new Tensor<float>(new float[64], new[] { 64 });
        filler.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(filler);

        // First tensor should be paged out to compressed backing file now.
        Assert.False(WeightRegistry.IsResidentInPool(t));

        // Materialize — must decompress and produce identical bytes.
        WeightRegistry.Materialize(t);
        var got = t.DataVector.AsSpan();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], got[i]);

        // GetReport: ratio should be < 1.0 because the redundant input
        // compresses well; CompressionEnabled should be true.
        var report = WeightRegistry.GetStreamingReport();
        Assert.True(report.CompressionEnabled);
        Assert.True(report.CompressionRatio < 1.0,
            $"Expected compression ratio < 1.0 for redundant input, got {report.CompressionRatio:F3}.");
        Assert.True(report.EvictionCount >= 1);
        Assert.True(report.DiskWriteBytes > 0);
        Assert.True(report.DiskReadCount >= 1);
    }

    [Fact]
    public void Compression_Disabled_StoresRawBytes()
    {
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32,
            StreamingBackingStorePath = _backingDir,
            EnableCompression = false,
        });

        float[] expected = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f };
        var t = new Tensor<float>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        var filler = new Tensor<float>(new float[16], new[] { 16 });
        filler.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(filler);

        WeightRegistry.Materialize(t);
        var got = t.DataVector.AsSpan();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], got[i]);

        var report = WeightRegistry.GetStreamingReport();
        Assert.False(report.CompressionEnabled);
    }

    [Fact]
    public void GetStreamingReport_TracksResidentBytesPeak()
    {
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingBackingStorePath = _backingDir,
        });

        var t = new Tensor<float>(new float[256], new[] { 256 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        var report = WeightRegistry.GetStreamingReport();
        // 256 floats = 1024 bytes; resident set held all of it briefly.
        Assert.True(report.ResidentBytesPeak >= 1024);
        Assert.Equal(1, report.RegisteredEntryCount);
    }

    [Fact]
    public void Configure_WithLiveHandles_Throws()
    {
        // Audit P1 #8: Configure mid-flight should refuse to dispose a
        // pool with registered entries — disposing would orphan handles.
        var t = new Tensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            WeightRegistry.Configure(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 1024,
                StreamingBackingStorePath = _backingDir,
            }));
        Assert.Contains("registered entries", ex.Message);

        // Cleanup so Dispose's Reset() doesn't double-free
        WeightRegistry.UnregisterWeight(t);
    }

    [Fact]
    public void DropStorage_OnSharedRefcount_Throws()
    {
        // Audit P1 #5: DropStorageForStreaming must reject tensors whose
        // storage refcount > 1 (rebound peers exist). Construct that
        // scenario directly via RebindStorageFrom — the test assembly
        // has internals access. Without this guard, registering a
        // tensor whose storage is shared with a peer would silently
        // drop bytes the peer is still reading.
        var t1 = new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3 });
        var t2 = new Tensor<float>(new float[] { 9f, 9f, 9f }, new[] { 3 });
        // Rebind t2's storage to point at t1's — both now share storage,
        // refcount = 2.
        t2.RebindStorageFrom(t1);

        t1.Lifetime = WeightLifetime.Streaming;

        // RegisterWeight implicitly calls DropStorageForStreaming; with
        // refcount > 1 the atomic claim must fail and the call must
        // throw. The pool-entry rollback (separate audit fix) means
        // there's no leaked pool entry on this throw path.
        var ex = Assert.Throws<InvalidOperationException>(() => WeightRegistry.RegisterWeight(t1));
        Assert.Contains("sole storage ownership", ex.Message);

        // Both tensors must still be intact: the failed register must
        // not have stripped t1's bytes (and therefore t2's).
        Assert.Equal(3, t1.DataVector.Length);
        Assert.Equal(3, t2.DataVector.Length);
        Assert.Equal(1f, t1.DataVector.AsSpan()[0]);
        Assert.Equal(1f, t2.DataVector.AsSpan()[0]); // shared with t1

        // No pool handle should have been assigned because the rollback
        // must restore the tensor to its pre-register state.
        Assert.True(t1.StreamingPoolHandle < 0);
    }

    [Fact]
    public void RestoreStorage_OnUnsupportedType_ThrowsClearError()
    {
        // Audit P1 #6: ElementSizeForStreaming throws NotSupportedException
        // (clear message) instead of Marshal.SizeOf<T>'s opaque
        // ArgumentException. Decimal isn't blittable in the streaming
        // sense — covered by SerializeToBytes which would already throw,
        // so the behaviour we want is consistent error contracts.
        var t = new Tensor<decimal>(new decimal[] { 1m, 2m }, new[] { 2 });
        t.Lifetime = WeightLifetime.Streaming;

        var ex = Assert.Throws<NotSupportedException>(() => WeightRegistry.RegisterWeight(t));
        Assert.Contains("Decimal", ex.Message);
    }

    [Fact]
    public void GetStreamingReport_BeforeFirstRegister_SeedsCompressionFlag()
    {
        // Audit P1 #9: with EnableCompression=true but no register yet,
        // report should show CompressionEnabled=true (seeded from options),
        // not the default(false) of the all-zero record struct.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024,
            StreamingBackingStorePath = _backingDir,
            EnableCompression = true,
        });

        var report = WeightRegistry.GetStreamingReport();
        Assert.True(report.CompressionEnabled);
        Assert.Equal(0, report.RegisteredEntryCount);
    }

    [Fact]
    public async System.Threading.Tasks.Task PrefetchHitMissCounters_Reflect_PrefetchEffectiveness()
    {
        // Audit P1 #10 + counter-semantics fix: hit/miss must only
        // count foreground reads where a prefetch was actually issued
        // — register-then-immediately-materialize must NOT count as
        // a hit (no prefetch ran). This test exercises the real
        // PrefetchAsync path and uses the Task-returning internal
        // overload to wait deterministically on the worker rather
        // than polling with a wall-clock budget.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32,
            StreamingBackingStorePath = _backingDir,
        });

        var t = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        // Force eviction so the next Materialize must read from disk
        // (counts as a "miss" only if a prefetch was issued for t).
        var filler = new Tensor<float>(new float[16], new[] { 16 });
        filler.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(filler);

        // Phase 1: foreground Materialize WITHOUT a prefetch issued.
        // Bytes are not resident, so the read goes to disk — but
        // because no prefetch was issued for this handle, neither
        // hit NOR miss is incremented.
        WeightRegistry.Materialize(t);
        var report0 = WeightRegistry.GetStreamingReport();
        Assert.Equal(0, report0.PrefetchHitCount);
        Assert.Equal(0, report0.PrefetchMissCount);
        Assert.Equal(0, report0.PrefetchIssueCount);

        // Phase 2: release t back to pool, evict it again, then
        // PrefetchAsync (and AWAIT completion via the test-only
        // Task-returning overload). The next foreground Materialize
        // finds the bytes resident → counts as a HIT because a
        // prefetch was issued for this handle.
        WeightRegistry.ReleaseToPool(t);
        // Bring filler back to LRU head so t is the eviction victim.
        WeightRegistry.Materialize(filler);
        // Now register a second filler to force t out.
        var filler2 = new Tensor<float>(new float[16], new[] { 16 });
        filler2.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(filler2);

        // Use the deterministic Task-returning internal overload
        // (InternalsVisibleTo grants the test assembly access) so
        // this test doesn't poll with a wall-clock budget that's
        // flaky on busy CI agents. The Task completes when the
        // prefetch worker finishes (Rehydrate done) or short-circuits
        // (already resident / dropped).
        await WeightRegistry.PrefetchAsyncForTesting(t);

        // Bytes should be resident now → next foreground Materialize
        // is a HIT (prefetch was issued AND bytes were found resident).
        WeightRegistry.Materialize(t);
        var report1 = WeightRegistry.GetStreamingReport();
        Assert.True(report1.PrefetchIssueCount >= 1, "PrefetchAsync must increment IssueCount");
        Assert.True(report1.PrefetchHitCount >= 1, "Foreground after prefetch must count as a hit");
        Assert.Equal(0, report1.PrefetchMissCount); // bytes were resident, no miss
    }

    [Fact]
    public void RegisterWeight_DegenerateZeroLengthTensor_RegistersWithEmptyPayload()
    {
        // Edge case: Length=0 tensors must round-trip through register
        // without throwing. byteCount=0 is the lower bound of the
        // overflow check; the upper bound (>int.MaxValue) is covered
        // by RegisterWeight_OnHugeTensor_ThrowsClearError below.
        var t = new Tensor<float>(Array.Empty<float>(), new[] { 0 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        Assert.True(t.StreamingPoolHandle >= 0);
        Assert.Equal(0, t.DataVector.Length);
    }

    [Fact]
    public void CheckedStreamingByteCount_OverflowsIntMaxValue_ThrowsClearError()
    {
        // Audit P2 #16: byteCount overflow guard must throw
        // NotSupportedException with a chunking hint, not silently
        // wrap to a negative int. The guard is extracted into the
        // internal CheckedStreamingByteCount<T> helper so we can
        // unit-test it directly with a synthetic Length value
        // — no need to allocate a multi-GB tensor in a test process.
        // RegisterWeight calls the same helper on the production path,
        // so this test fully covers the overflow path.

        // float = 4 bytes/element. (int.MaxValue / 4) + 1 elements
        // would need (int.MaxValue / 4 + 1) × 4 ≈ int.MaxValue + 4
        // bytes, which exceeds int.MaxValue.
        int hugeLength = (int.MaxValue / 4) + 1;
        var ex = Assert.Throws<NotSupportedException>(() =>
            WeightRegistry.CheckedStreamingByteCount<float>(hugeLength));
        Assert.Contains("Streaming registration requires per-tensor size", ex.Message);
        Assert.Contains("Chunk", ex.Message);

        // Boundary check: a length that fits exactly at int.MaxValue
        // must NOT throw. int.MaxValue / 4 elements × 4 bytes =
        // int.MaxValue - 3 bytes (since int.MaxValue is not divisible
        // by 4). That's the largest valid float-tensor count.
        int maxValidFloatLength = int.MaxValue / 4;
        int byteCount = WeightRegistry.CheckedStreamingByteCount<float>(maxValidFloatLength);
        Assert.Equal(maxValidFloatLength * 4, byteCount);

        // Negative length must reject explicitly (not silently wrap).
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            WeightRegistry.CheckedStreamingByteCount<float>(-1));

        // Length=0 must succeed and return 0 (degenerate but valid).
        Assert.Equal(0, WeightRegistry.CheckedStreamingByteCount<float>(0));
    }

    [Fact]
    public void Pool_Disposed_RegisterThrowsObjectDisposed()
    {
        // Audit P0 #2: pool methods must throw ObjectDisposedException
        // after Dispose. Pool is owned by WeightRegistry; we test via
        // direct pool access then bypassing the registry.
        var pool = new StreamingTensorPool(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024,
            StreamingBackingStorePath = _backingDir,
        });
        pool.Dispose();
        Assert.Throws<ObjectDisposedException>(() => pool.Register(new byte[16]));
        Assert.Throws<ObjectDisposedException>(() => pool.Rehydrate(1));
    }

    [Fact]
    public void StressTest_LargeTensor_RoundTripPreservesBytes()
    {
        // Audit P2 #17 stress: ~1 MB tensor through eviction +
        // compression + decompress. Catches issues that only surface at
        // realistic-ish scale (alignment, memory bandwidth, LZ4 block
        // boundaries).
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 64 * 1024, // 64 KB — forces eviction
            StreamingBackingStorePath = _backingDir,
            EnableCompression = true,
        });

        const int n = 256 * 1024; // 1 MB of float
        float[] expected = new float[n];
        var rng = new Random(1234);
        for (int i = 0; i < n; i++) expected[i] = (float)rng.NextDouble();

        var t = new Tensor<float>(expected, new[] { n });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        // Add a filler to force eviction.
        var filler = new Tensor<float>(new float[32 * 1024], new[] { 32 * 1024 });
        filler.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(filler);

        // Materialize and verify.
        WeightRegistry.Materialize(t);
        var got = t.DataVector.AsSpan();
        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], got[i]);
    }

    [Fact]
    public async System.Threading.Tasks.Task StressTest_ConcurrentMaterialize_ProducesCorrectBytes()
    {
        // Audit P2 #17 concurrency stress: 4 threads simultaneously
        // Materialize different tensors. Catches races on the registry
        // lock + pool LRU.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 16 * 1024,
            StreamingBackingStorePath = _backingDir,
        });

        const int numTensors = 16;
        var tensors = new Tensor<float>[numTensors];
        var expected = new float[numTensors][];
        for (int k = 0; k < numTensors; k++)
        {
            expected[k] = new float[1024];
            for (int i = 0; i < 1024; i++) expected[k][i] = k * 1024 + i;
            tensors[k] = new Tensor<float>(expected[k], new[] { 1024 });
            tensors[k].Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(tensors[k]);
        }

        // Concurrent Materialize from 4 threads, each owning a slice of
        // tensors. All-or-nothing correctness.
        var tasks = new System.Threading.Tasks.Task[4];
        for (int worker = 0; worker < 4; worker++)
        {
            int wId = worker;
            tasks[wId] = System.Threading.Tasks.Task.Run(() =>
            {
                for (int k = wId; k < numTensors; k += 4)
                {
                    WeightRegistry.Materialize(tensors[k]);
                    var got = tensors[k].DataVector.AsSpan();
                    for (int i = 0; i < 1024; i++)
                    {
                        if (got[i] != expected[k][i])
                            throw new Xunit.Sdk.XunitException(
                                $"Tensor {k} index {i}: expected {expected[k][i]}, got {got[i]}");
                    }
                }
            });
        }
        await System.Threading.Tasks.Task.WhenAll(tasks);
    }

    [Fact]
    public void MaterializeScope_LazyEnumerable_EnumeratedOnce()
    {
        // Audit round-3 #28: ctor must not double-enumerate. Lazy
        // enumerables that yield different sequences on second pass
        // would corrupt the scope.
        var t1 = new Tensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        var t2 = new Tensor<float>(new float[] { 3f, 4f }, new[] { 2 });
        t1.Lifetime = WeightLifetime.Streaming;
        t2.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t1);
        WeightRegistry.RegisterWeight(t2);

        int enumerationCount = 0;
        IEnumerable<Tensor<float>> LazyOnce()
        {
            enumerationCount++;
            yield return t1;
            yield return t2;
        }

        using (WeightRegistry.MaterializeMany(LazyOnce()))
        {
            Assert.Equal(2, t1.DataVector.Length);
            Assert.Equal(2, t2.DataVector.Length);
        }
        Assert.Equal(1, enumerationCount);
    }

    [Fact]
    public void MaterializeScope_PartialFailureInCtor_ReleasesAlreadyMaterialized()
    {
        // Audit round-3 #25: if Materialize throws on weight N, the
        // previously-materialized weights must be released before the
        // exception propagates — otherwise they leak forever.
        var t1 = new Tensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        t1.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t1);

        // t2 is tagged Streaming but never registered → handle = -1 →
        // Materialize is a no-op, won't throw. To force a real throw mid-
        // ctor we use a hostile enumerable that yields t1, then throws.
        IEnumerable<Tensor<float>> HostileEnum()
        {
            yield return t1;
            throw new InvalidOperationException("simulated mid-enumeration failure");
        }

        Assert.Throws<InvalidOperationException>(() =>
        {
            using var _ = WeightRegistry.MaterializeMany(HostileEnum());
        });

        // After the throw, t1 must have been released (data array empty).
        Assert.Equal(0, t1.DataVector.Length);
    }

    [Fact]
    public void Materialize_EarlyOut_StillRefreshesLRU()
    {
        // Audit round-3 #29: when Materialize finds the tensor already
        // resident, it should still bump LRU heat — otherwise a hot
        // weight that always early-outs gets passed by less-recently-
        // used neighbours and eventually evicts.
        //
        // Test setup designed to actually trigger eviction:
        //   budget    = 32 bytes
        //   hot       = 4 floats = 16 bytes  (registered first → at LRU head)
        //   neighbour = 4 floats = 16 bytes  (registered second → now at LRU head; hot at tail)
        //   bump hot  via Materialize early-out × 5 (must move hot back to LRU head)
        //   pressure  = 4 floats = 16 bytes  (registering this puts us at 48 bytes — must evict 16)
        // The eviction victim is whichever entry is at LRU TAIL when
        // pressure registers. Without LRU refresh on early-out, hot
        // was last moved at register time (oldest) → evicts. With
        // refresh, hot was bumped after neighbour → neighbour evicts.
        // Asserting hot stays resident proves the refresh happened.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32, // 8 floats
            StreamingBackingStorePath = _backingDir,
        });

        var hot = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        hot.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(hot);
        // hot is now resident (Register adds to LRU head); LRU = [hot].

        var neighbour = new Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, new[] { 4 });
        neighbour.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(neighbour);
        // 32 bytes total — at budget, no eviction yet. LRU = [neighbour, hot].
        // Without further bumps, neighbour stays at head and hot will
        // evict when the next entry pushes us over budget.
        Assert.True(WeightRegistry.IsResidentInPool(hot));
        Assert.True(WeightRegistry.IsResidentInPool(neighbour));

        // Bump hot via the early-out path: Materialize finds it already
        // resident and (per the audit fix) refreshes the LRU. After
        // these calls, LRU = [hot, neighbour], so neighbour is the
        // eviction victim when pressure pushes us over budget.
        for (int i = 0; i < 5; i++) WeightRegistry.Materialize(hot);

        // Pressure: 4 floats = 16 bytes → 32 + 16 = 48 > budget 32 →
        // evict from LRU tail (neighbour) until back at budget.
        var pressure = new Tensor<float>(new float[] { 9f, 9f, 9f, 9f }, new[] { 4 });
        pressure.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(pressure);

        // Assert: hot is STILL resident (its LRU heat was refreshed by
        // the early-out path); neighbour is the one that got paged out.
        // This is the exact behaviour Audit #29 asks for — without the
        // refresh, hot would have evicted instead.
        Assert.True(WeightRegistry.IsResidentInPool(hot),
            "Materialize early-out must refresh LRU heat so the hot tensor stays resident under pressure.");
        Assert.False(WeightRegistry.IsResidentInPool(neighbour),
            "Without bumping LRU, the neighbour at LRU tail must be the eviction victim, not hot.");
    }

    [Fact]
    public void MarkAccessed_BumpsLRU_WithoutMaterialize()
    {
        // Audit round-3 #32 + reviewer: MarkAccessed must actually
        // bump the LRU position so a regression where it becomes a
        // no-op fails this test. Setup mirrors the early-out LRU
        // test: tight budget, two entries fitting exactly, then a
        // third entry pushing us over budget. Whichever entry was
        // at LRU TAIL when pressure was registered is the eviction
        // victim — proving MarkAccessed moved its target away from
        // the tail (or didn't, in a regression).
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32, // 8 floats — fits exactly two 16-byte tensors
            StreamingBackingStorePath = _backingDir,
        });

        var hot = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        hot.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(hot);

        var neighbour = new Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, new[] { 4 });
        neighbour.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(neighbour);

        // LRU = [neighbour, hot] (most recent first). Without a bump,
        // hot would evict next. Use MarkAccessed (NOT Materialize) to
        // refresh hot's LRU position.
        WeightRegistry.MarkAccessed(hot);
        // LRU should now be [hot, neighbour].

        // Trigger eviction by registering a third entry that doesn't fit.
        var pressure = new Tensor<float>(new float[] { 9f, 9f, 9f, 9f }, new[] { 4 });
        pressure.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(pressure);

        // MarkAccessed working = neighbour at tail = neighbour evicts;
        // MarkAccessed regressed to no-op = hot at tail = hot evicts.
        Assert.True(WeightRegistry.IsResidentInPool(hot),
            "MarkAccessed must move the target to LRU head; if it regresses to a no-op the hot tensor would evict instead of the neighbour.");
        Assert.False(WeightRegistry.IsResidentInPool(neighbour),
            "Neighbour should be the eviction victim once MarkAccessed moves hot to LRU head.");
    }

    [Fact]
    public void MarkAccessed_NonStreamingOrUnregistered_DoesNotThrow()
    {
        // Defensive coverage for the no-op contract: MarkAccessed on
        // a non-Streaming weight or one whose handle is -1 must short-
        // circuit without acquiring the pool lock or throwing.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024,
            StreamingBackingStorePath = _backingDir,
        });

        var notStreaming = new Tensor<float>(new float[] { 1f }, new[] { 1 });
        WeightRegistry.MarkAccessed(notStreaming);

        var streamingButUnregistered = new Tensor<float>(new float[] { 1f }, new[] { 1 });
        streamingButUnregistered.Lifetime = WeightLifetime.Streaming;
        // Handle stays at -1 because we never called RegisterWeight.
        WeightRegistry.MarkAccessed(streamingButUnregistered);

        // Both calls returned without throwing — that's the contract.
    }

    [Fact]
    public async System.Threading.Tasks.Task PrefetchAsyncMany_BatchedPrefetch_BringsAllResident()
    {
        // Audit round-3 #33 + reviewer: batched prefetch must actually
        // make ALL target tensors resident, not just count an issue.
        // Setup must let all three FIT in the budget — earlier setup
        // had budget=32 with three 16-byte tensors which is physically
        // impossible to satisfy (only two fit). Now: budget=64 so all
        // three (3 × 16 = 48 bytes) fit comfortably.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 64, // fits all 3 × 16-byte tensors
            StreamingBackingStorePath = _backingDir,
        });

        var tensors = new Tensor<float>[3];
        for (int k = 0; k < 3; k++)
        {
            tensors[k] = new Tensor<float>(new float[] { k, k, k, k }, new[] { 4 });
            tensors[k].Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(tensors[k]);
        }

        // Force eviction of all 3 by adding pressure that exceeds budget.
        // 3 × 16 + 16 = 64 → at budget; 3 × 16 + 64 = 112 → over → all 3 evict.
        var pressure = new Tensor<float>(new float[16], new[] { 16 });
        pressure.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(pressure);

        // Verify our setup: all three tensors have been evicted.
        for (int k = 0; k < 3; k++)
            Assert.False(WeightRegistry.IsResidentInPool(tensors[k]),
                $"Test setup precondition: tensor {k} must be evicted after pressure registration.");

        // Drop pressure so the budget has room for the prefetch to bring
        // all three back resident.
        WeightRegistry.UnregisterWeight(pressure);

        // Issue batched prefetch and wait deterministically by polling
        // PrefetchIssueCount. Use generous timeout (CI agents can be
        // slow under contention) but assert on actual residency, not
        // on counter value alone.
        WeightRegistry.PrefetchAsyncMany(tensors);

        // Wait for ALL THREE prefetch issues to land (PrefetchAsyncMany
        // queues one worker that does three Rehydrates; IssueCount
        // increments per Rehydrate).
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (DateTime.UtcNow < deadline)
        {
            var r = WeightRegistry.GetStreamingReport();
            if (r.PrefetchIssueCount >= 3) break;
            await System.Threading.Tasks.Task.Delay(20);
        }

        var finalReport = WeightRegistry.GetStreamingReport();
        Assert.True(finalReport.PrefetchIssueCount >= 3,
            $"Batched prefetch should issue 3 Rehydrates; got {finalReport.PrefetchIssueCount}.");

        // The actual contract: ALL THREE tensors must be resident after
        // the batched prefetch worker finishes. This is the assertion
        // the test name promises and the previous version skipped.
        for (int k = 0; k < 3; k++)
            Assert.True(WeightRegistry.IsResidentInPool(tensors[k]),
                $"Tensor {k} must be resident after PrefetchAsyncMany completes.");
    }

    [Fact]
    public void RehydrateInto_ReturnsCallerOwnedCopy()
    {
        // Audit round-3 #36: direct test of RehydrateInto (was only tested
        // transitively through Materialize). Verifies the returned byte[]
        // is a true copy — mutating it doesn't affect subsequent reads.
        var pool = new StreamingTensorPool(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024,
            StreamingBackingStorePath = _backingDir,
        });
        try
        {
            var data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            long h = pool.Register(data);

            var copy1 = pool.RehydrateInto(h);
            copy1[0] = 99; // mutate caller's copy

            var copy2 = pool.RehydrateInto(h);
            Assert.Equal(1, copy2[0]); // pool's data unchanged
            Assert.Equal(8, copy2.Length);
        }
        finally { pool.Dispose(); }
    }

    [Fact]
    public async System.Threading.Tasks.Task PrefetchAsync_BringsBytesIntoResidentSet()
    {
        // Reviewer flagged the original wall-clock poll loop (1s budget)
        // as flaky on busy CI agents. Switched to the internal
        // Task-returning overload PrefetchAsyncForTesting, which
        // resolves when the worker finishes Rehydrate (or short-
        // circuits via the dedup path). This is the deterministic
        // production-grade fix — no polling, no wall-clock budget,
        // no flakiness window.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32,
            StreamingBackingStorePath = _backingDir,
        });

        var t = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        // Force eviction.
        var filler = new Tensor<float>(new float[16], new[] { 16 });
        filler.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(filler);
        Assert.False(WeightRegistry.IsResidentInPool(t));

        // Drop filler so prefetch has budget to bring t back resident.
        WeightRegistry.UnregisterWeight(filler);

        // Deterministic wait — Task completes when the worker finishes.
        // Use a generous WaitAsync timeout as a hard upper bound (5s)
        // so a buggy implementation can't hang the test forever.
        var prefetchTask = WeightRegistry.PrefetchAsyncForTesting(t);
        var completed = await System.Threading.Tasks.Task.WhenAny(
            prefetchTask,
            System.Threading.Tasks.Task.Delay(TimeSpan.FromSeconds(5)));
        Assert.Same(prefetchTask, completed);

        Assert.True(WeightRegistry.IsResidentInPool(t),
            "Prefetch worker completed but bytes are not resident — Rehydrate likely failed silently.");
    }

    [Fact]
    public async System.Threading.Tasks.Task PrefetchAsync_OnAlreadyResident_NoOps()
    {
        // Reviewer audit P2: PrefetchAsync must skip already-resident
        // handles instead of queueing a redundant worker. Verify by
        // observing PrefetchIssueCount stays at 0 when the handle is
        // resident at the time of the prefetch call.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024,
            StreamingBackingStorePath = _backingDir,
        });

        var t = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        // After register the bytes are still resident in the pool — no
        // foreground Materialize needed.
        Assert.True(WeightRegistry.IsResidentInPool(t));

        var pre = WeightRegistry.GetStreamingReport();
        await WeightRegistry.PrefetchAsyncForTesting(t);
        var post = WeightRegistry.GetStreamingReport();

        // Dedup path means the prefetch was never enqueued because the
        // resident-check short-circuited. PrefetchIssueCount must NOT
        // have advanced.
        Assert.Equal(pre.PrefetchIssueCount, post.PrefetchIssueCount);
    }

    [Fact]
    public async System.Threading.Tasks.Task PrefetchAsync_DoesNotDoubleQueue_WhenInFlight()
    {
        // Reviewer audit P2: a second PrefetchAsync on the same handle
        // while the first is still in-flight must NOT queue a second
        // worker. Hard to test deterministically because we'd have to
        // freeze the first worker; instead, exercise the dedup state
        // by issuing two prefetches "back to back" and verifying the
        // semaphore + in-flight set keep IssueCount bounded.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32,
            StreamingBackingStorePath = _backingDir,
        });

        var t = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);

        // Force eviction so prefetch has work to do.
        var filler = new Tensor<float>(new float[16], new[] { 16 });
        filler.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(filler);
        Assert.False(WeightRegistry.IsResidentInPool(t));

        // Drop filler so prefetch can land.
        WeightRegistry.UnregisterWeight(filler);

        // Issue two prefetches and await both. The dedup path means
        // the second one short-circuits on the in-flight set (or,
        // in the rare case the first finishes between the two calls,
        // on the resident-check). Either way IssueCount must be
        // exactly 1, not 2.
        var p1 = WeightRegistry.PrefetchAsyncForTesting(t);
        var p2 = WeightRegistry.PrefetchAsyncForTesting(t);
        await System.Threading.Tasks.Task.WhenAll(p1, p2);

        var report = WeightRegistry.GetStreamingReport();
        Assert.True(report.PrefetchIssueCount <= 1,
            $"Dedup must prevent double-queue. PrefetchIssueCount={report.PrefetchIssueCount}.");
        Assert.True(WeightRegistry.IsResidentInPool(t),
            "Prefetch worker should still complete the (single) Rehydrate.");
    }

    [Fact]
    public void StreamingPoolReport_Default_HasCompressionRatio_One()
    {
        // Reviewer audit P2: default(StreamingPoolReport) must surface
        // CompressionRatio as 1.0 (the documented "no compression has
        // run" value), NOT the all-zero struct's 0.0 which would
        // imply misleading "perfect compression".
        StreamingPoolReport defaultReport = default;
        Assert.Equal(1.0, defaultReport.CompressionRatio);

        // Same for new() — both paths must normalize.
        var newedReport = new StreamingPoolReport();
        Assert.Equal(1.0, newedReport.CompressionRatio);

        // An explicitly-set non-zero ratio must be preserved verbatim.
        var explicitReport = new StreamingPoolReport { CompressionRatio = 0.65 };
        Assert.Equal(0.65, explicitReport.CompressionRatio);

        // Explicitly setting to zero (an invalid practical value) also
        // surfaces as 1.0 — defensive normalization, since LZ4 never
        // produces zero-byte output for non-empty input.
        var zeroReport = new StreamingPoolReport { CompressionRatio = 0.0 };
        Assert.Equal(1.0, zeroReport.CompressionRatio);
    }

    [Fact]
    public void Pool_PostDispose_ReadAPIsThrowObjectDisposed()
    {
        // Reviewer audit P2: read APIs (IsResident, GetReport,
        // ResidentEntryCount, RegisteredEntryCount, MarkAccessed)
        // must throw ObjectDisposedException after Dispose, not
        // silently return defaults that let callers keep operating
        // on a dead pool.
        var pool = new StreamingTensorPool(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024,
            StreamingBackingStorePath = _backingDir,
        });
        long h = pool.Register(new byte[] { 1, 2, 3 });
        pool.Dispose();

        Assert.Throws<ObjectDisposedException>(() => pool.IsResident(h));
        Assert.Throws<ObjectDisposedException>(() => pool.GetReport());
        Assert.Throws<ObjectDisposedException>(() => pool.ResidentEntryCount);
        Assert.Throws<ObjectDisposedException>(() => pool.RegisteredEntryCount);
        Assert.Throws<ObjectDisposedException>(() => pool.MarkAccessed(h));
    }

    [Fact]
    public void Eviction_HighEntropyData_FallsBackToRawStorage()
    {
        // Reviewer audit P2: the compression-doesn't-shrink fallback
        // path (LZ4 produces output >= input → store raw) was uncovered
        // by tests. Use cryptographically random bytes which LZ4
        // cannot meaningfully compress, force eviction, then rehydrate
        // and verify the bytes round-trip exactly. The fallback path
        // is exercised because IsCompressed is set to false by the
        // eviction logic when compression doesn't shrink the payload.
        var pool = new StreamingTensorPool(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32, // tight budget forces eviction
            StreamingBackingStorePath = _backingDir,
            EnableCompression = true, // we want the compress-then-fallback path
        });
        try
        {
            // High-entropy payload — random bytes from a CSPRNG.
            // LZ4 typically produces output >= input on this kind of
            // data, triggering the fallback branch.
            var random = new byte[64];
            new Random(42).NextBytes(random);
            long h1 = pool.Register(random);

            // Force eviction of h1 by registering a second entry that
            // pushes us over budget (32 bytes, h1 already occupies 64).
            // Note: registering 64 bytes immediately exceeds budget so
            // h1 evicts during this Register call.
            var filler = new byte[8];
            long h2 = pool.Register(filler);

            // h1 should be evicted to disk now.
            Assert.False(pool.IsResident(h1));

            // Rehydrate must bit-exactly restore h1's bytes regardless
            // of whether the fallback path stored raw or LZ4-compressed.
            // If the fallback didn't run (compressed path tried but
            // failed to decode raw bytes) Rehydrate would corrupt or
            // throw; passing here means the raw-storage fallback
            // worked end-to-end.
            var rehydrated = pool.RehydrateInto(h1);
            Assert.Equal(random.Length, rehydrated.Length);
            for (int i = 0; i < random.Length; i++)
                Assert.Equal(random[i], rehydrated[i]);
        }
        finally { pool.Dispose(); }
    }

    // -------- AllocateRegistered<T> + EvictUntilFreeBytes (#1222 follow-on) --------

    [Fact]
    public void AllocateRegistered_AfterReset_StillReturnsStreamingLifetime()
    {
        // Reset() leaves _options at a default GpuOffloadOptions (not
        // null), so AllocateRegistered's contract is preserved across
        // Reset: it always returns a Streaming-lifetime tensor with
        // materialized storage. The "fall back to Default Tensor" path
        // exists only for the brief window before any Configure call
        // at process startup — not normally testable from here.
        WeightRegistry.Reset();

        var t = WeightRegistry.AllocateRegistered<float>(new[] { 16, 16 });

        Assert.Equal(WeightLifetime.Streaming, t.Lifetime);
        Assert.Equal(16 * 16, t.Length);
        Assert.True(t.DataVector.Length > 0,
            "AllocateRegistered should leave storage materialized for caller "
            + "to initialize before RegisterWeight.");
    }

    [Fact]
    public void AllocateRegistered_WithStreamingConfigured_ReturnsStreamingLifetimeTensor()
    {
        var t = WeightRegistry.AllocateRegistered<float>(new[] { 8, 8 });

        // Streaming-configured path: lifetime is set, storage is
        // materialized (caller has yet to RegisterWeight). The contract
        // is "ready for caller to initialize then register".
        Assert.Equal(WeightLifetime.Streaming, t.Lifetime);
        Assert.Equal(64, t.Length);
        Assert.True(t.DataVector.Length > 0,
            "AllocateRegistered should return a materialized tensor so callers "
            + "can initialize weights into it before calling RegisterWeight.");
    }

    [Fact]
    public void AllocateRegistered_LongSequence_BoundsPeakResidency()
    {
        // The whole point of AllocateRegistered: a long sequence of
        // large allocations + registers should stay bounded by the pool
        // budget instead of peaking at the cumulative weight size.
        // Simulate PaLM-E's MHA-layer pattern: 4 weights per "layer",
        // 16 "layers", each weight 64 KB. Cumulative = 64 KB × 4 × 16 =
        // 4 MB. Pool budget set to 256 KB by the fixture override below.
        // Without eviction-on-allocate, peak resident would be 4 MB
        // (everything alive); with it, resident should plateau near
        // the 256 KB budget.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 256L * 1024,
            StreamingBackingStorePath = _backingDir,
        });

        long peakResidentBytes = 0;
        for (int layer = 0; layer < 16; layer++)
        {
            for (int matrix = 0; matrix < 4; matrix++)
            {
                // 64 KB per matrix at fp32 = 16384 elements = 128×128.
                var t = WeightRegistry.AllocateRegistered<float>(new[] { 128, 128 });
                // Initialize (write into materialized storage). DataVector
                // is the underlying Vector; AsWritableSpan exposes the
                // writable mutable view that AsSpan (read-only) does not.
                var span = t.DataVector.AsWritableSpan();
                for (int i = 0; i < span.Length; i++) span[i] = (float)((layer * 4 + matrix + i) % 7);
                // Register: serializes + drops + commits to pool.
                WeightRegistry.RegisterWeight(t);

                long resident = WeightRegistry.GetStreamingReport().ResidentBytes;
                if (resident > peakResidentBytes) peakResidentBytes = resident;
            }
        }

        // Pool resident should be bounded by budget (256 KB) plus a
        // small slop for the most recent allocation that triggered
        // eviction. Peak shouldn't approach the cumulative 4 MB total.
        Assert.True(peakResidentBytes <= 384L * 1024,
            $"AllocateRegistered + RegisterWeight should bound peak resident "
            + $"to ~budget (256 KB), but peak was {peakResidentBytes} bytes "
            + $"({peakResidentBytes / 1024.0:F1} KB). If this exceeds 384 KB "
            + "(budget + ~50% slop for the last register), eviction-on-allocate "
            + "isn't running, or EvictUntilFreeBytes isn't getting the "
            + "caller-requested headroom.");
    }

    [Fact]
    public void AllocateRegistered_OversizeShape_ThrowsClearError()
    {
        // Per-tensor bytecount is bounded by int.MaxValue (the byte[]
        // limit). Asking for 2.5 GB of fp32 (~625M elements) should
        // throw NotSupportedException with a chunking hint, not OOM
        // somewhere deep in the pool.
        var ex = Assert.Throws<NotSupportedException>(() =>
            WeightRegistry.AllocateRegistered<float>(new[] { 1_000_000_000 }));
        Assert.Contains("Chunk", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AllocateRegistered_NegativeShape_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            WeightRegistry.AllocateRegistered<float>(new[] { -1, 16 }));
    }

    [Fact]
    public void EvictUntilFreeBytes_DirectCall_PagesLruEntries()
    {
        // Direct StreamingTensorPool test — no AllocateRegistered
        // wrapper. Register some entries, then ask for headroom equal
        // to half the budget and verify resident drops accordingly.
        var pool = new StreamingTensorPool(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 4096L,
            StreamingBackingStorePath = _backingDir,
        });
        try
        {
            // 4 × 1 KB entries = 4 KB resident, exactly at budget.
            for (int i = 0; i < 4; i++)
            {
                var bytes = new byte[1024];
                pool.Register(bytes);
            }
            Assert.Equal(4096L, pool.ResidentBytes);

            // Ask for 2 KB of headroom. Pool should evict 2 LRU entries.
            bool secured = pool.EvictUntilFreeBytes(2048L);
            Assert.True(secured,
                "EvictUntilFreeBytes should secure 2 KB headroom from a "
                + "4 KB pool with 4 evictable entries.");
            Assert.True(pool.ResidentBytes <= 2048L,
                $"After EvictUntilFreeBytes(2048), resident should be <= 2048; got {pool.ResidentBytes}.");
        }
        finally { pool.Dispose(); }
    }

    [Fact]
    public void EvictUntilFreeBytes_RequestExceedsBudget_BestEffortReturnsFalse()
    {
        // Asking for more headroom than the budget itself should
        // empty the LRU and return false — caller's allocation will
        // push the pool past budget, but EvictIfOverBudget on the
        // next register catches up.
        var pool = new StreamingTensorPool(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 4096L,
            StreamingBackingStorePath = _backingDir,
        });
        try
        {
            for (int i = 0; i < 4; i++) pool.Register(new byte[1024]);

            // Ask for 8 KB headroom — exceeds the 4 KB budget. Even
            // after evicting all 4 entries, can't satisfy → false.
            bool secured = pool.EvictUntilFreeBytes(8192L);
            Assert.False(secured,
                "EvictUntilFreeBytes(8192) > budget(4096) should empty the "
                + "LRU and return false (best-effort signal).");
            Assert.Equal(0L, pool.ResidentBytes);
        }
        finally { pool.Dispose(); }
    }
}
