// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// #2322: weight streaming rebuilds every weight from its stored bytes on each pass, and a large model allocated
/// its whole weight set again per pass (~6 GB of a 661M-parameter training step), and copied each write-back
/// through an extra array (~2.6 GB). Restore arrays are now recycled through a bounded pool and write-backs refill
/// the pool entry's own buffer. These tests pin the safety contract as well as the saving: a recycled array only
/// ever shows its new weight, a shed array is not reused while a gradient tape is open, caches keyed by the array
/// see the refill as a mutation, the cap bounds what is held, and an in-place write-back reads back exactly.
/// </summary>
[Collection("WeightRegistry")]
public class StreamingRestoreBufferRecyclingTests
{
    private const int L = 4096;                 // 16 KB per fp32 weight; every weight the same length
    private const int N = 16;                   // twice the resident cap, so owners are shed and re-materialized
    private const long Cap = 8L * L * sizeof(float);

    private static float Value(int weight, int element, int step) => ((weight * 31 + element * 7 + step * 13) % 509) * 0.25f - 60f;

    private static Tensor<float>[] RegisterWeights(int step)
    {
        var weights = new Tensor<float>[N];
        for (int i = 0; i < N; i++)
        {
            var t = new Tensor<float>(new[] { L });
            for (int j = 0; j < L; j++) t[j] = Value(i, j, step);
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);
            weights[i] = t;
        }
        return weights;
    }

    private static string Configure(long restorePoolBytes, StreamingStoreDtype dtype = StreamingStoreDtype.Lossless)
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-2322-" + Guid.NewGuid().ToString("N"));
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = Cap,
            TransparentAutoEviction = true,
            StreamingStoreDtype = dtype,
            StreamingRestoreBufferPoolBytes = restorePoolBytes,
        });
        return dir;
    }

    // The collection runs alone (DisableParallelization), so an open tape here was leaked by an earlier test; say so
    // rather than fail the pooling assertions that depend on there being none.
    private static void AssertNoTapeIsOpen()
        => Assert.False(AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps.AnyTapeActive(),
            "A gradient tape is still open from an earlier test, so shed arrays are parked rather than pooled.");

    private static void Cleanup(string dir)
    {
        WeightRegistry.Reset();
        try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>
    /// Several training-like steps, each inside a tape: materialize every weight, update it, let the owner budget
    /// shed the cold ones. From the second step on, restores are served from recycled arrays, and every weight still
    /// reads back exactly its own latest values, never a neighbour's.
    /// </summary>
    [Theory]
    [InlineData(StreamingStoreDtype.Lossless)]
    [InlineData(StreamingStoreDtype.FullPrecision)]
    public void RecycledArrays_CarryOnlyTheirOwnWeight_AcrossSteps(StreamingStoreDtype dtype)
    {
        string dir = Configure(restorePoolBytes: 4 * Cap, dtype);
        try
        {
            var weights = RegisterWeights(step: 0);
            long rentsBefore = StreamingRestoreBuffers<float>.RecycledRentCount;
            for (int step = 1; step <= 4; step++)
            {
                using (new GradientTape<float>())
                {
                    for (int i = 0; i < N; i++)
                    {
                        WeightRegistry.Materialize(weights[i]);
                        for (int j = 0; j < L; j++) weights[i][j] = Value(i, j, step);
                    }
                }

                for (int i = 0; i < N; i++)
                {
                    WeightRegistry.Materialize(weights[i]);
                    Assert.Equal(Value(i, 0, step), weights[i][0]);
                    Assert.Equal(Value(i, L / 2, step), weights[i][L / 2]);
                    Assert.Equal(Value(i, L - 1, step), weights[i][L - 1]);
                }
            }

            Assert.True(StreamingRestoreBuffers<float>.RecycledRentCount > rentsBefore,
                "No restore was served from a recycled array, so the allocation saving never engaged.");
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A closure recorded on an open tape may still read an array a shed weight released, so the array is parked,
    /// not pooled, until the outermost tape closes.
    /// </summary>
    [Fact]
    public void ShedArrays_WaitForTheOutermostTapeToClose()
    {
        string dir = Configure(restorePoolBytes: 4 * Cap);
        try
        {
            AssertNoTapeIsOpen();
            StreamingRestoreBuffers<float>.Clear();
            var weights = RegisterWeights(step: 0);
            using (new GradientTape<float>())
            {
                using (new GradientTape<float>())
                {
                    for (int i = 0; i < N; i++) WeightRegistry.Materialize(weights[i]);
                }
                Assert.True(StreamingRestoreBuffers<float>.ParkedBytes > 0, "Shed arrays were not parked while a tape was open.");
                Assert.Equal(0, StreamingRestoreBuffers<float>.PooledBytes);
            }

            Assert.Equal(0, StreamingRestoreBuffers<float>.ParkedBytes);
            Assert.True(StreamingRestoreBuffers<float>.PooledBytes > 0, "Closing the outermost tape did not release the parked arrays.");
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>Pooled and parked arrays together never exceed the configured cap; zero turns recycling off.</summary>
    [Theory]
    [InlineData(0L)]
    [InlineData(3L * L * sizeof(float))]
    public void TheCap_BoundsWhatRecyclingHolds(long cap)
    {
        string dir = Configure(restorePoolBytes: cap);
        try
        {
            StreamingRestoreBuffers<float>.Clear();
            var weights = RegisterWeights(step: 0);
            for (int round = 0; round < 3; round++)
                for (int i = 0; i < N; i++) WeightRegistry.Materialize(weights[i]);

            long held = StreamingRestoreBuffers<float>.PooledBytes + StreamingRestoreBuffers<float>.ParkedBytes;
            Assert.True(held <= cap, $"Recycling holds {held} bytes, above the {cap}-byte cap.");
            if (cap == 0) Assert.Equal(0, held);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A recycled array keeps its identity, so the derived-weight caches keyed by it (packed GEMM panels, int8 packs,
    /// transposed kernels) must see the refill as a mutation of that array.
    /// </summary>
    [Fact]
    public void ARecycledArray_InvalidatesTheCachesKeyedByIt()
    {
        string dir = Configure(restorePoolBytes: 4 * Cap);
        try
        {
            AssertNoTapeIsOpen();
            StreamingRestoreBuffers<float>.Clear();
            var weights = RegisterWeights(step: 0);
            for (int round = 0; round < 2; round++)
                for (int i = 0; i < N; i++) WeightRegistry.Materialize(weights[i]);

            // Take a pooled array and the version its caches were built against, then restore a weight into it.
            var pooled = StreamingRestoreBuffers<float>.Rent(L, out bool recycled);
            Assert.True(recycled, "Expected a pooled array after shedding owners outside any tape.");
            long versionBefore = SimdGemm.WeightArrayVersionOf(pooled);
            StreamingRestoreBuffers<float>.Park(pooled);

            // Outside any tape the parked array is pooled at once, and the pool is last-in first-out, so the next
            // restore of this length receives exactly it.
            var target = weights.First(w => w.DataVector.Length == 0);
            WeightRegistry.Materialize(target);
            var installed = target.GetReadOnlyDataArray();
            Assert.Same(pooled, installed);
            Assert.True(SimdGemm.WeightArrayVersionOf(installed) > versionBefore,
                "A recycled array was installed without bumping its cache version.");
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The GPU engine caches uploads by backing array and serves one while the tensor's GPU-cache version matches.
    /// A weight restored into a recycled array must therefore report a version that no upload of the array's earlier
    /// contents carries: a fresh storage restarting at its own version 0 matched such an upload.
    /// </summary>
    [Fact]
    public void ARecycledArray_DoesNotMatchAGpuUploadOfItsEarlierContents()
    {
        string dir = Configure(restorePoolBytes: 4 * Cap);
        try
        {
            AssertNoTapeIsOpen();
            StreamingRestoreBuffers<float>.Clear();
            var weights = RegisterWeights(step: 0);
            for (int round = 0; round < 2; round++)
                for (int i = 0; i < N; i++) WeightRegistry.Materialize(weights[i]);

            // An upload of the pooled array's current contents, recorded at the version the array had then. A tensor
            // over a caller array shares that array's epoch, as the restored weight's storage now does.
            var pooled = StreamingRestoreBuffers<float>.Rent(L, out bool recycled);
            Assert.True(recycled, "Expected a pooled array after shedding owners outside any tape.");
            int uploadedAt = new Tensor<float>(pooled, new[] { L }).GpuCacheVersion;
            StreamingRestoreBuffers<float>.Park(pooled);

            var target = weights.First(w => w.DataVector.Length == 0);
            WeightRegistry.Materialize(target);
            Assert.Same(pooled, target.GetReadOnlyDataArray());
            Assert.NotEqual(uploadedAt, target.GpuCacheVersion);
        }
        finally
        {
            Cleanup(dir);
        }
    }
    /// <summary>
    /// <c>Rehydrate</c> hands out a span over the pool's own buffer, documented to stay valid for the caller. A
    /// write-back must therefore install a new buffer instead of refilling a lent one: the span keeps the bytes it
    /// was given, and the entry reads back the new content. A prefetch only warms the entry, so it does not stop the
    /// in-place refill.
    /// </summary>
    [Fact]
    public void AWriteBack_LeavesALentSpanUnchanged_ButStillRefillsAfterPrefetch()
    {
        string dir = Path.Combine(Path.GetTempPath(), "aidotnet-2322-lent-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var pool = new StreamingTensorPool(new GpuOffloadOptions { StreamingBackingStorePath = dir, StreamingPoolMaxResidentBytes = 1 << 20 });
            byte[] original = Enumerable.Range(0, 1000).Select(i => (byte)(i % 251)).ToArray();
            long lent = pool.Register((byte[])original.Clone());
            long warmed = pool.Register((byte[])original.Clone());

            var span = pool.Rehydrate(lent, isPrefetch: false);
            pool.WarmForPrefetch(warmed);

            byte[] updated = Enumerable.Range(0, 950).Select(i => (byte)((i * 7) % 253)).ToArray();
            foreach (long handle in new[] { lent, warmed })
            {
                using var stream = new PooledChunkWriteStream();
                stream.Write(updated, 0, updated.Length);
                pool.ReplaceEntryData(handle, stream);
            }

            Assert.Equal(original, span.ToArray());
            Assert.Equal(updated, pool.Rehydrate(lent, isPrefetch: false).ToArray());
            // Only the prefetched entry was refilled in place.
            Assert.Equal(1, pool.WriteBackInPlaceCount);
        }
        finally
        {
            try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
    /// <summary>
    /// A write-back that fits the entry's resident buffer refills it in place; the entry then reads back exactly
    /// the new content, both resident and after paging to disk and back.
    /// </summary>
    [Fact]
    public void AnInPlaceWriteBack_ReadsBackExactly_ResidentAndAfterPaging()
    {
        string dir = Path.Combine(Path.GetTempPath(), "aidotnet-2322-pool-" + Guid.NewGuid().ToString("N"));
        try
        {
            // Room for one 1000-byte entry: registering a second pages the first out to the backing file.
            using var pool = new StreamingTensorPool(new GpuOffloadOptions { StreamingBackingStorePath = dir, StreamingPoolMaxResidentBytes = 1500 });
            long handle = pool.Register(Enumerable.Range(0, 1000).Select(i => (byte)(i % 251)).ToArray());

            byte[] smaller = Enumerable.Range(0, 950).Select(i => (byte)((i * 7) % 253)).ToArray();
            using (var stream = new PooledChunkWriteStream())
            {
                stream.Write(smaller, 0, smaller.Length);
                pool.ReplaceEntryData(handle, stream);
            }
            Assert.Equal(1, pool.WriteBackInPlaceCount);
            Assert.Equal(smaller, pool.Rehydrate(handle, isPrefetch: false).ToArray());

            // Page the refilled entry out (its slack beyond the content must not reach the file) and read it back.
            pool.Register(new byte[1000]);
            Assert.False(pool.IsResident(handle), "Registering a second entry should have paged the first out.");
            Assert.Equal(smaller, pool.Rehydrate(handle, isPrefetch: false).ToArray());
        }
        finally
        {
            try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
