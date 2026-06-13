// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Transparent weight-streaming tests (issue #430 follow-up): a streaming
/// weight whose bytes have been paged out auto-rehydrates the moment its data
/// is read through ANY public accessor — <see cref="Tensor{T}.AsSpan"/>, the
/// indexer, <see cref="TensorBase{T}.ToArray"/> — with no explicit
/// <c>WeightRegistry.Materialize</c> call. This is what makes streaming
/// "transparent": a model fits a bounded resident set without per-layer
/// orchestration code, because the read path itself pages weights in on demand
/// AND the pool's symmetric owner-drop frees cold weights' GC-heap copies so
/// the resident set stays bounded.
///
/// <para><c>[Collection("WeightRegistry")]</c> serializes against the other
/// streaming suites — all mutate the process-wide <see cref="WeightRegistry"/>
/// singleton.</para>
/// </summary>
[Collection("WeightRegistry")]
public class WeightStreamingTransparentAccessTests : IDisposable
{
    private readonly string _backingDir;

    public WeightStreamingTransparentAccessTests()
    {
        _backingDir = Path.Combine(Path.GetTempPath(), "aidotnet-wr-transparent-test-" + Guid.NewGuid().ToString("N"));
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 1024L * 1024 * 1024, // 1 GiB — generous headroom
            StreamingBackingStorePath = _backingDir,
            TransparentAutoEviction = true,
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
    public void AsSpan_OnDroppedStreamingWeight_AutoRehydrates()
    {
        float[] expected = { 1.5f, -2.25f, 3.125f, 4.0f };
        var t = new Tensor<float>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;

        // RegisterWeight drops the resident data after copying to the pool.
        WeightRegistry.RegisterWeight(t);
        Assert.Equal(0, t.DataVector.Length);

        // Reading via AsSpan must transparently page the bytes back in — no
        // explicit Materialize call.
        var span = t.AsSpan();
        Assert.Equal(expected.Length, span.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], span[i]);

        // Side effect: the tensor is now resident again.
        Assert.Equal(expected.Length, t.DataVector.Length);
    }

    [Fact]
    public void Memory_OnDroppedStreamingWeight_AutoRehydrates()
    {
        // .Memory / .Data are the buffer accessors engine + layer code use (e.g.
        // SelfAttentionLayer.Forward reaches the weight via .Memory). They must
        // auto-rehydrate a dropped streaming weight exactly like AsSpan — without
        // the gate, .Memory Slices empty storage and throws ArgumentOutOfRange.
        float[] expected = { 2.5f, 4.5f, 6.5f, 8.5f };
        var t = new Tensor<float>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        Assert.Equal(0, t.DataVector.Length);

        var span = t.Memory.Span; // must page the bytes back in transparently
        Assert.Equal(expected.Length, span.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], span[i]);
        Assert.Equal(expected.Length, t.DataVector.Length);
    }

    [Fact]
    public void Indexer_OnDroppedStreamingWeight_AutoRehydrates()
    {
        float[] expected = { 10f, 20f, 30f, 40f, 50f };
        var t = new Tensor<float>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        Assert.Equal(0, t.DataVector.Length);

        // A single-element read through the flat indexer must auto-rehydrate.
        Assert.Equal(30f, t[2]);
        Assert.Equal(50f, t[4]);
    }

    [Fact]
    public void ToArray_OnDroppedStreamingWeight_AutoRehydrates()
    {
        // ToArray() routes through EnsureMaterialized too, and GetDataArray's
        // live-backing fast path falls back to ToArray when storage is dropped
        // (storage.Length 0 != logical Length) — so this also covers the
        // GetDataArray engine-consumption path.
        double[] expected = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var t = new Tensor<double>(expected, new[] { expected.Length });
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        Assert.Equal(0, t.DataVector.Length);

        double[] got = t.ToArray();
        Assert.Equal(expected, got);
    }

    [Fact]
    public void AsSpan_AfterDiskEviction_AutoRehydratesFromBackingStore()
    {
        // Tight budget so the first weight is paged out to DISK by the second
        // weight's registration; transparent read must page it back from disk.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 16, // one 4-float tensor
            StreamingBackingStorePath = _backingDir,
            TransparentAutoEviction = true,
        });

        float[] first = { 7f, 8f, 9f, 10f };
        var a = new Tensor<float>(first, new[] { 4 });
        a.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(a);

        var b = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        b.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(b);

        // 'a' was evicted to backing store when 'b' registered.
        Assert.False(WeightRegistry.IsResidentInPool(a));

        // Transparent read pages it back from disk with exact bytes.
        var span = a.AsSpan();
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], span[i]);

        var report = WeightRegistry.GetStreamingReport();
        Assert.True(report.DiskReadCount >= 1, "Rehydrate should have read from the backing store.");
    }

    [Fact]
    public void TransparentForward_BoundsResidentSet_ByDroppingColdOwners()
    {
        // The core option-(b) guarantee. Simulate a forward pass that reads
        // weight after weight (a block loop) under a budget that fits only a
        // couple of weights. Every read auto-rehydrates; every rehydrate pages
        // the coldest weight out AND drops that weight's owning tensor's
        // resident _data. So after touching all N weights, only a bounded
        // handful are resident — NOT all N. Without the symmetric owner-drop,
        // every materialised weight would stay GC-resident and the resident
        // set would grow to all N (the OOM the feature exists to prevent).
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32, // two 4-float (16-byte) tensors
            StreamingBackingStorePath = _backingDir,
            TransparentAutoEviction = true,
        });

        const int n = 8;
        const int elems = 4;
        var weights = new Tensor<float>[n];
        var expected = new float[n][];
        for (int k = 0; k < n; k++)
        {
            expected[k] = new float[elems];
            for (int i = 0; i < elems; i++) expected[k][i] = k * 100 + i;
            weights[k] = new Tensor<float>(expected[k], new[] { elems });
            weights[k].Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(weights[k]);
            // Registration drops resident data immediately.
            Assert.Equal(0, weights[k].DataVector.Length);
        }

        // "Forward pass": read each weight in order via the transparent path.
        for (int k = 0; k < n; k++)
        {
            var span = weights[k].AsSpan(); // auto-rehydrate
            for (int i = 0; i < elems; i++)
                Assert.Equal(expected[k][i], span[i]);
        }

        // After the full sweep, count how many tensors still hold resident
        // _data. The bound: roughly the budget's worth of weights plus the
        // last-touched one — emphatically NOT all n. We assert a generous
        // upper bound (<= 4) that still proves cold owners were dropped: a
        // regression that leaves every materialised weight resident would
        // show all 8 resident here.
        int residentTensors = 0;
        for (int k = 0; k < n; k++)
            if (weights[k].DataVector.Length > 0) residentTensors++;

        Assert.True(residentTensors <= 4,
            $"Transparent streaming must bound the resident set by dropping cold owners; " +
            $"{residentTensors} of {n} tensors are still resident (expected a small handful).");
        Assert.True(residentTensors < n,
            "At least some cold weights must have been dropped — none were, so the owner-drop is not firing.");

        // The earliest-touched weights must have been dropped outright.
        Assert.Equal(0, weights[0].DataVector.Length);
    }

    [Fact]
    public void DroppedColdWeight_RemainsReadable_OnNextAccess()
    {
        // A weight dropped by the owner-drop sweep must still read correctly
        // when the "forward" touches it again — it simply re-pages from the
        // pool/disk. Proves the bound doesn't corrupt data.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32,
            StreamingBackingStorePath = _backingDir,
            TransparentAutoEviction = true,
        });

        float[] aData = { 1f, 2f, 3f, 4f };
        var a = new Tensor<float>(aData, new[] { 4 });
        a.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(a);

        var fillers = new Tensor<float>[4];
        for (int k = 0; k < fillers.Length; k++)
        {
            fillers[k] = new Tensor<float>(new float[] { k, k, k, k }, new[] { 4 });
            fillers[k].Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(fillers[k]);
        }

        // Touch 'a' (rehydrate), then touch all fillers so 'a' goes cold and
        // its owner gets dropped.
        _ = a.AsSpan();
        for (int k = 0; k < fillers.Length; k++) _ = fillers[k].AsSpan();

        // 'a' should have been dropped by now.
        Assert.Equal(0, a.DataVector.Length);

        // Re-access 'a' — must transparently re-page with exact bytes.
        var span = a.AsSpan();
        for (int i = 0; i < aData.Length; i++)
            Assert.Equal(aData[i], span[i]);
    }

    [Fact]
    public void NonStreamingWeight_ReadPath_Unaffected()
    {
        // A Default-lifetime tensor (every activation, every non-streamed
        // model) must read exactly as before — the gate short-circuits on the
        // Lifetime field with no pool interaction.
        float[] data = { 11f, 22f, 33f };
        var t = new Tensor<float>(data, new[] { 3 });
        // Lifetime stays Default.

        var span = t.AsSpan();
        Assert.Equal(3, span.Length);
        Assert.Equal(22f, span[1]);
        Assert.Equal(3, t.DataVector.Length); // never dropped
    }
}
