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
/// </summary>
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
}
