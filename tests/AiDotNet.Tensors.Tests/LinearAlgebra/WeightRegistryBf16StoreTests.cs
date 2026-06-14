// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// End-to-end bf16 streaming store: the registry quantizes float/double weights to
/// bf16 on register (halving/quartering the resident + disk bytes) and widens them
/// back to T on materialize, with the StreamingStoreDtype policy + execution-mode
/// hint driving the choice. Serialized against other registry tests (global state).
/// </summary>
[Collection("WeightRegistry")]
public class WeightRegistryBf16StoreTests
{
    private static (string dir, Tensor<float> t) SetupFloat(StreamingStoreDtype dtype, int n, bool? training)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = dtype,
        });
        WeightRegistry.SetStreamingExecutionTraining(training);
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) t[i] = (float)Math.Sin(i * 0.013) * 0.05f;
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        return (dir, t);
    }

    [Fact]
    public void Bf16Store_HalvesResidentBytes_AndRoundTripsWithinPrecision()
    {
        const int n = 512;
        var (dir, t) = SetupFloat(StreamingStoreDtype.Bf16, n, training: null);
        try
        {
            // Stored as bf16 → 2 bytes/elem, not fp32's 4.
            Assert.Equal(n * 2, WeightRegistry.StreamingPool.ResidentBytes);
            Assert.Equal((byte)1, t.StreamingStoreEncoding);

            var expected = new float[n];
            for (int i = 0; i < n; i++) expected[i] = (float)Math.Sin(i * 0.013) * 0.05f;

            WeightRegistry.Materialize(t); // bf16 → fp32 widen
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < n; i++) { double e = t[i] - expected[i]; sum2 += e * e; ref2 += (double)expected[i] * expected[i]; }
            Assert.True(Math.Sqrt(sum2 / ref2) < 0.01, "bf16 store round-trip RMS error should be <1%");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void FullPrecision_KeepsNativeBytes()
    {
        const int n = 512;
        var (dir, t) = SetupFloat(StreamingStoreDtype.FullPrecision, n, training: null);
        try
        {
            Assert.Equal(n * sizeof(float), WeightRegistry.StreamingPool.ResidentBytes);
            Assert.Equal((byte)0, t.StreamingStoreEncoding);
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void Auto_PicksBf16InInference_FullInTraining()
    {
        const int n = 256;
        var (d1, tInf) = SetupFloat(StreamingStoreDtype.Auto, n, training: false);
        try { Assert.Equal((byte)1, tInf.StreamingStoreEncoding); Assert.Equal(n * 2, WeightRegistry.StreamingPool.ResidentBytes); }
        finally { Cleanup(d1); }

        var (d2, tTrain) = SetupFloat(StreamingStoreDtype.Auto, n, training: true);
        try { Assert.Equal((byte)0, tTrain.StreamingStoreEncoding); Assert.Equal(n * sizeof(float), WeightRegistry.StreamingPool.ResidentBytes); }
        finally { Cleanup(d2); }

        // Unknown mode (null) must be SAFE = full precision (never silent bf16 masters).
        var (d3, tUnknown) = SetupFloat(StreamingStoreDtype.Auto, n, training: null);
        try { Assert.Equal((byte)0, tUnknown.StreamingStoreEncoding); }
        finally { Cleanup(d3); }
    }

    [Fact]
    public void Bf16Store_Double_Quarters_TheBytes()
    {
        const int n = 256;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16d-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Bf16,
        });
        WeightRegistry.SetStreamingExecutionTraining(null);
        try
        {
            var t = new Tensor<double>(new[] { n });
            for (int i = 0; i < n; i++) t[i] = Math.Cos(i * 0.02) * 0.1;
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);

            // fp64 → bf16 = 4x reduction (8 bytes → 2).
            Assert.Equal(n * 2, WeightRegistry.StreamingPool.ResidentBytes);

            WeightRegistry.Materialize(t);
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < n; i++) { double exp = Math.Cos(i * 0.02) * 0.1; double e = t[i] - exp; sum2 += e * e; ref2 += exp * exp; }
            Assert.True(Math.Sqrt(sum2 / ref2) < 0.01, "fp64→bf16 round-trip RMS error should be <1%");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void Int8Store_QuartersBytes_AndRoundTrips()
    {
        const int n = 1024;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int8-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int8,
        });
        WeightRegistry.SetStreamingExecutionTraining(null);
        try
        {
            var t = new Tensor<float>(new[] { n });
            for (int i = 0; i < n; i++) t[i] = (float)Math.Sin(i * 0.02) * 0.05f;
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);

            // int8 + 4-byte scale ≈ n bytes vs fp32's 4n → ~4x reduction.
            Assert.Equal(n + 4, WeightRegistry.StreamingPool.ResidentBytes);
            Assert.Equal((byte)2, t.StreamingStoreEncoding);

            WeightRegistry.Materialize(t);
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < n; i++) { double exp = Math.Sin(i * 0.02) * 0.05; double e = t[i] - exp; sum2 += e * e; ref2 += exp * exp; }
            Assert.True(Math.Sqrt(sum2 / ref2) < 0.03, "int8 store round-trip RMS error should be ~1-2%");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void LosslessStore_IsBitExact_AndSmallerThanFp32()
    {
        const int n = 4096;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-lossless-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Lossless,
        });
        WeightRegistry.SetStreamingExecutionTraining(null);
        try
        {
            var expected = new float[n];
            for (int i = 0; i < n; i++) expected[i] = (float)Math.Sin(i * 0.01) * 0.03f;
            var t = new Tensor<float>(new[] { n });
            for (int i = 0; i < n; i++) t[i] = expected[i];
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);

            Assert.Equal((byte)3, t.StreamingStoreEncoding);
            // Lossless compresses: resident bytes < raw fp32.
            Assert.True(WeightRegistry.StreamingPool.ResidentBytes < n * sizeof(float),
                $"lossless resident {WeightRegistry.StreamingPool.ResidentBytes} should be < {n * 4}");

            WeightRegistry.Materialize(t);
            // BIT-exact round-trip (lossless).
            for (int i = 0; i < n; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(t[i]));
        }
        finally { Cleanup(dir); }
    }

    private static void Cleanup(string dir)
    {
        WeightRegistry.SetStreamingExecutionTraining(null);
        WeightRegistry.Reset();
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }
}
