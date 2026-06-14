// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// No-upcast int8 streaming: an int8-stored weight materialized for inference stays as int8 +
/// per-row scales (the form the int8 GEMM consumes) instead of decoding to fp32. The fp32 view
/// is produced lazily only if a non-matmul path reads it. Serialized against other registry
/// tests (global state).
/// </summary>
[Collection("WeightRegistry")]
public class StreamingInt8NoUpcastTests
{
    private static float Val(int i) => (float)Math.Sin(i * 0.011) * 0.05f;

    [Fact]
    public void Int8Inference_Materialize_KeepsInt8_NoFp32Decode_LazyFallbackIsCorrect()
    {
        // Logical Linear weight [in, out]; the int8 store keeps it TRANSPOSED [out, in] + per-
        // output scales (the kernel layout), so Rows = out, K = in.
        const int inDim = 16, outDim = 32, n = inDim * outDim;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int8nu-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int8,
        });
        WeightRegistry.SetStreamingExecutionTraining(false); // inference
        try
        {
            var t = new Tensor<float>(new[] { inDim, outDim });
            for (int i = 0; i < n; i++) t[i] = Val(i);
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t); // stores per-output int8 (transposed), drops fp32 _data
            Assert.Equal(StreamingEncoding.Int8, t.StreamingStoreEncoding);

            WeightRegistry.Materialize(t);

            // No upcast: the int8 weight is attached (kernel-ready [out,in]), fp32 _data NOT decoded.
            Assert.NotNull(t.StreamingInt8);
            Assert.Equal(0, t.DataVector.Length);
            Assert.Equal(outDim, t.StreamingInt8!.Rows);   // per-output scales
            Assert.Equal(inDim, t.StreamingInt8.K);
            Assert.Equal(outDim, t.StreamingInt8.Scales.Length);
            Assert.Equal(n, t.StreamingInt8.Data.Length);
            Assert.True(t.StreamingInt8.TransposedFromLogical);

            // A second Materialize is a no-op (already materialized as int8) — no re-fetch.
            WeightRegistry.Materialize(t);
            Assert.NotNull(t.StreamingInt8);

            // Lazy fp32 fallback: reading a span dequantizes per-row + clears the int8 form.
            var span = t.AsSpan();
            Assert.Null(t.StreamingInt8);
            Assert.Equal(n, t.DataVector.Length);
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < n; i++) { double e = span[i] - Val(i); sum2 += e * e; ref2 += (double)Val(i) * Val(i); }
            Assert.True(Math.Sqrt(sum2 / ref2) < 0.03, "per-row int8 dequant should be within ~1-2% RMS");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void Int8Training_Materialize_DecodesToFp32_NoInt8Attached()
    {
        const int rows = 8, k = 16, n = rows * k;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int8tr-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int8,
        });
        WeightRegistry.SetStreamingExecutionTraining(true); // training → must be writable fp32
        try
        {
            var t = new Tensor<float>(new[] { rows, k });
            for (int i = 0; i < n; i++) t[i] = Val(i);
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);

            WeightRegistry.Materialize(t);

            // Training never uses the no-upcast int8 form (the optimizer writes the weight).
            Assert.Null(t.StreamingInt8);
            Assert.Equal(n, t.DataVector.Length);
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
