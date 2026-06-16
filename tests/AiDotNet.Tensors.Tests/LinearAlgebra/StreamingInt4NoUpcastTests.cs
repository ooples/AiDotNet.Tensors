// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// No-upcast int4 streaming: an int4-stored weight materialized for inference stays as int4
/// group-quant + group scales (the form the int4 GEMM consumes) instead of decoding to fp32, so
/// an int4-resident model keeps its ~8x-smaller footprint. The fp view is produced lazily only
/// if a non-matmul path reads it; training always decodes to writable fp. Serialized against the
/// other registry tests (global state).
/// </summary>
[Collection("WeightRegistry")]
public class StreamingInt4NoUpcastTests
{
    private static float Val(int i) => (float)Math.Sin(i * 0.011) * 0.05f;

    [Fact]
    public void Int4Inference_Materialize_KeepsInt4_NoFp32Decode_LazyFallbackIsCorrect()
    {
        // Logical Linear weight [in, out]; the int4 store keeps it TRANSPOSED [out, in] + group
        // scales (the kernel layout), so Rows = out, K = in.
        const int inDim = 64, outDim = 128, n = inDim * outDim;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int4nu-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int4,
        });
        WeightRegistry.SetStreamingExecutionTraining(false); // inference
        try
        {
            var t = new Tensor<float>(new[] { inDim, outDim });
            for (int i = 0; i < n; i++) t[i] = Val(i);
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t); // stores int4 group-quant (transposed), drops fp32 _data
            Assert.Equal(StreamingEncoding.Int4, t.StreamingStoreEncoding);

            WeightRegistry.Materialize(t);

            // No upcast: the int4 weight is attached (kernel-ready [out,in]), fp32 _data NOT decoded.
            Assert.NotNull(t.StreamingInt4);
            Assert.Equal(0, t.DataVector.Length);
            Assert.Equal(outDim, t.StreamingInt4!.Rows);   // [out, in]
            Assert.Equal(inDim, t.StreamingInt4.K);
            Assert.Equal(n, t.StreamingInt4.Data.Length);
            Assert.True(t.StreamingInt4.TransposedFromLogical);
            Assert.Equal(StreamingStoreCodec.DefaultInt4GroupSize, t.StreamingInt4.GroupSize);

            // A second Materialize is a no-op (already materialized as int4) — no re-fetch.
            WeightRegistry.Materialize(t);
            Assert.NotNull(t.StreamingInt4);

            // Lazy fp32 fallback: reading a span dequantizes (group-quant, transposed back to the
            // logical [in,out]) + clears the int4 form.
            var span = t.AsSpan();
            Assert.Null(t.StreamingInt4);
            Assert.Equal(n, t.DataVector.Length);
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < n; i++) { double e = span[i] - Val(i); sum2 += e * e; ref2 += (double)Val(i) * Val(i); }
            // The transpose-back must land each value at its logical index: a wrong transpose
            // would scramble this far past int4's ~12-15% group-quant RMS.
            Assert.True(Math.Sqrt(sum2 / ref2) < 0.20, "int4 group-quant lazy dequant should be within ~12-15% RMS");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void Int4Training_Materialize_DecodesToFp32_NoInt4Attached()
    {
        const int rows = 8, k = 128, n = rows * k;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int4tr-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int4,
        });
        WeightRegistry.SetStreamingExecutionTraining(true); // training → must be writable fp
        try
        {
            var t = new Tensor<float>(new[] { rows, k });
            for (int i = 0; i < n; i++) t[i] = Val(i);
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);

            WeightRegistry.Materialize(t);

            // Training never uses the no-upcast int4 form (the optimizer writes the weight).
            Assert.Null(t.StreamingInt4);
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
