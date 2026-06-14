// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// End-to-end no-upcast int8 compute: an int8-streamed weight (Linear pattern, [out,in])
/// flowing through the engine's transposed matmul (c = a·Wᵀ) routes to the int8 weight-only
/// GEMM — consuming the stored int8 + per-row scales directly, never upcasting to fp32. The
/// result matches the fp32 path on the same dequantized weight, and the weight stays in its
/// int8 form afterward (proving the fp32 decode was skipped). Serialized (global registry).
/// </summary>
// The int8 weight-only GEMM (SgemmWithInt8RowScaledCachedB) and the engine's routing hook
// live in NET5_0_OR_GREATER (AVX2 intrinsics) — net471 falls back to fp32 decode + Sgemm, so
// the no-upcast routing only applies (and is tested) there.
#if NET5_0_OR_GREATER
[Collection("WeightRegistry")]
public class StreamingInt8MatMulRoutingTests
{
    private struct Rng
    {
        private ulong _s;
        public Rng(ulong seed) { _s = seed | 1UL; }
        public float Next(double std) { ulong x = _s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; _s = x; double u = (x >> 11) * (1.0 / (1UL << 53)); return (float)((u - 0.5) * 2 * std); }
    }

    // Per-row symmetric int8 quant matching StreamingStoreCodec.EncodeInt8Float, returning the
    // dequantized fp32 weight (what both the int8 kernel and the fp32 reference should compute).
    private static float[] DequantizePerRow(float[] w, int n, int k)
    {
        var deq = new float[n * k];
        for (int r = 0; r < n; r++)
        {
            float amax = 0f;
            for (int j = 0; j < k; j++) { float a = Math.Abs(w[r * k + j]); if (a > amax) amax = a; }
            float scale = amax > 0f ? amax / 127f : 1f;
            float inv = 1f / scale;
            for (int j = 0; j < k; j++)
            {
                int q = (int)Math.Round(w[r * k + j] * inv);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                deq[r * k + j] = q * scale;
            }
        }
        return deq;
    }

    [Fact]
    public void Int8Weight_ThroughTransposedMatMul_RoutesToInt8Kernel_NoUpcast()
    {
        const int n = 64, k = 128, m = 8; // weight [out=64, in=128]; activation [batch=8, in=128]
        var engine = new CpuEngine();
        var rng = new Rng(2024);

        var wData = new float[n * k];
        for (int i = 0; i < wData.Length; i++) wData[i] = rng.Next(0.1);
        var aData = new float[m * k];
        for (int i = 0; i < aData.Length; i++) aData[i] = rng.Next(1.0);

        var a = new Tensor<float>(aData, new[] { m, k });

        // Reference: the engine's normal fp32 transposed-matmul on the DEQUANTIZED weight —
        // exactly what the int8 kernel computes, so the only difference is accumulation order.
        var deq = DequantizePerRow(wData, n, k);
        var wRef = new Tensor<float>(deq, new[] { n, k });
        var cRef = engine.TensorMatMulTransposed(a, wRef);

        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int8mm-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int8,
        });
        WeightRegistry.SetStreamingExecutionTraining(false); // inference
        try
        {
            var wStream = new Tensor<float>(new[] { n, k });
            for (int i = 0; i < wData.Length; i++) wStream[i] = wData[i];
            wStream.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(wStream); // per-row int8 store, fp32 dropped

            // The weight reaches the engine still paged-out → the int8 routing materializes it
            // AS int8 (no fp32 decode) and runs the int8 GEMM.
            var cInt8 = engine.TensorMatMulTransposed(a, wStream);

            // (1) Routed to int8 + no upcast: the weight is in its int8 form, fp32 _data empty.
            Assert.NotNull(wStream.StreamingInt8);
            Assert.Equal(0, wStream.DataVector.Length);

            // (2) Correct: matches the fp32 path on the dequantized weight.
            var rEng = cInt8.AsSpan();
            var rRef = cRef.AsSpan();
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < m * n; i++) { double e = rEng[i] - rRef[i]; sum2 += e * e; ref2 += (double)rRef[i] * rRef[i]; }
            double rel = Math.Sqrt(sum2 / Math.Max(1e-30, ref2));
            Assert.True(rel < 0.01, $"int8-routed matmul should match fp32-on-dequantized (rel {rel:E3}).");
        }
        finally
        {
            WeightRegistry.SetStreamingExecutionTraining(null);
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
#endif
