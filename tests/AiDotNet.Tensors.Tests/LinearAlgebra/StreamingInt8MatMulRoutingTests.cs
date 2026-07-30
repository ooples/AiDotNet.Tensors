// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// End-to-end no-upcast int8 compute through the CONSUMER's path: a Linear weight [in,out]
/// int8-streamed for inference, run through <c>Engine.FusedLinear(x, W, bias)</c> = x·W, routes
/// to the int8 weight-only GEMM — consuming the stored int8 + per-OUTPUT scales directly (W is
/// stored transposed [out,in], the kernel layout), never upcasting to fp32. The result matches
/// the fp32 FusedLinear on the same per-output-dequantized weight, and the weight stays int8
/// afterward (proving the fp32 decode was skipped). Serialized (global registry).
/// </summary>
// The int8 weight-only GEMM + the FusedLinear int8 hook live in NET5_0_OR_GREATER (AVX2).
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

    // Per-OUTPUT-channel symmetric int8 quant of a Linear weight W[in,out] (per column), matching
    // the registry's transpose-then-per-row encode. Returns the dequantized fp32 weight.
    private static float[] DequantizePerOutput(float[] w, int inDim, int outDim)
    {
        var deq = new float[inDim * outDim];
        for (int o = 0; o < outDim; o++)
        {
            float amax = 0f;
            for (int i = 0; i < inDim; i++) { float a = Math.Abs(w[i * outDim + o]); if (a > amax) amax = a; }
            float scale = amax > 0f ? amax / 127f : 1f;
            float inv = 1f / scale;
            for (int i = 0; i < inDim; i++)
            {
                int q = (int)Math.Round(w[i * outDim + o] * inv);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                deq[i * outDim + o] = q * scale;
            }
        }
        return deq;
    }

    [Fact]
    public void Int8Weight_ThroughFusedLinear_RoutesToInt8Kernel_NoUpcast()
    {
        const int inDim = 128, outDim = 64, batch = 8;
        var engine = new CpuEngine();
        var rng = new Rng(2024);

        var wData = new float[inDim * outDim];          // logical [in, out]
        for (int i = 0; i < wData.Length; i++) wData[i] = rng.Next(0.1);
        var xData = new float[batch * inDim];
        for (int i = 0; i < xData.Length; i++) xData[i] = rng.Next(1.0);
        var biasData = new float[outDim];
        for (int i = 0; i < biasData.Length; i++) biasData[i] = rng.Next(0.3);

        var x = new Tensor<float>(xData, new[] { batch, inDim });
        var bias = new Tensor<float>(biasData, new[] { outDim });

        // Reference: fp32 FusedLinear on the per-output-dequantized weight — exactly what the
        // int8 kernel computes (only accumulation order differs).
        var deq = DequantizePerOutput(wData, inDim, outDim);
        var wRef = new Tensor<float>(deq, new[] { inDim, outDim });
        var cRef = engine.FusedLinear(x, wRef, bias, FusedActivationType.None);

        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int8fl-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int8,
        });
        WeightRegistry.SetStreamingExecutionTraining(false); // inference
        try
        {
            var wStream = new Tensor<float>(new[] { inDim, outDim });
            for (int i = 0; i < wData.Length; i++) wStream[i] = wData[i];
            wStream.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(wStream); // per-output int8 store (transposed), fp32 dropped

            // The weight reaches FusedLinear still paged-out → routed AS int8 (no fp32 decode).
            var cInt8 = engine.FusedLinear(x, wStream, bias, FusedActivationType.None);

            // (1) Routed to int8 + no upcast: weight is in int8 form, fp32 _data empty.
            Assert.NotNull(wStream.StreamingInt8);
            Assert.Equal(0, wStream.DataVector.Length);

            // (2) Correct: matches fp32 FusedLinear on the per-output-dequantized weight.
            var rEng = cInt8.AsSpan();
            var rRef = cRef.AsSpan();
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < batch * outDim; i++) { double e = rEng[i] - rRef[i]; sum2 += e * e; ref2 += (double)rRef[i] * rRef[i]; }
            double rel = Math.Sqrt(sum2 / Math.Max(1e-30, ref2));
            Assert.True(rel < 0.01, $"int8-routed FusedLinear should match fp32-on-dequantized (rel {rel:E3}).");
        }
        finally
        {
            WeightRegistry.SetStreamingExecutionTraining(null);
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// The RAM-aware <see cref="StreamingStoreDtype.Auto"/> path (not the explicit Int8 opt-in):
    /// a large inference weight, under a footprint+cap where bf16 would overflow the resident cap
    /// but int8 fits, is AUTO-selected to int8 AND routed through the same no-upcast int8 kernel —
    /// producing correct output. This proves the new Auto escalation doesn't just tag the encoding
    /// byte but actually computes through the int8 GEMM (the behavioural end-to-end, not just policy).
    /// </summary>
    [Fact]
    public void Auto_RamAware_SelectsInt8_ForLargeInferenceWeight_AndRoutesToInt8Kernel()
    {
        const int inDim = 256, outDim = 128, batch = 8; // 32,768 weights ≥ MinAutoQuantElements, rank 2
        var engine = new CpuEngine();
        var rng = new Rng(7777);

        var wData = new float[inDim * outDim];
        for (int i = 0; i < wData.Length; i++) wData[i] = rng.Next(0.1);
        var xData = new float[batch * inDim];
        for (int i = 0; i < xData.Length; i++) xData[i] = rng.Next(1.0);
        var biasData = new float[outDim];
        for (int i = 0; i < biasData.Length; i++) biasData[i] = rng.Next(0.3);

        var x = new Tensor<float>(xData, new[] { batch, inDim });
        var bias = new Tensor<float>(biasData, new[] { outDim });

        var deq = DequantizePerOutput(wData, inDim, outDim);
        var wRef = new Tensor<float>(deq, new[] { inDim, outDim });
        var cRef = engine.FusedLinear(x, wRef, bias, FusedActivationType.None);

        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-autoint8fl-{Guid.NewGuid():N}");
        // Footprint 4 MiB, cap 1 MiB: bf16 (÷2 = 2 MiB) overflows the cap, int8 (÷4 = 1 MiB) fits →
        // Auto picks int8 for this large weight. The cap comfortably holds the single 33 KB weight
        // (no thrash); it only drives the PRECISION choice.
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1L * 1024 * 1024,
            ExpectedStreamingFootprintBytes = 4L * 1024 * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Auto,
        });
        WeightRegistry.SetStreamingExecutionTraining(false); // inference (read-only)
        try
        {
            var wStream = new Tensor<float>(new[] { inDim, outDim });
            for (int i = 0; i < wData.Length; i++) wStream[i] = wData[i];
            wStream.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(wStream);

            // (0) Auto CHOSE int8 (not bf16) for this large weight under memory pressure.
            Assert.Equal(StreamingEncoding.Int8, wStream.StreamingStoreEncoding);

            var cInt8 = engine.FusedLinear(x, wStream, bias, FusedActivationType.None);

            // (1) Routed to int8, no upcast: int8 form present, fp32 _data empty.
            Assert.NotNull(wStream.StreamingInt8);
            Assert.Equal(0, wStream.DataVector.Length);

            // (2) Correct: matches fp32 FusedLinear on the per-output-dequantized weight.
            var rEng = cInt8.AsSpan();
            var rRef = cRef.AsSpan();
            double sum2 = 0, ref2 = 0;
            for (int i = 0; i < batch * outDim; i++) { double e = rEng[i] - rRef[i]; sum2 += e * e; ref2 += (double)rRef[i] * rRef[i]; }
            double rel = Math.Sqrt(sum2 / Math.Max(1e-30, ref2));
            Assert.True(rel < 0.01, $"Auto-int8-routed FusedLinear should match fp32-on-dequantized (rel {rel:E3}).");
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
