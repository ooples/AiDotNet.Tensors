// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The bias epilogue of the no-upcast int4 linear (FusedLinear and FusedLinearInto), and the bf16 bytes a device reads
/// from a bf16 streaming store. Serialized against the other registry tests (global state).
/// </summary>
[Collection("WeightRegistry")]
public class StreamingStoreLinearBiasTests
{
    private const int M = 4, K = 64, N = 32;

    private static float Value(int i, int salt) => (float)Math.Sin(i * 0.013 + salt) * 0.05f;

    private static string Configure(StreamingStoreDtype dtype)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-slb-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = dtype,
        });
        WeightRegistry.SetStreamingExecutionTraining(false);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        WeightRegistry.SetStreamingExecutionTraining(null);
        WeightRegistry.Reset();
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }

    private static Tensor<float> Streamed(int salt)
    {
        var w = new Tensor<float>(new[] { K, N });
        for (int i = 0; i < w.Length; i++) w[i] = Value(i, salt);
        w.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(w);
        return w;
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void Int4_linear_adds_the_bias_and_activation_like_the_fp32_linear_on_the_decoded_weight()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { M, K });
        for (int i = 0; i < x.Length; i++) x[i] = Value(i, 7) * 20f;
        var bias = new Tensor<float>(new[] { N });
        for (int i = 0; i < N; i++) bias[i] = 0.3f - 0.02f * i;

        var dir = Configure(StreamingStoreDtype.Int4);
        try
        {
            // The reference reads a twin weight's fp32 view (its lazy int4 decode): what the int4 GEMM computes.
            var twin = Streamed(salt: 1);
            WeightRegistry.Materialize(twin);
            var decoded = new Tensor<float>(twin.AsSpan().ToArray(), new[] { K, N });
            var expected = engine.FusedLinear(x, decoded, bias, FusedActivationType.ReLU);

            var w = Streamed(salt: 1);
            var viaFused = engine.FusedLinear(x, w, bias, FusedActivationType.ReLU);
            Assert.NotNull(w.StreamingInt4);   // routed as int4, never decoded
            var into = new Tensor<float>(new[] { M, N });
            engine.FusedLinearInto(into, x, w, bias, FusedActivationType.ReLU);
            Assert.NotNull(w.StreamingInt4);

            // The int4 kernel's result differs from the decoded-weight reference by the CPU's kernel choice (1.1e-3 on an
            // output of 0.44 on the CI runner): a relative bound, as the int8 routing test uses. A lost bias (up to 0.3
            // on outputs of about 0.5) is far outside it.
            var e = expected.AsSpan(); var a = viaFused.AsSpan(); var b = into.AsSpan();
            double ref2 = 0, errA = 0, errB = 0;
            bool biasShows = false;
            for (int i = 0; i < e.Length; i++)
            {
                ref2 += (double)e[i] * e[i];
                errA += (double)(a[i] - e[i]) * (a[i] - e[i]);
                errB += (double)(b[i] - e[i]) * (b[i] - e[i]);
                biasShows |= e[i] > 0.25f;
            }
            Assert.True(biasShows, "the bias must move the output");
            Assert.True(Math.Sqrt(errA / ref2) < 0.01, $"FusedLinear: relative error {Math.Sqrt(errA / ref2):E3}");
            Assert.True(Math.Sqrt(errB / ref2) < 0.01, $"FusedLinearInto: relative error {Math.Sqrt(errB / ref2):E3}");
        }
        finally { Cleanup(dir); }
    }
#endif

    [Fact]
    public void A_bf16_store_reads_back_as_two_bytes_per_element_and_any_other_weight_reads_null()
    {
        var dir = Configure(StreamingStoreDtype.Bf16);
        try
        {
            var plain = new Tensor<float>(new[] { K, N });
            Assert.Null(WeightRegistry.TryReadBf16Store(plain));   // not streamed
            Assert.Throws<ArgumentNullException>(() => WeightRegistry.TryReadBf16Store<float>(null!));

            var w = new Tensor<float>(new[] { K, N });
            for (int i = 0; i < w.Length; i++) w[i] = (i % 17 - 8) / 64f;   // exact in bf16
            w.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(w);
            var bytes = WeightRegistry.TryReadBf16Store(w);
            Assert.NotNull(bytes);
            Assert.Equal(2 * K * N, bytes!.Length);
            for (int i = 0; i < K * N; i++)
            {
                int bits = (bytes[2 * i] | bytes[2 * i + 1] << 8) << 16;
                Assert.Equal((i % 17 - 8) / 64f, BitConverter.ToSingle(BitConverter.GetBytes(bits), 0));
            }
        }
        finally { Cleanup(dir); }

        dir = Configure(StreamingStoreDtype.Int4);
        try { Assert.Null(WeightRegistry.TryReadBf16Store(Streamed(salt: 2))); }   // streamed, but not as bf16
        finally { Cleanup(dir); }
    }
}
