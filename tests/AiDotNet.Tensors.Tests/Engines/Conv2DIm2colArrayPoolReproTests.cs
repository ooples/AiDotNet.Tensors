using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Repro for the native ArrayPool&lt;float&gt;.Shared corruption that crashes the test host in the
/// float Conv2DIm2colGemm small-N transposed SGEMM path (the AccessViolation seen across every
/// CNN/diffusion CI shard). Uses the exact small-N shapes ResNet50 (32x32) hits right before the
/// crash: small N (=4/16 output patches), large outC, via Conv2DInto (the route ConvolutionalLayer
/// uses). Deterministic.
/// </summary>
public sealed class Conv2DIm2colArrayPoolReproTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // input [1,inC,H,W], kernel [outC,inC,kH,kW], stride s -> output [1,outC,oHW,oHW]; runs Conv2DInto.
    private static void Conv(CpuEngine eng, int inC, int hw, int outC, int k, int stride, int seed)
    {
        int oHW = (hw - k) / stride + 1;
        var input = Rand(new[] { 1, inC, hw, hw }, seed);
        var kernel = Rand(new[] { outC, inC, k, k }, seed + 1);
        var output = new Tensor<float>(new[] { 1, outC, oHW, oHW });
        eng.Conv2DInto(output, input, kernel, stride, 0, 1);

        // Validate the WHOLE output buffer, not just element 0: every value must be
        // finite (pool corruption / a mismatched-tile overrun shows up as NaN/Inf
        // partway through the buffer), and the conv of random non-zero input*kernel
        // must produce a non-zero result (an all-zero buffer means the GEMM was
        // silently skipped / wrote nothing — a no-op regression the single-element
        // check would miss).
        var span = output.AsWritableSpan();
        bool anyNonZero = false;
        for (int i = 0; i < span.Length; i++)
        {
            float v = span[i];
            Assert.False(float.IsNaN(v) || float.IsInfinity(v),
                $"non-finite output at index {i} (inC={inC} hw={hw} outC={outC} k={k} s={stride})");
            if (v != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero,
            $"conv produced an all-zero output buffer (inC={inC} hw={hw} outC={outC} k={k} s={stride})");
    }

    [Fact]
    public void Direct_SmallM_Sgemm_DoesNotCorruptArrayPool()
    {
        // Isolate the small-N conv's transposed SGEMM: M=4 (<=NParallelSmallMMaxM=8), large N.
        const int M = 4, K = 512, N = 1024;
        var rng = new Random(7);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);

        // Naive row-major reference (a,b are fixed across iterations). Asserting against
        // this validates not just "no NaN/crash" but that the shape that used to slip into
        // the broken mr=4 PackBoth fallback now produces the CORRECT product through the
        // Streaming reroute — over the ENTIRE C buffer, not element 0 alone.
        var expected = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double acc = 0;
                for (int p = 0; p < K; p++) acc += (double)a[i * K + p] * b[p * N + j];
                expected[i * N + j] = (float)acc;
            }

        for (int it = 0; it < 300; it++)
        {
            var c = new float[M * N];
            AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(a, b, c, M, K, N); // 6-arg overload (the one Conv2DIm2colGemm uses)
            for (int idx = 0; idx < c.Length; idx++)
            {
                float v = c[idx];
                Assert.False(float.IsNaN(v) || float.IsInfinity(v), $"non-finite C[{idx}] on iteration {it}");
                // K=512 fp32 accumulation differs from the naive order by a few ULPs scaled
                // by the magnitude (~5-6 here); 1e-2 absolute is comfortably above that and
                // well below any corruption-sized error.
                Assert.True(Math.Abs(v - expected[idx]) < 1e-2f,
                    $"C[{idx}]={v} != expected {expected[idx]} on iteration {it}");
            }
            // Force ArrayPool traffic between GEMMs so corruption surfaces as an AV on Rent.
            var scratch = System.Buffers.ArrayPool<float>.Shared.Rent(4096);
            System.Buffers.ArrayPool<float>.Shared.Return(scratch);
        }
    }

    [Fact]
    public void SmallN_ResNetShapeConvs_DoNotCorruptArrayPool()
    {
        var engine = new CpuEngine();
        // The exact small-N shapes ResNet50@32x32 runs right before the host crash:
        //   inC=512 4x4 -> outC=1024 1x1 s2  (K=512  N=4)
        //   inC=256 2x2 -> outC=1024 1x1     (K=256  N=4)
        //   inC=256 4x4 -> outC=256  3x3 s2  (K=2304 N=4)
        //   inC=128 4x4 -> outC=512  1x1     (K=128  N=16)
        //   inC=512 4x4 -> outC=128  1x1     (K=512  N=16)
        for (int i = 0; i < 80; i++)
        {
            Conv(engine, 512, 4, 1024, 1, 2, i * 7 + 1);
            Conv(engine, 256, 2, 1024, 1, 1, i * 7 + 2);
            Conv(engine, 256, 4, 256, 3, 2, i * 7 + 3);
            Conv(engine, 128, 4, 512, 1, 1, i * 7 + 4);
            Conv(engine, 512, 4, 128, 1, 1, i * 7 + 5);
        }
    }
}
