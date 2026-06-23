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
        float v = output[0, 0, 0, 0];
        Assert.False(float.IsNaN(v) || float.IsInfinity(v));
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
        for (int it = 0; it < 300; it++)
        {
            var c = new float[M * N];
            AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(a, b, c, M, K, N); // 6-arg overload (the one Conv2DIm2colGemm uses)
            Assert.False(float.IsNaN(c[0]) || float.IsInfinity(c[0]));
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
