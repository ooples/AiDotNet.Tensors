#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IShortAttentionForwardBackend"/>: head-interleaved attention forward against a double-precision reference,
/// including key counts that are not a multiple of the kernel's 32-key block, causal masks with positive and negative
/// offsets (a row that sees no key outputs zero), and the engine's inference path through MultiHeadAttentionCore.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ShortAttentionForwardTests
{
    private static float[] Values(int n, int seed)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((((i + seed) * 7919) % 61) - 30) * 0.03f;
        return v;
    }

    private static double[] Reference(float[] q, float[] k, float[] v, int b, int h, int sq, int sk, int d, double scale,
        bool causal, int offset)
    {
        int width = h * d;
        var o = new double[b * sq * width];
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
                for (int i = 0; i < sq; i++)
                {
                    var s = new double[sk];
                    double max = double.NegativeInfinity;
                    int last = causal ? i + offset : sk - 1;
                    for (int j = 0; j < sk && j <= last; j++)
                    {
                        double dot = 0;
                        for (int x = 0; x < d; x++)
                            dot += (double)q[(bi * sq + i) * width + hi * d + x] * k[(bi * sk + j) * width + hi * d + x];
                        s[j] = dot * scale;
                        max = Math.Max(max, s[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < sk && j <= last; j++) { s[j] = Math.Exp(s[j] - max); sum += s[j]; }
                    for (int x = 0; x < d; x++)
                    {
                        double acc = 0;
                        for (int j = 0; j < sk && j <= last; j++) acc += s[j] * v[(bi * sk + j) * width + hi * d + x];
                        o[(bi * sq + i) * width + hi * d + x] = sum > 0 ? acc / sum : 0;
                    }
                }
        return o;
    }

    [SkippableFact]
    public void OpenCl_ShortAttention_MatchesReference()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var attention = (IShortAttentionForwardBackend)backend;
        var cases = new (int B, int H, int Sq, int Sk, int D, bool Causal, int Offset)[]
        {
            (2, 3, 65, 65, 64, false, 0), (3, 2, 7, 7, 8, false, 0), (1, 1, 130, 130, 64, true, 0),
            (2, 2, 5, 40, 16, true, 35), (1, 2, 9, 9, 33, true, -3), (1, 1, 1, 1, 1, false, 0),
        };
        foreach (var (b, h, sq, sk, d, causal, offset) in cases)
        {
            Assert.True(attention.SupportsShortAttention(d));
            float[] q = Values(b * sq * h * d, 1), k = Values(b * sk * h * d, 2), v = Values(b * sk * h * d, 3);
            double scale = 1.0 / Math.Sqrt(d);
            var expected = Reference(q, k, v, b, h, sq, sk, d, scale, causal, offset);
            using var qb = backend.AllocateBuffer(q);
            using var kb = backend.AllocateBuffer(k);
            using var vb = backend.AllocateBuffer(v);
            using var ob = backend.AllocateBuffer(expected.Length);
            Assert.True(attention.TryShortAttentionForward(qb, kb, vb, ob, b, h, sq, sk, d, (float)scale, causal, offset));
            var actual = backend.DownloadBuffer(ob);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 * (1 + Math.Abs(expected[i])),
                    $"B={b} H={h} Sq={sq} Sk={sk} D={d} causal={causal}/{offset} [{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    [SkippableFact]
    public void Engine_MultiHeadAttentionCore_Inference_MatchesCpu()
    {
        DirectGpuTensorEngine? engine = null;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or TypeInitializationException) { }
        Skip.If(engine is null || !engine.IsGpuAvailable, "No DirectGpu backend available.");
        using (engine)
        {
            Skip.IfNot(engine.GetBackend() is IShortAttentionForwardBackend, "The backend has no short-sequence attention.");
            const int b = 2, s = 65, heads = 3, d = 64;
            var q = new Tensor<float>(Values(b * s * heads * d, 4), new[] { b, s, heads * d });
            var k = new Tensor<float>(Values(b * s * heads * d, 5), new[] { b, s, heads * d });
            var v = new Tensor<float>(Values(b * s * heads * d, 6), new[] { b, s, heads * d });
            var expected = new CpuEngine().MultiHeadAttentionCore(q, k, v, heads).ToArray();
            var actual = engine.MultiHeadAttentionCore(q, k, v, heads).ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 * (1 + Math.Abs(expected[i])),
                    $"[{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    // The fused inference op MultiHeadAttentionLayer calls (projections, attention, output projection): its attention
    // now runs the short-sequence kernel.
    [SkippableFact]
    public void Engine_MultiHeadAttentionForward_Inference_MatchesCpu()
    {
        DirectGpuTensorEngine? engine = null;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or TypeInitializationException) { }
        Skip.If(engine is null || !engine.IsGpuAvailable, "No DirectGpu backend available.");
        using (engine)
        {
            Skip.IfNot(engine.GetBackend() is IShortAttentionForwardBackend, "The backend has no short-sequence attention.");
            const int b = 3, s = 65, width = 192, heads = 3;
            var x = new Tensor<float>(Values(b * s * width, 7), new[] { b, s, width });
            var wq = new Tensor<float>(Values(width * width, 8), new[] { width, width });
            var wk = new Tensor<float>(Values(width * width, 9), new[] { width, width });
            var wv = new Tensor<float>(Values(width * width, 10), new[] { width, width });
            var wo = new Tensor<float>(Values(width * width, 11), new[] { width, width });
            var expected = new CpuEngine().MultiHeadAttentionForward(x, wq, wk, wv, wo, heads).ToArray();
            var actual = engine.MultiHeadAttentionForward(x, wq, wk, wv, wo, heads).ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3 * (1 + Math.Abs(expected[i])),
                    $"[{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }
}
#endif