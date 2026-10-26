#if NETCOREAPP
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1549: the FP16 flash-attention forward against an fp64 reference of the same FP16-rounded inputs, over head
/// dimensions, causal masking (with an offset), grouped-query heads and ragged sequence lengths, for both accumulate
/// forms; and the FP16-accumulate kernel's overflow redo (scores or P V partials past 65504 must give the FP32 result).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class PtxFlashAttentionTests
{
    private readonly ITestOutputHelper _output;

    public PtxFlashAttentionTests(ITestOutputHelper output) => _output = output;

    private static DirectPtxRuntime RequireAmpere()
    {
        DirectPtxRuntime runtime;
        try
        {
            runtime = new DirectPtxRuntime();
        }
        catch (Exception ex) when (ex is DllNotFoundException or InvalidOperationException or NotSupportedException)
        {
            Skip.If(true, $"No CUDA device: {ex.Message}");
            throw;
        }
        if (runtime.ComputeCapabilityMajor < 8)
        {
            runtime.Dispose();
            Skip.If(true, "Needs compute capability 8.0+.");
        }
        return runtime;
    }

    private static Half[] RandomHalf(Random rng, int count, double range) =>
        Enumerable.Range(0, count).Select(_ => (Half)((rng.NextDouble() * 2 - 1) * range)).ToArray();

    // fp64 attention of the FP16 inputs: O[b, h, i, :] = sum_j softmax_j(scale * q_i . k_j) v_j over the visible keys.
    private static double[] Reference(Half[] q, Half[] k, Half[] v, int batch, int hq, int hkv, int sq, int sk, int d,
        double scale, bool causal, int offset)
    {
        var o = new double[batch * hq * sq * d];
        var logits = new double[sk];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < hq; h++)
            {
                int kvh = h / (hq / hkv);
                for (int i = 0; i < sq; i++)
                {
                    int qBase = ((b * hq + h) * sq + i) * d;
                    double max = double.NegativeInfinity;
                    int visible = causal ? Math.Min(sk, i + offset + 1) : sk;
                    for (int j = 0; j < visible; j++)
                    {
                        int kBase = ((b * hkv + kvh) * sk + j) * d;
                        double dot = 0;
                        for (int c = 0; c < d; c++) dot += (double)q[qBase + c] * (double)k[kBase + c];
                        logits[j] = dot * scale;
                        max = Math.Max(max, logits[j]);
                    }
                    if (visible <= 0) continue;   // no visible key: zeros
                    double sum = 0;
                    for (int j = 0; j < visible; j++) sum += logits[j] = Math.Exp(logits[j] - max);
                    for (int j = 0; j < visible; j++)
                    {
                        int vBase = ((b * hkv + kvh) * sk + j) * d;
                        double p = logits[j] / sum;
                        for (int c = 0; c < d; c++) o[qBase + c] += p * (double)v[vBase + c];
                    }
                }
            }
        return o;
    }

    private static float[] Run(DirectPtxRuntime runtime, PtxFlashAttentionKernel kernel, Half[] q, Half[] k, Half[] v,
        int batch, int hq, int hkv, int sq, int sk, float scale, int offset)
    {
        using var dq = runtime.AllocateBytes((nuint)(q.Length * 2));
        using var dk = runtime.AllocateBytes((nuint)(k.Length * 2));
        using var dv = runtime.AllocateBytes((nuint)(v.Length * 2));
        using var dout = runtime.AllocateBytes((nuint)(q.Length * 4));
        using var dlse = runtime.AllocateBytes((nuint)(batch * hq * sq * 4));
        dq.Upload<Half>(q);
        dk.Upload<Half>(k);
        dv.Upload<Half>(v);
        kernel.Launch(dq.Pointer, dk.Pointer, dv.Pointer, dout.Pointer, dlse.Pointer, batch, hq, hkv, sq, sk, scale, offset);
        runtime.Synchronize();
        var result = new float[q.Length];
        dout.Download<float>(result);
        return result;
    }

    [SkippableTheory]
    [InlineData(64, false, false)]
    [InlineData(64, true, false)]
    [InlineData(64, false, true)]
    [InlineData(64, true, true)]
    [InlineData(128, false, false)]
    [InlineData(128, true, false)]
    [InlineData(128, false, true)]
    [InlineData(128, true, true)]
    public void MatchesTheReference_OverShapesHeadsAndMasks(int d, bool causal, bool fp16Accumulate)
    {
        using var runtime = RequireAmpere();
        using var kernel = new PtxFlashAttentionKernel(runtime, d, causal, fp16Accumulate);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        _output.WriteLine($"d{d} causal={causal} f16acc={fp16Accumulate}: {kernel.Audit.Function.RegistersPerThread} registers");
        // (batch, Hq, Hkv, Sq, Sk, causal offset): one tile, ragged lengths over several tiles, grouped heads, Sk > Sq.
        foreach (var (b, hq, hkv, sq, sk, offset) in new[]
                 {
                     (1, 1, 1, 64, 64, 0), (2, 4, 2, 100, 130, 0), (1, 2, 1, 37, 200, causal ? 163 : 0), (1, 1, 1, 257, 257, 0),
                 })
        {
            var rng = new Random(d * 1000 + sq * 7 + sk);
            Half[] q = RandomHalf(rng, b * hq * sq * d, 1.0), k = RandomHalf(rng, b * hkv * sk * d, 1.0), v = RandomHalf(rng, b * hkv * sk * d, 1.0);
            float scale = (float)(1.0 / Math.Sqrt(d));
            float[] actual = Run(runtime, kernel, q, k, v, b, hq, hkv, sq, sk, scale, offset);
            double[] expected = Reference(q, k, v, b, hq, hkv, sq, sk, d, scale, causal, offset);
            double worst = 0;
            for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
            _output.WriteLine($"  B{b} Hq{hq} Hkv{hkv} Sq{sq} Sk{sk} offset {offset}: max |err| {worst:E2}");
            // P is rounded to FP16 (2^-11 relative) before P V; FP16 accumulation adds a rounding per 64-key tile.
            Assert.True(worst <= (fp16Accumulate ? 6e-3 : 3e-3), $"max |err| {worst:E2}");
        }
    }

    // Opt-in (AIDOTNET_RUN_FLASH_PROBE=1): both accumulate forms on the PyTorch SDPA comparison shapes (FP16 Q/K/V in
    // [-1, 1), scale 1 / sqrt(D)); best of 5 rounds of 20 launches, host-timed around a synchronize.
    [SkippableFact]
    public void Probe_Throughput()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_FLASH_PROBE") == "1", "Set AIDOTNET_RUN_FLASH_PROBE=1.");
        using var runtime = RequireAmpere();
        foreach (var (b, h, s, d) in new[] { (8, 16, 512, 64), (8, 16, 1024, 64), (4, 16, 2048, 64), (2, 16, 4096, 64), (8, 16, 1024, 128), (2, 16, 4096, 128) })
            foreach (bool causal in new[] { false, true })
            {
                var rng = new Random(1);
                int n = b * h * s * d;
                using var dq = runtime.AllocateBytes((nuint)(n * 2));
                using var dk = runtime.AllocateBytes((nuint)(n * 2));
                using var dv = runtime.AllocateBytes((nuint)(n * 2));
                using var dout = runtime.AllocateBytes((nuint)(n * 4));
                dq.Upload<Half>(RandomHalf(rng, n, 1.0));
                dk.Upload<Half>(RandomHalf(rng, n, 1.0));
                dv.Upload<Half>(RandomHalf(rng, n, 1.0));
                double flops = 4.0 * b * h * s * (double)s * d * (causal ? 0.5 : 1.0);
                string line = $"B{b} H{h} S{s} D{d} causal={causal}:";
                foreach (bool f16 in new[] { false, true })
                {
                    using var kernel = new PtxFlashAttentionKernel(runtime, d, causal, f16);
                    void Launch() => kernel.Launch(dq.Pointer, dk.Pointer, dv.Pointer, dout.Pointer, IntPtr.Zero, b, h, h, s, s, (float)(1 / Math.Sqrt(d)));
                    for (int i = 0; i < 3; i++) Launch();
                    runtime.Synchronize();
                    double best = double.MaxValue;
                    for (int round = 0; round < 5; round++)
                    {
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        for (int i = 0; i < 20; i++) Launch();
                        runtime.Synchronize();
                        best = Math.Min(best, watch.Elapsed.TotalMilliseconds * 1000 / 20);
                    }
                    line += $" {(f16 ? "f16acc" : "f32acc")} {best:F1}us ({flops / best / 1e6:F1} TFLOPS, {kernel.Audit.Function.RegistersPerThread} regs)";
                }
                _output.WriteLine(line);
            }
    }

    // Overflow-sized inputs: Q = K = 40 makes a score's FP16 partial reach 40 * 40 * D (102400 at D = 64, past 65504);
    // V near 3000 with near-uniform probabilities makes a 64-key P V partial reach about 64 * 3000. The FP16-accumulate
    // kernel must redo those blocks and match the FP32-accumulate kernel bit for bit; a NaN input stays NaN.
    [SkippableTheory]
    [InlineData(64)]
    [InlineData(128)]
    public void Fp16Accumulate_RedoesOverflowingBlocks_AndKeepsNaN(int d)
    {
        using var runtime = RequireAmpere();
        using var f16 = new PtxFlashAttentionKernel(runtime, d, isCausal: false, fp16Accumulate: true);
        using var f32 = new PtxFlashAttentionKernel(runtime, d, isCausal: false, fp16Accumulate: false);
        const int b = 1, h = 1, sq = 128, sk = 128;
        var rng = new Random(3);
        float scale = 1f / 64;
        // Large scores in every row of the first 64-row block only; the second block stays small.
        Half[] q = RandomHalf(rng, sq * d, 0.5);
        for (int i = 0; i < 64 * d; i++) q[i] = (Half)40f;
        Half[] k = Enumerable.Repeat((Half)40f, sk * d).ToArray();
        for (int i = 0; i < sk * d; i += 3) k[i] = (Half)20f;   // per-key variety; every score still > 84000
        Half[] v = RandomHalf(rng, sk * d, 1.0);
        float[] got16 = Run(runtime, f16, q, k, v, b, h, h, sq, sk, scale, 0);
        float[] got32 = Run(runtime, f32, q, k, v, b, h, h, sq, sk, scale, 0);
        Assert.All(got16, x => Assert.True(float.IsFinite(x)));
        Assert.Equal(got32.Take(64 * d), got16.Take(64 * d));   // the redone block is the FP32 kernel's result exactly

        Half[] bigV = Enumerable.Range(0, sk * d).Select(i => (Half)(i % 2 == 0 ? 3000f : 2900f)).ToArray();
        Half[] small = RandomHalf(rng, sq * d, 0.5);
        got16 = Run(runtime, f16, small, RandomHalf(new Random(4), sk * d, 0.5), bigV, b, h, h, sq, sk, scale, 0);
        got32 = Run(runtime, f32, small, RandomHalf(new Random(4), sk * d, 0.5), bigV, b, h, h, sq, sk, scale, 0);
        Assert.Equal(got32, got16);

        small[5] = Half.NaN;   // row 0
        got16 = Run(runtime, f16, small, RandomHalf(new Random(4), sk * d, 0.5), v, b, h, h, sq, sk, scale, 0);
        Assert.All(got16.Take(d), x => Assert.True(float.IsNaN(x)));
        Assert.All(got16.Skip(64 * d), x => Assert.True(float.IsFinite(x)));
    }
}
#endif