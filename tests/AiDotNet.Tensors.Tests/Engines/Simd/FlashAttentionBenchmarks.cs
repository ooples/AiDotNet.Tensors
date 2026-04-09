using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Flash Attention benchmarks at realistic sequence lengths.
/// Compares Flash Attention (O(N) memory) vs naive BLAS-based attention.
/// </summary>
public class FlashAttentionBenchmarks
{
    private readonly ITestOutputHelper _output;

    public FlashAttentionBenchmarks(ITestOutputHelper output) => _output = output;

    [Theory(Skip = "Performance benchmark — run manually via run-compilation-benchmarks.sh")]
    [InlineData(64, 32, 50)]
    [InlineData(256, 64, 30)]
    [InlineData(512, 64, 20)]
    [InlineData(1024, 64, 10)]
    [InlineData(2048, 64, 5)]
    public void FlashVsNaive_VaryingSeqLen(int seqLen, int headDim, int iters)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var q = CreateRandomArray(seqLen * headDim, rng);
        var k = CreateRandomArray(seqLen * headDim, rng);
        var v = CreateRandomArray(seqLen * headDim, rng);
        float scale = 1f / MathF.Sqrt(headDim);

        // Warmup
        var flashOut = new float[seqLen * headDim];
        FusedAttention.FlashAttentionForward(q, k, v, flashOut, seqLen, seqLen, headDim, scale);

        // Flash Attention timing
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            FusedAttention.FlashAttentionForward(q, k, v, flashOut, seqLen, seqLen, headDim, scale);
        sw.Stop();
        double flashMs = sw.Elapsed.TotalMilliseconds / iters;

        // Naive attention timing
        var naiveOut = NaiveAttention(q, k, v, seqLen, seqLen, headDim, scale);
        sw.Restart();
        for (int i = 0; i < iters; i++)
            naiveOut = NaiveAttention(q, k, v, seqLen, seqLen, headDim, scale);
        sw.Stop();
        double naiveMs = sw.Elapsed.TotalMilliseconds / iters;

        // Accuracy check
        float maxDiff = 0f;
        for (int i = 0; i < flashOut.Length; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(flashOut[i] - naiveOut[i]));

        double speedup = naiveMs / flashMs;
        long flashMemory = seqLen * headDim * sizeof(float); // O(N) — only output
        long naiveMemory = (long)seqLen * seqLen * sizeof(float) + seqLen * headDim * sizeof(float); // O(N^2) scores + output

        _output.WriteLine($"SeqLen={seqLen,5} HeadDim={headDim} | Flash: {flashMs,8:F3}ms  Naive: {naiveMs,8:F3}ms  " +
            $"Speedup: {speedup,5:F2}x  MaxDiff: {maxDiff:E2}  " +
            $"FlashMem: {flashMemory / 1024}KB  NaiveMem: {naiveMemory / 1024}KB");
    }

    [Fact(Skip = "Performance benchmark — run manually")]
    public void FlashAttention_CausalMask_Benchmark()
    {
        int seqLen = 512, headDim = 64, iters = 20;
        var rng = RandomHelper.CreateSeededRandom(99);
        var q = CreateRandomArray(seqLen * headDim, rng);
        var k = CreateRandomArray(seqLen * headDim, rng);
        var v = CreateRandomArray(seqLen * headDim, rng);
        var output = new float[seqLen * headDim];
        float scale = 1f / MathF.Sqrt(headDim);

        // Warmup
        FusedAttention.FlashAttentionForward(q, k, v, output, seqLen, seqLen, headDim, scale, isCausal: true);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            FusedAttention.FlashAttentionForward(q, k, v, output, seqLen, seqLen, headDim, scale, isCausal: true);
        sw.Stop();

        _output.WriteLine($"Causal Flash Attention [seq={seqLen}, d={headDim}]: {sw.Elapsed.TotalMilliseconds / iters:F3}ms");
    }

    private static float[] NaiveAttention(float[] q, float[] k, float[] v,
        int seqQ, int seqK, int headDim, float scale)
    {
        var scores = new float[seqQ * seqK];
        var output = new float[seqQ * headDim];

        for (int i = 0; i < seqQ; i++)
            for (int j = 0; j < seqK; j++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                    dot += q[i * headDim + d] * k[j * headDim + d];
                scores[i * seqK + j] = dot * scale;
            }

        for (int i = 0; i < seqQ; i++)
        {
            float maxVal = float.NegativeInfinity;
            for (int j = 0; j < seqK; j++)
                if (scores[i * seqK + j] > maxVal) maxVal = scores[i * seqK + j];
            float sumExp = 0f;
            for (int j = 0; j < seqK; j++)
            {
                scores[i * seqK + j] = MathF.Exp(scores[i * seqK + j] - maxVal);
                sumExp += scores[i * seqK + j];
            }
            for (int j = 0; j < seqK; j++)
                scores[i * seqK + j] /= sumExp;
        }

        for (int i = 0; i < seqQ; i++)
            for (int d = 0; d < headDim; d++)
            {
                float sum = 0f;
                for (int j = 0; j < seqK; j++)
                    sum += scores[i * seqK + j] * v[j * headDim + d];
                output[i * headDim + d] = sum;
            }

        return output;
    }

    private static float[] CreateRandomArray(int length, Random rng)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }
}
