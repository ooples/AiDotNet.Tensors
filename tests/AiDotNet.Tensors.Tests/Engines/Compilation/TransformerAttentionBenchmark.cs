using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// End-to-end Transformer attention benchmark at realistic sequence lengths.
/// Multi-head attention: Q@K^T → scale → softmax → @V with head splitting.
/// </summary>
[Trait("Category", "Benchmark")]
public class TransformerAttentionBenchmark
{
    private readonly ITestOutputHelper _output;
    public TransformerAttentionBenchmark(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(128, 64, 8, 20)]    // Short context, 8 heads
    [InlineData(512, 64, 8, 10)]    // Medium context
    [InlineData(2048, 64, 8, 3)]    // Long context
    public void MultiHeadAttention_FlashVsNaive(int seqLen, int headDim, int numHeads, int iters)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        int modelDim = headDim * numHeads;
        int warmup = 2;

        // Generate Q, K, V for all heads (flattened)
        var q = CreateRandomArray(seqLen * modelDim, rng);
        var k = CreateRandomArray(seqLen * modelDim, rng);
        var v = CreateRandomArray(seqLen * modelDim, rng);
        float scale = 1f / MathF.Sqrt(headDim);

        // Flash Attention per head
        var flashOutput = new float[seqLen * modelDim];
        double flashMs = MeasureMultiHead(() =>
        {
            for (int h = 0; h < numHeads; h++)
            {
                var qHead = ExtractHead(q, seqLen, headDim, numHeads, h);
                var kHead = ExtractHead(k, seqLen, headDim, numHeads, h);
                var vHead = ExtractHead(v, seqLen, headDim, numHeads, h);
                var outHead = new float[seqLen * headDim];
                FusedAttention.FlashAttentionForward(qHead, kHead, vHead, outHead,
                    seqLen, seqLen, headDim, scale);
                InsertHead(flashOutput, outHead, seqLen, headDim, numHeads, h);
            }
        }, warmup, iters);

        // Naive attention per head
        double naiveMs = MeasureMultiHead(() =>
        {
            for (int h = 0; h < numHeads; h++)
            {
                var qHead = ExtractHead(q, seqLen, headDim, numHeads, h);
                var kHead = ExtractHead(k, seqLen, headDim, numHeads, h);
                var vHead = ExtractHead(v, seqLen, headDim, numHeads, h);
                NaiveAttention(qHead, kHead, vHead, seqLen, headDim, scale);
            }
        }, warmup, iters);

        double speedup = naiveMs / flashMs;
        long naiveMemPerHead = (long)seqLen * seqLen * 4; // O(N^2) score matrix
        long flashMemPerHead = (long)seqLen * headDim * 4; // O(N) output only

        _output.WriteLine($"Seq={seqLen,5} Heads={numHeads} HeadDim={headDim} | " +
            $"Flash: {flashMs,8:F2}ms  Naive: {naiveMs,8:F2}ms  Speedup: {speedup:F2}x  " +
            $"NaiveMem/head: {naiveMemPerHead / 1024}KB  FlashMem/head: {flashMemPerHead / 1024}KB");
    }

    private static double MeasureMultiHead(Action action, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static float[] ExtractHead(float[] data, int seqLen, int headDim, int numHeads, int headIdx)
    {
        int modelDim = headDim * numHeads;
        var head = new float[seqLen * headDim];
        for (int s = 0; s < seqLen; s++)
            Array.Copy(data, s * modelDim + headIdx * headDim, head, s * headDim, headDim);
        return head;
    }

    private static void InsertHead(float[] output, float[] head, int seqLen, int headDim, int numHeads, int headIdx)
    {
        int modelDim = headDim * numHeads;
        for (int s = 0; s < seqLen; s++)
            Array.Copy(head, s * headDim, output, s * modelDim + headIdx * headDim, headDim);
    }

    private static float[] NaiveAttention(float[] q, float[] k, float[] v,
        int seqLen, int headDim, float scale)
    {
        var scores = new float[seqLen * seqLen];
        var output = new float[seqLen * headDim];

        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < seqLen; j++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                    dot += q[i * headDim + d] * k[j * headDim + d];
                scores[i * seqLen + j] = dot * scale;
            }

        for (int i = 0; i < seqLen; i++)
        {
            float maxVal = float.NegativeInfinity;
            for (int j = 0; j < seqLen; j++)
                if (scores[i * seqLen + j] > maxVal) maxVal = scores[i * seqLen + j];
            float sumExp = 0f;
            for (int j = 0; j < seqLen; j++)
            {
                scores[i * seqLen + j] = MathF.Exp(scores[i * seqLen + j] - maxVal);
                sumExp += scores[i * seqLen + j];
            }
            for (int j = 0; j < seqLen; j++)
                scores[i * seqLen + j] /= sumExp;
        }

        for (int i = 0; i < seqLen; i++)
            for (int d = 0; d < headDim; d++)
            {
                float sum = 0f;
                for (int j = 0; j < seqLen; j++)
                    sum += scores[i * seqLen + j] * v[j * headDim + d];
                output[i * headDim + d] = sum;
            }

        return output;
    }

    private static float[] CreateRandomArray(int length, Random rng)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }
}
