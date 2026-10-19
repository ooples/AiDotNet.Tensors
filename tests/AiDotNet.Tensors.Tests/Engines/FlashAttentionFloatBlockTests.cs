using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float FlashAttention runs in fixed 64-row query blocks per head with the softmax computed in place (#653): the
/// output and log-sum-exp statistics against a double-precision reference across several blocks and a partial one, with
/// and without a causal mask and an additive bias, and bit-identical results at every thread count.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]   // varies the process-wide MaxDegreeOfParallelism
public class FlashAttentionFloatBlockTests
{
    private const int Batch = 2, Heads = 3, SeqQ = 150, SeqK = 97, HeadDim = 24;

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        return t;
    }

    private static (double[] Output, double[] Stats) Reference(Tensor<float> q, Tensor<float> k, Tensor<float> v,
        Tensor<float>? bias, bool causal, double scale)
    {
        var output = new double[Batch * Heads * SeqQ * HeadDim];
        var stats = new double[Batch * Heads * SeqQ];
        var row = new double[SeqK];
        for (int b = 0; b < Batch; b++)
        for (int h = 0; h < Heads; h++)
        for (int i = 0; i < SeqQ; i++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < SeqK; j++)
            {
                double s = 0;
                for (int d = 0; d < HeadDim; d++) s += (double)q[b, h, i, d] * k[b, h, j, d];
                s *= scale;
                if (bias is not null) s += bias[b, h, i, j];
                if (causal && j > i) s = double.NegativeInfinity;
                row[j] = s;
                if (s > max) max = s;
            }
            double sum = 0;
            for (int j = 0; j < SeqK; j++) { row[j] = double.IsNegativeInfinity(row[j]) ? 0 : Math.Exp(row[j] - max); sum += row[j]; }
            stats[(b * Heads + h) * SeqQ + i] = max + Math.Log(sum);
            for (int d = 0; d < HeadDim; d++)
            {
                double acc = 0;
                for (int j = 0; j < SeqK; j++) acc += row[j] / sum * v[b, h, j, d];
                output[((b * Heads + h) * SeqQ + i) * HeadDim + d] = acc;
            }
        }
        return (output, stats);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void MatchesADoubleReference_AndIsIndependentOfThreadCount(bool causal, bool withBias)
    {
        var q = Random(1, Batch, Heads, SeqQ, HeadDim);
        var k = Random(2, Batch, Heads, SeqK, HeadDim);
        var v = Random(3, Batch, Heads, SeqK, HeadDim);
        var bias = withBias ? Random(4, Batch, Heads, SeqQ, SeqK) : null;
        double scale = 1.0 / Math.Sqrt(HeadDim);
        var engine = new CpuEngine();
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        Tensor<float> serialOut, parallelOut, serialStats, parallelStats;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            serialOut = engine.FlashAttention(q, k, v, scale, causal, out serialStats, bias);
            CpuParallelSettings.MaxDegreeOfParallelism = 7;
            parallelOut = engine.FlashAttention(q, k, v, scale, causal, out parallelStats, bias);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }

        for (int i = 0; i < serialOut.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(serialOut.GetFlat(i)) == BitConverter.SingleToInt32Bits(parallelOut.GetFlat(i)),
                $"output[{i}] differs between 1 and 7 threads");
        for (int i = 0; i < serialStats.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(serialStats.GetFlat(i)) == BitConverter.SingleToInt32Bits(parallelStats.GetFlat(i)),
                $"stats[{i}] differs between 1 and 7 threads");

        var (expectedOut, expectedStats) = Reference(q, k, v, bias, causal, scale);
        for (int i = 0; i < expectedOut.Length; i++)
            Assert.True(Math.Abs(expectedOut[i] - parallelOut.GetFlat(i)) <= 2e-5,
                $"output[{i}]: expected {expectedOut[i]:G9}, got {parallelOut.GetFlat(i):G9}");
        for (int i = 0; i < expectedStats.Length; i++)
            Assert.True(Math.Abs(expectedStats[i] - parallelStats.GetFlat(i)) <= 2e-5 * Math.Max(1, Math.Abs(expectedStats[i])),
                $"stats[{i}]: expected {expectedStats[i]:G9}, got {parallelStats.GetFlat(i):G9}");
    }
}
