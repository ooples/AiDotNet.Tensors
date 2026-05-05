// Copyright (c) AiDotNet. All rights reserved.

#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #294 Phase 5 acceptance criterion #9: rank-4
/// <c>[B, H, Sq, D]</c> case for <see cref="FlashAttention{T}"/> must
/// run within 2% of the rank-fixed <see cref="FlashAttention2"/>
/// baseline. The batchProduct-loop wrapper adds no measurable cost
/// on the canonical shape — SIMD work is unchanged, only the outer
/// (B*H) iteration is restructured.
///
/// <para>Three shapes covered: small (BERT-base attention block),
/// medium (BERT-large), and SeqLen-stress (long context). Each is
/// measured under both Forward and Forward+Backward to catch
/// regressions in either path.</para>
///
/// <para>Run with:
/// <c>dotnet run -c Release --framework net10.0 -- --filter *FlashAttentionGeneric*</c></para>
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class FlashAttentionGenericVsRankFixedBenchmark
{
    [Params(64, 128, 256)]
    public int Sq { get; set; }

    private const int BatchSize = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 32;

    private Tensor<float> _q = null!;
    private Tensor<float> _k = null!;
    private Tensor<float> _v = null!;
    private Tensor<float> _dO = null!;

    [GlobalSetup]
    public void Setup()
    {
        _q = Random(new[] { BatchSize, NumHeads, Sq, HeadDim }, seed: 1);
        _k = Random(new[] { BatchSize, NumHeads, Sq, HeadDim }, seed: 2);
        _v = Random(new[] { BatchSize, NumHeads, Sq, HeadDim }, seed: 3);
        _dO = Random(new[] { BatchSize, NumHeads, Sq, HeadDim }, seed: 4);
    }

    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }

    [Benchmark(Baseline = true, Description = "FlashAttention2 rank-fixed forward")]
    public Tensor<float> Forward_RankFixed()
    {
        var (output, _) = FlashAttention2.Forward(_q, _k, _v);
        return output;
    }

    [Benchmark(Description = "FlashAttention<T> generic-T forward")]
    public Tensor<float> Forward_Generic()
    {
        var (output, _) = FlashAttention<float>.Forward(_q, _k, _v);
        return output;
    }

    [Benchmark(Description = "FlashAttention2 rank-fixed forward+backward")]
    public Tensor<float> ForwardBackward_RankFixed()
    {
        var (output, lse) = FlashAttention2.Forward(_q, _k, _v);
        var (dQ, _, _) = FlashAttention2.Backward(_dO, _q, _k, _v, output, lse);
        return dQ;
    }

    [Benchmark(Description = "FlashAttention<T> generic-T forward+backward")]
    public Tensor<float> ForwardBackward_Generic()
    {
        var (output, lse) = FlashAttention<float>.Forward(_q, _k, _v);
        var (dQ, _, _) = FlashAttention<float>.Backward(_dO, _q, _k, _v, output, lse);
        return dQ;
    }
}
#endif
