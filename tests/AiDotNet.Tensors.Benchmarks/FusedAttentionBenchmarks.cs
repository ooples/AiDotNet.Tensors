#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #889: one attention training step (forward, then the tape's backward for Q, K and V) through the materialising
/// ScaledDotProductAttention, which returns and keeps the [B, H, S, S] attention weights, versus the weightless fused
/// FlashAttention-2 overload, at representative sequence lengths. [MemoryDiagnoser] reports the bytes each allocates.
///
/// Run:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --fused-attention
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class FusedAttentionBenchmarks
{
    [Params(256, 1024, 2048)]
    public int Sequence { get; set; }

    [Params(false, true)]
    public bool Causal { get; set; }

    private const int Batch = 1, Heads = 8, HeadDim = 64;
    private CpuEngine _engine = null!;
    private Tensor<float> _q = null!, _k = null!, _v = null!, _w = null!;
    private Tensor<bool>? _causalMask;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();
        var rng = new Random(889);
        Tensor<float> R(params int[] shape)
        {
            var t = new Tensor<float>(shape);
            for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
            return t;
        }
        _q = R(Batch, Heads, Sequence, HeadDim);
        _k = R(Batch, Heads, Sequence, HeadDim);
        _v = R(Batch, Heads, Sequence, HeadDim);
        _w = R(Batch, Heads, Sequence, HeadDim);
        if (Causal)
        {
            var allowed = new bool[Batch * Heads * Sequence * Sequence];
            for (int bh = 0, at = 0; bh < Batch * Heads; bh++)
                for (int i = 0; i < Sequence; i++)
                    for (int j = 0; j < Sequence; j++, at++) allowed[at] = j <= i;
            _causalMask = new Tensor<bool>(allowed, new[] { Batch, Heads, Sequence, Sequence });
        }
    }

    private float Step(bool fused)
    {
        using var tape = new GradientTape<float>();
        var o = fused
            ? _engine.ScaledDotProductAttention(_q, _k, _v, null, null, Causal)
            : _engine.ScaledDotProductAttention(_q, _k, _v, _causalMask, null, out _);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(o, _w), new[] { 0, 1, 2, 3 }, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { _q, _k, _v });
        return grads[_q][0] + grads[_k][0] + grads[_v][0];
    }

    [Benchmark(Baseline = true)]
    public float Materialised() => Step(fused: false);

    [Benchmark]
    public float Fused() => Step(fused: true);
}
#endif
