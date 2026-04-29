#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Distributions;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Distribution-sampling throughput. Each benchmark draws <c>BatchSize</c> samples per
/// invocation. We report ops/sec (= samples/sec when EventSize == 1, samples · EventSize/sec
/// for multivariate).
///
/// The "BeatPyTorch" claim from issue #213 is "2× win on bulk categorical / normal /
/// multivariate-normal sampling". This file generates the AiDotNet-side numbers; the
/// PyTorch reference is gathered separately (TorchSharp's torch.Tensor.normal_, etc.)
/// and recorded in the PR description.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class DistributionSamplingBenchmarks
{
    /// <summary>Number of independent distributions to sample in parallel.</summary>
    [Params(1024, 16_384, 262_144)]
    public int BatchSize;

    private NormalDistribution _normal = null!;
    private CategoricalDistribution _cat = null!;
    private GammaDistribution _gamma = null!;
    private DirichletDistribution _dir = null!;
    private DiagonalMultivariateNormalDistribution _mvn = null!;
    private Random _rng = null!;

    /// <summary>Build the distributions once per param sweep.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var loc = new float[BatchSize]; var scale = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) { loc[i] = 0f; scale[i] = 1f; }
        _normal = new NormalDistribution(loc, scale);

        const int K = 8;
        var probs = new float[BatchSize * K];
        for (int b = 0; b < BatchSize; b++)
            for (int i = 0; i < K; i++) probs[b * K + i] = 1f / K;
        _cat = new CategoricalDistribution(probs, K);

        var conc = new float[BatchSize]; var rate = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) { conc[i] = 2f; rate[i] = 1f; }
        _gamma = new GammaDistribution(conc, rate);

        var dirConc = new float[BatchSize * K];
        for (int i = 0; i < dirConc.Length; i++) dirConc[i] = 1f;
        _dir = new DirichletDistribution(dirConc, K);

        const int D = 16;
        var mvnLoc = new float[BatchSize * D]; var mvnScale = new float[BatchSize * D];
        for (int i = 0; i < mvnScale.Length; i++) mvnScale[i] = 1f;
        _mvn = new DiagonalMultivariateNormalDistribution(mvnLoc, mvnScale, D);

        _rng = new Random(42);
    }

    /// <summary>Univariate Normal — the simplest hot path.</summary>
    [Benchmark]
    public float[] NormalSample() => _normal.Sample(_rng);

    /// <summary>Categorical sampling — branch-heavy cumulative-prob lookup.</summary>
    [Benchmark]
    public float[] CategoricalSample() => _cat.Sample(_rng);

    /// <summary>Gamma sampling via Marsaglia-Tsang rejection.</summary>
    [Benchmark]
    public float[] GammaSample() => _gamma.Sample(_rng);

    /// <summary>Dirichlet sampling — K Gammas + normalize.</summary>
    [Benchmark]
    public float[] DirichletSample() => _dir.Sample(_rng);

    /// <summary>Diagonal MVN sampling — D independent Gaussians per batch element.</summary>
    [Benchmark]
    public float[] DiagonalMvnSample() => _mvn.Sample(_rng);
}
#endif
