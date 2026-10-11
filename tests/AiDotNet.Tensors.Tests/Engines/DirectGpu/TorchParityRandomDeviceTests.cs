using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1112 tier 7: the random ops sample on the device with a counter-based Philox generator. The draws cannot equal the
/// CPU's System.Random stream, so each op is checked for what it promises: its distribution's moments over many draws,
/// the same tensor for the same seed (and a different one for another seed), its structural invariants (support,
/// permutations, masks), the CPU's exceptions for invalid parameters, device launches, and no host fallback.
/// </summary>
public abstract class TorchParityRandomDeviceTestsBase
{
    private const int N = 200_000;
    private readonly IDirectGpuTensorEngineFixture _fixture;
    protected TorchParityRandomDeviceTestsBase(IDirectGpuTensorEngineFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "This backend has no PyTorch-parity device kernels.");
        return gpu;
    }

    private static long TotalFallbacks() => GpuLaunchProbe.Fallbacks.Sum(entry =>
    {
        int x = entry.IndexOf("x ", StringComparison.Ordinal);
        return x > 0 && long.TryParse(entry.Substring(0, x), out long count) ? count : 0L;
    });

    private static Tensor<float> OnDevice(Func<Tensor<float>> run)
    {
        long launches = GpuLaunchProbe.Count, fallbacks = TotalFallbacks();
        var r = run();
        _ = r[0];
        Assert.True(GpuLaunchProbe.Count > launches, "no device work launched");
        Assert.Equal(fallbacks, TotalFallbacks());
        return r;
    }

    private static double[] Values(Tensor<float> t) => Enumerable.Range(0, t.Length).Select(i => (double)t[i]).ToArray();

    private static (double Mean, double Variance) Moments(double[] v)
    {
        double mean = v.Average();
        return (mean, v.Sum(x => (x - mean) * (x - mean)) / (v.Length - 1));
    }

    // |sample mean - mu| within 5 standard errors.
    private static void AssertMean(string what, double[] v, double mu, double variance)
        => Assert.True(Math.Abs(v.Average() - mu) <= 5 * Math.Sqrt(variance / v.Length), $"{what}: mean {v.Average()} vs {mu}");

    private static void AssertSeeded(Func<int, Tensor<float>> draw)
    {
        var a = Values(draw(7));
        var b = Values(draw(7));
        var c = Values(draw(8));
        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
    }

    public static TheoryData<string> Distributions() => new()
    {
        "uniform", "cauchy", "exponential", "geometric", "logNormal", "randLike", "randnLike", "randint", "randintLike", "normal",
    };

    [SkippableTheory]
    [MemberData(nameof(Distributions))]
    public void ShapeDistributions_HaveTheirMoments_AndAreSeeded(string op)
    {
        var gpu = GpuOrSkip();
        var like = new Tensor<float>(new[] { 400, 500 });
        var mean = new Tensor<float>(new[] { N });
        var std = new Tensor<float>(new[] { N });
        for (int i = 0; i < N; i++) { mean[i] = 1.5f; std[i] = 0.5f; }
        Func<int, Tensor<float>> draw = op switch
        {
            "uniform" => s => gpu.TensorUniform<float>(new[] { N }, 2, 5, s),
            "cauchy" => s => gpu.TensorCauchy<float>(new[] { N }, 1, 2, s),
            "exponential" => s => gpu.TensorExponential<float>(new[] { N }, 2, s),
            "geometric" => s => gpu.TensorGeometric<float>(new[] { N }, 0.3, s),
            "logNormal" => s => gpu.TensorLogNormal<float>(new[] { N }, 0.1, 0.5, s),
            "randLike" => s => gpu.TensorRandLike(like, s),
            "randnLike" => s => gpu.TensorRandnLike(like, s),
            "randint" => s => gpu.TensorRandint<float>(-3, 7, new[] { N }, s),
            "randintLike" => s => gpu.TensorRandintLike(like, 10, 13, s),
            _ => s => gpu.TensorNormal(mean, std, s),
        };
        var v = Values(OnDevice(() => draw(1)));
        switch (op)
        {
            case "uniform": Assert.All(v, x => Assert.InRange(x, 2, 5)); AssertMean(op, v, 3.5, 9.0 / 12); break;
            case "cauchy":
                var sorted = v.OrderBy(x => x).ToArray();
                Assert.InRange(sorted[N / 2], 0.95, 1.05);                          // median
                Assert.InRange(sorted[3 * N / 4] - sorted[N / 4], 3.9, 4.1);        // IQR = 2 sigma
                break;
            case "exponential": Assert.All(v, x => Assert.True(x >= 0)); AssertMean(op, v, 0.5, 0.25); break;
            case "geometric":
                Assert.All(v, x => Assert.True(x >= 1 && x == Math.Floor(x)));
                AssertMean(op, v, 1 / 0.3, 0.7 / (0.3 * 0.3));
                break;
            case "logNormal":
                Assert.All(v, x => Assert.True(x > 0));
                AssertMean(op, v, Math.Exp(0.1 + 0.125), (Math.Exp(0.25) - 1) * Math.Exp(0.2 + 0.25));
                break;
            case "randLike": Assert.All(v, x => Assert.InRange(x, 0, 1)); AssertMean(op, v, 0.5, 1.0 / 12); break;
            case "randnLike":
                AssertMean(op, v, 0, 1);
                Assert.InRange(Moments(v).Variance, 0.98, 1.02);
                break;
            case "randint":
                Assert.All(v, x => Assert.True(x >= -3 && x <= 6 && x == Math.Floor(x)));
                Assert.Equal(10, v.Distinct().Count());
                AssertMean(op, v, 1.5, (100 - 1) / 12.0);
                break;
            case "randintLike": Assert.All(v, x => Assert.True(x >= 10 && x <= 12 && x == Math.Floor(x))); break;
            default:
                AssertMean(op, v, 1.5, 0.25);
                Assert.InRange(Moments(v).Variance, 0.245, 0.255);
                break;
        }
        AssertSeeded(draw);
    }

    [SkippableFact]
    public void Bernoulli_PoissonAndBinomial_FollowTheirParameters()
    {
        var gpu = GpuOrSkip();
        var p = new Tensor<float>(new[] { N });
        var rates = new Tensor<float>(new[] { N });
        var counts = new Tensor<float>(new[] { N });
        var probs = new Tensor<float>(new[] { N });
        for (int i = 0; i < N; i++)
        {
            p[i] = i % 3 == 0 ? 0f : i % 3 == 1 ? 1f : 0.3f;
            rates[i] = i % 3 == 0 ? 0f : i % 3 == 1 ? 3f : 25f;           // zero, Knuth, PTRS
            counts[i] = i % 4 == 0 ? 0f : i % 4 == 1 ? 5f : 100f;          // zero, waiting times, BTRS (n q >= 10)
            probs[i] = i % 4 == 3 ? 1f : i % 4 == 1 ? 0.3f : 0.4f;
        }
        var b = Values(OnDevice(() => gpu.TensorBernoulli(p, 3)));
        Assert.All(Enumerable.Range(0, N).Where(i => i % 3 == 0), i => Assert.Equal(0.0, b[i]));
        Assert.All(Enumerable.Range(0, N).Where(i => i % 3 == 1), i => Assert.Equal(1.0, b[i]));
        AssertMean("bernoulli", Enumerable.Range(0, N).Where(i => i % 3 == 2).Select(i => b[i]).ToArray(), 0.3, 0.21);

        var poisson = Values(OnDevice(() => gpu.TensorPoisson(rates, 4)));
        Assert.All(poisson, x => Assert.True(x >= 0 && x == Math.Floor(x)));
        Assert.All(Enumerable.Range(0, N).Where(i => i % 3 == 0), i => Assert.Equal(0.0, poisson[i]));
        foreach (var (residue, rate) in new[] { (1, 3.0), (2, 25.0) })
        {
            var group = Enumerable.Range(0, N).Where(i => i % 3 == residue).Select(i => poisson[i]).ToArray();
            AssertMean($"poisson {rate}", group, rate, rate);
            Assert.InRange(Moments(group).Variance, rate * 0.95, rate * 1.05);
        }

        var binomial = Values(OnDevice(() => gpu.TensorBinomial(counts, probs, 5)));
        for (int i = 0; i < N; i++)
        {
            Assert.True(binomial[i] >= 0 && binomial[i] <= counts[i] && binomial[i] == Math.Floor(binomial[i]));
            if (i % 4 == 0) Assert.Equal(0.0, binomial[i]);
            if (i % 4 == 3) Assert.Equal(100.0, binomial[i]);
        }
        AssertMean("binomial(5, .3)", Enumerable.Range(0, N).Where(i => i % 4 == 1).Select(i => binomial[i]).ToArray(), 1.5, 1.05);
        AssertMean("binomial(100, .4)", Enumerable.Range(0, N).Where(i => i % 4 == 2).Select(i => binomial[i]).ToArray(), 40, 24);

        AssertSeeded(s => gpu.TensorPoisson(rates, s));
        AssertSeeded(s => gpu.TensorBinomial(counts, probs, s));
    }

    // For p below ~1e-16, log(1 - p) rounds to 0: the binomial waiting-time walk then never advanced (the CPU looped
    // forever) and geometric draws were infinite. Both engines now use log1p(-p) and a bounded walk.
    [SkippableFact]
    public void TinyProbabilities_GiveFiniteDraws_OnBothEngines()
    {
        var gpu = GpuOrSkip();
        var counts = new Tensor<float>(new[] { 1000 });
        var probs = new Tensor<float>(new[] { 1000 });
        for (int i = 0; i < counts.Length; i++) { counts[i] = 5; probs[i] = 1e-30f; }
        foreach (IEngine e in new IEngine[] { new CpuEngine(), gpu })
        {
            var b = Values(e.TensorBinomial(counts, probs, 3));
            Assert.All(b, x => Assert.Equal(0.0, x));
            var g = Values(e.TensorGeometric<float>(new[] { 1000 }, 1e-17, 3));
            Assert.All(g, x => Assert.True(x >= 1 && !double.IsInfinity(x) && !double.IsNaN(x), $"geometric draw {x}"));
            // Mean 1e17: a draw below 1e10 has probability 1e-7. log(1 - p) == 0 collapsed every draw to 1.
            Assert.True(g.Count(x => x > 1e10) > 990, $"geometric(1e-17) draws too small: median {g.OrderBy(x => x).ElementAt(500)}");
        }
    }

    [SkippableFact]
    public void InvalidParameters_ThrowTheCpusExceptions()
    {
        var gpu = GpuOrSkip();
        var bad = new Tensor<float>(new[] { 1000 });
        for (int i = 0; i < bad.Length; i++) bad[i] = 0.5f;
        bad[700] = 1.5f;
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorBernoulli(bad, 1));
        bad[700] = -2f;
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorPoisson(bad, 1));
        var counts = new Tensor<float>(new[] { 1000 });
        for (int i = 0; i < counts.Length; i++) counts[i] = 4;
        counts[300] = 2.5f;
        var e = Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorBinomial(counts, bad.Clone(), 1));
        Assert.Equal("count", e.ParamName);
        counts[300] = 4;
        Assert.Equal("p", Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorBinomial(counts, bad, 1)).ParamName);

        var weights = new Tensor<float>(new[] { 3, 4 });
        for (int i = 0; i < weights.Length; i++) weights[i] = 1;
        weights[1, 2] = float.NaN;
        Assert.Throws<ArgumentException>(() => gpu.TensorMultinomial(weights, 2, true, 1));
        weights[1, 2] = 0; weights[1, 0] = 0; weights[1, 1] = 0;   // row 1 has one positive weight
        var exhausted = Assert.Throws<ArgumentException>(() => gpu.TensorMultinomial(weights, 2, false, 1));
        Assert.Contains("row 1", exhausted.Message);
    }

    [SkippableFact]
    public void Multinomial_Randperm_AndRrelu_KeepTheirInvariants()
    {
        var gpu = GpuOrSkip();
        const int rows = 2000, categories = 6;
        var weights = new Tensor<float>(new[] { rows, categories });
        float[] w = { 1, 0, 2, 3, 0, 4 };
        for (int r = 0; r < rows; r++) for (int c = 0; c < categories; c++) weights[r, c] = w[c];
        var without = OnDevice(() => gpu.TensorMultinomial(weights, 4, false, 9));
        for (int r = 0; r < rows; r++)
        {
            var picks = Enumerable.Range(0, 4).Select(k => (int)without[r, k]).ToArray();
            Assert.Equal(4, picks.Distinct().Count());
            Assert.All(picks, c => Assert.True(w[c] > 0));
        }
        var with = OnDevice(() => gpu.TensorMultinomial(weights, 50, true, 9));
        var counts = new double[categories];
        for (int i = 0; i < with.Length; i++) counts[(int)with[i]]++;
        for (int c = 0; c < categories; c++)
        {
            double expected = w[c] / 10.0 * with.Length;
            Assert.True(Math.Abs(counts[c] - expected) <= 5 * Math.Sqrt(expected + 1), $"category {c}: {counts[c]} vs {expected}");
        }
        AssertSeeded(s => gpu.TensorMultinomial(weights, 4, false, s));

        var perm = Values(OnDevice(() => gpu.TensorRandperm<float>(5000, 11)));
        Assert.Equal(Enumerable.Range(0, 5000).Select(i => (double)i), perm.OrderBy(x => x));
        AssertSeeded(s => gpu.TensorRandperm<float>(5000, s));

        var x = new Tensor<float>(new[] { N });
        var rng = new Random(2);
        for (int i = 0; i < N; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        var cpu = new CpuEngine();
        var evalCpu = Values(cpu.TensorRrelu(x, 0.1, 0.3));
        var evalGpu = Values(OnDevice(() => gpu.TensorRrelu(x, 0.1, 0.3)));
        for (int i = 0; i < N; i++) Assert.True(Math.Abs(evalCpu[i] - evalGpu[i]) <= 1e-6 * Math.Max(1, Math.Abs(evalCpu[i])));
        var train = Values(OnDevice(() => gpu.TensorRrelu(x, 0.1, 0.3, training: true, seed: 4)));
        var slopes = Enumerable.Range(0, N).Where(i => x[i] < 0).Select(i => train[i] / x[i]).ToArray();
        for (int i = 0; i < N; i++) if (x[i] > 0) Assert.Equal((double)x[i], train[i]);
        Assert.All(slopes, s => Assert.InRange(s, 0.1 - 1e-6, 0.3 + 1e-6));
        AssertMean("rrelu slope", slopes, 0.2, 0.04 / 12);
        AssertSeeded(s => gpu.TensorRrelu(x, 0.1, 0.3, training: true, seed: s));
    }

    [SkippableFact]
    public void Dropouts_KeepOrReplaceEachUnit_WithTheCpusConstants()
    {
        var gpu = GpuOrSkip();
        var x = new Tensor<float>(new[] { 50, 40, 10, 10 });
        var rng = new Random(5);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 + 0.5);
        Assert.Same(x, gpu.TensorAlphaDropout(x, 0.3, training: false));
        Assert.Same(x, gpu.TensorChannelDropout(x, 0.3, training: false, spatialDims: 2));

        const double p = 0.3, sat = -1.7580993408473766;
        double a = 1 / Math.Sqrt((1 - p) * (1 + p * sat * sat)), b = -a * sat * p;
        var alpha = OnDevice(() => gpu.TensorAlphaDropout(x, p, training: true, seed: 6));
        int dropped = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double kept = a * x[i] + b, gone = a * sat + b;
            bool isKept = Math.Abs(alpha[i] - kept) <= 1e-5 * Math.Max(1, Math.Abs(kept));
            Assert.True(isKept || Math.Abs(alpha[i] - gone) <= 1e-5, $"alpha[{i}] = {alpha[i]}");
            if (!isKept) dropped++;
        }
        Assert.InRange(dropped / (double)x.Length, 0.29, 0.31);

        var channel = OnDevice(() => gpu.TensorChannelDropout(x, p, training: true, spatialDims: 2, seed: 6));
        var feature = OnDevice(() => gpu.TensorFeatureAlphaDropout(x, p, training: true, seed: 6));
        int channelsDropped = 0;
        for (int c = 0; c < 50 * 40; c++)
        {
            bool first = channel[c * 100] == 0;
            if (first) channelsDropped++;
            bool featureDropped = Math.Abs(feature[c * 100] - (a * sat + b)) <= 1e-5;
            for (int k = 0; k < 100; k++)
            {
                int i = c * 100 + k;
                Assert.Equal(first, channel[i] == 0);
                if (!first) Assert.True(Math.Abs(channel[i] - x[i] / (1 - p)) <= 1e-5 * Math.Max(1.0, Math.Abs(channel[i])));
                Assert.Equal(featureDropped, Math.Abs(feature[i] - (a * sat + b)) <= 1e-5);
            }
        }
        Assert.InRange(channelsDropped / 2000.0, 0.25, 0.35);
        AssertSeeded(s => gpu.TensorChannelDropout(x, p, training: true, spatialDims: 2, seed: s));
    }
}

/// <summary>The cases above on the machine's default DirectGpu backend.</summary>
[Collection("VulkanGlobalState")]
public class TorchParityRandomDeviceTests : TorchParityRandomDeviceTestsBase, IClassFixture<DirectGpuTensorEngineTestFixture>
{
    public TorchParityRandomDeviceTests(DirectGpuTensorEngineTestFixture fixture) : base(fixture) { }
}

/// <summary>The cases above on the Vulkan backend (#1127).</summary>
[Collection("VulkanGlobalState")]
public class TorchParityRandomVulkanDeviceTests : TorchParityRandomDeviceTestsBase, IClassFixture<VulkanTensorEngineTestFixture>
{
    public TorchParityRandomVulkanDeviceTests(VulkanTensorEngineTestFixture fixture) : base(fixture) { }
}
