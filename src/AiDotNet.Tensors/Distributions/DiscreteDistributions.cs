using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>Bernoulli distribution.</summary>
public sealed class BernoulliDistribution : DistributionBase
{
    private readonly float[] _probs;
    /// <summary>Per-batch success probability. Returns a defensive copy so callers can't
    /// mutate the validated parameters after construction.</summary>
    public float[] Probs => (float[])_probs.Clone();
    /// <inheritdoc />
    public override int BatchSize => _probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => new IntegerIntervalConstraint(0, 1);
    /// <summary>Build a Bernoulli from per-batch probabilities.</summary>
    public BernoulliDistribution(float[] probs)
    {
        if (probs == null) throw new ArgumentNullException(nameof(probs));
        for (int i = 0; i < probs.Length; i++)
        {
            float p = probs[i];
            if (float.IsNaN(p) || float.IsInfinity(p) || p < 0f || p > 1f)
                throw new ArgumentException("probs must be finite and in [0, 1].");
        }
        _probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) x[i] = rng.NextDouble() < _probs[i] ? 1f : 0f;
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float p = _probs[i]; float v = value[i];
            if (v == 1f) lp[i] = p > 0f ? MathF.Log(p) : float.NegativeInfinity;
            else if (v == 0f) lp[i] = p < 1f ? MathF.Log(1f - p) : float.NegativeInfinity;
            else lp[i] = float.NegativeInfinity;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float p = _probs[i];
            float a = p > 0f ? -p * MathF.Log(p) : 0f;
            float b = p < 1f ? -(1f - p) * MathF.Log(1f - p) : 0f;
            h[i] = a + b;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => (float[])_probs.Clone();
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = _probs[i] * (1f - _probs[i]); return v; } }
    /// <summary>Direct read access to the validated probs array — used by KL helpers
    /// to avoid the per-access Clone() overhead of the public Probs getter.</summary>
    internal float[] ProbsRaw => _probs;
}

/// <summary>Binomial distribution: number of successes in <c>totalCount</c> independent Bernoulli trials.</summary>
public sealed class BinomialDistribution : DistributionBase
{
    private readonly int[] _totalCount;
    private readonly float[] _probs;
    /// <summary>Number of trials (per batch). Returns a defensive copy.</summary>
    public int[] TotalCount => (int[])_totalCount.Clone();
    /// <summary>Per-batch success probability. Returns a defensive copy.</summary>
    public float[] Probs => (float[])_probs.Clone();
    /// <inheritdoc />
    public override int BatchSize => _probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support
    {
        get
        {
            int max = 0;
            for (int i = 0; i < _totalCount.Length; i++) if (_totalCount[i] > max) max = _totalCount[i];
            return new IntegerIntervalConstraint(0, max);
        }
    }
    /// <summary>Build a binomial from per-batch totalCount and probs.</summary>
    public BinomialDistribution(int[] totalCount, float[] probs)
    {
        if (totalCount == null) throw new ArgumentNullException(nameof(totalCount));
        if (probs == null) throw new ArgumentNullException(nameof(probs));
        if (totalCount.Length != probs.Length) throw new ArgumentException();
        for (int i = 0; i < totalCount.Length; i++)
        {
            if (totalCount[i] < 0) throw new ArgumentException("totalCount >= 0.");
            float p = probs[i];
            if (float.IsNaN(p) || float.IsInfinity(p) || p < 0f || p > 1f)
                throw new ArgumentException("probs must be finite and in [0, 1].");
        }
        _totalCount = (int[])totalCount.Clone(); _probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            int k = 0;
            for (int t = 0; t < _totalCount[i]; t++) if (rng.NextDouble() < _probs[i]) k++;
            x[i] = k;
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            int n = _totalCount[i]; float p = _probs[i]; float k = value[i];
            if (k < 0f || k > n || k != MathF.Floor(k)) { lp[i] = float.NegativeInfinity; continue; }
            if (p == 0f) { lp[i] = k == 0f ? 0f : float.NegativeInfinity; continue; }
            if (p == 1f) { lp[i] = k == n  ? 0f : float.NegativeInfinity; continue; }
            float logBinom = SpecialFunctions.Lgamma(n + 1f) - SpecialFunctions.Lgamma(k + 1f) - SpecialFunctions.Lgamma(n - k + 1f);
            lp[i] = logBinom + k * MathF.Log(p) + (n - k) * MathF.Log(1f - p);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            int n = _totalCount[i];
            float p = _probs[i];
            if (p == 0f || p == 1f) { h[i] = 0f; continue; }
            double s = 0;
            for (int k = 0; k <= n; k++)
            {
                float logBinom = SpecialFunctions.Lgamma(n + 1f) - SpecialFunctions.Lgamma(k + 1f) - SpecialFunctions.Lgamma(n - k + 1f);
                float lp2 = logBinom + k * MathF.Log(p) + (n - k) * MathF.Log(1f - p);
                if (!float.IsNegativeInfinity(lp2)) s += MathF.Exp(lp2) * lp2;
            }
            h[i] = (float)(-s);
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = _totalCount[i] * _probs[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = _totalCount[i] * _probs[i] * (1f - _probs[i]); return v; } }
}

/// <summary>Categorical distribution over <c>K</c> classes per batch element.</summary>
public sealed class CategoricalDistribution : DistributionBase
{
    private readonly float[] _probs;
    /// <summary>Per-batch probability vectors. Layout: <c>[batch, K]</c> row-major.
    /// Returns a defensive copy so callers can't mutate validated parameters.</summary>
    public float[] Probs => (float[])_probs.Clone();
    /// <summary>Number of categories.</summary>
    public int K { get; }
    /// <inheritdoc />
    public override int BatchSize => _probs.Length / K;
    /// <inheritdoc />
    public override int EventSize => 1; // sample is a single category index
    /// <inheritdoc />
    public override IConstraint Support => new IntegerIntervalConstraint(0, K - 1);
    /// <summary>Build a categorical from per-batch prob vectors.</summary>
    public CategoricalDistribution(float[] probs, int k)
    {
        if (probs == null) throw new ArgumentNullException(nameof(probs));
        if (k <= 0) throw new ArgumentException("K > 0.");
        if (probs.Length % k != 0) throw new ArgumentException("probs.Length must be a multiple of K.");
        int batch = probs.Length / k;
        for (int b = 0; b < batch; b++)
        {
            float sum = 0f;
            for (int i = 0; i < k; i++)
            {
                float p = probs[b * k + i];
                if (float.IsNaN(p) || float.IsInfinity(p) || p < 0f)
                    throw new ArgumentException($"probs[{b}, {i}] must be finite and ≥ 0.");
                sum += p;
            }
            if (float.IsNaN(sum) || float.IsInfinity(sum) || MathF.Abs(sum - 1f) > 1e-3f)
                throw new ArgumentException($"probs row {b} must sum to 1 (got {sum}).");
        }
        _probs = (float[])probs.Clone(); K = k;
    }
    /// <summary>Direct read access to the validated probs array — used by KL helpers
    /// to avoid the per-access Clone() overhead of the public Probs getter.</summary>
    internal float[] ProbsRaw => _probs;
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        int batch = BatchSize;
        var x = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            float u = (float)rng.NextDouble();
            float cum = 0f;
            int chosen = K - 1;
            for (int i = 0; i < K; i++)
            {
                cum += _probs[b * K + i];
                if (u <= cum) { chosen = i; break; }
            }
            x[b] = chosen;
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        int batch = BatchSize;
        var lp = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            float v = value[b];
            if (v < 0f || v >= K || v != MathF.Floor(v)) { lp[b] = float.NegativeInfinity; continue; }
            int k = (int)v;
            float p = _probs[b * K + k];
            lp[b] = p > 0f ? MathF.Log(p) : float.NegativeInfinity;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        int batch = BatchSize;
        var h = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            float s = 0f;
            for (int i = 0; i < K; i++)
            {
                float p = _probs[b * K + i];
                if (p > 0f) s -= p * MathF.Log(p);
            }
            h[b] = s;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            int batch = BatchSize;
            var m = new float[batch];
            for (int b = 0; b < batch; b++)
            {
                float s = 0f;
                for (int i = 0; i < K; i++) s += i * _probs[b * K + i];
                m[b] = s;
            }
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            int batch = BatchSize;
            var v = new float[batch];
            var mean = Mean;
            for (int b = 0; b < batch; b++)
            {
                float s = 0f;
                for (int i = 0; i < K; i++)
                {
                    float diff = i - mean[b];
                    s += _probs[b * K + i] * diff * diff;
                }
                v[b] = s;
            }
            return v;
        }
    }
}

/// <summary>One-hot categorical: same parameters as <see cref="CategoricalDistribution"/> but samples are length-K one-hot vectors.</summary>
public sealed class OneHotCategoricalDistribution : DistributionBase
{
    private readonly CategoricalDistribution _inner;
    /// <summary>Number of categories.</summary>
    public int K => _inner.K;
    /// <summary>Per-batch probability vectors.</summary>
    public float[] Probs => _inner.Probs;
    /// <inheritdoc />
    public override int BatchSize => _inner.BatchSize;
    /// <inheritdoc />
    public override int EventSize => K;
    /// <inheritdoc />
    /// <remarks>One-hot vectors only — <c>SimplexConstraint</c> is too broad because LogProb
    /// rejects non-binary entries and multi-hot vectors. <see cref="OneHotConstraint"/> matches
    /// the Sample/LogProb contract exactly.</remarks>
    public override IConstraint Support => new OneHotConstraint(K);
    /// <summary>Build a one-hot categorical from per-batch prob vectors.</summary>
    public OneHotCategoricalDistribution(float[] probs, int k) { _inner = new CategoricalDistribution(probs, k); }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var idx = _inner.Sample(rng);
        var x = new float[BatchSize * K];
        for (int b = 0; b < BatchSize; b++) x[b * K + (int)idx[b]] = 1f;
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        // Reduce the one-hot to an index, then call inner log-prob. The previous version
        // accepted vectors like [1, 1, 0] (multi-hot) by stopping at the first 1; we now
        // require exactly one entry to be 1 and the rest to be 0, returning -∞ otherwise.
        var idx = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            int chosen = -1;
            bool valid = true;
            for (int i = 0; i < K; i++)
            {
                float v = value[b * K + i];
                if (v == 1f)
                {
                    if (chosen >= 0) { valid = false; break; } // multi-hot
                    chosen = i;
                }
                else if (v != 0f)
                {
                    valid = false; break; // non-binary entry
                }
            }
            idx[b] = valid && chosen >= 0 ? chosen : -1;
        }
        return _inner.LogProb(idx);
    }
    /// <inheritdoc />
    public override float[] Entropy() => _inner.Entropy();
    /// <inheritdoc />
    public override float[] Mean => (float[])_inner.ProbsRaw.Clone();  // never expose internal storage
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var probs = _inner.ProbsRaw;  // borrow the inner's validated array — no per-iteration clone
            var v = new float[BatchSize * K];
            for (int b = 0; b < BatchSize; b++)
                for (int i = 0; i < K; i++)
                {
                    float p = probs[b * K + i];
                    v[b * K + i] = p * (1f - p);
                }
            return v;
        }
    }
}

/// <summary>Geometric distribution: number of failures before first success.</summary>
public sealed class GeometricDistribution : DistributionBase
{
    private readonly float[] _probs;
    /// <summary>Per-batch success probability. Returns a defensive copy.</summary>
    public float[] Probs => (float[])_probs.Clone();
    /// <inheritdoc />
    public override int BatchSize => _probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeIntegerConstraint.Instance;
    /// <summary>Build a geometric from per-batch probs.</summary>
    public GeometricDistribution(float[] probs)
    {
        if (probs == null) throw new ArgumentNullException(nameof(probs));
        for (int i = 0; i < probs.Length; i++)
        {
            float p = probs[i];
            if (float.IsNaN(p) || float.IsInfinity(p) || !(p > 0f && p <= 1f))
                throw new ArgumentException("probs must be finite and in (0, 1].");
        }
        _probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble();
            if (u < 1e-12) u = 1e-12;
            x[i] = MathF.Floor(MathF.Log((float)u) / MathF.Log(1f - _probs[i]));
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float k = value[i]; float p = _probs[i];
            if (k < 0 || k != MathF.Floor(k)) { lp[i] = float.NegativeInfinity; continue; }
            if (p == 1f) { lp[i] = k == 0f ? 0f : float.NegativeInfinity; continue; }
            lp[i] = k * MathF.Log(1f - p) + MathF.Log(p);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float p = _probs[i];
            if (p == 1f) { h[i] = 0f; continue; }
            h[i] = -(p * MathF.Log(p) + (1f - p) * MathF.Log(1f - p)) / p;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = (1f - _probs[i]) / _probs[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = (1f - _probs[i]) / (_probs[i] * _probs[i]); return v; } }
}

/// <summary>Poisson distribution: f(k) = e^-λ · λ^k / k!.</summary>
public sealed class PoissonDistribution : DistributionBase
{
    private readonly float[] _rate;
    /// <summary>Per-batch rate λ ≥ 0. Returns a defensive copy.</summary>
    public float[] Rate => (float[])_rate.Clone();
    /// <inheritdoc />
    public override int BatchSize => _rate.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeIntegerConstraint.Instance;
    /// <summary>Build a Poisson from per-batch rate.</summary>
    public PoissonDistribution(float[] rate)
    {
        if (rate == null) throw new ArgumentNullException(nameof(rate));
        for (int i = 0; i < rate.Length; i++)
        {
            float r = rate[i];
            if (float.IsNaN(r) || float.IsInfinity(r) || !(r >= 0f))
                throw new ArgumentException("rate must be finite and ≥ 0.");
        }
        _rate = (float[])rate.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) x[i] = SamplePoisson(rng, _rate[i]);
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float k = value[i];
            if (k < 0 || k != MathF.Floor(k)) { lp[i] = float.NegativeInfinity; continue; }
            if (_rate[i] == 0f) { lp[i] = k == 0f ? 0f : float.NegativeInfinity; continue; }
            lp[i] = k * MathF.Log(_rate[i]) - _rate[i] - SpecialFunctions.Lgamma(k + 1f);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float lambda = _rate[i];
            int kmax = (int)MathF.Ceiling(lambda + 10f * MathF.Sqrt(lambda + 1f));
            double s = 0;
            for (int k = 0; k <= kmax; k++)
            {
                double lp = k * Math.Log(lambda) - lambda - SpecialFunctions.Lgamma(k + 1f);
                if (lp > -50.0) s += Math.Exp(lp) * lp;
            }
            h[i] = (float)(-s);
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => (float[])_rate.Clone();
    /// <inheritdoc />
    public override float[] Variance => (float[])_rate.Clone();

    /// <summary>Knuth's Poisson sampler for small λ; transformed-rejection for large λ.</summary>
    internal static float SamplePoisson(Random rng, float lambda)
    {
        if (lambda < 30f)
        {
            double L = Math.Exp(-lambda);
            int count = 0; double p = 1;
            do { count++; p *= rng.NextDouble(); } while (p > L);
            return count - 1;
        }
        // Atkinson's transformed-rejection (textbook approach for large λ).
        double c = 0.767 - 3.36 / lambda;
        double beta = Math.PI / Math.Sqrt(3.0 * lambda);
        double alpha = beta * lambda;
        double k = Math.Log(c) - lambda - Math.Log(beta);
        while (true)
        {
            double u = rng.NextDouble();
            double x = (alpha - Math.Log((1 - u) / u)) / beta;
            int n = (int)Math.Floor(x + 0.5);
            if (n < 0) continue;
            double v = rng.NextDouble();
            double y = alpha - beta * x;
            double lhs = y + Math.Log(v / Math.Pow(1 + Math.Exp(y), 2));
            double rhs = k + n * Math.Log(lambda) - SpecialFunctions.Lgamma(n + 1f);
            if (lhs <= rhs) return n;
        }
    }
}

/// <summary>Negative binomial distribution: number of failures before <c>totalCount</c> successes.</summary>
public sealed class NegativeBinomialDistribution : DistributionBase
{
    private readonly float[] _totalCount;
    private readonly float[] _probs;
    /// <summary>Number of successes (per batch). Real-valued; PyTorch parity. Returns a defensive copy.</summary>
    public float[] TotalCount => (float[])_totalCount.Clone();
    /// <summary>Per-batch success probability. Returns a defensive copy.</summary>
    public float[] Probs => (float[])_probs.Clone();
    /// <inheritdoc />
    public override int BatchSize => _probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeIntegerConstraint.Instance;
    /// <summary>Build a negative binomial.</summary>
    public NegativeBinomialDistribution(float[] totalCount, float[] probs)
    {
        if (totalCount == null) throw new ArgumentNullException(nameof(totalCount));
        if (probs == null) throw new ArgumentNullException(nameof(probs));
        if (totalCount.Length != probs.Length) throw new ArgumentException();
        for (int i = 0; i < probs.Length; i++)
        {
            float r = totalCount[i]; float p = probs[i];
            if (float.IsNaN(r) || float.IsInfinity(r) || !(r > 0f))
                throw new ArgumentException("totalCount must be finite and > 0.");
            if (float.IsNaN(p) || float.IsInfinity(p) || !(p > 0f && p < 1f))
                throw new ArgumentException("probs must be finite and in (0, 1).");
        }
        _totalCount = (float[])totalCount.Clone(); _probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float scale = (1f - _probs[i]) / _probs[i];
            float lambda = GammaDistribution.MarsagliaTsang(rng, _totalCount[i]) * scale;
            x[i] = PoissonDistribution.SamplePoisson(rng, lambda);
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float k = value[i]; float r = _totalCount[i]; float p = _probs[i];
            if (k < 0f || k != MathF.Floor(k)) { lp[i] = float.NegativeInfinity; continue; }
            lp[i] = SpecialFunctions.Lgamma(r + k) - SpecialFunctions.Lgamma(k + 1f) - SpecialFunctions.Lgamma(r)
                  + r * MathF.Log(p) + k * MathF.Log(1f - p);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) h[i] = float.NaN;
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = _totalCount[i] * (1f - _probs[i]) / _probs[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = _totalCount[i] * (1f - _probs[i]) / (_probs[i] * _probs[i]); return v; } }
}

/// <summary>Multinomial distribution: <c>totalCount</c> draws from a categorical.</summary>
public sealed class MultinomialDistribution : DistributionBase
{
    private readonly int[] _totalCount;
    private readonly float[] _probs;
    /// <summary>Trials per batch element. Returns a defensive copy.</summary>
    public int[] TotalCount => (int[])_totalCount.Clone();
    /// <summary>Per-batch probability vectors, layout <c>[batch, K]</c>. Returns a defensive copy.</summary>
    public float[] Probs => (float[])_probs.Clone();
    /// <summary>Number of categories.</summary>
    public int K { get; }
    /// <inheritdoc />
    public override int BatchSize => _totalCount.Length;
    /// <inheritdoc />
    public override int EventSize => K;
    /// <inheritdoc />
    /// <remarks>Each row must be a non-negative integer count vector summing to the
    /// matching <see cref="TotalCount"/> entry — a plain non-negative constraint
    /// would over-advertise the support.</remarks>
    public override IConstraint Support => new IntegerSimplexConstraint(K, _totalCount);
    /// <summary>Build a Multinomial from totalCount + probs.</summary>
    public MultinomialDistribution(int[] totalCount, float[] probs, int k)
    {
        if (totalCount == null) throw new ArgumentNullException(nameof(totalCount));
        if (probs == null) throw new ArgumentNullException(nameof(probs));
        if (k <= 0) throw new ArgumentException("K > 0.");
        if (probs.Length != totalCount.Length * k) throw new ArgumentException("probs length mismatch.");
        int batch = totalCount.Length;
        for (int b = 0; b < batch; b++)
        {
            float sum = 0f;
            for (int i = 0; i < k; i++)
            {
                float p = probs[b * k + i];
                if (float.IsNaN(p) || float.IsInfinity(p) || p < 0f)
                    throw new ArgumentException($"probs[{b}, {i}] = {p} must be finite and ≥ 0.");
                sum += p;
            }
            if (float.IsNaN(sum) || float.IsInfinity(sum) || MathF.Abs(sum - 1f) > 1e-3f)
                throw new ArgumentException($"probs row {b} must sum to 1 (got {sum}).");
            if (totalCount[b] < 0) throw new ArgumentException($"totalCount[{b}] = {totalCount[b]} must be >= 0.");
        }
        _totalCount = (int[])totalCount.Clone(); _probs = (float[])probs.Clone(); K = k;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize * K];
        for (int b = 0; b < BatchSize; b++)
        {
            for (int t = 0; t < _totalCount[b]; t++)
            {
                float u = (float)rng.NextDouble();
                float cum = 0f; int chosen = K - 1;
                for (int i = 0; i < K; i++)
                {
                    cum += _probs[b * K + i];
                    if (u <= cum) { chosen = i; break; }
                }
                x[b * K + chosen]++;
            }
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            int n = _totalCount[b];
            long sumK = 0;
            bool valid = true;
            for (int i = 0; i < K; i++)
            {
                float k = value[b * K + i];
                if (k < 0f || k != MathF.Floor(k)) { valid = false; break; }
                sumK += (long)k;
            }
            if (!valid || sumK != n) { lp[b] = float.NegativeInfinity; continue; }

            float l = SpecialFunctions.Lgamma(n + 1f);
            for (int i = 0; i < K; i++)
            {
                float k = value[b * K + i];
                float p = _probs[b * K + i];
                l -= SpecialFunctions.Lgamma(k + 1f);
                if (p > 0f) l += k * MathF.Log(p);
                else if (k > 0f) l = float.NegativeInfinity;
            }
            lp[b] = l;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) h[i] = float.NaN;
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize * K];
            for (int b = 0; b < BatchSize; b++)
                for (int i = 0; i < K; i++)
                    m[b * K + i] = _totalCount[b] * _probs[b * K + i];
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize * K];
            for (int b = 0; b < BatchSize; b++)
                for (int i = 0; i < K; i++)
                {
                    float p = _probs[b * K + i];
                    v[b * K + i] = _totalCount[b] * p * (1f - p);
                }
            return v;
        }
    }
}
