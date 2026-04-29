using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>Bernoulli distribution.</summary>
public sealed class BernoulliDistribution : DistributionBase
{
    /// <summary>Per-batch success probability.</summary>
    public float[] Probs { get; }
    /// <inheritdoc />
    public override int BatchSize => Probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => new IntegerIntervalConstraint(0, 1);
    /// <summary>Build a Bernoulli from per-batch probabilities.</summary>
    public BernoulliDistribution(float[] probs)
    {
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] < 0f || probs[i] > 1f) throw new ArgumentException("probs ∈ [0, 1].");
        Probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) x[i] = rng.NextDouble() < Probs[i] ? 1f : 0f;
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float p = Probs[i]; float v = value[i];
            // Bernoulli support is {0, 1}. Branch explicitly so:
            //  - non-binary inputs return -∞ (out of support)
            //  - the deterministic edges p ∈ {0, 1} are not contaminated by 0·-∞ = NaN
            //    that the generic v·log(p) + (1-v)·log(1-p) form would otherwise produce.
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
            float p = Probs[i];
            float a = p > 0f ? -p * MathF.Log(p) : 0f;
            float b = p < 1f ? -(1f - p) * MathF.Log(1f - p) : 0f;
            h[i] = a + b;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => (float[])Probs.Clone();
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = Probs[i] * (1f - Probs[i]); return v; } }
}

/// <summary>Binomial distribution: number of successes in <c>totalCount</c> independent Bernoulli trials.</summary>
public sealed class BinomialDistribution : DistributionBase
{
    /// <summary>Number of trials (per batch).</summary>
    public int[] TotalCount { get; }
    /// <summary>Per-batch success probability.</summary>
    public float[] Probs { get; }
    /// <inheritdoc />
    public override int BatchSize => Probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support
    {
        get
        {
            int max = 0;
            for (int i = 0; i < TotalCount.Length; i++) if (TotalCount[i] > max) max = TotalCount[i];
            return new IntegerIntervalConstraint(0, max);
        }
    }
    /// <summary>Build a binomial from per-batch totalCount and probs.</summary>
    public BinomialDistribution(int[] totalCount, float[] probs)
    {
        if (totalCount.Length != probs.Length) throw new ArgumentException();
        for (int i = 0; i < totalCount.Length; i++)
        {
            if (totalCount[i] < 0) throw new ArgumentException("totalCount >= 0.");
            if (probs[i] < 0f || probs[i] > 1f) throw new ArgumentException("probs ∈ [0, 1].");
        }
        TotalCount = (int[])totalCount.Clone(); Probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            int k = 0;
            for (int t = 0; t < TotalCount[i]; t++) if (rng.NextDouble() < Probs[i]) k++;
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
            int n = TotalCount[i]; float p = Probs[i]; float k = value[i];
            // Discrete support: k must be a non-negative integer in [0, n]. Lgamma alone
            // would silently accept fractional counts, so we filter them out explicitly.
            if (k < 0f || k > n || k != MathF.Floor(k)) { lp[i] = float.NegativeInfinity; continue; }
            // Deterministic edges: only k == 0 (for p == 0) or k == n (for p == 1) have mass.
            // Skip the generic formula so we don't get 0 · -∞ = NaN at those edges.
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
        // No simple closed form; sum over the support.
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            int n = TotalCount[i];
            float p = Probs[i];
            // Deterministic ⇒ entropy is 0 (no uncertainty).
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
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = TotalCount[i] * Probs[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = TotalCount[i] * Probs[i] * (1f - Probs[i]); return v; } }
}

/// <summary>Categorical distribution over <c>K</c> classes per batch element.</summary>
public sealed class CategoricalDistribution : DistributionBase
{
    /// <summary>Per-batch probability vectors. Layout: <c>[batch, K]</c> row-major.</summary>
    public float[] Probs { get; }
    /// <summary>Number of categories.</summary>
    public int K { get; }
    /// <inheritdoc />
    public override int BatchSize => Probs.Length / K;
    /// <inheritdoc />
    public override int EventSize => 1; // sample is a single category index
    /// <inheritdoc />
    public override IConstraint Support => new IntegerIntervalConstraint(0, K - 1);
    /// <summary>Build a categorical from per-batch prob vectors.</summary>
    public CategoricalDistribution(float[] probs, int k)
    {
        if (k <= 0) throw new ArgumentException("K > 0.");
        if (probs.Length % k != 0) throw new ArgumentException("probs.Length must be a multiple of K.");
        // Validate each row sums close to 1 and entries are non-negative.
        int batch = probs.Length / k;
        for (int b = 0; b < batch; b++)
        {
            float sum = 0f;
            for (int i = 0; i < k; i++)
            {
                if (probs[b * k + i] < 0f) throw new ArgumentException("probs >= 0.");
                sum += probs[b * k + i];
            }
            if (MathF.Abs(sum - 1f) > 1e-3f) throw new ArgumentException($"probs row {b} must sum to 1 (got {sum}).");
        }
        Probs = (float[])probs.Clone(); K = k;
    }
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
                cum += Probs[b * K + i];
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
            // Reject non-integer indices outright (truncation via (int) would silently
            // accept e.g. 1.7 as category 1, which is wrong for a discrete support).
            if (v < 0f || v >= K || v != MathF.Floor(v)) { lp[b] = float.NegativeInfinity; continue; }
            int k = (int)v;
            float p = Probs[b * K + k];
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
                float p = Probs[b * K + i];
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
                for (int i = 0; i < K; i++) s += i * Probs[b * K + i];
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
                    s += Probs[b * K + i] * diff * diff;
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
    public override IConstraint Support => new SimplexConstraint(K);
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
    public override float[] Mean => Probs;
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize * K];
            for (int b = 0; b < BatchSize; b++)
                for (int i = 0; i < K; i++)
                {
                    float p = Probs[b * K + i];
                    v[b * K + i] = p * (1f - p);
                }
            return v;
        }
    }
}

/// <summary>Geometric distribution: number of failures before first success.</summary>
public sealed class GeometricDistribution : DistributionBase
{
    /// <summary>Per-batch success probability.</summary>
    public float[] Probs { get; }
    /// <inheritdoc />
    public override int BatchSize => Probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <summary>Build a geometric from per-batch probs.</summary>
    public GeometricDistribution(float[] probs)
    {
        for (int i = 0; i < probs.Length; i++)
            if (!(probs[i] > 0f && probs[i] <= 1f)) throw new ArgumentException("probs ∈ (0, 1].");
        Probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble();
            if (u < 1e-12) u = 1e-12;
            x[i] = MathF.Floor(MathF.Log((float)u) / MathF.Log(1f - Probs[i]));
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
            float k = value[i];
            if (k < 0 || k != MathF.Floor(k)) { lp[i] = float.NegativeInfinity; continue; }
            lp[i] = k * MathF.Log(1f - Probs[i]) + MathF.Log(Probs[i]);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float p = Probs[i];
            h[i] = -(p * MathF.Log(p) + (1f - p) * MathF.Log(1f - p)) / p;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = (1f - Probs[i]) / Probs[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = (1f - Probs[i]) / (Probs[i] * Probs[i]); return v; } }
}

/// <summary>Poisson distribution: f(k) = e^-λ · λ^k / k!.</summary>
public sealed class PoissonDistribution : DistributionBase
{
    /// <summary>Per-batch rate λ &gt; 0.</summary>
    public float[] Rate { get; }
    /// <inheritdoc />
    public override int BatchSize => Rate.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <summary>Build a Poisson from per-batch rate.</summary>
    public PoissonDistribution(float[] rate)
    {
        for (int i = 0; i < rate.Length; i++) if (!(rate[i] >= 0f)) throw new ArgumentException("rate >= 0.");
        Rate = (float[])rate.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) x[i] = SamplePoisson(rng, Rate[i]);
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
            // Degenerate Poisson(λ=0): the distribution is a point mass at 0.
            // log p(0|0) = 0; log p(k>0|0) = -∞. Avoids log(0) = -∞ contaminating k=0.
            if (Rate[i] == 0f) { lp[i] = k == 0f ? 0f : float.NegativeInfinity; continue; }
            lp[i] = k * MathF.Log(Rate[i]) - Rate[i] - SpecialFunctions.Lgamma(k + 1f);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        // Numerical sum (closed form involves ψ).
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float lambda = Rate[i];
            // Truncate at lambda + 10·sqrt(lambda) — captures essentially all mass.
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
    public override float[] Mean => (float[])Rate.Clone();
    /// <inheritdoc />
    public override float[] Variance => (float[])Rate.Clone();

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
    /// <summary>Number of successes (per batch). Real-valued; PyTorch parity.</summary>
    public float[] TotalCount { get; }
    /// <summary>Per-batch success probability.</summary>
    public float[] Probs { get; }
    /// <inheritdoc />
    public override int BatchSize => Probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <summary>Build a negative binomial.</summary>
    public NegativeBinomialDistribution(float[] totalCount, float[] probs)
    {
        if (totalCount.Length != probs.Length) throw new ArgumentException();
        for (int i = 0; i < probs.Length; i++)
        {
            if (!(totalCount[i] > 0f)) throw new ArgumentException("totalCount > 0.");
            // Reject p == 0 outright — Sample divides by p and (1−p)/p would blow up.
            // p == 1 is also disallowed because all mass would be on k = 0.
            if (!(probs[i] > 0f && probs[i] < 1f)) throw new ArgumentException("probs ∈ (0, 1).");
        }
        TotalCount = (float[])totalCount.Clone(); Probs = (float[])probs.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        // Sample λ ∼ Gamma(r, (1-p)/p); k ∼ Poisson(λ).
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float scale = (1f - Probs[i]) / Probs[i];
            float lambda = GammaDistribution.MarsagliaTsang(rng, TotalCount[i]) * scale;
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
            float k = value[i]; float r = TotalCount[i]; float p = Probs[i];
            // k must be a non-negative integer; Lgamma alone would silently accept fractions.
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
        for (int i = 0; i < BatchSize; i++) h[i] = float.NaN;  // no closed form; user can MC-estimate
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = TotalCount[i] * (1f - Probs[i]) / Probs[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = TotalCount[i] * (1f - Probs[i]) / (Probs[i] * Probs[i]); return v; } }
}

/// <summary>Multinomial distribution: <c>totalCount</c> draws from a categorical.</summary>
public sealed class MultinomialDistribution : DistributionBase
{
    /// <summary>Trials per batch element.</summary>
    public int[] TotalCount { get; }
    /// <summary>Per-batch probability vectors, layout <c>[batch, K]</c>.</summary>
    public float[] Probs { get; }
    /// <summary>Number of categories.</summary>
    public int K { get; }
    /// <inheritdoc />
    public override int BatchSize => TotalCount.Length;
    /// <inheritdoc />
    public override int EventSize => K;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <summary>Build a Multinomial from totalCount + probs.</summary>
    public MultinomialDistribution(int[] totalCount, float[] probs, int k)
    {
        if (k <= 0) throw new ArgumentException("K > 0.");
        if (probs.Length != totalCount.Length * k) throw new ArgumentException("probs length mismatch.");
        // Each row of probs must be a valid probability vector (non-negative, sum to 1).
        // Same validation surface as CategoricalDistribution.
        int batch = totalCount.Length;
        for (int b = 0; b < batch; b++)
        {
            float sum = 0f;
            for (int i = 0; i < k; i++)
            {
                if (probs[b * k + i] < 0f) throw new ArgumentException($"probs[{b}, {i}] = {probs[b * k + i]} must be ≥ 0.");
                sum += probs[b * k + i];
            }
            if (MathF.Abs(sum - 1f) > 1e-3f)
                throw new ArgumentException($"probs row {b} must sum to 1 (got {sum}).");
            if (totalCount[b] < 0) throw new ArgumentException($"totalCount[{b}] = {totalCount[b]} must be >= 0.");
        }
        TotalCount = (int[])totalCount.Clone(); Probs = (float[])probs.Clone(); K = k;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = new float[BatchSize * K];
        for (int b = 0; b < BatchSize; b++)
        {
            for (int t = 0; t < TotalCount[b]; t++)
            {
                float u = (float)rng.NextDouble();
                float cum = 0f; int chosen = K - 1;
                for (int i = 0; i < K; i++)
                {
                    cum += Probs[b * K + i];
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
            int n = TotalCount[b];
            // Validate the count vector: each k_i must be a non-negative integer and
            // Σ k_i must equal n. Otherwise the count vector is impossible under this
            // multinomial — the analytical formula would otherwise return a finite (and
            // wrong) value for fractional or mis-summed inputs.
            float sumK = 0f;
            bool valid = true;
            for (int i = 0; i < K; i++)
            {
                float k = value[b * K + i];
                if (k < 0f || k != MathF.Floor(k)) { valid = false; break; }
                sumK += k;
            }
            if (!valid || sumK != n) { lp[b] = float.NegativeInfinity; continue; }

            float l = SpecialFunctions.Lgamma(n + 1f);
            for (int i = 0; i < K; i++)
            {
                float k = value[b * K + i];
                float p = Probs[b * K + i];
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
                    m[b * K + i] = TotalCount[b] * Probs[b * K + i];
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
                    float p = Probs[b * K + i];
                    v[b * K + i] = TotalCount[b] * p * (1f - p);
                }
            return v;
        }
    }
}
