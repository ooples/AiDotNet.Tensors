using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Multivariate normal distribution N(μ, Σ). Supports diagonal covariance via the
/// dedicated <see cref="DiagonalMultivariateNormalDistribution"/> for the common
/// case; this class accepts a full covariance matrix per batch element. Cholesky
/// factor <c>L</c> is computed at construction time and reused for sampling.
/// </summary>
public sealed class MultivariateNormalDistribution : DistributionBase
{
    /// <summary>Per-batch mean vectors. Layout <c>[batch, D]</c> row-major.</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch covariance matrices. Layout <c>[batch, D, D]</c> row-major.</summary>
    public float[] Covariance { get; }
    /// <summary>Lower-triangular Cholesky factor of <see cref="Covariance"/>.</summary>
    public float[] CholeskyL { get; }
    /// <summary>Event dimensionality.</summary>
    public int D { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length / D;
    /// <inheritdoc />
    public override int EventSize => D;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build an MVN from per-batch loc + covariance.</summary>
    public MultivariateNormalDistribution(float[] loc, float[] covariance, int d)
    {
        if (d <= 0) throw new ArgumentException("D > 0.");
        if (loc.Length % d != 0) throw new ArgumentException("loc.Length must be a multiple of D.");
        int batch = loc.Length / d;
        if (covariance.Length != batch * d * d) throw new ArgumentException("covariance shape mismatch.");
        Loc = (float[])loc.Clone(); Covariance = (float[])covariance.Clone(); D = d;
        CholeskyL = new float[batch * d * d];
        for (int b = 0; b < batch; b++) Cholesky(covariance, b * d * d, CholeskyL, b * d * d, d);
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        int batch = BatchSize;
        var x = new float[batch * D];
        var z = new float[D];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < D; i++) z[i] = (float)NormalDistribution.Gaussian(rng);
            // x = L · z + μ
            for (int i = 0; i < D; i++)
            {
                float acc = 0f;
                for (int j = 0; j <= i; j++) acc += CholeskyL[b * D * D + i * D + j] * z[j];
                x[b * D + i] = Loc[b * D + i] + acc;
            }
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        int batch = BatchSize;
        var lp = new float[batch];
        var diff = new float[D];
        var sol = new float[D];
        const float HalfLog2Pi = 0.91893853321f;
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < D; i++) diff[i] = value[b * D + i] - Loc[b * D + i];
            // Solve L · sol = diff (forward substitution).
            for (int i = 0; i < D; i++)
            {
                float s = diff[i];
                for (int j = 0; j < i; j++) s -= CholeskyL[b * D * D + i * D + j] * sol[j];
                sol[i] = s / CholeskyL[b * D * D + i * D + i];
            }
            float quad = 0f;
            for (int i = 0; i < D; i++) quad += sol[i] * sol[i];
            float halfLogDet = 0f;
            for (int i = 0; i < D; i++) halfLogDet += MathF.Log(CholeskyL[b * D * D + i * D + i]);
            lp[b] = -0.5f * quad - halfLogDet - D * HalfLog2Pi;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        int batch = BatchSize;
        var h = new float[batch];
        const float HalfLog2PiE = 1.41893853321f;
        for (int b = 0; b < batch; b++)
        {
            float halfLogDet = 0f;
            for (int i = 0; i < D; i++) halfLogDet += MathF.Log(CholeskyL[b * D * D + i * D + i]);
            h[b] = D * HalfLog2PiE + halfLogDet;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => (float[])Loc.Clone();
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            int batch = BatchSize;
            var v = new float[batch * D];
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < D; i++)
                    v[b * D + i] = Covariance[b * D * D + i * D + i];
            return v;
        }
    }

    private static void Cholesky(float[] src, int srcOff, float[] dst, int dstOff, int n)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                dst[dstOff + i * n + j] = 0f;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double s = src[srcOff + i * n + j];
                for (int k = 0; k < j; k++) s -= dst[dstOff + i * n + k] * dst[dstOff + j * n + k];
                if (i == j)
                {
                    if (s <= 0) throw new ArgumentException("covariance is not positive definite.");
                    dst[dstOff + i * n + i] = (float)Math.Sqrt(s);
                }
                else
                {
                    dst[dstOff + i * n + j] = (float)(s / dst[dstOff + j * n + j]);
                }
            }
        }
    }
}

/// <summary>Diagonal multivariate normal — specialised when Σ = diag(σ²).</summary>
public sealed class DiagonalMultivariateNormalDistribution : DistributionBase
{
    /// <summary>Per-batch loc, layout <c>[batch, D]</c>.</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch scale (per-dim std), layout <c>[batch, D]</c>.</summary>
    public float[] Scale { get; }
    /// <summary>Event dimensionality.</summary>
    public int D { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length / D;
    /// <inheritdoc />
    public override int EventSize => D;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a diagonal MVN.</summary>
    public DiagonalMultivariateNormalDistribution(float[] loc, float[] scale, int d)
    {
        if (loc.Length != scale.Length) throw new ArgumentException();
        if (loc.Length % d != 0) throw new ArgumentException("D must divide loc.Length.");
        for (int i = 0; i < scale.Length; i++) if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        Loc = (float[])loc.Clone(); Scale = (float[])scale.Clone(); D = d;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[Loc.Length];
        for (int i = 0; i < x.Length; i++)
            x[i] = Loc[i] + Scale[i] * (float)NormalDistribution.Gaussian(rng);
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        int batch = BatchSize;
        var lp = new float[batch];
        const float LogTwoPi = 1.83787706641f;
        for (int b = 0; b < batch; b++)
        {
            float acc = 0f;
            for (int i = 0; i < D; i++)
            {
                float z = (value[b * D + i] - Loc[b * D + i]) / Scale[b * D + i];
                acc -= 0.5f * (z * z + LogTwoPi) + MathF.Log(Scale[b * D + i]);
            }
            lp[b] = acc;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        int batch = BatchSize;
        var h = new float[batch];
        const float HalfLog2PiE = 1.41893853321f;
        for (int b = 0; b < batch; b++)
        {
            float acc = D * HalfLog2PiE;
            for (int i = 0; i < D; i++) acc += MathF.Log(Scale[b * D + i]);
            h[b] = acc;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => (float[])Loc.Clone();
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[Scale.Length];
            for (int i = 0; i < v.Length; i++) v[i] = Scale[i] * Scale[i];
            return v;
        }
    }
    /// <inheritdoc />
    public override float[] StdDev => (float[])Scale.Clone();
}

/// <summary>Dirichlet distribution on the K-simplex.</summary>
public sealed class DirichletDistribution : DistributionBase
{
    /// <summary>Per-batch concentration vectors. Layout <c>[batch, K]</c>.</summary>
    public float[] Concentration { get; }
    /// <summary>Number of categories.</summary>
    public int K { get; }
    /// <inheritdoc />
    public override int BatchSize => Concentration.Length / K;
    /// <inheritdoc />
    public override int EventSize => K;
    /// <inheritdoc />
    public override IConstraint Support => new SimplexConstraint(K);
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a Dirichlet from per-batch concentration vectors.</summary>
    public DirichletDistribution(float[] concentration, int k)
    {
        if (k <= 0) throw new ArgumentException("K > 0.");
        if (concentration.Length % k != 0) throw new ArgumentException("concentration.Length % K != 0.");
        for (int i = 0; i < concentration.Length; i++) if (!(concentration[i] > 0f)) throw new ArgumentException("α > 0.");
        Concentration = (float[])concentration.Clone(); K = k;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        int batch = BatchSize;
        var x = new float[batch * K];
        for (int b = 0; b < batch; b++)
        {
            float sum = 0f;
            for (int i = 0; i < K; i++)
            {
                x[b * K + i] = GammaDistribution.MarsagliaTsang(rng, Concentration[b * K + i]);
                sum += x[b * K + i];
            }
            for (int i = 0; i < K; i++) x[b * K + i] /= sum;
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
            float a0 = 0f;
            float l = 0f;
            for (int i = 0; i < K; i++)
            {
                float a = Concentration[b * K + i];
                a0 += a;
                l += (a - 1f) * MathF.Log(value[b * K + i]) - SpecialFunctions.Lgamma(a);
            }
            l += SpecialFunctions.Lgamma(a0);
            lp[b] = l;
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
            float a0 = 0f;
            for (int i = 0; i < K; i++) a0 += Concentration[b * K + i];
            float lnB = -SpecialFunctions.Lgamma(a0);
            for (int i = 0; i < K; i++) lnB += SpecialFunctions.Lgamma(Concentration[b * K + i]);
            float dgA0 = SpecialFunctions.Digamma(a0);
            float term = (a0 - K) * dgA0;
            for (int i = 0; i < K; i++)
            {
                float a = Concentration[b * K + i];
                term -= (a - 1f) * SpecialFunctions.Digamma(a);
            }
            h[b] = lnB + term;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            int batch = BatchSize;
            var m = new float[batch * K];
            for (int b = 0; b < batch; b++)
            {
                float a0 = 0f;
                for (int i = 0; i < K; i++) a0 += Concentration[b * K + i];
                for (int i = 0; i < K; i++) m[b * K + i] = Concentration[b * K + i] / a0;
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
            var v = new float[batch * K];
            for (int b = 0; b < batch; b++)
            {
                float a0 = 0f;
                for (int i = 0; i < K; i++) a0 += Concentration[b * K + i];
                float denom = a0 * a0 * (a0 + 1f);
                for (int i = 0; i < K; i++)
                {
                    float a = Concentration[b * K + i];
                    v[b * K + i] = a * (a0 - a) / denom;
                }
            }
            return v;
        }
    }
}
