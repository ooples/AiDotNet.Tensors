using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;
using AiDotNet.Tensors.Distributions.Transforms;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Wraps a base distribution and reinterprets the rightmost <c>reinterpretedBatchDims</c>
/// of its batch shape as event dims. Mirrors <c>torch.distributions.Independent</c>: useful
/// for treating a length-K independent normal collection as a single K-dimensional event.
/// </summary>
public sealed class IndependentDistribution : DistributionBase
{
    /// <summary>Underlying base distribution.</summary>
    public IDistribution Base { get; }
    /// <inheritdoc />
    public override int BatchSize { get; }
    /// <inheritdoc />
    public override int EventSize { get; }
    /// <inheritdoc />
    public override IConstraint Support => Base.Support;
    /// <inheritdoc />
    public override bool HasRSample => Base.HasRSample;

    /// <summary>Wrap <paramref name="@base"/> and treat its full batch as <c>newBatchSize</c> independent groups
    /// of <c>EventSize</c>. <paramref name="@base"/>.BatchSize must equal <c>newBatchSize · newEventSize</c>.</summary>
    public IndependentDistribution(IDistribution @base, int newBatchSize, int newEventSize)
    {
        Base = @base ?? throw new ArgumentNullException(nameof(@base));
        if (newBatchSize * newEventSize != @base.BatchSize)
            throw new ArgumentException(
                $"newBatchSize·newEventSize ({newBatchSize * newEventSize}) must equal base.BatchSize ({@base.BatchSize}).");
        if (@base.EventSize != 1)
            throw new ArgumentException("Independent expects a scalar-event base distribution.");
        BatchSize = newBatchSize;
        EventSize = newEventSize;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => Base.Sample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng) => Base.RSample(rng);
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        // Base.LogProb expects [base.BatchSize] = [BatchSize · EventSize] values.
        var lpFlat = Base.LogProb(value);
        var lp = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            float acc = 0f;
            for (int e = 0; e < EventSize; e++) acc += lpFlat[b * EventSize + e];
            lp[b] = acc;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var hFlat = Base.Entropy();
        var h = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            float acc = 0f;
            for (int e = 0; e < EventSize; e++) acc += hFlat[b * EventSize + e];
            h[b] = acc;
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => Base.Mean;
    /// <inheritdoc />
    public override float[] Variance => Base.Variance;
}

/// <summary>
/// Push a base distribution through a chain of <see cref="ITransform"/> operations.
/// Sample = transform(base.Sample); log_prob = base.log_prob(inverse(y)) − Σ log|det J|.
/// </summary>
public sealed class TransformedDistribution : DistributionBase
{
    /// <summary>Base distribution.</summary>
    public IDistribution Base { get; }
    /// <summary>Transforms applied left-to-right.</summary>
    public ITransform[] Transforms { get; }
    /// <inheritdoc />
    public override int BatchSize => Base.BatchSize;
    /// <inheritdoc />
    public override int EventSize => Base.EventSize;
    /// <inheritdoc />
    public override IConstraint Support => Transforms.Length == 0 ? Base.Support : Transforms[Transforms.Length - 1].Codomain;
    /// <inheritdoc />
    public override bool HasRSample => Base.HasRSample;

    /// <summary>Build a transformed distribution.</summary>
    public TransformedDistribution(IDistribution baseDist, params ITransform[] transforms)
    {
        Base = baseDist ?? throw new ArgumentNullException(nameof(baseDist));
        Transforms = transforms ?? Array.Empty<ITransform>();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        var x = Base.Sample(rng);
        foreach (var t in Transforms) x = t.Forward(x);
        return x;
    }
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = Base.RSample(rng);
        foreach (var t in Transforms) x = t.Forward(x);
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        // Walk transforms in reverse to get pre-image and accumulate log|det J|.
        float[] y = value;
        var ldj = new float[value.Length];
        for (int i = Transforms.Length - 1; i >= 0; i--)
        {
            var x = Transforms[i].Inverse(y);
            var d = Transforms[i].LogAbsDetJacobian(x, y);
            for (int j = 0; j < ldj.Length; j++) ldj[j] += d[j];
            y = x;
        }
        var baseLp = Base.LogProb(y);
        // Reduce ldj across event dims (they currently match value shape).
        var lp = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            float acc = baseLp[b];
            for (int e = 0; e < EventSize; e++) acc -= ldj[b * EventSize + e];
            lp[b] = acc;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        // Closed form only for constant-Jacobian transforms; otherwise leave NaN.
        bool constant = true; foreach (var t in Transforms) if (!t.ConstantJacobian) { constant = false; break; }
        if (!constant)
        {
            var nan = new float[BatchSize];
            for (int i = 0; i < nan.Length; i++) nan[i] = float.NaN;
            return nan;
        }
        var h = Base.Entropy();
        // Probe a length-(B·E) zero-vector through each transform to extract its constant log|det J|.
        var probe = new float[BatchSize * EventSize];
        var lj = new float[probe.Length];
        var x = probe;
        foreach (var t in Transforms)
        {
            var y = t.Forward(x);
            var d = t.LogAbsDetJacobian(x, y);
            for (int i = 0; i < lj.Length; i++) lj[i] += d[i];
            x = y;
        }
        var hOut = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            float acc = h[b];
            for (int e = 0; e < EventSize; e++) acc += lj[b * EventSize + e];
            hOut[b] = acc;
        }
        return hOut;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            // Push base mean through transforms — only exact for affine but a useful default.
            var m = Base.Mean;
            foreach (var t in Transforms) m = t.Forward(m);
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        // No general closed form. Return NaN; users compute MC estimates.
        get { var v = new float[BatchSize * EventSize]; for (int i = 0; i < v.Length; i++) v[i] = float.NaN; return v; }
    }
}

/// <summary>
/// Mixture of <c>K</c> distributions of the same family (e.g. Gaussian Mixture Model).
/// Sampling: draw a component index from <see cref="Mixing"/>, then sample from that component.
/// LogProb: log Σ_k π_k · p_k(x).
/// </summary>
public sealed class MixtureSameFamilyDistribution : DistributionBase
{
    /// <summary>Mixing categorical: probability of each component, layout <c>[batch, K]</c>.</summary>
    public CategoricalDistribution Mixing { get; }
    /// <summary>Component distribution; its batch shape is <c>[batch · K]</c>, scalar event.</summary>
    public IDistribution Component { get; }
    /// <inheritdoc />
    public override int BatchSize => Mixing.BatchSize;
    /// <inheritdoc />
    public override int EventSize => Component.EventSize;
    /// <inheritdoc />
    public override IConstraint Support => Component.Support;

    /// <summary>Build a mixture from a categorical mixing distribution and a component distribution
    /// whose batch shape equals <c>Mixing.BatchSize · Mixing.K</c>.</summary>
    public MixtureSameFamilyDistribution(CategoricalDistribution mixing, IDistribution component)
    {
        Mixing = mixing; Component = component;
        if (component.BatchSize != mixing.BatchSize * mixing.K)
            throw new ArgumentException(
                $"component.BatchSize ({component.BatchSize}) must equal mixing.BatchSize · K ({mixing.BatchSize * mixing.K}).");
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        int K = Mixing.K;
        var idx = Mixing.Sample(rng);
        var compSamples = Component.Sample(rng);
        // Pick per-batch the sample from chosen component.
        var x = new float[BatchSize * EventSize];
        for (int b = 0; b < BatchSize; b++)
        {
            int chosen = (int)idx[b];
            for (int e = 0; e < EventSize; e++)
                x[b * EventSize + e] = compSamples[(b * K + chosen) * EventSize + e];
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        int K = Mixing.K;
        // Broadcast value to all components: shape [batch · K, event].
        var broad = new float[BatchSize * K * EventSize];
        for (int b = 0; b < BatchSize; b++)
            for (int k = 0; k < K; k++)
                for (int e = 0; e < EventSize; e++)
                    broad[(b * K + k) * EventSize + e] = value[b * EventSize + e];
        var compLp = Component.LogProb(broad); // length [batch · K]
        var lp = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            // log Σ exp(log π_k + log p_k(x)) — numerically stable via log-sum-exp.
            float maxV = float.NegativeInfinity;
            for (int k = 0; k < K; k++)
            {
                float t = MathF.Log(MathF.Max(Mixing.Probs[b * K + k], 1e-30f)) + compLp[b * K + k];
                if (t > maxV) maxV = t;
            }
            double s = 0;
            for (int k = 0; k < K; k++)
            {
                float t = MathF.Log(MathF.Max(Mixing.Probs[b * K + k], 1e-30f)) + compLp[b * K + k];
                s += Math.Exp(t - maxV);
            }
            lp[b] = (float)(maxV + Math.Log(s));
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < h.Length; i++) h[i] = float.NaN; // mixtures: no closed form
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            int K = Mixing.K;
            var m = new float[BatchSize * EventSize];
            var cm = Component.Mean;
            for (int b = 0; b < BatchSize; b++)
                for (int e = 0; e < EventSize; e++)
                {
                    float acc = 0f;
                    for (int k = 0; k < K; k++)
                        acc += Mixing.Probs[b * K + k] * cm[(b * K + k) * EventSize + e];
                    m[b * EventSize + e] = acc;
                }
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        get { var v = new float[BatchSize * EventSize]; for (int i = 0; i < v.Length; i++) v[i] = float.NaN; return v; }
    }
}

/// <summary>Relaxed (Gumbel-softmax) Bernoulli — continuous proxy for sampling Bernoulli with reparam grads.</summary>
public sealed class RelaxedBernoulliDistribution : DistributionBase
{
    /// <summary>Per-batch logits (log p / (1 - p)).</summary>
    public float[] Logits { get; }
    /// <summary>Temperature τ &gt; 0 controlling the relaxation.</summary>
    public float[] Temperature { get; }
    /// <inheritdoc />
    public override int BatchSize => Logits.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => UnitIntervalConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a relaxed Bernoulli.</summary>
    public RelaxedBernoulliDistribution(float[] logits, float[] temperature)
    {
        if (logits.Length != temperature.Length) throw new ArgumentException();
        for (int i = 0; i < temperature.Length; i++) if (!(temperature[i] > 0f)) throw new ArgumentException("τ > 0.");
        Logits = logits; Temperature = temperature;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble();
            if (u < 1e-12) u = 1e-12; if (u > 1 - 1e-12) u = 1 - 1e-12;
            float l = MathF.Log((float)u) - MathF.Log(1f - (float)u);
            float v = (Logits[i] + l) / Temperature[i];
            x[i] = 1f / (1f + MathF.Exp(-v));
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
            float t = Temperature[i]; float y = value[i];
            float logy = MathF.Log(y); float log1my = MathF.Log(1f - y);
            float diff = Logits[i] - t * (logy - log1my);
            lp[i] = MathF.Log(t) - logy - log1my + diff - 2f * Softplus(diff);
        }
        return lp;
    }
    private static float Softplus(float v) => v > 20f ? v : MathF.Log(1f + MathF.Exp(v));
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < h.Length; i++) h[i] = float.NaN;
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        // Closed-form mean isn't simple; report sigmoid(logits) as a conventional placeholder.
        get
        {
            var m = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++) m[i] = 1f / (1f + MathF.Exp(-Logits[i]));
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = float.NaN; return v; } }
}

/// <summary>Relaxed One-Hot Categorical (Gumbel-softmax). Reparameterisable continuous proxy
/// for categorical sampling.</summary>
public sealed class RelaxedOneHotCategoricalDistribution : DistributionBase
{
    /// <summary>Per-batch logits over K categories. Layout <c>[batch, K]</c>.</summary>
    public float[] Logits { get; }
    /// <summary>Per-batch temperature.</summary>
    public float[] Temperature { get; }
    /// <summary>Number of categories.</summary>
    public int K { get; }
    /// <inheritdoc />
    public override int BatchSize => Temperature.Length;
    /// <inheritdoc />
    public override int EventSize => K;
    /// <inheritdoc />
    public override IConstraint Support => new SimplexConstraint(K);
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a relaxed one-hot categorical.</summary>
    public RelaxedOneHotCategoricalDistribution(float[] logits, float[] temperature, int k)
    {
        if (k <= 0) throw new ArgumentException();
        if (logits.Length != temperature.Length * k) throw new ArgumentException();
        for (int i = 0; i < temperature.Length; i++) if (!(temperature[i] > 0f)) throw new ArgumentException("τ > 0.");
        Logits = logits; Temperature = temperature; K = k;
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
            // y_k = softmax((logits + g) / τ) where g ∼ Gumbel(0, 1).
            float maxV = float.NegativeInfinity;
            for (int i = 0; i < K; i++)
            {
                double u = rng.NextDouble();
                if (u < 1e-12) u = 1e-12;
                float g = -MathF.Log(-MathF.Log((float)u));
                float v = (Logits[b * K + i] + g) / Temperature[b];
                x[b * K + i] = v;
                if (v > maxV) maxV = v;
            }
            double s = 0;
            for (int i = 0; i < K; i++) { x[b * K + i] = MathF.Exp(x[b * K + i] - maxV); s += x[b * K + i]; }
            for (int i = 0; i < K; i++) x[b * K + i] = (float)(x[b * K + i] / s);
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
            float t = Temperature[b];
            // log Γ(K)·τ^(K-1) − K·log Σ exp((α-τ logy)) + Σ (α_k - (τ+1) log y_k) − ...
            // We use the Maddison/Mnih/Teh formulation:
            // log p(y) = log((K-1)!) + (K-1)·log τ + Σ(α_k − τ·log y_k) − K·log Σ exp(α_k − τ·log y_k).
            float c = SpecialFunctions.Lgamma(K) + (K - 1) * MathF.Log(t);
            // Compute each α_k − τ·log y_k. Use log-sum-exp for stability.
            float maxV = float.NegativeInfinity;
            var z = new float[K];
            for (int i = 0; i < K; i++)
            {
                z[i] = Logits[b * K + i] - t * MathF.Log(MathF.Max(value[b * K + i], 1e-30f));
                if (z[i] > maxV) maxV = z[i];
            }
            double sumExp = 0;
            for (int i = 0; i < K; i++) sumExp += Math.Exp(z[i] - maxV);
            double lse = maxV + Math.Log(sumExp);
            double term2 = 0;
            for (int i = 0; i < K; i++) term2 += z[i];
            lp[b] = (float)(c + term2 - K * lse);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < h.Length; i++) h[i] = float.NaN;
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            // softmax(logits) — the categorical mean of a one-hot.
            var m = new float[BatchSize * K];
            for (int b = 0; b < BatchSize; b++)
            {
                float maxV = float.NegativeInfinity;
                for (int i = 0; i < K; i++) if (Logits[b * K + i] > maxV) maxV = Logits[b * K + i];
                double s = 0;
                for (int i = 0; i < K; i++) { m[b * K + i] = MathF.Exp(Logits[b * K + i] - maxV); s += m[b * K + i]; }
                for (int i = 0; i < K; i++) m[b * K + i] = (float)(m[b * K + i] / s);
            }
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize * K]; for (int i = 0; i < v.Length; i++) v[i] = float.NaN; return v; } }
}
