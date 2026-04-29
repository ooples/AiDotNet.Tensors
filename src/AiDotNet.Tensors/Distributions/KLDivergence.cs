using System;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Open registry of analytical KL-divergence formulas. Mirrors PyTorch's
/// <c>torch.distributions.kl.register_kl</c>: users can register new pairs at runtime
/// and the dispatch picks the most-specific formula. The library ships analytical
/// pairs for every common distribution combination (~10 by default); users can extend.
/// </summary>
public static class KLDivergence
{
    private static readonly ConcurrentDictionary<(Type p, Type q), Func<IDistribution, IDistribution, float[]>> _registry = new();

    static KLDivergence()
    {
        Register<NormalDistribution, NormalDistribution>(KlNormalNormal);
        Register<BernoulliDistribution, BernoulliDistribution>(KlBernoulliBernoulli);
        Register<CategoricalDistribution, CategoricalDistribution>(KlCategoricalCategorical);
        Register<UniformDistribution, UniformDistribution>(KlUniformUniform);
        Register<ExponentialDistribution, ExponentialDistribution>(KlExpExp);
        Register<LaplaceDistribution, LaplaceDistribution>(KlLaplaceLaplace);
        Register<GammaDistribution, GammaDistribution>(KlGammaGamma);
        Register<BetaDistribution, BetaDistribution>(KlBetaBeta);
        Register<DirichletDistribution, DirichletDistribution>(KlDirichletDirichlet);
        Register<DiagonalMultivariateNormalDistribution, DiagonalMultivariateNormalDistribution>(KlDiagMvnDiagMvn);
        Register<MultivariateNormalDistribution, MultivariateNormalDistribution>(KlMvnMvn);
    }

    /// <summary>Register an analytical KL-divergence formula for a (P, Q) type pair.</summary>
    public static void Register<TP, TQ>(Func<TP, TQ, float[]> fn)
        where TP : IDistribution
        where TQ : IDistribution
    {
        _registry[(typeof(TP), typeof(TQ))] = (p, q) => fn((TP)p, (TQ)q);
    }

    /// <summary>
    /// Compute KL(p || q) using a registered analytical formula if one exists, otherwise
    /// throw with a helpful message. (Monte-Carlo fallback is the user's responsibility:
    /// drawing N samples and computing <c>p.LogProb(x) - q.LogProb(x)</c>.)
    /// </summary>
    public static float[] Compute(IDistribution p, IDistribution q)
    {
        if (p == null) throw new ArgumentNullException(nameof(p));
        if (q == null) throw new ArgumentNullException(nameof(q));
        if (_registry.TryGetValue((p.GetType(), q.GetType()), out var fn))
            return fn(p, q);
        throw new NotSupportedException(
            $"No analytical KL divergence registered for ({p.GetType().Name}, {q.GetType().Name}). " +
            "Register one via KLDivergence.Register or use Monte-Carlo estimation.");
    }

    /// <summary>True iff an analytical formula is registered for the given type pair.</summary>
    public static bool IsRegistered<TP, TQ>() where TP : IDistribution where TQ : IDistribution =>
        _registry.ContainsKey((typeof(TP), typeof(TQ)));

    // -------- Built-in analytical formulas --------

    private static float[] KlNormalNormal(NormalDistribution p, NormalDistribution q)
    {
        if (p.BatchSize != q.BatchSize) throw new ArgumentException("batch shapes must match.");
        var kl = new float[p.BatchSize];
        for (int i = 0; i < p.BatchSize; i++)
        {
            float diff = p.Loc[i] - q.Loc[i];
            float var1 = p.Scale[i] * p.Scale[i];
            float var2 = q.Scale[i] * q.Scale[i];
            kl[i] = MathF.Log(q.Scale[i] / p.Scale[i]) + (var1 + diff * diff) / (2f * var2) - 0.5f;
        }
        return kl;
    }

    private static float[] KlBernoulliBernoulli(BernoulliDistribution p, BernoulliDistribution q)
    {
        if (p.BatchSize != q.BatchSize) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int i = 0; i < p.BatchSize; i++)
        {
            float pp = p.Probs[i], qq = q.Probs[i];
            float a = pp > 0f ? pp * (MathF.Log(pp) - MathF.Log(qq)) : 0f;
            float b = pp < 1f ? (1f - pp) * (MathF.Log(1f - pp) - MathF.Log(1f - qq)) : 0f;
            kl[i] = a + b;
        }
        return kl;
    }

    private static float[] KlCategoricalCategorical(CategoricalDistribution p, CategoricalDistribution q)
    {
        if (p.BatchSize != q.BatchSize || p.K != q.K) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int b = 0; b < p.BatchSize; b++)
        {
            float s = 0f;
            for (int k = 0; k < p.K; k++)
            {
                float pk = p.Probs[b * p.K + k]; float qk = q.Probs[b * q.K + k];
                if (pk > 0f) s += pk * (MathF.Log(pk) - MathF.Log(MathF.Max(qk, 1e-30f)));
            }
            kl[b] = s;
        }
        return kl;
    }

    private static float[] KlUniformUniform(UniformDistribution p, UniformDistribution q)
    {
        if (p.BatchSize != q.BatchSize) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int i = 0; i < p.BatchSize; i++)
        {
            // KL(U(a,b) || U(c,d)) = log((d-c)/(b-a)) if [a,b] ⊂ [c,d], else +∞.
            if (p.Low[i] >= q.Low[i] && p.High[i] <= q.High[i])
                kl[i] = MathF.Log((q.High[i] - q.Low[i]) / (p.High[i] - p.Low[i]));
            else
                kl[i] = float.PositiveInfinity;
        }
        return kl;
    }

    private static float[] KlExpExp(ExponentialDistribution p, ExponentialDistribution q)
    {
        if (p.BatchSize != q.BatchSize) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int i = 0; i < p.BatchSize; i++)
            kl[i] = MathF.Log(p.Rate[i] / q.Rate[i]) + q.Rate[i] / p.Rate[i] - 1f;
        return kl;
    }

    private static float[] KlLaplaceLaplace(LaplaceDistribution p, LaplaceDistribution q)
    {
        if (p.BatchSize != q.BatchSize) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int i = 0; i < p.BatchSize; i++)
        {
            float diff = MathF.Abs(p.Loc[i] - q.Loc[i]);
            kl[i] = MathF.Log(q.Scale[i] / p.Scale[i])
                  + (p.Scale[i] * MathF.Exp(-diff / p.Scale[i]) + diff) / q.Scale[i] - 1f;
        }
        return kl;
    }

    private static float[] KlGammaGamma(GammaDistribution p, GammaDistribution q)
    {
        if (p.BatchSize != q.BatchSize) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int i = 0; i < p.BatchSize; i++)
        {
            float ap = p.Concentration[i]; float aq = q.Concentration[i];
            float bp = p.Rate[i]; float bq = q.Rate[i];
            kl[i] = (ap - aq) * SpecialFunctions.Digamma(ap)
                  - SpecialFunctions.Lgamma(ap) + SpecialFunctions.Lgamma(aq)
                  + aq * (MathF.Log(bp) - MathF.Log(bq))
                  + ap * (bq - bp) / bp;
        }
        return kl;
    }

    private static float[] KlBetaBeta(BetaDistribution p, BetaDistribution q)
    {
        if (p.BatchSize != q.BatchSize) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int i = 0; i < p.BatchSize; i++)
        {
            float a1p = p.Concentration1[i]; float a0p = p.Concentration0[i];
            float a1q = q.Concentration1[i]; float a0q = q.Concentration0[i];
            float sumP = a1p + a0p; float sumQ = a1q + a0q;
            kl[i] = SpecialFunctions.LogBeta(a1q, a0q) - SpecialFunctions.LogBeta(a1p, a0p)
                  + (a1p - a1q) * SpecialFunctions.Digamma(a1p)
                  + (a0p - a0q) * SpecialFunctions.Digamma(a0p)
                  + (a1q - a1p + a0q - a0p) * SpecialFunctions.Digamma(sumP);
        }
        return kl;
    }

    private static float[] KlDirichletDirichlet(DirichletDistribution p, DirichletDistribution q)
    {
        if (p.BatchSize != q.BatchSize || p.K != q.K) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int b = 0; b < p.BatchSize; b++)
        {
            float a0p = 0f, a0q = 0f;
            for (int i = 0; i < p.K; i++)
            {
                a0p += p.Concentration[b * p.K + i];
                a0q += q.Concentration[b * p.K + i];
            }
            float term = SpecialFunctions.Lgamma(a0p) - SpecialFunctions.Lgamma(a0q);
            for (int i = 0; i < p.K; i++)
            {
                float ap = p.Concentration[b * p.K + i];
                float aq = q.Concentration[b * p.K + i];
                term += SpecialFunctions.Lgamma(aq) - SpecialFunctions.Lgamma(ap)
                      + (ap - aq) * (SpecialFunctions.Digamma(ap) - SpecialFunctions.Digamma(a0p));
            }
            kl[b] = term;
        }
        return kl;
    }

    private static float[] KlDiagMvnDiagMvn(DiagonalMultivariateNormalDistribution p, DiagonalMultivariateNormalDistribution q)
    {
        if (p.BatchSize != q.BatchSize || p.D != q.D) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        for (int b = 0; b < p.BatchSize; b++)
        {
            float trace = 0f, quad = 0f, halfLogRatio = 0f;
            for (int i = 0; i < p.D; i++)
            {
                float vp = p.Scale[b * p.D + i] * p.Scale[b * p.D + i];
                float vq = q.Scale[b * p.D + i] * q.Scale[b * p.D + i];
                float diff = p.Loc[b * p.D + i] - q.Loc[b * p.D + i];
                trace += vp / vq;
                quad  += diff * diff / vq;
                halfLogRatio += MathF.Log(q.Scale[b * p.D + i]) - MathF.Log(p.Scale[b * p.D + i]);
            }
            kl[b] = 0.5f * (trace + quad - p.D) + halfLogRatio;
        }
        return kl;
    }

    private static float[] KlMvnMvn(MultivariateNormalDistribution p, MultivariateNormalDistribution q)
    {
        if (p.BatchSize != q.BatchSize || p.D != q.D) throw new ArgumentException();
        var kl = new float[p.BatchSize];
        var diff = new float[p.D];
        var sol = new float[p.D];
        for (int b = 0; b < p.BatchSize; b++)
        {
            // KL = 0.5 · ( tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − D + log(|Σ_q| / |Σ_p|) )
            float halfLogP = 0f, halfLogQ = 0f;
            for (int i = 0; i < p.D; i++)
            {
                halfLogP += MathF.Log(p.CholeskyL[b * p.D * p.D + i * p.D + i]);
                halfLogQ += MathF.Log(q.CholeskyL[b * p.D * p.D + i * p.D + i]);
            }
            // (μ_q - μ_p) and forward-substitute against L_q.
            for (int i = 0; i < p.D; i++) diff[i] = p.Loc[b * p.D + i] - q.Loc[b * p.D + i];
            for (int i = 0; i < p.D; i++)
            {
                float s = diff[i];
                for (int j = 0; j < i; j++) s -= q.CholeskyL[b * p.D * p.D + i * p.D + j] * sol[j];
                sol[i] = s / q.CholeskyL[b * p.D * p.D + i * p.D + i];
            }
            float quad = 0f;
            for (int i = 0; i < p.D; i++) quad += sol[i] * sol[i];
            // tr(Σ_q⁻¹ Σ_p): requires solving L_q · Y = L_p column-by-column then summing ‖Y_:i‖².
            float trace = 0f;
            for (int j = 0; j < p.D; j++)
            {
                // Y_:j satisfies L_q Y_:j = L_p_:j  (only j..D-1 entries non-zero in L_p_:j).
                for (int i = 0; i < p.D; i++) sol[i] = 0f;
                for (int i = 0; i < p.D; i++)
                {
                    float rhs = i >= j ? p.CholeskyL[b * p.D * p.D + i * p.D + j] : 0f;
                    float s = rhs;
                    for (int k = 0; k < i; k++) s -= q.CholeskyL[b * p.D * p.D + i * p.D + k] * sol[k];
                    sol[i] = s / q.CholeskyL[b * p.D * p.D + i * p.D + i];
                }
                for (int i = 0; i < p.D; i++) trace += sol[i] * sol[i];
            }
            kl[b] = 0.5f * (trace + quad - p.D) + halfLogQ - halfLogP;
        }
        return kl;
    }
}
