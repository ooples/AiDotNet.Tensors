using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// D-Adapt-Adam (Defazio &amp; Mishchenko, 2023) — Adam variant that estimates the
/// optimal step size online. The user supplies an upper-bound learning rate
/// (default 1.0) and the optimizer adapts a scalar <c>d_t</c> upward across training,
/// so no manual LR schedule is required.
///
/// Algorithm (per the paper, single-d / Adam variant):
///   m_t = β₁·m + (1-β₁)·d·g
///   v_t = β₂·v + (1-β₂)·(d·g)²
///   s_t = β₂·s + (1-β₂)·d·g
///   numerator   += d·⟨g, s⟩
///   denominator  = β₂·denominator + (1-β₂)·‖s‖₁
///   d_hat = numerator / denominator
///   d_{t+1} = max(d, d_hat / d_coef)        // d only ever grows
///   p ← p − lr·d_{t+1}·m / (sqrt(v) + eps·d_{t+1})
/// </summary>
public sealed class DAdaptAdamOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1.0, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 0.0, ["d0"] = 1e-6, ["growth_rate"] = double.PositiveInfinity,
    };
    private static readonly string[] _stateNames = new[] { "step" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <summary>The current adapted scalar <c>d</c> (one per param group).</summary>
    public double[] CurrentD { get; private set; } = Array.Empty<double>();

    /// <inheritdoc />
    public override void Step()
    {
        if (CurrentD.Length != ParamGroups.Count)
        {
            CurrentD = new double[ParamGroups.Count];
            for (int i = 0; i < CurrentD.Length; i++) CurrentD[i] = ParamGroups[i].GetOption("d0", 1e-6);
        }

        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            double d = CurrentD[gi];

            // Group-wide accumulators for the d-update.
            double sk_l1 = 0;
            double numerator = 0;

            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f) for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];

                var slot = GetOrCreateState(gi, pi, p.Length);
                if (!slot.ContainsKey("exp_avg"))     slot["exp_avg"]     = OptimizerStateValue.FromTensor(new float[p.Length]);
                if (!slot.ContainsKey("exp_avg_sq"))  slot["exp_avg_sq"]  = OptimizerStateValue.FromTensor(new float[p.Length]);
                if (!slot.ContainsKey("s"))           slot["s"]           = OptimizerStateValue.FromTensor(new float[p.Length]);

                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;
                var s = slot["s"].Tensor!;

                double dg, dgSq;
                for (int i = 0; i < p.Length; i++)
                {
                    dg = d * grad[i];
                    dgSq = dg * dg;
                    m[i] = (float)(b1 * m[i] + (1.0 - b1) * dg);
                    v[i] = (float)(b2 * v[i] + (1.0 - b2) * dgSq);
                    // s acts as a running 'gradient times d' L1 sum signal
                    s[i] = (float)(b2 * s[i] + (1.0 - b2) * (lr * dg));
                    sk_l1 += Math.Abs(s[i]);
                    numerator += lr * dg * s[i];
                }

                // Apply the parameter update with current d.
                for (int i = 0; i < p.Length; i++)
                {
                    float denom = (float)(MathF.Sqrt(v[i]) + eps * d);
                    p[i] -= lr * m[i] / denom;
                }
            }

            // d update — only ever grows. growth_rate caps how fast d can rise per step.
            if (sk_l1 > 0)
            {
                double dHat = numerator / ((1.0 - b2) * sk_l1);
                double growthCap = g.GetOption("growth_rate", double.PositiveInfinity);
                double newD = Math.Min(Math.Max(d, dHat), d * growthCap);
                CurrentD[gi] = newD;
            }
        }
    }
}

/// <summary>
/// Prodigy (Mishchenko &amp; Defazio, 2024) — improved D-Adaptation that adapts <c>d</c>
/// faster by tracking a richer denominator (β₂-running of d·s rather than s alone)
/// and by incorporating a small bias-correction. Identical state surface to
/// <see cref="DAdaptAdamOptimizer"/>; the difference is in the d-update formula.
/// </summary>
public sealed class ProdigyOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1.0, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 0.0, ["d0"] = 1e-6,
        ["beta3"] = 0.0,  // 0 disables β3 EMA; >0 uses √β2 variant from the paper
        ["growth_rate"] = double.PositiveInfinity,
    };
    private static readonly string[] _stateNames = new[] { "step" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <summary>Current adapted scalar d (one per param group).</summary>
    public double[] CurrentD { get; private set; } = Array.Empty<double>();

    /// <summary>Numerator EMA used for the d-update (one per group).</summary>
    public double[] DNumerator { get; private set; } = Array.Empty<double>();

    /// <inheritdoc />
    public override void Step()
    {
        if (CurrentD.Length != ParamGroups.Count)
        {
            CurrentD = new double[ParamGroups.Count];
            DNumerator = new double[ParamGroups.Count];
            for (int i = 0; i < CurrentD.Length; i++) CurrentD[i] = ParamGroups[i].GetOption("d0", 1e-6);
        }

        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            double d = CurrentD[gi];
            // Prodigy-specific: beta3 — when 0, paper recommends sqrt(beta2).
            double b3 = g.GetOption("beta3", 0.0);
            if (b3 == 0.0) b3 = Math.Sqrt(b2);

            // Per-step delta to apply to the running numerator/denominator.
            double dDelta = 0;
            double sk_l1 = 0;

            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f) for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];

                var slot = GetOrCreateState(gi, pi, p.Length);
                if (!slot.ContainsKey("exp_avg"))     slot["exp_avg"]     = OptimizerStateValue.FromTensor(new float[p.Length]);
                if (!slot.ContainsKey("exp_avg_sq"))  slot["exp_avg_sq"]  = OptimizerStateValue.FromTensor(new float[p.Length]);
                if (!slot.ContainsKey("s"))           slot["s"]           = OptimizerStateValue.FromTensor(new float[p.Length]);
                if (!slot.ContainsKey("p0"))
                {
                    // Snapshot the initial parameters; the d-update uses ⟨p_t − p_0, g_t⟩.
                    var p0 = new float[p.Length];
                    Array.Copy(p, p0, p.Length);
                    slot["p0"] = OptimizerStateValue.FromTensor(p0);
                }

                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;
                var s = slot["s"].Tensor!;
                var p0v = slot["p0"].Tensor!;

                for (int i = 0; i < p.Length; i++)
                {
                    double dg = d * grad[i];
                    m[i]   = (float)(b1 * m[i]   + (1.0 - b1) * dg);
                    v[i]   = (float)(b2 * v[i]   + (1.0 - b2) * dg * dg);
                    s[i]   = (float)(b3 * s[i]   + (1.0 - b3) * lr * dg);
                    sk_l1 += Math.Abs(s[i]);
                    dDelta += d * dg * (p0v[i] - p[i]);
                }

                for (int i = 0; i < p.Length; i++)
                {
                    float denom = (float)(MathF.Sqrt(v[i]) + eps * d);
                    p[i] -= lr * m[i] / denom;
                }
            }

            DNumerator[gi] = b3 * DNumerator[gi] + dDelta;
            if (sk_l1 > 0)
            {
                double dHat = DNumerator[gi] / ((1.0 - b3) * sk_l1);
                double growthCap = g.GetOption("growth_rate", double.PositiveInfinity);
                double newD = Math.Min(Math.Max(d, dHat), d * growthCap);
                CurrentD[gi] = newD;
            }
        }
    }
}
