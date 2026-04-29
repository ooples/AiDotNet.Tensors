using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions.Transforms;

/// <summary>
/// Planar flow (Rezende &amp; Mohamed, 2015): y = x + u · h(wᵀx + b) where h = tanh.
/// Per-flow parameters u, w ∈ ℝᴰ, b ∈ ℝ. Element-wise log|det J| = log|1 + u'·w·h'(wᵀx+b)|
/// where u' is u corrected to enforce invertibility.
///
/// <para><b>Limitation:</b> the analytical inverse of a planar flow has no closed form,
/// so wrapping this transform in a <see cref="TransformedDistribution"/> and calling
/// <c>LogProb(y)</c> on it will throw — the score path needs to invert through every
/// transform. The expected usage pattern is the variational-flow workflow where you draw
/// <c>z ∼ base</c>, apply <c>y = T(z)</c>, and compute
/// <c>log p_y(y) = log p_z(z) − log|det J|</c> directly while you still hold <c>z</c>:
/// no inverse needed. <see cref="ScoreFromBase"/> bundles that pattern.</para>
/// </summary>
public sealed class PlanarFlowTransform : ITransform
{
    /// <summary>u parameter (length D).</summary>
    public float[] U { get; }
    /// <summary>w parameter (length D).</summary>
    public float[] W { get; }
    /// <summary>scalar bias.</summary>
    public float B { get; }
    /// <summary>Event dimension D.</summary>
    public int D => W.Length;

    /// <summary>Build a planar flow.</summary>
    public PlanarFlowTransform(float[] u, float[] w, float b)
    {
        if (u == null) throw new ArgumentNullException(nameof(u));
        if (w == null) throw new ArgumentNullException(nameof(w));
        if (u.Length != w.Length) throw new ArgumentException("u and w must have the same length.");
        if (w.Length == 0) throw new ArgumentException("w must be non-empty.", nameof(w));
        // Reject the degenerate w = 0 case — CorrectedU's w² normaliser would divide by 0.
        float w2 = 0f;
        for (int i = 0; i < w.Length; i++) w2 += w[i] * w[i];
        if (!(w2 > 0f))
            throw new ArgumentException("w must have non-zero norm (the planar flow is undefined when w == 0).", nameof(w));
        U = (float[])u.Clone(); W = (float[])w.Clone(); B = b;
    }

    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => RealConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public bool IsDimensionPreserving => true;

    private float[] CorrectedU()
    {
        // Enforce wᵀu ≥ -1 to guarantee invertibility (Rezende & Mohamed Appendix A.1).
        float wTu = 0f, w2 = 0f;
        for (int i = 0; i < D; i++) { wTu += W[i] * U[i]; w2 += W[i] * W[i]; }
        float m = -1f + (float)Math.Log(1.0 + Math.Exp(wTu)) - wTu;
        var u = new float[D];
        for (int i = 0; i < D; i++) u[i] = U[i] + m * W[i] / w2;
        return u;
    }

    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (x.Length % D != 0)
            throw new ArgumentException(
                $"x.Length ({x.Length}) must be a multiple of D ({D}).", nameof(x));
        int batch = x.Length / D;
        var u = CorrectedU();
        var y = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float pre = B;
            for (int i = 0; i < D; i++) pre += W[i] * x[b * D + i];
            float h = MathF.Tanh(pre);
            for (int i = 0; i < D; i++) y[b * D + i] = x[b * D + i] + u[i] * h;
        }
        return y;
    }

    /// <inheritdoc />
    public float[] Inverse(float[] y) =>
        throw new NotSupportedException("Planar flow inverse has no closed form; use a different flow or a numerical inverse.");

    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (x.Length % D != 0)
            throw new ArgumentException(
                $"x.Length ({x.Length}) must be a multiple of D ({D}).", nameof(x));
        int batch = x.Length / D;
        var u = CorrectedU();
        var ldj = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float pre = B;
            for (int i = 0; i < D; i++) pre += W[i] * x[b * D + i];
            float dh = 1f - MathF.Tanh(pre) * MathF.Tanh(pre); // sech²
            float wTu = 0f;
            for (int i = 0; i < D; i++) wTu += W[i] * u[i];
            float perBatch = MathF.Log(MathF.Abs(1f + wTu * dh));
            for (int i = 0; i < D; i++) ldj[b * D + i] = perBatch / D; // distribute equally per dim
        }
        return ldj;
    }

    /// <summary>
    /// Score the transformed sample <c>y = Forward(z)</c> using the base log-density
    /// <paramref name="logProbBase"/> at the pre-image <paramref name="z"/>:
    /// <c>log p_y(y) = log p_z(z) − Σ log|det J(z)|</c>. Avoids needing the inverse,
    /// which has no closed form for planar flows. Length = batch.
    /// </summary>
    public float[] ScoreFromBase(float[] z, float[] logProbBase)
    {
        if (z == null) throw new ArgumentNullException(nameof(z));
        if (logProbBase == null) throw new ArgumentNullException(nameof(logProbBase));
        if (z.Length % D != 0)
            throw new ArgumentException($"z.Length ({z.Length}) must be a multiple of D ({D}).", nameof(z));
        int batch = z.Length / D;
        if (logProbBase.Length != batch)
            throw new ArgumentException($"logProbBase.Length ({logProbBase.Length}) must equal batch ({batch}).", nameof(logProbBase));
        var y = Forward(z);
        var ldj = LogAbsDetJacobian(z, y);
        var lp = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            float acc = logProbBase[b];
            for (int i = 0; i < D; i++) acc -= ldj[b * D + i];
            lp[b] = acc;
        }
        return lp;
    }
}

/// <summary>
/// Radial flow (Rezende &amp; Mohamed, 2015): y = x + β · h(α, r) · (x − x₀) with r = ‖x − x₀‖.
/// <para><b>Limitation:</b> like <see cref="PlanarFlowTransform"/>, the inverse requires a 1-D
/// root solve and is not bundled. Score samples via the variational pattern
/// <see cref="ScoreFromBase"/> instead of <c>TransformedDistribution.LogProb</c>.</para>
/// </summary>
public sealed class RadialFlowTransform : ITransform
{
    /// <summary>Reference point x₀ (length D).</summary>
    public float[] X0 { get; }
    /// <summary>Scale α &gt; 0.</summary>
    public float Alpha { get; }
    /// <summary>Strength β (corrected internally).</summary>
    public float Beta { get; }
    /// <summary>Event dimension D.</summary>
    public int D => X0.Length;

    /// <summary>Build a radial flow.</summary>
    public RadialFlowTransform(float[] x0, float alpha, float beta)
    {
        if (x0 == null) throw new ArgumentNullException(nameof(x0));
        if (x0.Length == 0)
            throw new ArgumentException("x0 must be non-empty (the flow dimension D is x0.Length).", nameof(x0));
        if (alpha <= 0f) throw new ArgumentException("alpha > 0.");
        X0 = (float[])x0.Clone(); Alpha = alpha; Beta = beta;
    }

    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => RealConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public bool IsDimensionPreserving => true;

    private float CorrectedBeta()
    {
        // β ≥ −α to guarantee invertibility.
        return -Alpha + (float)Math.Log(1.0 + Math.Exp(Beta));
    }

    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (x.Length % D != 0)
            throw new ArgumentException($"x.Length ({x.Length}) must be a multiple of D ({D}).", nameof(x));
        int batch = x.Length / D;
        float beta = CorrectedBeta();
        var y = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float r = 0f;
            for (int i = 0; i < D; i++) { var d = x[b * D + i] - X0[i]; r += d * d; }
            r = MathF.Sqrt(r);
            float h = 1f / (Alpha + r);
            for (int i = 0; i < D; i++) y[b * D + i] = x[b * D + i] + beta * h * (x[b * D + i] - X0[i]);
        }
        return y;
    }

    /// <inheritdoc />
    public float[] Inverse(float[] y) =>
        throw new NotSupportedException("Radial flow inverse requires a 1-D root solve; not bundled.");

    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (x.Length % D != 0)
            throw new ArgumentException($"x.Length ({x.Length}) must be a multiple of D ({D}).", nameof(x));
        int batch = x.Length / D;
        float beta = CorrectedBeta();
        var ldj = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float r = 0f;
            for (int i = 0; i < D; i++) { var d = x[b * D + i] - X0[i]; r += d * d; }
            r = MathF.Sqrt(r);
            float h = 1f / (Alpha + r);
            float hPrime = -h * h;
            float perBatch = (D - 1) * MathF.Log(MathF.Abs(1f + beta * h)) + MathF.Log(MathF.Abs(1f + beta * h + beta * hPrime * r));
            for (int i = 0; i < D; i++) ldj[b * D + i] = perBatch / D;
        }
        return ldj;
    }

    /// <summary>Same forward-only score helper as <see cref="PlanarFlowTransform.ScoreFromBase"/>.</summary>
    public float[] ScoreFromBase(float[] z, float[] logProbBase)
    {
        if (z == null) throw new ArgumentNullException(nameof(z));
        if (logProbBase == null) throw new ArgumentNullException(nameof(logProbBase));
        if (z.Length % D != 0)
            throw new ArgumentException($"z.Length ({z.Length}) must be a multiple of D ({D}).", nameof(z));
        int batch = z.Length / D;
        if (logProbBase.Length != batch)
            throw new ArgumentException($"logProbBase.Length ({logProbBase.Length}) must equal batch ({batch}).", nameof(logProbBase));
        var y = Forward(z);
        var ldj = LogAbsDetJacobian(z, y);
        var lp = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            float acc = logProbBase[b];
            for (int i = 0; i < D; i++) acc -= ldj[b * D + i];
            lp[b] = acc;
        }
        return lp;
    }
}

/// <summary>
/// RealNVP coupling layer (Dinh et al., 2017): split x into [x_a, x_b] by a binary mask;
/// pass x_a through unchanged and apply x_b ← x_b · exp(s(x_a)) + t(x_a) where s, t are
/// affine functions of x_a (here we expose simple linear s(x_a) = sScale·x_a + sBias and
/// likewise for t — sufficient for unit tests; users plug deeper networks via composition).
/// </summary>
public sealed class RealNvpCouplingTransform : ITransform
{
    /// <summary>Boolean mask of length D — true entries are passed unchanged; false entries are transformed.</summary>
    public bool[] Mask { get; }
    /// <summary>Linear scale parameters s for the transformed entries.</summary>
    public float[] SScale { get; }
    /// <summary>Linear bias parameters s for the transformed entries.</summary>
    public float[] SBias { get; }
    /// <summary>Linear scale parameters t for the transformed entries.</summary>
    public float[] TScale { get; }
    /// <summary>Linear bias parameters t for the transformed entries.</summary>
    public float[] TBias { get; }
    /// <summary>Event dimension.</summary>
    public int D => Mask.Length;

    /// <summary>Build a RealNVP coupling layer.</summary>
    public RealNvpCouplingTransform(bool[] mask, float[] sScale, float[] sBias, float[] tScale, float[] tBias)
    {
        if (sScale.Length != mask.Length || sBias.Length != mask.Length
            || tScale.Length != mask.Length || tBias.Length != mask.Length)
            throw new ArgumentException("mask, s, t parameters must all be length D.");
        Mask = mask; SScale = sScale; SBias = sBias; TScale = tScale; TBias = tBias;
    }

    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => RealConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public bool IsDimensionPreserving => true;

    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        int batch = x.Length / D;
        var y = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            // First compute the active "context" sum from masked-true entries.
            float ctxS = 0f, ctxT = 0f;
            for (int i = 0; i < D; i++) if (Mask[i]) { ctxS += SScale[i] * x[b * D + i]; ctxT += TScale[i] * x[b * D + i]; }
            for (int i = 0; i < D; i++)
            {
                if (Mask[i]) y[b * D + i] = x[b * D + i];
                else
                {
                    float s = ctxS + SBias[i];
                    float t = ctxT + TBias[i];
                    y[b * D + i] = x[b * D + i] * MathF.Exp(s) + t;
                }
            }
        }
        return y;
    }

    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        int batch = y.Length / D;
        var x = new float[y.Length];
        for (int b = 0; b < batch; b++)
        {
            float ctxS = 0f, ctxT = 0f;
            for (int i = 0; i < D; i++) if (Mask[i]) { ctxS += SScale[i] * y[b * D + i]; ctxT += TScale[i] * y[b * D + i]; }
            for (int i = 0; i < D; i++)
            {
                if (Mask[i]) x[b * D + i] = y[b * D + i];
                else
                {
                    float s = ctxS + SBias[i];
                    float t = ctxT + TBias[i];
                    x[b * D + i] = (y[b * D + i] - t) * MathF.Exp(-s);
                }
            }
        }
        return x;
    }

    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        int batch = x.Length / D;
        var ldj = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float ctxS = 0f;
            for (int i = 0; i < D; i++) if (Mask[i]) ctxS += SScale[i] * x[b * D + i];
            float perEvent = 0f;
            for (int i = 0; i < D; i++) if (!Mask[i]) perEvent += ctxS + SBias[i];
            for (int i = 0; i < D; i++) ldj[b * D + i] = perEvent / D;
        }
        return ldj;
    }
}
