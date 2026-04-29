using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions.Transforms;

/// <summary>
/// Masked Autoregressive Flow (MAF) transform (Papamakarios et al., 2017).
///
/// True autoregressive flow: μ_i and s_i are computed at evaluation time from the
/// already-known prefix x_{1:i-1} via a user-supplied <see cref="Conditioner"/>
/// callback, so the same transform handles arbitrary inputs (not just the one its
/// parameters happened to be precomputed from).
///
/// Forward (analytic, parallel across event dim):
///   y_i = (x_i − μ_i(x_{1:i-1})) · exp(−s_i(x_{1:i-1}))
/// Inverse (sequential — the cost MAF accepts to make log_prob fast):
///   x_i = y_i · exp(s_i(x_{1:i-1})) + μ_i(x_{1:i-1})
/// log|det J| = −Σ s_i.
/// </summary>
public sealed class MaskedAutoregressiveFlowTransform : ITransform
{
    /// <summary>Conditioner: given the dim index <c>i</c> and the already-known prefix
    /// <c>x_{0..i-1}</c> (length <c>i</c>), returns <c>(μ_i, s_i)</c>. Same callback is
    /// invoked during forward, inverse, and log|det J| evaluation so the autoregressive
    /// contract holds for arbitrary inputs.</summary>
    public Func<int, float[], (float Mu, float LogScale)> Conditioner { get; }
    /// <summary>Event dimension D.</summary>
    public int D { get; }

    /// <summary>Build a MAF transform with a per-dim conditioner.</summary>
    public MaskedAutoregressiveFlowTransform(int d, Func<int, float[], (float Mu, float LogScale)> conditioner)
    {
        if (d <= 0) throw new ArgumentOutOfRangeException(nameof(d));
        Conditioner = conditioner ?? throw new ArgumentNullException(nameof(conditioner));
        D = d;
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
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (x.Length % D != 0) throw new ArgumentException($"x.Length must be a multiple of D={D}.");
        int batch = x.Length / D;
        var y = new float[x.Length];
        var prefix = new float[D];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < D; i++) prefix[i] = x[b * D + i];
            for (int i = 0; i < D; i++)
            {
                var slice = new float[i];
                Array.Copy(prefix, 0, slice, 0, i);
                var (mu, s) = Conditioner(i, slice);
                y[b * D + i] = (prefix[i] - mu) * MathF.Exp(-s);
            }
        }
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        if (y == null) throw new ArgumentNullException(nameof(y));
        if (y.Length % D != 0) throw new ArgumentException($"y.Length must be a multiple of D={D}.");
        int batch = y.Length / D;
        var x = new float[y.Length];
        var prefix = new float[D];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < D; i++)
            {
                var slice = new float[i];
                Array.Copy(prefix, 0, slice, 0, i);
                var (mu, s) = Conditioner(i, slice);
                prefix[i] = y[b * D + i] * MathF.Exp(s) + mu;
                x[b * D + i] = prefix[i];
            }
        }
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (x.Length % D != 0) throw new ArgumentException();
        var ldj = new float[x.Length];
        int batch = x.Length / D;
        var prefix = new float[D];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < D; i++) prefix[i] = x[b * D + i];
            float per = 0f;
            for (int i = 0; i < D; i++)
            {
                var slice = new float[i];
                Array.Copy(prefix, 0, slice, 0, i);
                var (_, s) = Conditioner(i, slice);
                per -= s;
            }
            for (int i = 0; i < D; i++) ldj[b * D + i] = per / D;
        }
        return ldj;
    }
}

/// <summary>
/// Neural Spline Flow (NSF) transform (Durkan et al., 2019), rational-quadratic spline variant.
/// Maps each event dim through a monotonic rational-quadratic spline parameterised by K knot
/// positions / heights / derivatives. Inputs outside <c>[Lower, Upper]</c> pass through
/// unchanged (linear tails at the boundary slopes).
/// </summary>
public sealed class NeuralSplineFlowTransform : ITransform
{
    /// <summary>Lower bound of the spline support.</summary>
    public float Lower { get; }
    /// <summary>Upper bound of the spline support.</summary>
    public float Upper { get; }
    /// <summary>Number of bins K (must be ≥ 1).</summary>
    public int Bins { get; }
    /// <summary>Per-batch knot widths Δx_k (length BatchSize·K, sums to Upper - Lower per batch).</summary>
    public float[] Widths { get; }
    /// <summary>Per-batch knot heights Δy_k (length BatchSize·K, sums to Upper - Lower per batch).</summary>
    public float[] Heights { get; }
    /// <summary>Per-batch knot derivatives d_k at internal knots (length BatchSize·(K+1), positive).</summary>
    public float[] Derivatives { get; }
    /// <summary>Event dimension (per-element spline; the same parameters are applied independently to each entry).</summary>
    public int D { get; }

    /// <summary>Build an NSF rational-quadratic spline transform.</summary>
    public NeuralSplineFlowTransform(
        float lower, float upper, int bins,
        float[] widths, float[] heights, float[] derivatives, int d)
    {
        if (upper <= lower) throw new ArgumentException();
        if (bins < 1) throw new ArgumentException();
        if (widths.Length != heights.Length) throw new ArgumentException();
        if (widths.Length % bins != 0) throw new ArgumentException("widths.Length must be a multiple of bins.");
        int batch = widths.Length / bins;
        if (derivatives.Length != batch * (bins + 1)) throw new ArgumentException();
        for (int i = 0; i < derivatives.Length; i++) if (!(derivatives[i] > 0f)) throw new ArgumentException("derivatives > 0.");
        Lower = lower; Upper = upper; Bins = bins; Widths = widths; Heights = heights; Derivatives = derivatives; D = d;
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
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            int batchIdx = (i / D) % (Widths.Length / Bins);  // share params across event dims
            y[i] = ApplySpline(x[i], batchIdx, forward: true, out _);
        }
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        var x = new float[y.Length];
        for (int i = 0; i < y.Length; i++)
        {
            int batchIdx = (i / D) % (Widths.Length / Bins);
            x[i] = ApplySpline(y[i], batchIdx, forward: false, out _);
        }
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        var ldj = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            int batchIdx = (i / D) % (Widths.Length / Bins);
            ApplySpline(x[i], batchIdx, forward: true, out float dlog);
            ldj[i] = dlog;
        }
        return ldj;
    }

    private float ApplySpline(float input, int batchIdx, bool forward, out float logDeriv)
    {
        // Linear identity outside [Lower, Upper].
        if (input < Lower || input > Upper) { logDeriv = 0f; return input; }

        // Compute cumulative knot positions for this batch.
        float xCum = Lower, yCum = Lower;
        int bin = -1;
        float xWidth = 0f, yHeight = 0f;
        for (int k = 0; k < Bins; k++)
        {
            float w = Widths[batchIdx * Bins + k];
            float h = Heights[batchIdx * Bins + k];
            float xNext = xCum + w;
            float yNext = yCum + h;
            // The forward bin contains input; the inverse bin contains input on y-axis.
            bool hit = forward
                ? input >= xCum && input <= xNext
                : input >= yCum && input <= yNext;
            if (hit) { bin = k; xWidth = w; yHeight = h; break; }
            xCum = xNext; yCum = yNext;
        }
        if (bin < 0) { logDeriv = 0f; return input; }

        float dk  = Derivatives[batchIdx * (Bins + 1) + bin];
        float dk1 = Derivatives[batchIdx * (Bins + 1) + bin + 1];
        float s = yHeight / xWidth;

        if (forward)
        {
            float xi = (input - xCum) / xWidth;
            float numer = yHeight * (s * xi * xi + dk * xi * (1f - xi));
            float denom = s + (dk + dk1 - 2f * s) * xi * (1f - xi);
            float y = yCum + numer / denom;
            // Derivative for log_abs_det:
            float t = (s * s) * (dk1 * xi * xi + 2f * s * xi * (1f - xi) + dk * (1f - xi) * (1f - xi));
            logDeriv = MathF.Log(t) - 2f * MathF.Log(denom);
            return y;
        }
        else
        {
            // Inverse via solving the quadratic numer/denom = (y − yCum)/yHeight for xi.
            float yRel = (input - yCum) / yHeight;
            float a = yHeight * (s - dk) + (input - yCum) * (dk + dk1 - 2f * s);
            float b = yHeight * dk - (input - yCum) * (dk + dk1 - 2f * s);
            float c = -s * (input - yCum);
            float disc = b * b - 4f * a * c;
            if (disc < 0) disc = 0;
            float xi = 2f * c / (-b - MathF.Sqrt(disc));
            float x = xCum + xi * xWidth;
            float t = (s * s) * (dk1 * xi * xi + 2f * s * xi * (1f - xi) + dk * (1f - xi) * (1f - xi));
            float denom = s + (dk + dk1 - 2f * s) * xi * (1f - xi);
            logDeriv = -(MathF.Log(t) - 2f * MathF.Log(denom));
            _ = yRel;
            return x;
        }
    }
}
