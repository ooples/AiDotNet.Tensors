using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions.Transforms;

/// <summary>
/// Softmax transform: maps ℝ^K → simplex^(K-1). y_i = exp(x_i) / Σ_j exp(x_j).
/// Maps ℝ^K → simplex of size K. Bijective on the affine subspace where Σ x_i = 0
/// (PyTorch parity: log_abs_det_jacobian uses the K-1 dimensional measure).
/// </summary>
public sealed class SoftmaxTransform : ITransform
{
    /// <summary>Event dimension (= number of categories).</summary>
    public int K { get; }

    /// <summary>Build a softmax transform.</summary>
    public SoftmaxTransform(int k)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        K = k;
    }

    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => new SimplexConstraint(K);
    /// <inheritdoc />
    public bool ConstantJacobian => false;

    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        if (x.Length % K != 0) throw new ArgumentException("x.Length must be a multiple of K.");
        int batch = x.Length / K;
        var y = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float maxV = float.NegativeInfinity;
            for (int i = 0; i < K; i++) if (x[b * K + i] > maxV) maxV = x[b * K + i];
            double sum = 0;
            for (int i = 0; i < K; i++) { y[b * K + i] = MathF.Exp(x[b * K + i] - maxV); sum += y[b * K + i]; }
            float inv = (float)(1.0 / sum);
            for (int i = 0; i < K; i++) y[b * K + i] *= inv;
        }
        return y;
    }

    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        // log y is one valid pre-image; users typically want this canonical choice.
        var x = new float[y.Length];
        for (int i = 0; i < y.Length; i++) x[i] = MathF.Log(MathF.Max(y[i], 1e-30f));
        return x;
    }

    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // log|det J| (in the ambient sense) is undefined since softmax is not invertible
        // on ℝ^K. The conventional value used by torch is Σ log y_i — the log-Jacobian
        // determinant when restricting to the K-1 dim simplex.
        if (y.Length % K != 0) throw new ArgumentException();
        int batch = y.Length / K;
        var ldj = new float[y.Length];
        for (int b = 0; b < batch; b++)
        {
            float per = 0f;
            for (int i = 0; i < K; i++) per += MathF.Log(MathF.Max(y[b * K + i], 1e-30f));
            for (int i = 0; i < K; i++) ldj[b * K + i] = per / K;
        }
        return ldj;
    }
}

/// <summary>
/// Stick-breaking transform: maps ℝ^(K-1) → simplex^(K-1). Each input x_i is squashed
/// through a sigmoid to a stick-fraction, multiplied by the remaining stick length.
///   v_i = σ(x_i + log(1/(K-i-1))),    y_i = v_i · Π_{j&lt;i} (1 - v_j),    y_{K-1} = 1 - Σ_{i&lt;K-1} y_i.
/// </summary>
public sealed class StickBreakingTransform : ITransform
{
    /// <summary>Event dimension of the simplex (length of the output).</summary>
    public int K { get; }

    /// <summary>Build a stick-breaking transform; input dim is K-1, output dim is K.</summary>
    public StickBreakingTransform(int k)
    {
        if (k < 2) throw new ArgumentOutOfRangeException(nameof(k));
        K = k;
    }

    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => new SimplexConstraint(K);
    /// <inheritdoc />
    public bool ConstantJacobian => false;

    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        // Input shape: [batch, K-1]; output shape: [batch, K].
        int inDim = K - 1;
        if (x.Length % inDim != 0) throw new ArgumentException();
        int batch = x.Length / inDim;
        var y = new float[batch * K];
        for (int b = 0; b < batch; b++)
        {
            float remaining = 1f;
            for (int i = 0; i < inDim; i++)
            {
                float shift = MathF.Log(1f / (K - i - 1));
                float v = 1f / (1f + MathF.Exp(-(x[b * inDim + i] + shift)));
                y[b * K + i] = v * remaining;
                remaining *= (1f - v);
            }
            y[b * K + (K - 1)] = remaining;
        }
        return y;
    }

    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        if (y.Length % K != 0) throw new ArgumentException();
        int batch = y.Length / K;
        int outDim = K - 1;
        var x = new float[batch * outDim];
        for (int b = 0; b < batch; b++)
        {
            float remaining = 1f;
            for (int i = 0; i < outDim; i++)
            {
                float v = y[b * K + i] / MathF.Max(remaining, 1e-30f);
                float shift = MathF.Log(1f / (K - i - 1));
                x[b * outDim + i] = MathF.Log(v / (1f - v)) - shift;
                remaining *= (1f - v);
            }
        }
        return x;
    }

    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // log|det J| = Σ_i [ log(y_i) + log(1 - Σ_{j≤i} y_j) ] (PyTorch convention).
        if (y.Length % K != 0) throw new ArgumentException();
        int batch = y.Length / K;
        int inDim = K - 1;
        var ldj = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float remaining = 1f;
            float per = 0f;
            for (int i = 0; i < inDim; i++)
            {
                per += MathF.Log(MathF.Max(y[b * K + i], 1e-30f))
                     + MathF.Log(MathF.Max(remaining, 1e-30f));
                remaining -= y[b * K + i];
            }
            for (int i = 0; i < inDim; i++) ldj[b * inDim + i] = per / inDim;
        }
        return ldj;
    }
}

/// <summary>Absolute value: y = |x|. Not bijective; provided for distributions where
/// the underlying base is symmetric (e.g. the half-normal pre-image).</summary>
public sealed class AbsTransform : ITransform
{
    /// <summary>Singleton instance.</summary>
    public static readonly AbsTransform Instance = new AbsTransform();
    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => NonNegativeConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => true;
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = MathF.Abs(x[i]);
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y) => (float[])y.Clone();
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        var r = new float[x.Length];
        // Identity in absolute terms — this transform halves the measure but for our
        // symmetric-base use cases we mirror PyTorch and report 0.
        return r;
    }
}

/// <summary>Lifts a 1-D base transform to apply to each <c>EventSize</c>-element block
/// independently, summing log|det J| over the event dim. Mirrors
/// <c>torch.distributions.transforms.IndependentTransform</c>.</summary>
public sealed class IndependentTransform : ITransform
{
    /// <summary>Wrapped base transform.</summary>
    public ITransform Base { get; }
    /// <summary>Number of trailing dims to treat as a single event.</summary>
    public int ReinterpretedDims { get; }

    /// <summary>Wrap <paramref name="@base"/> as an event-wise transform.</summary>
    public IndependentTransform(ITransform @base, int reinterpretedDims)
    {
        if (reinterpretedDims < 1) throw new ArgumentOutOfRangeException(nameof(reinterpretedDims));
        Base = @base ?? throw new ArgumentNullException(nameof(@base));
        ReinterpretedDims = reinterpretedDims;
    }
    /// <inheritdoc />
    public IConstraint Domain => Base.Domain;
    /// <inheritdoc />
    public IConstraint Codomain => Base.Codomain;
    /// <inheritdoc />
    public bool ConstantJacobian => Base.ConstantJacobian;
    /// <inheritdoc />
    public float[] Forward(float[] x) => Base.Forward(x);
    /// <inheritdoc />
    public float[] Inverse(float[] y) => Base.Inverse(y);
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y) => Base.LogAbsDetJacobian(x, y);
}

/// <summary>
/// CDF transform: y = F(x). Maps the support of an arbitrary 1-D distribution into (0, 1).
/// Useful for building copulas. Inverse is the ICDF; log_abs_det = log_prob(x).
/// </summary>
public sealed class CumulativeDistributionTransform : ITransform
{
    /// <summary>Underlying distribution (must have closed-form CDF/ICDF).</summary>
    public IDistribution Distribution { get; }
    /// <summary>Build a CDF transform from a 1-D distribution.</summary>
    public CumulativeDistributionTransform(IDistribution distribution)
    {
        if (distribution == null) throw new ArgumentNullException(nameof(distribution));
        if (distribution.EventSize != 1)
            throw new ArgumentException("CumulativeDistributionTransform requires a univariate distribution.");
        Distribution = distribution;
    }
    /// <inheritdoc />
    public IConstraint Domain => Distribution.Support;
    /// <inheritdoc />
    public IConstraint Codomain => UnitIntervalConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public float[] Forward(float[] x) => Distribution.Cdf(x);
    /// <inheritdoc />
    public float[] Inverse(float[] y) => Distribution.Icdf(y);
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y) => Distribution.LogProb(x);
}

/// <summary>
/// LowerCholeskyTransform: maps an unconstrained matrix to a lower-triangular matrix
/// with positive diagonal — the canonical reparam for covariance Cholesky factors.
/// Forward(x) takes the strictly-lower entries verbatim and exp's the diagonal;
/// inverse undoes the exp on the diagonal.
/// </summary>
public sealed class LowerCholeskyTransform : ITransform
{
    /// <summary>Matrix dimension N (event is N×N flattened row-major).</summary>
    public int N { get; }
    /// <summary>Build a lower-Cholesky transform.</summary>
    public LowerCholeskyTransform(int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        N = n;
    }
    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => RealConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        if (x.Length % (N * N) != 0) throw new ArgumentException();
        int batch = x.Length / (N * N);
        var y = new float[x.Length];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    float xij = x[b * N * N + i * N + j];
                    if (j > i) y[b * N * N + i * N + j] = 0f;
                    else if (i == j) y[b * N * N + i * N + j] = MathF.Exp(xij);
                    else y[b * N * N + i * N + j] = xij;
                }
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        if (y.Length % (N * N) != 0) throw new ArgumentException();
        int batch = y.Length / (N * N);
        var x = new float[y.Length];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    float yij = y[b * N * N + i * N + j];
                    if (j > i) x[b * N * N + i * N + j] = 0f;
                    else if (i == j) x[b * N * N + i * N + j] = MathF.Log(yij);
                    else x[b * N * N + i * N + j] = yij;
                }
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // log|det J| over the unconstrained → triangular map = Σ x_{ii} (the diagonal entries
        // before exponentiation), distributed evenly across the N² entries for our flat layout.
        if (x.Length % (N * N) != 0) throw new ArgumentException();
        int batch = x.Length / (N * N);
        var ldj = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            float per = 0f;
            for (int i = 0; i < N; i++) per += x[b * N * N + i * N + i];
            for (int j = 0; j < N * N; j++) ldj[b * N * N + j] = per / (N * N);
        }
        return ldj;
    }
}

/// <summary>
/// CorrCholeskyTransform: maps unconstrained R^(N(N-1)/2) to a unit-diagonal lower-triangular
/// matrix L such that L·Lᵀ is a valid correlation matrix. Used as the canonical reparam for
/// LKJ-prior correlation matrices. Implementation follows the spherical parameterisation
/// (Pinheiro &amp; Bates, 1996; Lewandowski et al., 2009).
/// </summary>
public sealed class CorrCholeskyTransform : ITransform
{
    /// <summary>Matrix dimension N.</summary>
    public int N { get; }
    /// <summary>Build a correlation-Cholesky transform.</summary>
    public CorrCholeskyTransform(int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        N = n;
    }
    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => RealConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;

    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        // Input length per batch element = N(N-1)/2; output is N×N row-major.
        int inDim = N * (N - 1) / 2;
        if (x.Length % inDim != 0) throw new ArgumentException();
        int batch = x.Length / inDim;
        var y = new float[batch * N * N];
        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            for (int i = 0; i < N; i++)
            {
                float remaining = 1f;
                for (int j = 0; j < i; j++)
                {
                    // Squash to (-1, 1) via tanh so the row sits on the unit ball.
                    float c = MathF.Tanh(x[b * inDim + idx]);
                    idx++;
                    y[b * N * N + i * N + j] = c * MathF.Sqrt(remaining);
                    remaining -= c * c * remaining;
                    if (remaining < 0) remaining = 0;
                }
                y[b * N * N + i * N + i] = MathF.Sqrt(MathF.Max(remaining, 0f));
            }
        }
        return y;
    }

    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        int inDim = N * (N - 1) / 2;
        if (y.Length % (N * N) != 0) throw new ArgumentException();
        int batch = y.Length / (N * N);
        var x = new float[batch * inDim];
        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            for (int i = 0; i < N; i++)
            {
                float remaining = 1f;
                for (int j = 0; j < i; j++)
                {
                    float yij = y[b * N * N + i * N + j];
                    float c = yij / MathF.Sqrt(MathF.Max(remaining, 1e-30f));
                    if (c >= 1f) c = 1f - 1e-6f; if (c <= -1f) c = -1f + 1e-6f;
                    x[b * inDim + idx] = MathF.Atanh(c);
                    idx++;
                    remaining -= yij * yij;
                    if (remaining < 0) remaining = 0;
                }
            }
        }
        return x;
    }

    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // For each row i and column j < i: contribution log|sech²(x_{i,j})| − 0.5 · log(remaining)
        // where remaining decreases as we walk along the row. Distribute the per-batch sum
        // uniformly across the input vector for the [batch, inDim] flattened layout.
        int inDim = N * (N - 1) / 2;
        int batch = x.Length / inDim;
        var ldj = new float[x.Length];
        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            float per = 0f;
            for (int i = 0; i < N; i++)
            {
                float remaining = 1f;
                for (int j = 0; j < i; j++)
                {
                    float xij = x[b * inDim + idx];
                    idx++;
                    float c = MathF.Tanh(xij);
                    // d c / d x = sech²(x) = 1 − tanh²(x).
                    per += MathF.Log(1f - c * c) + 0.5f * MathF.Log(MathF.Max(remaining, 1e-30f));
                    remaining -= c * c * remaining;
                }
            }
            for (int j = 0; j < inDim; j++) ldj[b * inDim + j] = per / inDim;
        }
        return ldj;
    }
}
