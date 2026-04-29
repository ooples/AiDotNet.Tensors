// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NN.Parametrizations;

/// <summary>
/// Parametrization framework — wrap a parameter with a reparam (e.g.
/// "always positive" via exp, or "spectral-normalized" via power
/// iteration) and the wrapper transparently emits the parameterized
/// value on every forward call. Mirrors <c>torch.nn.utils.parametrize</c>.
///
/// <para>Lifecycle:
/// <list type="number">
///   <item><see cref="Parametrize{T}.Wrap"/> — bind a reparam to a parameter.</item>
///   <item><see cref="IParametrization{T}.Forward"/> — called every time the wrapped value is needed.</item>
///   <item><see cref="Parametrize{T}.Remove"/> — collapse the reparam into the underlying parameter; the wrapper goes away.</item>
/// </list>
/// </para>
/// </summary>
public interface IParametrization<T>
{
    /// <summary>Maps the underlying parameter to its reparameterized
    /// form. Called every forward pass that touches the parameter.</summary>
    Tensor<T> Forward(Tensor<T> raw);
}

/// <summary>Wrapper around a base parameter + a parametrization. The
/// effective tensor is <c>parametrization.Forward(Raw)</c>.</summary>
public sealed class ParametrizedParameter<T>
{
    /// <summary>The underlying raw tensor — the optimizer updates this
    /// directly, and the parametrization runs on top.</summary>
    public Tensor<T> Raw { get; private set; }

    /// <summary>The active parametrization. Replaceable via
    /// <see cref="Parametrize{T}.Wrap"/>.</summary>
    public IParametrization<T> Parametrization { get; private set; }

    internal ParametrizedParameter(Tensor<T> raw, IParametrization<T> parametrization)
    {
        Raw = raw;
        Parametrization = parametrization;
    }

    /// <summary>Returns the parameterized tensor —
    /// <c>parametrization.Forward(Raw)</c>.</summary>
    public Tensor<T> Forward() => Parametrization.Forward(Raw);
}

/// <summary>Static facade over the parametrization framework.</summary>
public static class Parametrize<T>
{
    /// <summary>Wrap <paramref name="raw"/> with <paramref name="parametrization"/>.
    /// Returns a <see cref="ParametrizedParameter{T}"/> that exposes
    /// <c>Forward()</c> for callers and keeps <c>Raw</c> available for
    /// optimizer updates.</summary>
    public static ParametrizedParameter<T> Wrap(Tensor<T> raw, IParametrization<T> parametrization)
    {
        if (raw is null) throw new ArgumentNullException(nameof(raw));
        if (parametrization is null) throw new ArgumentNullException(nameof(parametrization));
        return new ParametrizedParameter<T>(raw, parametrization);
    }

    /// <summary>Consolidate the parametrization into the raw tensor —
    /// equivalent to <c>parametrize.remove_parametrizations</c>. The
    /// wrapper becomes a plain <see cref="Tensor{T}"/> matching the
    /// last forward output.</summary>
    public static Tensor<T> Remove(ParametrizedParameter<T> wrapped)
        => wrapped.Forward();
}

/// <summary>Weight normalization — factors a tensor into direction
/// (<c>v / ||v||</c>) + magnitude <c>g</c> so the optimizer sees the
/// magnitude as a scalar variable. Mirrors
/// <c>torch.nn.utils.weight_norm</c>.</summary>
public sealed class WeightNorm<T> : IParametrization<T>
{
    /// <summary>Per-row magnitude scale <c>g</c>; broadcast against the
    /// direction vector at forward time.</summary>
    public Tensor<T> G { get; }

    /// <summary>Axis along which to compute the L2 norm.</summary>
    public int Dim { get; }

    /// <summary>Constructs a weight-norm parametrization. Initialises
    /// <see cref="G"/> from the input's per-row L2 norm.</summary>
    public WeightNorm(Tensor<T> initialV, int dim = 0)
    {
        Dim = dim;
        var ops = MathHelper.GetNumericOperations<T>();
        if (initialV.Rank != 2)
            throw new ArgumentException("WeightNorm currently supports rank-2 inputs only.", nameof(initialV));

        int outer = initialV._shape[1 - dim];
        int inner = initialV._shape[dim];
        G = new Tensor<T>(new[] { outer });
        var src = initialV.AsSpan();
        var gSpan = G.AsWritableSpan();
        for (int o = 0; o < outer; o++)
        {
            double n = 0;
            for (int i = 0; i < inner; i++)
            {
                double v = ops.ToDouble(dim == 0 ? src[i * outer + o] : src[o * inner + i]);
                n += v * v;
            }
            gSpan[o] = ops.FromDouble(Math.Sqrt(n));
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> raw)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (raw.Rank != 2)
            throw new InvalidOperationException("WeightNorm forward: raw must be rank 2.");
        int outer = raw._shape[1 - Dim];
        int inner = raw._shape[Dim];
        var output = new Tensor<T>((int[])raw._shape.Clone());
        var src = raw.AsSpan();
        var dst = output.AsWritableSpan();
        var gSpan = G.AsSpan();
        for (int o = 0; o < outer; o++)
        {
            double n = 0;
            for (int i = 0; i < inner; i++)
            {
                double v = ops.ToDouble(Dim == 0 ? src[i * outer + o] : src[o * inner + i]);
                n += v * v;
            }
            double normInv = 1.0 / Math.Max(1e-12, Math.Sqrt(n));
            double g = ops.ToDouble(gSpan[o]);
            for (int i = 0; i < inner; i++)
            {
                int idx = Dim == 0 ? i * outer + o : o * inner + i;
                dst[idx] = ops.FromDouble(ops.ToDouble(src[idx]) * normInv * g);
            }
        }
        return output;
    }
}

/// <summary>Spectral normalization via power iteration. Each forward
/// runs <paramref name="iters"/> iterations of <c>u ← Wᵀv / ||Wᵀv||</c>
/// then divides W by its top singular value. Mirrors
/// <c>torch.nn.utils.spectral_norm</c>.</summary>
public sealed class SpectralNorm<T> : IParametrization<T>
{
    private readonly int _iters;
    private readonly IEngine _engine;
    private Tensor<T> _u;

    /// <summary>The most-recent estimated top singular vector. Updated
    /// in place each forward call.</summary>
    public Tensor<T> U => _u;

    /// <summary>Constructs spectral-norm with <paramref name="iters"/>
    /// power-iteration steps per forward.</summary>
    public SpectralNorm(int outFeatures, int iters = 1)
    {
        _iters = iters;
        _engine = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();
        _u = new Tensor<T>(new[] { outFeatures });
        var span = _u.AsWritableSpan();
        // Init u to a fixed vector (1, 0, …, 0) — converges from any
        // non-orthogonal start. Deterministic for tests.
        span[0] = ops.One;
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> raw)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (raw.Rank != 2)
            throw new InvalidOperationException("SpectralNorm forward: raw must be rank 2.");
        int outF = raw._shape[0];
        int inF = raw._shape[1];

        var u = _u;
        Tensor<T> v = new Tensor<T>(new[] { inF });

        for (int iter = 0; iter < _iters; iter++)
        {
            // v = Wᵀu / ||Wᵀu||
            var rawT = _engine.TensorTranspose(raw);
            v = MatVec(rawT, u, ops, inF, outF);
            NormalizeInPlace(v, ops);
            // u = Wv / ||Wv||
            u = MatVec(raw, v, ops, outF, inF);
            NormalizeInPlace(u, ops);
        }
        _u = u;

        // Top singular value σ = uᵀWv (one inner product after the loop).
        double sigma = 0;
        var rawSpan = raw.AsSpan();
        var uSpan = u.AsSpan();
        var vSpan = v.AsSpan();
        for (int o = 0; o < outF; o++)
            for (int i = 0; i < inF; i++)
                sigma += ops.ToDouble(uSpan[o]) * ops.ToDouble(rawSpan[o * inF + i]) * ops.ToDouble(vSpan[i]);

        if (sigma < 1e-12) sigma = 1.0;
        var output = new Tensor<T>((int[])raw._shape.Clone());
        var dst = output.AsWritableSpan();
        for (int i = 0; i < dst.Length; i++)
            dst[i] = ops.FromDouble(ops.ToDouble(rawSpan[i]) / sigma);
        return output;
    }

    private static Tensor<T> MatVec(Tensor<T> M, Tensor<T> v, Interfaces.INumericOperations<T> ops, int outDim, int inDim)
    {
        var output = new Tensor<T>(new[] { outDim });
        var Mspan = M.AsSpan();
        var vSpan = v.AsSpan();
        var dst = output.AsWritableSpan();
        for (int r = 0; r < outDim; r++)
        {
            double acc = 0;
            for (int c = 0; c < inDim; c++)
                acc += ops.ToDouble(Mspan[r * inDim + c]) * ops.ToDouble(vSpan[c]);
            dst[r] = ops.FromDouble(acc);
        }
        return output;
    }

    private static void NormalizeInPlace(Tensor<T> t, Interfaces.INumericOperations<T> ops)
    {
        var span = t.AsWritableSpan();
        double n = 0;
        for (int i = 0; i < span.Length; i++)
        {
            double v = ops.ToDouble(span[i]);
            n += v * v;
        }
        n = Math.Max(1e-12, Math.Sqrt(n));
        for (int i = 0; i < span.Length; i++)
            span[i] = ops.FromDouble(ops.ToDouble(span[i]) / n);
    }
}

/// <summary>Orthogonal parametrization via the Cayley transform —
/// <c>(I − A) · (I + A)^(-1)</c> where A is the skew-symmetrized raw.
/// Mirrors <c>torch.nn.utils.orthogonal</c>'s default Cayley path.
/// Matrix-exp variant is a follow-up gated on
/// <c>matrix_exp</c> from the linalg parity issue.</summary>
public sealed class OrthogonalParametrization<T> : IParametrization<T>
{
    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> raw)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (raw.Rank != 2 || raw._shape[0] != raw._shape[1])
            throw new InvalidOperationException("OrthogonalParametrization requires a square matrix.");
        int n = raw._shape[0];

        // Skew-symmetrize: A = (raw − rawᵀ) / 2.
        var A = new Tensor<T>(new[] { n, n });
        var aSpan = A.AsWritableSpan();
        var src = raw.AsSpan();
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++)
            {
                double diff = (ops.ToDouble(src[r * n + c]) - ops.ToDouble(src[c * n + r])) * 0.5;
                aSpan[r * n + c] = ops.FromDouble(diff);
            }

        // Build (I + A) and (I − A).
        var iPlusA = new double[n * n];
        var iMinusA = new double[n * n];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                double a = ops.ToDouble(aSpan[r * n + c]);
                double diag = r == c ? 1.0 : 0.0;
                iPlusA[r * n + c] = diag + a;
                iMinusA[r * n + c] = diag - a;
            }
        }

        // Solve (I + A) · X = (I − A) for X via Gaussian elimination.
        // Result X = (I + A)^(-1) · (I − A) is orthogonal.
        var X = SolveGeneral(iPlusA, iMinusA, n);
        var output = new Tensor<T>(new[] { n, n });
        var dst = output.AsWritableSpan();
        for (int i = 0; i < X.Length; i++) dst[i] = ops.FromDouble(X[i]);
        return output;
    }

    private static double[] SolveGeneral(double[] A, double[] B, int n)
    {
        // Augment A | B and run partial-pivot Gaussian elimination.
        // Returns X = A^(-1) · B as a flat row-major matrix.
        var aug = new double[n * (2 * n)];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++) aug[r * (2 * n) + c] = A[r * n + c];
            for (int c = 0; c < n; c++) aug[r * (2 * n) + n + c] = B[r * n + c];
        }
        for (int p = 0; p < n; p++)
        {
            // Pivot row.
            int pivot = p;
            double pivotMag = Math.Abs(aug[p * (2 * n) + p]);
            for (int r = p + 1; r < n; r++)
            {
                double m = Math.Abs(aug[r * (2 * n) + p]);
                if (m > pivotMag) { pivotMag = m; pivot = r; }
            }
            if (pivotMag < 1e-12)
                throw new InvalidOperationException("OrthogonalParametrization: (I + A) is singular.");
            if (pivot != p)
            {
                for (int c = 0; c < 2 * n; c++)
                {
                    var tmp = aug[p * (2 * n) + c];
                    aug[p * (2 * n) + c] = aug[pivot * (2 * n) + c];
                    aug[pivot * (2 * n) + c] = tmp;
                }
            }
            double inv = 1.0 / aug[p * (2 * n) + p];
            for (int c = 0; c < 2 * n; c++) aug[p * (2 * n) + c] *= inv;
            for (int r = 0; r < n; r++)
            {
                if (r == p) continue;
                double factor = aug[r * (2 * n) + p];
                if (factor == 0) continue;
                for (int c = 0; c < 2 * n; c++)
                    aug[r * (2 * n) + c] -= factor * aug[p * (2 * n) + c];
            }
        }
        var result = new double[n * n];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++)
                result[r * n + c] = aug[r * (2 * n) + n + c];
        return result;
    }
}
