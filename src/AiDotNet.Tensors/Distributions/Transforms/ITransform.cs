using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions.Transforms;

/// <summary>
/// Bijective transform with a tractable log-absolute-determinant of the Jacobian.
/// Composing distributions with transforms is the foundation of normalising flows
/// (e.g. RealNVP, MAF) and the <c>TransformedDistribution</c> wrapper.
/// </summary>
public interface ITransform
{
    /// <summary>Domain of the forward map.</summary>
    IConstraint Domain { get; }
    /// <summary>Codomain (= range) of the forward map.</summary>
    IConstraint Codomain { get; }
    /// <summary>True if the Jacobian determinant is constant in <c>x</c>.</summary>
    bool ConstantJacobian { get; }

    /// <summary>Forward map y = f(x).</summary>
    float[] Forward(float[] x);
    /// <summary>Inverse map x = f⁻¹(y).</summary>
    float[] Inverse(float[] y);
    /// <summary>log |det df/dx| evaluated at x. Same length as <paramref name="x"/>.</summary>
    float[] LogAbsDetJacobian(float[] x, float[] y);
}

/// <summary>Affine transform y = loc + scale · x.</summary>
public sealed class AffineTransform : ITransform
{
    /// <summary>Translation.</summary>
    public float Loc { get; }
    /// <summary>Scaling factor (must be non-zero).</summary>
    public float Scale { get; }
    /// <summary>Build an affine transform.</summary>
    public AffineTransform(float loc, float scale)
    {
        if (scale == 0f) throw new ArgumentException("scale must be non-zero.");
        Loc = loc; Scale = scale;
    }
    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => RealConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => true;
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = Loc + Scale * x[i];
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        var x = new float[y.Length];
        for (int i = 0; i < y.Length; i++) x[i] = (y[i] - Loc) / Scale;
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        float v = MathF.Log(MathF.Abs(Scale));
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = v;
        return r;
    }
}

/// <summary>Exponential transform y = exp(x). Maps ℝ → (0, ∞).</summary>
public sealed class ExpTransform : ITransform
{
    /// <summary>Singleton instance.</summary>
    public static readonly ExpTransform Instance = new ExpTransform();
    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => PositiveConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = MathF.Exp(x[i]);
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        var x = new float[y.Length];
        for (int i = 0; i < y.Length; i++) x[i] = MathF.Log(y[i]);
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // d(exp x)/dx = exp(x) = y. log|det| = x.
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = x[i];
        return r;
    }
}

/// <summary>Sigmoid transform y = σ(x). Maps ℝ → (0, 1).</summary>
public sealed class SigmoidTransform : ITransform
{
    /// <summary>Singleton instance.</summary>
    public static readonly SigmoidTransform Instance = new SigmoidTransform();
    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => UnitIntervalConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            float xi = x[i];
            y[i] = xi >= 0
                ? 1f / (1f + MathF.Exp(-xi))
                : MathF.Exp(xi) / (1f + MathF.Exp(xi));
        }
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        var x = new float[y.Length];
        for (int i = 0; i < y.Length; i++) x[i] = MathF.Log(y[i] / (1f - y[i]));
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // log σ'(x) = -x - 2·softplus(-x) = log y + log(1-y)
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = MathF.Log(y[i]) + MathF.Log(1f - y[i]);
        return r;
    }
}

/// <summary>Tanh transform y = tanh(x). Maps ℝ → (-1, 1).</summary>
public sealed class TanhTransform : ITransform
{
    /// <summary>Singleton instance.</summary>
    public static readonly TanhTransform Instance = new TanhTransform();
    /// <inheritdoc />
    public IConstraint Domain => RealConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => new IntervalConstraint(-1f, 1f);
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = MathF.Tanh(x[i]);
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        var x = new float[y.Length];
        for (int i = 0; i < y.Length; i++) x[i] = 0.5f * MathF.Log((1f + y[i]) / (1f - y[i]));
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // log(1 - tanh(x)²) = log(sech(x)²) — written using softplus for numerical stability.
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++)
        {
            float xi = x[i];
            // 2·log(2) - x - 2·softplus(-x) using log1p(exp(-2|x|)) for numerical stability
            float absX = MathF.Abs(xi);
            r[i] = 2f * (MathF.Log(2f) - absX - Softplus(-2f * absX));
        }
        return r;
    }
    private static float Softplus(float v) => v > 20f ? v : MathF.Log(1f + MathF.Exp(v));
}

/// <summary>Power transform y = x^exponent. Maps (0, ∞) → (0, ∞) when exponent ≠ 0.</summary>
public sealed class PowerTransform : ITransform
{
    /// <summary>Exponent.</summary>
    public float Exponent { get; }
    /// <summary>Build a power transform.</summary>
    public PowerTransform(float exponent)
    {
        if (exponent == 0f) throw new ArgumentException("exponent must be non-zero.");
        Exponent = exponent;
    }
    /// <inheritdoc />
    public IConstraint Domain => PositiveConstraint.Instance;
    /// <inheritdoc />
    public IConstraint Codomain => PositiveConstraint.Instance;
    /// <inheritdoc />
    public bool ConstantJacobian => false;
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = MathF.Pow(x[i], Exponent);
        return y;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        var x = new float[y.Length];
        float invExp = 1f / Exponent;
        for (int i = 0; i < y.Length; i++) x[i] = MathF.Pow(y[i], invExp);
        return x;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // d(x^e)/dx = e · x^(e-1). log|det| = log|e| + (e-1) log x.
        var r = new float[x.Length];
        float logE = MathF.Log(MathF.Abs(Exponent));
        for (int i = 0; i < r.Length; i++) r[i] = logE + (Exponent - 1f) * MathF.Log(x[i]);
        return r;
    }
}

/// <summary>Composition of a list of transforms applied left-to-right.</summary>
public sealed class ComposeTransform : ITransform
{
    /// <summary>Sub-transforms applied in order.</summary>
    public ITransform[] Parts { get; }
    /// <summary>Build a composition.</summary>
    public ComposeTransform(params ITransform[] parts)
    {
        if (parts == null || parts.Length == 0) throw new ArgumentException("at least one transform required.");
        Parts = parts;
    }
    /// <inheritdoc />
    public IConstraint Domain => Parts[0].Domain;
    /// <inheritdoc />
    public IConstraint Codomain => Parts[Parts.Length - 1].Codomain;
    /// <inheritdoc />
    public bool ConstantJacobian
    {
        get { foreach (var p in Parts) if (!p.ConstantJacobian) return false; return true; }
    }
    /// <inheritdoc />
    public float[] Forward(float[] x)
    {
        var v = x;
        foreach (var p in Parts) v = p.Forward(v);
        return v;
    }
    /// <inheritdoc />
    public float[] Inverse(float[] y)
    {
        var v = y;
        for (int i = Parts.Length - 1; i >= 0; i--) v = Parts[i].Inverse(v);
        return v;
    }
    /// <inheritdoc />
    public float[] LogAbsDetJacobian(float[] x, float[] y)
    {
        // Walk left-to-right, accumulating log|det| at each intermediate step.
        var sum = new float[x.Length];
        var v = x;
        foreach (var p in Parts)
        {
            var next = p.Forward(v);
            var ld = p.LogAbsDetJacobian(v, next);
            for (int i = 0; i < sum.Length; i++) sum[i] += ld[i];
            v = next;
        }
        return sum;
    }
}
