// Copyright (c) AiDotNet. All rights reserved.
// Forward-mode AD rules (JVP) for the core differentiable ops.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff.ForwardAD;

/// <summary>
/// Forward-mode AD operations over <see cref="Dual{T}"/>. Each method
/// computes the primal via the given <see cref="IEngine"/> and the
/// tangent via the op's JVP rule (the transpose of its VJP backward).
/// </summary>
/// <remarks>
/// <para><b>Rule-by-rule correctness:</b> every method here is the
/// direct symbolic derivative of the primal op — no finite
/// differences. For <c>z = f(x, y)</c> with tangents
/// <c>dx</c>, <c>dy</c>, we compute <c>dz = ∂f/∂x · dx + ∂f/∂y · dy</c>.</para>
/// <para><b>Cross-validation with reverse mode:</b> for any op with
/// both a JVP (here) and a VJP (<see cref="BackwardFunctions{T}"/>),
/// a Jacobian built by forward-mode <c>jacfwd</c> must agree with one
/// built by reverse-mode <c>jacrev</c> to within numerical tolerance.
/// This is the main integration test for every op in this file.</para>
/// <para><b>Why a static helper instead of instance methods on
/// <see cref="Dual{T}"/>:</b> keeps <c>Dual&lt;T&gt;</c> a simple data
/// carrier and lets callers choose which engine to run under. Also
/// allows the SIMD-packed dual optimization to land later as a
/// drop-in <c>DualOps&lt;T&gt;</c> replacement.</para>
/// </remarks>
/// <typeparam name="T">Element numeric type.</typeparam>
public static class DualOps<T>
{
    // ─── Binary arithmetic ────────────────────────────────────────────

    /// <summary>
    /// <c>(a + b).primal = a.primal + b.primal</c>;
    /// <c>(a + b).tangent = a.tangent + b.tangent</c>.
    /// </summary>
    public static Dual<T> Add(IEngine engine, Dual<T> a, Dual<T> b)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return new Dual<T>(
            engine.TensorAdd(a.Primal, b.Primal),
            engine.TensorAdd(a.Tangent, b.Tangent));
    }

    /// <summary>
    /// <c>(a - b).tangent = a.tangent - b.tangent</c>.
    /// </summary>
    public static Dual<T> Subtract(IEngine engine, Dual<T> a, Dual<T> b)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return new Dual<T>(
            engine.TensorSubtract(a.Primal, b.Primal),
            engine.TensorSubtract(a.Tangent, b.Tangent));
    }

    /// <summary>
    /// Product rule: <c>d(a·b) = da·b + a·db</c>.
    /// </summary>
    public static Dual<T> Multiply(IEngine engine, Dual<T> a, Dual<T> b)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorMultiply(a.Primal, b.Primal);
        var t1 = engine.TensorMultiply(a.Tangent, b.Primal);
        var t2 = engine.TensorMultiply(a.Primal, b.Tangent);
        var tangent = engine.TensorAdd(t1, t2);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// Quotient rule: <c>d(a/b) = (da·b - a·db) / b²</c>.
    /// </summary>
    public static Dual<T> Divide(IEngine engine, Dual<T> a, Dual<T> b)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorDivide(a.Primal, b.Primal);
        var num1 = engine.TensorMultiply(a.Tangent, b.Primal);
        var num2 = engine.TensorMultiply(a.Primal, b.Tangent);
        var num = engine.TensorSubtract(num1, num2);
        var denom = engine.TensorMultiply(b.Primal, b.Primal);
        var tangent = engine.TensorDivide(num, denom);
        return new Dual<T>(primal, tangent);
    }

    // ─── Unary arithmetic ─────────────────────────────────────────────

    /// <summary>
    /// <c>d(-x) = -dx</c>.
    /// </summary>
    public static Dual<T> Negate(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return new Dual<T>(engine.TensorNegate(x.Primal), engine.TensorNegate(x.Tangent));
    }

    /// <summary>
    /// <c>d(α·x) = α·dx</c>.
    /// </summary>
    public static Dual<T> Scale(IEngine engine, Dual<T> x, T scalar)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return new Dual<T>(
            engine.TensorMultiplyScalar(x.Primal, scalar),
            engine.TensorMultiplyScalar(x.Tangent, scalar));
    }

    /// <summary>
    /// <c>d(x²) = 2·x·dx</c>. Uses an explicit two-op form to avoid
    /// allocating a scratch tensor for the scale factor.
    /// </summary>
    public static Dual<T> Square(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorMultiply(x.Primal, x.Primal);
        var two = MathHelper.GetNumericOperations<T>().FromDouble(2.0);
        var twoX = engine.TensorMultiplyScalar(x.Primal, two);
        var tangent = engine.TensorMultiply(twoX, x.Tangent);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(√x) = dx / (2·√x)</c>.
    /// </summary>
    public static Dual<T> Sqrt(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorSqrt(x.Primal);
        var two = MathHelper.GetNumericOperations<T>().FromDouble(2.0);
        var twoSqrt = engine.TensorMultiplyScalar(primal, two);
        var tangent = engine.TensorDivide(x.Tangent, twoSqrt);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(exp(x)) = exp(x)·dx</c>.
    /// </summary>
    public static Dual<T> Exp(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorExp(x.Primal);
        var tangent = engine.TensorMultiply(primal, x.Tangent);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(log(x)) = dx / x</c>.
    /// </summary>
    public static Dual<T> Log(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorLog(x.Primal);
        var tangent = engine.TensorDivide(x.Tangent, x.Primal);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(sin(x)) = cos(x)·dx</c>.
    /// </summary>
    public static Dual<T> Sin(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorSin(x.Primal);
        var cos = engine.TensorCos(x.Primal);
        var tangent = engine.TensorMultiply(cos, x.Tangent);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(cos(x)) = -sin(x)·dx</c>.
    /// </summary>
    public static Dual<T> Cos(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorCos(x.Primal);
        var negSin = engine.TensorNegate(engine.TensorSin(x.Primal));
        var tangent = engine.TensorMultiply(negSin, x.Tangent);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(tanh(x)) = (1 - tanh²(x))·dx</c>. Uses the stable form
    /// that avoids recomputing tanh.
    /// </summary>
    public static Dual<T> Tanh(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorTanh(x.Primal);
        var tanhSq = engine.TensorMultiply(primal, primal);
        var one = MathHelper.GetNumericOperations<T>().One;
        var data = new T[primal.Length];
        for (int i = 0; i < data.Length; i++) data[i] = one;
        var ones = new Tensor<T>(data, (int[])primal._shape.Clone());
        var oneMinusTanhSq = engine.TensorSubtract(ones, tanhSq);
        var tangent = engine.TensorMultiply(oneMinusTanhSq, x.Tangent);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(sigmoid(x)) = sigmoid(x)·(1 - sigmoid(x))·dx</c>.
    /// </summary>
    public static Dual<T> Sigmoid(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorSigmoid(x.Primal);
        var one = MathHelper.GetNumericOperations<T>().One;
        var data = new T[primal.Length];
        for (int i = 0; i < data.Length; i++) data[i] = one;
        var ones = new Tensor<T>(data, (int[])primal._shape.Clone());
        var oneMinusS = engine.TensorSubtract(ones, primal);
        var sTimesOneMinusS = engine.TensorMultiply(primal, oneMinusS);
        var tangent = engine.TensorMultiply(sTimesOneMinusS, x.Tangent);
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(ReLU(x)) = (x &gt; 0)·dx</c> — the tangent passes through
    /// where the primal is positive and is zeroed otherwise.
    /// </summary>
    public static Dual<T> ReLU(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorReLU(x.Primal);
        var ops = MathHelper.GetNumericOperations<T>();
        var zero = ops.Zero;
        var one = ops.One;
        var maskData = new T[x.Primal.Length];
        var primalData = x.Primal.AsSpan();
        for (int i = 0; i < maskData.Length; i++)
            maskData[i] = ops.GreaterThan(primalData[i], zero) ? one : zero;
        var mask = new Tensor<T>(maskData, (int[])x.Primal._shape.Clone());
        var tangent = engine.TensorMultiply(mask, x.Tangent);
        return new Dual<T>(primal, tangent);
    }

    // ─── Linear algebra ───────────────────────────────────────────────

    /// <summary>
    /// Matrix multiply with product rule:
    /// <c>d(A·B) = dA·B + A·dB</c>.
    /// </summary>
    public static Dual<T> MatMul(IEngine engine, Dual<T> a, Dual<T> b)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primal = engine.TensorMatMul(a.Primal, b.Primal);
        var t1 = engine.TensorMatMul(a.Tangent, b.Primal);
        var t2 = engine.TensorMatMul(a.Primal, b.Tangent);
        var tangent = engine.TensorAdd(t1, t2);
        return new Dual<T>(primal, tangent);
    }

    // ─── Reductions ───────────────────────────────────────────────────

    /// <summary>
    /// <c>d(sum(x)) = sum(dx)</c> — reductions are linear so the
    /// tangent of the reduction is the reduction of the tangent.
    /// Returned dual has shape <c>[1]</c>.
    /// </summary>
    public static Dual<T> Sum(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primalScalar = engine.TensorSum(x.Primal);
        var tangentScalar = engine.TensorSum(x.Tangent);
        var primal = new Tensor<T>(new[] { primalScalar }, new[] { 1 });
        var tangent = new Tensor<T>(new[] { tangentScalar }, new[] { 1 });
        return new Dual<T>(primal, tangent);
    }

    /// <summary>
    /// <c>d(mean(x)) = mean(dx)</c>. Returned dual has shape <c>[1]</c>.
    /// </summary>
    public static Dual<T> Mean(IEngine engine, Dual<T> x)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var primalScalar = engine.TensorMean(x.Primal);
        var tangentScalar = engine.TensorMean(x.Tangent);
        var primal = new Tensor<T>(new[] { primalScalar }, new[] { 1 });
        var tangent = new Tensor<T>(new[] { tangentScalar }, new[] { 1 });
        return new Dual<T>(primal, tangent);
    }
}
