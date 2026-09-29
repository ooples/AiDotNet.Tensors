// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Backward functions for the native complex tensor ops, which previously recorded nothing on a tape (eager) and a
/// null backward in a compiled graph, so no gradient could flow through a complex computation.
/// </summary>
/// <remarks>
/// Convention (as PyTorch): for a real loss L of a complex tensor z = x + i·y the gradient carried on the tape is
/// ∂L/∂x + i·∂L/∂y. For y = f(z) this propagates as grad_z = conj(∂y/∂z)·g + (∂y/∂z̄)·conj(g), which for the linear /
/// bilinear ops here reduces to adjoints: a holomorphic linear map A back-propagates with A^H, conjugation conjugates
/// the gradient, and x·conj(y) is holomorphic in x and anti-holomorphic in y.
/// </remarks>
internal static class ComplexBackwardFunctions<T>
{
    private static void Accumulate(Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads,
        Tensor<Complex<T>> input, Tensor<Complex<T>> grad, IEngine engine)
        => DifferentiableOps.AccumulateGrad(grads, input, grad, engine);

    /// <summary>a + b: the gradient passes unchanged to both operands.</summary>
    internal static void AddBackward(Tensor<Complex<T>> gradOutput, Tensor<Complex<T>>[] inputs, Tensor<Complex<T>> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads)
    {
        Accumulate(grads, inputs[0], gradOutput, engine);
        Accumulate(grads, inputs[1], gradOutput, engine);
    }

    /// <summary>s·a with a real s: grad_a = s·g.</summary>
    internal static void ScaleBackward(Tensor<Complex<T>> gradOutput, Tensor<Complex<T>>[] inputs, Tensor<Complex<T>> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads)
        => Accumulate(grads, inputs[0], engine.NativeComplexScale(gradOutput, (T)savedState[0]), engine);

    /// <summary>conj(a): grad_a = conj(g).</summary>
    internal static void ConjugateBackward(Tensor<Complex<T>> gradOutput, Tensor<Complex<T>>[] inputs, Tensor<Complex<T>> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads)
        => Accumulate(grads, inputs[0], engine.NativeComplexConjugate(gradOutput), engine);

    /// <summary>a·b: grad_a = g·conj(b), grad_b = g·conj(a).</summary>
    internal static void MultiplyBackward(Tensor<Complex<T>> gradOutput, Tensor<Complex<T>>[] inputs, Tensor<Complex<T>> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads)
    {
        Accumulate(grads, inputs[0], engine.NativeComplexCrossSpectral(gradOutput, inputs[1]), engine);
        Accumulate(grads, inputs[1], engine.NativeComplexCrossSpectral(gradOutput, inputs[0]), engine);
    }

    /// <summary>x·conj(y): grad_x = g·y, grad_y = x·conj(g).</summary>
    internal static void CrossSpectralBackward(Tensor<Complex<T>> gradOutput, Tensor<Complex<T>>[] inputs, Tensor<Complex<T>> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads)
    {
        Accumulate(grads, inputs[0], engine.NativeComplexMultiply(gradOutput, inputs[1]), engine);
        Accumulate(grads, inputs[1], engine.NativeComplexCrossSpectral(inputs[0], gradOutput), engine);
    }

    /// <summary>Unnormalised forward FFT (per last-axis row of length N): the adjoint is N·IFFT(g).</summary>
    internal static void FftBackward(Tensor<Complex<T>> gradOutput, Tensor<Complex<T>>[] inputs, Tensor<Complex<T>> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads)
    {
        int n = (int)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        Accumulate(grads, inputs[0], engine.NativeComplexScale(engine.NativeComplexIFFT(gradOutput), ops.FromDouble(n)), engine);
    }

    /// <summary>1/N-normalised inverse FFT: the adjoint is FFT(g)/N.</summary>
    internal static void IfftBackward(Tensor<Complex<T>> gradOutput, Tensor<Complex<T>>[] inputs, Tensor<Complex<T>> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<Complex<T>>, Tensor<Complex<T>>> grads)
    {
        int n = (int)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        Accumulate(grads, inputs[0], engine.NativeComplexScale(engine.NativeComplexFFTComplex(gradOutput), ops.FromDouble(1.0 / n)), engine);
    }
}
