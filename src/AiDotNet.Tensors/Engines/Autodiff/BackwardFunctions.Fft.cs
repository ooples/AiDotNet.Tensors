using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>The complex transform a tape edge recorded by <c>RecordFftTapeEdges</c> differentiates.</summary>
internal enum FftTransform
{
    /// <summary><c>IEngine.FFT</c>: unnormalised forward transform along the last axis.</summary>
    Forward,

    /// <summary><c>IEngine.IFFT</c>: inverse transform along the last axis, scaled by 1/n.</summary>
    Inverse,

    /// <summary><c>IEngine.FFT2D</c>: unnormalised forward transform over the last two axes.</summary>
    Forward2D,

    /// <summary><c>IEngine.IFFT2D</c>: inverse transform over the last two axes, scaled by 1/(height·width).</summary>
    Inverse2D,
}

/// <summary>
/// Backward of the real and imaginary outputs of <c>IEngine.FFT</c>, <c>IFFT</c>, <c>FFT2D</c> and <c>IFFT2D</c>.
/// </summary>
/// <remarks>
/// <para>
/// Each transform is a complex linear map <c>y = A·x</c>. Read as a real map of the stacked real and imaginary parts,
/// its transpose is <c>Aᴴ</c>, so for an output gradient <c>g = g_re + i·g_im</c> the input gradient is
/// <c>Aᴴ·g</c>, returned as its real part to the real input and its imaginary part to the imaginary input (torch's
/// convention for real-valued losses of complex results). For the unnormalised forward DFT <c>Aᴴ = n·IDFT</c>; for the
/// 1/n inverse <c>Aᴴ = DFT / n</c>; the 2-D cases are the same over <c>height·width</c> points.
/// </para>
/// <para>
/// The backward is the adjoint transform applied to the gradient with recorded engine ops, so under <c>createGraph</c>
/// the gradient is itself differentiable (with respect to the seed; a linear map has no second derivative in its
/// input). Each output is its own edge: the real output's edge seeds <c>g_im = 0</c>, the imaginary output's
/// <c>g_re = 0</c>, and the two contributions accumulate.
/// </para>
/// </remarks>
internal static partial class BackwardFunctions<T>
{
    internal static void FftBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        RequireSavedState(nameof(FftBackward), savedState, 2);
        var transform = savedState[0] is FftTransform t
            ? t
            : throw new ArgumentException(
                $"{nameof(FftBackward)}: savedState[0] (transform) is {Describe(savedState[0])}, expected {nameof(FftTransform)}.",
                nameof(savedState));
        bool imaginaryOutput = SavedBool(nameof(FftBackward), savedState, 1, "imaginaryOutput");

        var shape = gradOutput._shape;
        var zeros = new Tensor<T>(shape);
        var seedReal = imaginaryOutput ? zeros : gradOutput;
        var seedImag = imaginaryOutput ? gradOutput : zeros;

        int points = transform is FftTransform.Forward2D or FftTransform.Inverse2D
            ? shape[^2] * shape[^1]
            : shape[^1];
        Tensor<T> real, imag;
        double scale;
        switch (transform)
        {
            case FftTransform.Forward:
                engine.IFFT(seedReal, seedImag, out real, out imag);
                scale = points;
                break;
            case FftTransform.Inverse:
                engine.FFT(seedReal, seedImag, out real, out imag);
                scale = 1.0 / points;
                break;
            case FftTransform.Forward2D:
                engine.IFFT2D(seedReal, seedImag, out real, out imag);
                scale = points;
                break;
            case FftTransform.Inverse2D:
                engine.FFT2D(seedReal, seedImag, out real, out imag);
                scale = 1.0 / points;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(savedState), transform, "Unknown FFT transform.");
        }

        var factor = MathHelper.GetNumericOperations<T>().FromDouble(scale);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], engine.TensorMultiplyScalar(real, factor), engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], engine.TensorMultiplyScalar(imag, factor), engine);
    }
}
