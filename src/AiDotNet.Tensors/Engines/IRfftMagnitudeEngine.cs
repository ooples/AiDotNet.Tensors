using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Optional engine support for real FFT magnitude and its gradient.</summary>
public interface IRfftMagnitudeEngine
{
    /// <summary>Unnormalized real FFT magnitude along the last axis: sqrt(real²+imaginary²+epsilon).</summary>
    /// <remarks>The output keeps leading dimensions and has last dimension floor(N/2)+1.
    /// Windowing is explicit caller work. This operation records its analytic real-input gradient.
    /// GPU support is capability checked; unsupported storage or geometry never causes a host fallback.</remarks>
    /// <param name="input">Real input with at least one dimension and a positive last dimension.</param>
    /// <param name="epsilon">Finite nonnegative magnitude regularizer. At zero epsilon, derivatives at a zero spectrum
    /// retain the undefined IEEE behavior of the corresponding square-root composition.</param>
    Tensor<T> TensorRfftMagnitude<T>(Tensor<T> input, double epsilon = 1e-9);

    /// <summary>Writes real FFT magnitude to a preallocated tensor with the exact output shape.</summary>
    void TensorRfftMagnitudeInto<T>(Tensor<T> output, Tensor<T> input, double epsilon = 1e-9);

    /// <summary>Writes the analytic real FFT magnitude VJP to an input-shaped destination.</summary>
    /// <param name="gradOutput">The magnitude-shaped upstream gradient.</param>
    /// <param name="input">The original real input.</param>
    /// <param name="gradInput">The destination with the exact input shape.</param>
    /// <param name="epsilon">The forward magnitude regularizer.</param>
    /// <param name="accumulate">Adds the complete VJP when true; otherwise overwrites the destination.</param>
    void TensorRfftMagnitudeBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gradInput,
        double epsilon = 1e-9, bool accumulate = false);
}

/// <summary>Real FFT magnitude operations for engines that implement <see cref="IRfftMagnitudeEngine"/>.</summary>
public static class RfftMagnitudeEngineExtensions
{
    internal static IRfftMagnitudeEngine Require(IEngine engine)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return engine.As<IRfftMagnitudeEngine>()
            ?? throw new NotSupportedException("The engine does not implement real FFT magnitude.");
    }

    /// <inheritdoc cref="IRfftMagnitudeEngine.TensorRfftMagnitude{T}"/>
    public static Tensor<T> TensorRfftMagnitude<T>(this IEngine engine, Tensor<T> input, double epsilon = 1e-9)
        => Require(engine).TensorRfftMagnitude(input, epsilon);

    /// <inheritdoc cref="IRfftMagnitudeEngine.TensorRfftMagnitudeInto{T}"/>
    public static void TensorRfftMagnitudeInto<T>(this IEngine engine, Tensor<T> output, Tensor<T> input, double epsilon = 1e-9)
        => Require(engine).TensorRfftMagnitudeInto(output, input, epsilon);

    /// <inheritdoc cref="IRfftMagnitudeEngine.TensorRfftMagnitudeBackwardInto{T}"/>
    public static void TensorRfftMagnitudeBackwardInto<T>(this IEngine engine, Tensor<T> gradOutput, Tensor<T> input,
        Tensor<T> gradInput, double epsilon = 1e-9, bool accumulate = false)
        => Require(engine).TensorRfftMagnitudeBackwardInto(gradOutput, input, gradInput, epsilon, accumulate);
}
