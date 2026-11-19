using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

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
