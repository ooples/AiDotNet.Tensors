using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial interface IEngine
{
    /// <summary>Normalizes each first-dimension row of direction and scales it by gain: v*g/sqrt(sum(v*v)+epsilon).</summary>
    /// <param name="direction">A tensor with at least one dimension; each first-dimension slice is one row.</param>
    /// <param name="gain">A rank-one tensor with one value per row.</param>
    /// <param name="epsilon">A finite nonnegative value added inside the square root.</param>
    /// <returns>A tensor with direction's shape. This operation records its analytic gradient.</returns>
    Tensor<T> TensorWeightNorm<T>(Tensor<T> direction, Tensor<T> gain, double epsilon = 1e-24);

    /// <summary>Writes weight normalization into an existing output with direction's exact shape.</summary>
    void TensorWeightNormInto<T>(Tensor<T> output, Tensor<T> direction, Tensor<T> gain, double epsilon = 1e-24);

    /// <summary>Writes the analytic weight-normalization VJP directly into the requested destinations.</summary>
    /// <remarks>Null destinations are unrequested. Direction writes complete before gain writes, including when
    /// destination storage overlaps. An accumulate flag adds that contribution to the corresponding destination.</remarks>
    void TensorWeightNormBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> direction, Tensor<T> gain,
        Tensor<T>? gradDirection, Tensor<T>? gradGain, double epsilon = 1e-24,
        bool accumulateDirection = false, bool accumulateGain = false);
}
