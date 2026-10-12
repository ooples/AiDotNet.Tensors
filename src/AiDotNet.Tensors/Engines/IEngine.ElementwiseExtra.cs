using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Tensor-level unary maths that previously existed only over spans and <see cref="Vector{T}"/>; every one records on
/// the gradient tape.
/// </summary>
public partial interface IEngine
{
    /// <summary>Elementwise inverse hyperbolic cosine (torch.acosh): NaN below 1.</summary>
    Tensor<T> TensorAcosh<T>(Tensor<T> tensor);

    /// <summary>Elementwise inverse hyperbolic sine (torch.asinh).</summary>
    Tensor<T> TensorAsinh<T>(Tensor<T> tensor);

    /// <summary>Elementwise 2^x (torch.exp2).</summary>
    Tensor<T> TensorExp2<T>(Tensor<T> tensor);

    /// <summary>Elementwise e^x - 1, accurate near 0 (torch.expm1).</summary>
    Tensor<T> TensorExpm1<T>(Tensor<T> tensor);

    /// <summary>Elementwise base-10 logarithm (torch.log10).</summary>
    Tensor<T> TensorLog10<T>(Tensor<T> tensor);

    /// <summary>Elementwise log(1 + x), accurate near 0 (torch.log1p).</summary>
    Tensor<T> TensorLog1p<T>(Tensor<T> tensor);

    /// <summary>Elementwise base-2 logarithm (torch.log2).</summary>
    Tensor<T> TensorLog2<T>(Tensor<T> tensor);

    /// <summary>Elementwise reciprocal square root, 1 / sqrt(x) (torch.rsqrt).</summary>
    Tensor<T> TensorRsqrt<T>(Tensor<T> tensor);

    /// <summary>Elementwise tangent (torch.tan).</summary>
    Tensor<T> TensorTan<T>(Tensor<T> tensor);

    /// <summary>Elementwise truncation toward zero (torch.trunc / torch.fix); its gradient is zero.</summary>
    Tensor<T> TensorTrunc<T>(Tensor<T> tensor);
}
