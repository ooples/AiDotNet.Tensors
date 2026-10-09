using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Conversions between real and native complex tensors matching PyTorch. PyTorch's <c>view_as_*</c> return views;
/// these return copies with the same shapes and values.
/// </summary>
public partial interface IEngine
{
    /// <summary>The real parts (<c>torch.real</c>).</summary>
    Tensor<T> TensorReal<T>(Tensor<Complex<T>> input);

    /// <summary>The imaginary parts (<c>torch.imag</c>).</summary>
    Tensor<T> TensorImag<T>(Tensor<Complex<T>> input);

    /// <summary>real + i·imag from two same-shaped tensors (<c>torch.complex</c>).</summary>
    Tensor<Complex<T>> TensorComplex<T>(Tensor<T> real, Tensor<T> imag);

    /// <summary>Appends a trailing axis of size 2 holding (real, imag) (<c>torch.view_as_real</c>).</summary>
    Tensor<T> TensorViewAsReal<T>(Tensor<Complex<T>> input);

    /// <summary>Reads a trailing axis of size 2 as (real, imag) (<c>torch.view_as_complex</c>).</summary>
    Tensor<Complex<T>> TensorViewAsComplex<T>(Tensor<T> input);

    /// <summary>The angle of a real number: 0 for x ≥ 0, π for x &lt; 0, NaN for NaN (<c>torch.angle</c>).</summary>
    Tensor<T> TensorAngle<T>(Tensor<T> input);
}
