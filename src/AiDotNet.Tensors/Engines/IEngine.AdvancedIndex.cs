using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Advanced (NumPy-style) indexing.</summary>
public partial interface IEngine
{
    /// <summary>
    /// <c>tensor[indices[0], indices[1], ...]</c> with index tensors (torch <c>aten::index</c>): a null entry keeps the
    /// whole dimension, the index tensors broadcast together, consecutive index tensors replace their dimensions in
    /// place and separated ones move the broadcast dimensions to the front. Negative indices count from the end.
    /// Differentiable.
    /// </summary>
    Tensor<T> TensorIndex<T>(Tensor<T> tensor, Tensor<int>?[] indices);

    /// <summary><c>tensor[mask]</c> for a boolean mask over the leading dimensions: the selected positions, in row-major order.</summary>
    Tensor<T> TensorIndex<T>(Tensor<T> tensor, Tensor<bool> mask);
}
