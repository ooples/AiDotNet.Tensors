using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Activations, fused add-multiply ops, index-mapping ops and predicates matching PyTorch.</summary>
public partial interface IEngine
{
    /// <summary>max(0, x) + min(0, α(exp(x/α) - 1)) (<c>torch.celu</c>).</summary>
    Tensor<T> TensorCelu<T>(Tensor<T> tensor, double alpha = 1.0);

    /// <summary>x clamped to [min, max], with gradient 1 inside (<c>torch.nn.functional.hardtanh</c>).</summary>
    Tensor<T> TensorHardtanh<T>(Tensor<T> tensor, double minValue = -1.0, double maxValue = 1.0);

    /// <summary>log(sigmoid(x)), computed without overflow (<c>torch.nn.functional.logsigmoid</c>).</summary>
    Tensor<T> TensorLogSigmoid<T>(Tensor<T> tensor);

    /// <summary>x / (1 + |x|) (<c>torch.nn.functional.softsign</c>).</summary>
    Tensor<T> TensorSoftsign<T>(Tensor<T> tensor);

    /// <summary>softmax(-x) along <paramref name="axis"/> (<c>torch.nn.functional.softmin</c>).</summary>
    Tensor<T> TensorSoftmin<T>(Tensor<T> tensor, int axis = -1);

    /// <summary>
    /// Randomized leaky ReLU: in training, negative inputs are scaled by a slope drawn uniformly from
    /// [lower, upper] per element; in evaluation by the mean slope (<c>torch.rrelu</c> / <c>rrelu_with_noise</c>).
    /// </summary>
    Tensor<T> TensorRrelu<T>(Tensor<T> tensor, double lower = 1.0 / 8, double upper = 1.0 / 3, bool training = false, int? seed = null);

    /// <summary>other - α·input (<c>torch.rsub</c>).</summary>
    Tensor<T> TensorRsub<T>(Tensor<T> input, Tensor<T> other, double alpha = 1);

    /// <summary>input + value·t1·t2 (<c>torch.addcmul</c>).</summary>
    Tensor<T> TensorAddcmul<T>(Tensor<T> input, Tensor<T> tensor1, Tensor<T> tensor2, double value = 1);

    /// <summary>input + value·t1/t2 (<c>torch.addcdiv</c>).</summary>
    Tensor<T> TensorAddcdiv<T>(Tensor<T> input, Tensor<T> tensor1, Tensor<T> tensor2, double value = 1);

    /// <summary>β·input + α·(mat·vec) (<c>torch.addmv</c>).</summary>
    Tensor<T> TensorAddmv<T>(Tensor<T> input, Tensor<T> mat, Tensor<T> vec, double beta = 1, double alpha = 1);

    /// <summary>β·input + α·(vec1 ⊗ vec2) (<c>torch.addr</c>).</summary>
    Tensor<T> TensorAddr<T>(Tensor<T> input, Tensor<T> vec1, Tensor<T> vec2, double beta = 1, double alpha = 1);

    /// <summary>β·input + α·(batch1 @ batch2), batched (<c>torch.baddbmm</c>).</summary>
    Tensor<T> TensorBaddbmm<T>(Tensor<T> input, Tensor<T> batch1, Tensor<T> batch2, double beta = 1, double alpha = 1);

    /// <summary>β·input + α·Σᵦ(batch1ᵦ @ batch2ᵦ) (<c>torch.addbmm</c>).</summary>
    Tensor<T> TensorAddbmm<T>(Tensor<T> input, Tensor<T> batch1, Tensor<T> batch2, double beta = 1, double alpha = 1);

    /// <summary>The tensor sorted along axis 0 (<c>torch.msort</c>).</summary>
    Tensor<T> TensorMsort<T>(Tensor<T> tensor);

    /// <summary>A 2-D matrix with the flattened input on diagonal <paramref name="offset"/> (<c>torch.diagflat</c>).</summary>
    Tensor<T> TensorDiagflat<T>(Tensor<T> tensor, int offset = 0);

    /// <summary>
    /// <paramref name="input"/> with the diagonal (<paramref name="offset"/>, over <paramref name="dim1"/> and
    /// <paramref name="dim2"/>) replaced by <paramref name="src"/> (<c>torch.diagonal_scatter</c>).
    /// </summary>
    Tensor<T> TensorDiagonalScatter<T>(Tensor<T> input, Tensor<T> src, int offset = 0, int dim1 = 0, int dim2 = 1);

    /// <summary>Whether the element type is floating point (<c>torch.is_floating_point</c>).</summary>
    bool TensorIsFloatingPoint<T>(Tensor<T> tensor);

    /// <summary>Whether the element type is signed (<c>torch.is_signed</c>).</summary>
    bool TensorIsSigned<T>(Tensor<T> tensor);

    /// <summary>Whether the element type is complex (<c>torch.is_complex</c>).</summary>
    bool TensorIsComplex<T>(Tensor<T> tensor);

    /// <summary>Whether a single-element tensor is non-zero (<c>torch.is_nonzero</c>); other sizes are refused.</summary>
    bool TensorIsNonzero<T>(Tensor<T> tensor);

    /// <summary>Whether two tensors have the same shape (<c>torch.is_same_size</c>).</summary>
    bool TensorIsSameSize<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>The shape the given shapes broadcast to (<c>torch.broadcast_shapes</c>).</summary>
    int[] BroadcastShapes(params int[][] shapes);
}
