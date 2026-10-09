using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>The general N-d (1-3 spatial axes) convolution entry point matching PyTorch's <c>torch.convolution</c>.</summary>
public partial interface IEngine
{
    /// <summary>
    /// Convolution or transposed convolution with per-axis stride, padding, dilation and output padding, and
    /// <paramref name="groups"/> channel groups (<c>torch.convolution</c>). <paramref name="input"/> is
    /// <c>[batch, channels, spatial...]</c>; <paramref name="weight"/> is <c>[out, in/groups, k...]</c>, or
    /// <c>[in, out/groups, k...]</c> when <paramref name="transposed"/>.
    /// </summary>
    Tensor<T> TensorConvolution<T>(Tensor<T> input, Tensor<T> weight, Tensor<T>? bias, int[] stride, int[] padding,
        int[] dilation, bool transposed = false, int[]? outputPadding = null, int groups = 1);
}
