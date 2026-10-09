using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Pooling variants matching PyTorch: adaptive average/max pooling over 1-3 axes, power-average (Lp) pooling,
/// max pooling with indices, fractional max pooling and max unpooling. Pooled axes are always the trailing ones.
/// </summary>
public partial interface IEngine
{
    /// <summary>Adaptive average pooling over the last axis (<c>torch.adaptive_avg_pool1d</c>).</summary>
    Tensor<T> TensorAdaptiveAvgPool1D<T>(Tensor<T> input, int outputSize);

    /// <summary>Adaptive average pooling over the last three axes (<c>adaptive_avg_pool3d</c>).</summary>
    Tensor<T> TensorAdaptiveAvgPool3D<T>(Tensor<T> input, int[] outputSize);

    /// <summary>Adaptive max pooling over the last axis (<c>torch.adaptive_max_pool1d</c>).</summary>
    Tensor<T> TensorAdaptiveMaxPool1D<T>(Tensor<T> input, int outputSize);

    /// <summary>Adaptive max pooling over the last three axes (<c>adaptive_max_pool3d</c>).</summary>
    Tensor<T> TensorAdaptiveMaxPool3D<T>(Tensor<T> input, int[] outputSize);

    /// <summary>
    /// Adaptive max pooling over the trailing <c>outputSize.Length</c> axes, also returning each maximum's flat
    /// index within its input plane (<c>adaptive_max_pool{1,2,3}d(return_indices=True)</c>).
    /// </summary>
    (Tensor<T> Output, Tensor<int> Indices) TensorAdaptiveMaxPoolWithIndices<T>(Tensor<T> input, int[] outputSize);

    /// <summary>Max pooling over the last axis with indices (<c>torch.max_pool1d_with_indices</c>).</summary>
    (Tensor<T> Output, Tensor<int> Indices) TensorMaxPool1DWithIndices<T>(Tensor<T> input, int kernelSize,
        int stride = 0, int padding = 0, int dilation = 1, bool ceilMode = false);

    /// <summary>
    /// Power-average pooling (sum x^p)^(1/p) over the trailing <c>kernelSize.Length</c> axes
    /// (<c>lp_pool{1,2,3}d</c>); stride defaults to the kernel.
    /// </summary>
    Tensor<T> TensorLpPool<T>(Tensor<T> input, double power, int[] kernelSize, int[]? stride = null);

    /// <summary>
    /// Fractional max pooling with pseudo-random window starts per plane (<c>fractional_max_pool{2,3}d</c>);
    /// returns the output and each maximum's flat index within its input plane.
    /// </summary>
    (Tensor<T> Output, Tensor<int> Indices) TensorFractionalMaxPool<T>(Tensor<T> input, int[] kernelSize,
        int[] outputSize, int? seed = null);

    /// <summary>
    /// Scatters each pooled value back to its flat index in a zero plane of <paramref name="outputSize"/>
    /// (<c>max_unpool{1,2,3}d</c>).
    /// </summary>
    Tensor<T> TensorMaxUnpool<T>(Tensor<T> input, Tensor<int> indices, int[] outputSize);
}
