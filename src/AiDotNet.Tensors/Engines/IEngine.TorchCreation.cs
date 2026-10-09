using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>How <see cref="IEngine.TensorQuantile{T}"/> picks a value between the two order statistics around q.</summary>
public enum QuantileInterpolation
{
    /// <summary>Linear interpolation between the neighbours (PyTorch's default).</summary>
    Linear,
    /// <summary>The lower neighbour.</summary>
    Lower,
    /// <summary>The higher neighbour.</summary>
    Higher,
    /// <summary>The nearer neighbour (ties to the even index, as PyTorch rounds).</summary>
    Nearest,
    /// <summary>The mean of the two neighbours.</summary>
    Midpoint,
}

/// <summary>
/// Creation, window, index, shape and statistics ops matching their PyTorch counterparts. Random ops take an
/// optional seed (a cryptographically seeded generator when it is null).
/// </summary>
public partial interface IEngine
{
    /// <summary>The minimum over <paramref name="axes"/> (<c>torch.amin</c>); the gradient follows ReduceMax's tie rule.</summary>
    Tensor<T> TensorAmin<T>(Tensor<T> tensor, int[] axes, bool keepDims = false);

    /// <summary>1 where every element over <paramref name="axes"/> (all when null) is non-zero, else 0 (<c>torch.all</c>).</summary>
    Tensor<T> TensorAll<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false);

    /// <summary>1 where any element over <paramref name="axes"/> (all when null) is non-zero, else 0 (<c>torch.any</c>).</summary>
    Tensor<T> TensorAny<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false);

    /// <summary>start, start + step, … stopping before <paramref name="end"/> (<c>torch.arange</c>).</summary>
    Tensor<T> TensorArange<T>(double start, double end, double step = 1);

    /// <summary>start, start + step, … up to and including <paramref name="end"/> (the deprecated <c>torch.range</c>).</summary>
    Tensor<T> TensorRange<T>(double start, double end, double step = 1);

    /// <summary><paramref name="steps"/> values base^t for t evenly spaced from start to end (<c>torch.logspace</c>).</summary>
    Tensor<T> TensorLogspace<T>(double start, double end, int steps, double logBase = 10);

    /// <summary>Zeros with <paramref name="tensor"/>'s shape (<c>torch.zeros_like</c>).</summary>
    Tensor<T> TensorZerosLike<T>(Tensor<T> tensor);

    /// <summary>Ones with <paramref name="tensor"/>'s shape (<c>torch.ones_like</c>).</summary>
    Tensor<T> TensorOnesLike<T>(Tensor<T> tensor);

    /// <summary><paramref name="value"/> with <paramref name="tensor"/>'s shape (<c>torch.full_like</c>).</summary>
    Tensor<T> TensorFullLike<T>(Tensor<T> tensor, T value);

    /// <summary>A tensor with <paramref name="tensor"/>'s shape; its values are zero here (<c>torch.empty_like</c>).</summary>
    Tensor<T> TensorEmptyLike<T>(Tensor<T> tensor);

    /// <summary>Uniform [0, 1) samples with <paramref name="tensor"/>'s shape (<c>torch.rand_like</c>).</summary>
    Tensor<T> TensorRandLike<T>(Tensor<T> tensor, int? seed = null);

    /// <summary>Standard normal samples with <paramref name="tensor"/>'s shape (<c>torch.randn_like</c>).</summary>
    Tensor<T> TensorRandnLike<T>(Tensor<T> tensor, int? seed = null);

    /// <summary>Uniform integers in [low, high) with the given shape (<c>torch.randint</c>).</summary>
    Tensor<T> TensorRandint<T>(long low, long high, int[] shape, int? seed = null);

    /// <summary>Uniform integers in [low, high) with <paramref name="tensor"/>'s shape (<c>torch.randint_like</c>).</summary>
    Tensor<T> TensorRandintLike<T>(Tensor<T> tensor, long low, long high, int? seed = null);

    /// <summary>A random permutation of 0 … n-1 (<c>torch.randperm</c>).</summary>
    Tensor<T> TensorRandperm<T>(int n, int? seed = null);

    /// <summary>The Hann window; periodic (the DFT-even form) by default, as in PyTorch (<c>torch.hann_window</c>).</summary>
    Tensor<T> TensorHannWindow<T>(int length, bool periodic = true);

    /// <summary>The generalized Hamming window α - β·cos(2πn/N) (<c>torch.hamming_window</c>).</summary>
    Tensor<T> TensorHammingWindow<T>(int length, bool periodic = true, double alpha = 0.54, double beta = 0.46);

    /// <summary>The Blackman window (<c>torch.blackman_window</c>).</summary>
    Tensor<T> TensorBlackmanWindow<T>(int length, bool periodic = true);

    /// <summary>The Bartlett (triangular) window (<c>torch.bartlett_window</c>).</summary>
    Tensor<T> TensorBartlettWindow<T>(int length, bool periodic = true);

    /// <summary>The Kaiser window with shape parameter <paramref name="beta"/> (<c>torch.kaiser_window</c>).</summary>
    Tensor<T> TensorKaiserWindow<T>(int length, bool periodic = true, double beta = 12);

    /// <summary>The [2, N] row/column indices of the lower triangle of a row × col matrix (<c>torch.tril_indices</c>).</summary>
    Tensor<T> TensorTrilIndices<T>(int row, int col, int offset = 0);

    /// <summary>The [2, N] row/column indices of the upper triangle of a row × col matrix (<c>torch.triu_indices</c>).</summary>
    Tensor<T> TensorTriuIndices<T>(int row, int col, int offset = 0);

    /// <summary>Flat indices converted to one coordinate tensor per dimension of <paramref name="shape"/> (<c>torch.unravel_index</c>).</summary>
    Tensor<T>[] TensorUnravelIndex<T>(Tensor<T> indices, int[] shape);

    /// <summary>The length-r combinations of a 1-D tensor's elements, one per row (<c>torch.combinations</c>).</summary>
    Tensor<T> TensorCombinations<T>(Tensor<T> tensor, int r = 2, bool withReplacement = false);

    /// <summary>
    /// <paramref name="chunks"/> pieces along <paramref name="dim"/>, each ⌈size/chunks⌉ long except the last, possibly
    /// fewer pieces (<c>torch.chunk</c>).
    /// </summary>
    Tensor<T>[] TensorChunk<T>(Tensor<T> tensor, int chunks, int dim = 0);

    /// <summary>Pieces of the given <paramref name="sizes"/> along <paramref name="dim"/> (<c>torch.split_with_sizes</c>).</summary>
    Tensor<T>[] TensorSplitWithSizes<T>(Tensor<T> tensor, int[] sizes, int dim = 0);

    /// <summary>Axis <paramref name="dim"/> expanded into <paramref name="sizes"/> (<c>torch.unflatten</c>).</summary>
    Tensor<T> TensorUnflatten<T>(Tensor<T> tensor, int dim, int[] sizes);

    /// <summary>The slice at <paramref name="index"/> along <paramref name="dim"/>, that axis removed (<c>torch.select</c>).</summary>
    Tensor<T> TensorSelect<T>(Tensor<T> tensor, int dim, int index);

    /// <summary><paramref name="tensor"/> reshaped to <paramref name="other"/>'s shape (<c>view_as</c> / <c>reshape_as</c>).</summary>
    Tensor<T> TensorViewAs<T>(Tensor<T> tensor, Tensor<T> other);

    /// <summary><paramref name="tensor"/> summed down to the broadcast-compatible <paramref name="size"/> (<c>Tensor.sum_to_size</c>).</summary>
    Tensor<T> TensorSumToSize<T>(Tensor<T> tensor, int[] size);

    /// <summary>
    /// The q-th quantile over <paramref name="dim"/> (over all elements when null), differentiable through the order
    /// statistics it reads (<c>torch.quantile</c>).
    /// </summary>
    Tensor<T> TensorQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false,
        QuantileInterpolation interpolation = QuantileInterpolation.Linear);

    /// <summary><see cref="TensorQuantile{T}"/> ignoring NaN (NaN for an all-NaN slice) (<c>torch.nanquantile</c>).</summary>
    Tensor<T> TensorNanQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false,
        QuantileInterpolation interpolation = QuantileInterpolation.Linear);

    /// <summary>(standard deviation, mean) over <paramref name="axes"/> with Bessel correction <paramref name="correction"/> (<c>torch.std_mean</c>).</summary>
    (Tensor<T> Std, Tensor<T> Mean) TensorStdMean<T>(Tensor<T> tensor, int[]? axes = null, int correction = 1, bool keepDims = false);

    /// <summary>(variance, mean) over <paramref name="axes"/> with Bessel correction <paramref name="correction"/> (<c>torch.var_mean</c>).</summary>
    (Tensor<T> Var, Tensor<T> Mean) TensorVarMean<T>(Tensor<T> tensor, int[]? axes = null, int correction = 1, bool keepDims = false);

    /// <summary>The covariance matrix of the rows (variables) of a [variables, observations] tensor (<c>torch.cov</c>).</summary>
    Tensor<T> TensorCov<T>(Tensor<T> tensor, int correction = 1);

    /// <summary>The Pearson correlation matrix of the rows of a [variables, observations] tensor (<c>torch.corrcoef</c>).</summary>
    Tensor<T> TensorCorrcoef<T>(Tensor<T> tensor);

    /// <summary>The n-th forward difference along <paramref name="dim"/> (<c>torch.diff</c>).</summary>
    Tensor<T> TensorDiff<T>(Tensor<T> tensor, int n = 1, int dim = -1);

    /// <summary>The trapezoid-rule integral along <paramref name="dim"/> with spacing <paramref name="dx"/> (<c>torch.trapezoid</c> / <c>trapz</c>).</summary>
    Tensor<T> TensorTrapezoid<T>(Tensor<T> y, double dx = 1, int dim = -1);

    /// <summary>The cumulative trapezoid-rule integral along <paramref name="dim"/> (<c>torch.cumulative_trapezoid</c>).</summary>
    Tensor<T> TensorCumulativeTrapezoid<T>(Tensor<T> y, double dx = 1, int dim = -1);
}
