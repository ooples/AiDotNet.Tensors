using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NN.Losses;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Random sampling, dropout variants and composed ops matching PyTorch. Every random op takes an optional seed (a
/// cryptographically seeded generator when it is null).
/// </summary>
public partial interface IEngine
{
    /// <summary>1 with probability p per element, else 0 (<c>torch.bernoulli</c>).</summary>
    Tensor<T> TensorBernoulli<T>(Tensor<T> probabilities, int? seed = null);

    /// <summary>Binomial(count, probability) draws per element (<c>torch.binomial</c>).</summary>
    Tensor<T> TensorBinomial<T>(Tensor<T> count, Tensor<T> probability, int? seed = null);

    /// <summary>Poisson(rate) draws per element (<c>torch.poisson</c>).</summary>
    Tensor<T> TensorPoisson<T>(Tensor<T> rates, int? seed = null);

    /// <summary>Normal(mean, std) draws per element (<c>torch.normal</c>).</summary>
    Tensor<T> TensorNormal<T>(Tensor<T> mean, Tensor<T> std, int? seed = null);

    /// <summary>Uniform [low, high) draws (<c>Tensor.uniform_</c>).</summary>
    Tensor<T> TensorUniform<T>(int[] shape, double low = 0, double high = 1, int? seed = null);

    /// <summary>Cauchy(median, sigma) draws (<c>Tensor.cauchy_</c>).</summary>
    Tensor<T> TensorCauchy<T>(int[] shape, double median = 0, double sigma = 1, int? seed = null);

    /// <summary>Exponential(rate) draws (<c>Tensor.exponential_</c>).</summary>
    Tensor<T> TensorExponential<T>(int[] shape, double rate = 1, int? seed = null);

    /// <summary>Geometric(p) draws, the number of trials to the first success (≥ 1) (<c>Tensor.geometric_</c>).</summary>
    Tensor<T> TensorGeometric<T>(int[] shape, double p, int? seed = null);

    /// <summary>exp(Normal(mean, std)) draws (<c>Tensor.log_normal_</c>).</summary>
    Tensor<T> TensorLogNormal<T>(int[] shape, double mean = 1, double std = 2, int? seed = null);

    /// <summary>
    /// <paramref name="numSamples"/> category indices per row of a 1-D or 2-D weight tensor (<c>torch.multinomial</c>).
    /// </summary>
    Tensor<T> TensorMultinomial<T>(Tensor<T> probabilities, int numSamples, bool replacement = false, int? seed = null);

    /// <summary>SELU-compatible alpha dropout, keeping mean and variance (<c>torch.alpha_dropout</c>).</summary>
    Tensor<T> TensorAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null);

    /// <summary>Alpha dropout of whole channels (<c>torch.feature_alpha_dropout</c>).</summary>
    Tensor<T> TensorFeatureAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null);

    /// <summary>
    /// Zeroes whole channels with probability p and scales the rest by 1/(1-p) (<c>dropout1d/2d/3d</c>,
    /// <c>feature_dropout</c>). The channel axis is the one before the trailing <paramref name="spatialDims"/> axes.
    /// </summary>
    Tensor<T> TensorChannelDropout<T>(Tensor<T> tensor, double p, bool training, int spatialDims, int? seed = null);

    /// <summary>y = x1ᵀ·W·x2 + b per output feature: x1 [..., in1], x2 [..., in2], W [out, in1, in2] (<c>torch.nn.functional.bilinear</c>).</summary>
    Tensor<T> TensorBilinear<T>(Tensor<T> input1, Tensor<T> input2, Tensor<T> weight, Tensor<T>? bias = null);

    /// <summary>Interleaves channel groups: [N, g·k, ...] → [N, k·g, ...] (<c>torch.nn.functional.channel_shuffle</c>).</summary>
    Tensor<T> TensorChannelShuffle<T>(Tensor<T> tensor, int groups);

    /// <summary>[N, C, H·r, W·r] → [N, C·r², H, W], the inverse of PixelShuffle (<c>torch.nn.functional.pixel_unshuffle</c>).</summary>
    Tensor<T> TensorPixelUnshuffle<T>(Tensor<T> tensor, int downscaleFactor);

    /// <summary>
    /// Local response normalization across channels: x / (k + α·mean of x² over a window of <paramref name="size"/>)^β
    /// (<c>torch.nn.functional.local_response_norm</c>).
    /// </summary>
    Tensor<T> TensorLocalResponseNorm<T>(Tensor<T> tensor, int size, double alpha = 1e-4, double beta = 0.75, double k = 1);

    /// <summary>log(1 + exp(-y·x)) reduced (<c>torch.nn.functional.soft_margin_loss</c>).</summary>
    Tensor<T> TensorSoftMarginLoss<T>(Tensor<T> input, Tensor<T> target, LossReduction reduction = LossReduction.Mean);

    /// <summary>Each slice along <paramref name="dim"/> rescaled so its p-norm is at most <paramref name="maxNorm"/> (<c>torch.renorm</c>).</summary>
    Tensor<T> TensorRenorm<T>(Tensor<T> tensor, double p, int dim, double maxNorm);

    /// <summary>The p-norm over every axis except <paramref name="dim"/>, kept broadcastable (<c>torch.norm_except_dim</c>); dim = -1 takes the whole norm.</summary>
    Tensor<T> TensorNormExceptDim<T>(Tensor<T> tensor, double pow = 2, int dim = 0);

    /// <summary>
    /// The numerical gradient along each axis (central differences inside, one-sided at the edges) with uniform
    /// <paramref name="spacing"/> (<c>torch.gradient</c>).
    /// </summary>
    Tensor<T>[] TensorGradient<T>(Tensor<T> tensor, double spacing = 1, int[]? dims = null);

    /// <summary>Sequences of different lengths padded to the longest and stacked (<c>torch.nn.utils.rnn.pad_sequence</c>).</summary>
    Tensor<T> TensorPadSequence<T>(Tensor<T>[] sequences, bool batchFirst = false, double paddingValue = 0);

    /// <summary>The [size, rank] indices of the first <paramref name="size"/> non-zero elements, padded with <paramref name="fillValue"/> (<c>torch.nonzero_static</c>).</summary>
    Tensor<int> TensorNonzeroStatic<T>(Tensor<T> tensor, int size, int fillValue = -1);

    /// <summary>The distinct slices along <paramref name="dim"/>, sorted (<c>torch.unique(dim=...)</c>).</summary>
    Tensor<T> TensorUniqueDim<T>(Tensor<T> tensor, int dim);

    /// <summary>
    /// tensor with source's slices reduced into the slices at <paramref name="index"/> along <paramref name="dim"/>
    /// (<c>Tensor.index_reduce</c>).
    /// </summary>
    Tensor<T> TensorIndexReduce<T>(Tensor<T> tensor, int dim, Tensor<int> index, Tensor<T> source, ScatterReduceMode reduce, bool includeSelf = true);

    /// <summary>Transposed 1-D convolution: input [N, Cin, L], kernel [Cin, Cout, K] (<c>torch.nn.functional.conv_transpose1d</c>).</summary>
    Tensor<T> TensorConvTranspose1D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int outputPadding = 0);

    /// <summary>1-D convolution in time-batch-channel layout: input [T, B, Cin], weight [K, Cin, Cout] (<c>torch.conv_tbc</c>).</summary>
    Tensor<T> TensorConvTbc<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias, int pad = 0);
}
