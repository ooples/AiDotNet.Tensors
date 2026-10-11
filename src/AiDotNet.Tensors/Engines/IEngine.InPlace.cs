using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// In-place forms (torch <c>op_</c>) that are more than "compute the out-of-place result and write it back": random
/// fills, autograd state, layout changes and callables. The rest are in <c>IEngine.InPlace.Generated.cs</c>.
/// </summary>
public partial interface IEngine
{
    /// <summary>Sets every element to zero (torch <c>zero_</c>).</summary>
    void TensorZeroInPlace<T>(Tensor<T> tensor);

    /// <summary>Copies <paramref name="source"/>, broadcast to the tensor's shape, into <paramref name="tensor"/> (torch <c>copy_</c>); the gradient flows to the source.</summary>
    void TensorCopyInPlace<T>(Tensor<T> tensor, Tensor<T> source);

    /// <summary>Detaches the tensor from its history (torch <c>detach_</c>): gradients of later uses stop here.</summary>
    void TensorDetachInPlace<T>(Tensor<T> tensor);

    /// <summary>
    /// torch <c>requires_grad_</c>. The gradient tape records every op and returns gradients for whichever tensors are
    /// asked for, so <c>true</c> is already the case; <c>false</c> detaches the tensor from its history.
    /// </summary>
    void TensorRequiresGradInPlace<T>(Tensor<T> tensor, bool requiresGrad = true);

    /// <summary>Zeroes each element with probability <paramref name="dropoutRate"/> and scales the rest by 1/(1-p) (torch <c>dropout_</c>).</summary>
    void DropoutInPlace<T>(Tensor<T> input, double dropoutRate = 0.5, bool training = true);

    /// <summary>Fills the main diagonal with <paramref name="value"/> (torch <c>fill_diagonal_</c>); <paramref name="wrap"/> continues it down a tall matrix.</summary>
    void TensorFillDiagonalInPlace<T>(Tensor<T> tensor, T value, bool wrap = false);

    /// <summary>Replaces every element x with <paramref name="func"/>(x) (torch <c>apply_</c>).</summary>
    /// <remarks>Evaluated on the host by every engine, for the reason given on <see cref="TensorMapInPlace{T}"/>.</remarks>
    void TensorApplyInPlace<T>(Tensor<T> tensor, Func<T, T> func);

    /// <summary>Replaces every element x with <paramref name="func"/>(x, y), y from <paramref name="other"/> broadcast to the tensor (torch <c>map_</c>).</summary>
    /// <remarks>The function is a managed delegate, so every engine (a GPU engine included) evaluates it on the host, as torch's <c>map_</c> is CPU-only; there is no device kernel to dispatch to.</remarks>
    void TensorMapInPlace<T>(Tensor<T> tensor, Tensor<T> other, Func<T, T, T> func);

    /// <summary>Replaces every element s with <paramref name="func"/>(s, x, y), from <paramref name="x"/> and <paramref name="y"/> broadcast to the tensor (torch <c>map2_</c>).</summary>
    /// <remarks>Evaluated on the host by every engine, for the reason given on <see cref="TensorMapInPlace{T}"/>.</remarks>
    void TensorMap2InPlace<T>(Tensor<T> tensor, Tensor<T> x, Tensor<T> y, Func<T, T, T, T> func);

    /// <summary>
    /// Rescales each row of <paramref name="weight"/> named in <paramref name="indices"/> whose p-norm exceeds
    /// <paramref name="maxNorm"/> down to it (torch <c>embedding_renorm_</c>, the <c>max_norm</c> of an embedding lookup).
    /// </summary>
    void TensorEmbeddingRenormInPlace<T>(Tensor<T> weight, Tensor<int> indices, double maxNorm, double normType = 2.0);

    /// <summary>Fills with Bernoulli(<paramref name="p"/>) samples, 0 or 1 (torch <c>bernoulli_</c>).</summary>
    void TensorBernoulliInPlace<T>(Tensor<T> tensor, double p = 0.5, int? seed = null);

    /// <summary>Fills with Bernoulli samples, each with the probability at its position in <paramref name="p"/> (torch <c>bernoulli_</c>).</summary>
    void TensorBernoulliInPlace<T>(Tensor<T> tensor, Tensor<T> p, int? seed = null);

    /// <summary>Fills with Cauchy(<paramref name="median"/>, <paramref name="sigma"/>) samples (torch <c>cauchy_</c>).</summary>
    void TensorCauchyInPlace<T>(Tensor<T> tensor, double median = 0, double sigma = 1, int? seed = null);

    /// <summary>Fills with exponential samples of rate <paramref name="rate"/> (torch <c>exponential_</c>).</summary>
    void TensorExponentialInPlace<T>(Tensor<T> tensor, double rate = 1, int? seed = null);

    /// <summary>Fills with geometric samples of success probability <paramref name="p"/> (torch <c>geometric_</c>).</summary>
    void TensorGeometricInPlace<T>(Tensor<T> tensor, double p, int? seed = null);

    /// <summary>Fills with log-normal samples whose logarithm has the given mean and standard deviation (torch <c>log_normal_</c>).</summary>
    void TensorLogNormalInPlace<T>(Tensor<T> tensor, double mean = 1, double std = 2, int? seed = null);

    /// <summary>Fills with Normal(<paramref name="mean"/>, <paramref name="std"/>) samples (torch <c>normal_</c>).</summary>
    void TensorNormalInPlace<T>(Tensor<T> tensor, double mean = 0, double std = 1, int? seed = null);

    /// <summary>Fills with Uniform[<paramref name="low"/>, <paramref name="high"/>) samples (torch <c>uniform_</c>).</summary>
    void TensorUniformInPlace<T>(Tensor<T> tensor, double low = 0, double high = 1, int? seed = null);

    /// <summary>Fills with integers drawn uniformly from [<paramref name="low"/>, <paramref name="high"/>) (torch <c>random_</c>).</summary>
    void TensorRandomIntegersInPlace<T>(Tensor<T> tensor, long low, long high, int? seed = null);

    /// <summary>Fills with integers drawn uniformly from [0, <paramref name="high"/>) (torch <c>random_(to)</c>).</summary>
    void TensorRandomIntegersInPlace<T>(Tensor<T> tensor, long high, int? seed = null);

    /// <summary>Removes size-1 dimensions in place, as <see cref="TensorSqueeze{T}"/> does out of place (torch <c>squeeze_</c>).</summary>
    void TensorSqueezeInPlace<T>(Tensor<T> tensor, int axis = -1);

    /// <summary>Inserts a size-1 dimension at <paramref name="axis"/> in place (torch <c>unsqueeze_</c>).</summary>
    void TensorUnsqueezeInPlace<T>(Tensor<T> tensor, int axis);

    /// <summary>Swaps two dimensions in place (torch <c>swapaxes_</c> / <c>swapdims_</c>).</summary>
    void TensorSwapAxesInPlace<T>(Tensor<T> tensor, int axis1, int axis2);

    /// <summary>Swaps dimensions <paramref name="dim0"/> and <paramref name="dim1"/> in place (torch <c>transpose_</c>).</summary>
    void TensorTransposeInPlace<T>(Tensor<T> tensor, int dim0, int dim1);

    /// <summary>Transposes a tensor of at most 2 dimensions in place (torch <c>t_</c>).</summary>
    void TensorTInPlace<T>(Tensor<T> tensor);

    /// <summary>Gives the tensor an arbitrary shape and strides over its own storage (torch <c>as_strided_</c>).</summary>
    void TensorAsStridedInPlace<T>(Tensor<T> tensor, int[] size, int[] stride, int? storageOffset = null);

    /// <summary>Makes the tensor share <paramref name="source"/>'s storage, shape and strides (torch <c>set_</c>).</summary>
    void TensorSetInPlace<T>(Tensor<T> tensor, Tensor<T> source);

    /// <summary>Makes the tensor a view of <paramref name="source"/>'s storage with the given layout (torch <c>set_(storage, offset, size, stride)</c>).</summary>
    void TensorSetInPlace<T>(Tensor<T> tensor, Tensor<T> source, int storageOffset, int[] size, int[] stride);

    /// <summary>
    /// Resizes the tensor to <paramref name="shape"/> (torch <c>resize_</c>): the storage is kept when the new size fits
    /// and grown otherwise; existing elements keep their row-major values and new ones are zero.
    /// </summary>
    void TensorResizeInPlace<T>(Tensor<T> tensor, int[] shape);

    /// <summary>Resizes the tensor to <paramref name="other"/>'s shape (torch <c>resize_as_</c>).</summary>
    void TensorResizeAsInPlace<T>(Tensor<T> tensor, Tensor<T> other);
}
