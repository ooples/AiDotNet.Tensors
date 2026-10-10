using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Grouped and scaled matrix products (torch.nn.functional.grouped_mm, scaled_mm, scaled_grouped_mm).</summary>
public partial interface IEngine
{
    /// <summary>
    /// A grouped matrix product, as in mixture-of-experts layers (torch.nn.functional.grouped_mm).
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item>[G, M, K] x [G, K, N] -> [G, M, N]: G independent products; <paramref name="offsets"/> must be null.</item>
    /// <item>[Mtot, K] x [G, K, N] -> [Mtot, N]: rows [offsets[i-1], offsets[i]) use b[i] (the expert forward); rows after
    /// the last offset are zero.</item>
    /// <item>[M, Ktot] x [Ktot, N] -> [G, M, N]: group i contracts over [offsets[i-1], offsets[i]) (the expert weight
    /// gradient).</item>
    /// <item>[G, M, K] x [K, Ntot] -> [M, Ntot]: columns [offsets[i-1], offsets[i]) use a[i].</item>
    /// </list>
    /// <para><paramref name="offsets"/> holds the end of each group, non-decreasing. <paramref name="bias"/> is [N] or
    /// [G, N] and is added to each group's product. Differentiable.</para>
    /// </remarks>
    Tensor<T> TensorGroupedMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<int>? offsets = null, Tensor<T>? bias = null);

    /// <summary>
    /// The product of two scaled low-precision matrices (torch.nn.functional.scaled_mm): mm(a * scaleA, b * scaleB) + bias,
    /// with each operand's decoding scales laid out by its <see cref="ScalingRecipe"/> and <see cref="ScaleSwizzle"/>.
    /// </summary>
    /// <remarks>
    /// <paramref name="a"/> is [M, K] and <paramref name="b"/> [K, N], given as their decoded values. Block scales of
    /// <paramref name="b"/> are laid out over its transpose [N, K], as torch passes a column-major b. Not differentiable.
    /// </remarks>
    Tensor<T> TensorScaledMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> scaleA, ScalingRecipe recipeA,
        Tensor<T> scaleB, ScalingRecipe recipeB, ScaleSwizzle swizzleA = ScaleSwizzle.None, ScaleSwizzle swizzleB = ScaleSwizzle.None,
        Tensor<T>? bias = null);

    /// <summary>
    /// <see cref="TensorGroupedMatMul{T}"/> of scaled low-precision operands (torch.nn.functional.scaled_grouped_mm).
    /// A 3-D operand's scales carry a leading group axis; with a jagged contraction dimension (2-D x 2-D) row-wise
    /// scales are per group, [G, M] for a and [G, N] for b. Not differentiable.
    /// </summary>
    Tensor<T> TensorScaledGroupedMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> scaleA, ScalingRecipe recipeA,
        Tensor<T> scaleB, ScalingRecipe recipeB, ScaleSwizzle swizzleA = ScaleSwizzle.None, ScaleSwizzle swizzleB = ScaleSwizzle.None,
        Tensor<T>? bias = null, Tensor<int>? offsets = null);
}
