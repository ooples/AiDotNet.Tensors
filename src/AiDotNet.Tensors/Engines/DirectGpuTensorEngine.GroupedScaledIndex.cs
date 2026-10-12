using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// GPU dispatch of advanced indexing and the grouped / scaled GEMMs.
/// </summary>
/// <remarks>
/// <para>Each is a composition whose compute steps are engine calls on <c>this</c>, so on a GPU engine they run as
/// device kernels: advanced indexing computes the flat source positions on the host (from the small index tensors)
/// and gathers with <see cref="IEngine.TensorTake{T}"/>; the grouped GEMM is <see cref="IEngine.TensorNarrow{T}"/> /
/// <see cref="IEngine.TensorMatMul{T}"/> / <see cref="IEngine.TensorConcatenate{T}"/>; the scaled GEMMs apply their
/// scales with <see cref="IEngine.TensorMultiply{T}"/> and multiply with <see cref="IEngine.TensorMatMul{T}"/>. The
/// residency probe checks that each launches device kernels; a dedicated fused kernel would replace the composition
/// here without changing callers.</para>
/// </remarks>
public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> TensorIndex<T>(Tensor<T> tensor, Tensor<int>?[] indices) => base.TensorIndex(tensor, indices);

    /// <inheritdoc/>
    public override Tensor<T> TensorIndex<T>(Tensor<T> tensor, Tensor<bool> mask) => base.TensorIndex(tensor, mask);

    /// <inheritdoc/>
    public override Tensor<T> TensorGroupedMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<int>? offsets = null, Tensor<T>? bias = null)
        => base.TensorGroupedMatMul(a, b, offsets, bias);

    /// <inheritdoc/>
    public override Tensor<T> TensorScaledMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> scaleA, ScalingRecipe recipeA,
        Tensor<T> scaleB, ScalingRecipe recipeB, ScaleSwizzle swizzleA = ScaleSwizzle.None, ScaleSwizzle swizzleB = ScaleSwizzle.None,
        Tensor<T>? bias = null)
        => base.TensorScaledMatMul(a, b, scaleA, recipeA, scaleB, recipeB, swizzleA, swizzleB, bias);

    /// <inheritdoc/>
    public override Tensor<T> TensorScaledGroupedMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> scaleA, ScalingRecipe recipeA,
        Tensor<T> scaleB, ScalingRecipe recipeB, ScaleSwizzle swizzleA = ScaleSwizzle.None, ScaleSwizzle swizzleB = ScaleSwizzle.None,
        Tensor<T>? bias = null, Tensor<int>? offsets = null)
        => base.TensorScaledGroupedMatMul(a, b, scaleA, recipeA, scaleB, recipeB, swizzleA, swizzleB, bias, offsets);
}
