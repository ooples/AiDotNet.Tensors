using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Volumetric grid sampling matching PyTorch's <c>torch.grid_sampler_3d</c>.</summary>
public partial interface IEngine
{
    /// <summary>
    /// Samples <paramref name="input"/> <c>[N, C, D, H, W]</c> at the normalized (x, y, z) locations in
    /// <paramref name="grid"/> <c>[N, D', H', W', 3]</c> (x indexes W, z indexes D), trilinearly or by nearest
    /// voxel, with zeros/border/reflection out-of-range handling (<c>torch.grid_sampler_3d</c>,
    /// <c>F.grid_sample</c> on 5-D input). Differentiable in both the input and the grid.
    /// </summary>
    Tensor<T> TensorGridSample3D<T>(Tensor<T> input, Tensor<T> grid, GridSampleMode mode = GridSampleMode.Bilinear,
        GridSamplePadding padding = GridSamplePadding.Zeros, bool alignCorners = false);
}
