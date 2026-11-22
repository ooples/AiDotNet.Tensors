namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : ISmallTopKRowsBackend
{
    /// <inheritdoc/>
    public int MaxSmallTopK => SmallTopKRows.MaxK;

    /// <inheritdoc/>
    public void TopKRowsSmall(IGpuBuffer input, IGpuBuffer values, IGpuBuffer indices, int rows, int cols, int k)
    {
        if (!SmallTopKRows.Validate(input, values, indices, rows, cols, k)) return;
        GlslDispatchN(VulkanInstantNgpKernels.TopKRowsSmall, rows,
            [input, values, indices], [(uint)rows, (uint)cols, (uint)k]);
    }
}
