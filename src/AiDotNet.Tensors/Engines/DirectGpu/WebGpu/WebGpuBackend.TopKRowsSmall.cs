#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : ISmallTopKRowsBackend
{
    /// <inheritdoc/>
    public int MaxSmallTopK => SmallTopKRows.MaxK;

    /// <inheritdoc/>
    public void TopKRowsSmall(IGpuBuffer input, IGpuBuffer values, IGpuBuffer indices, int rows, int cols, int k)
    {
        if (!SmallTopKRows.Validate(input, values, indices, rows, cols, k)) return;
        Dispatch3BufferAsync("ResidentTopKRowsSmall", WebGpuInstantNgpKernels.TopKRowsSmall, "main",
            input, values, indices,
            [System.BitConverter.Int32BitsToSingle(rows), System.BitConverter.Int32BitsToSingle(cols),
                System.BitConverter.Int32BitsToSingle(k), 0f], rows).GetAwaiter().GetResult();
    }
}
#endif
