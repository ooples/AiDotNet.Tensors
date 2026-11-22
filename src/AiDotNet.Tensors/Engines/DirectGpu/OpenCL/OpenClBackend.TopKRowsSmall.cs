namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : ISmallTopKRowsBackend
{
    /// <inheritdoc/>
    public int MaxSmallTopK => SmallTopKRows.MaxK;

    /// <inheritdoc/>
    public void TopKRowsSmall(IGpuBuffer input, IGpuBuffer values, IGpuBuffer indices, int rows, int cols, int k)
    {
        if (!SmallTopKRows.Validate(input, values, indices, rows, cols, k)) return;
        var kernel = GetInstantNgpKernel("resident_topk_rows_small");
        kernel.SetArg(0, InstantNgpHandle(input));
        kernel.SetArg(1, InstantNgpHandle(values));
        kernel.SetArg(2, InstantNgpHandle(indices));
        kernel.SetArg(3, rows);
        kernel.SetArg(4, cols);
        kernel.SetArg(5, k);
        kernel.Execute1D(rows, CalculateOptimalWorkGroupSize1D(rows));
    }
}
