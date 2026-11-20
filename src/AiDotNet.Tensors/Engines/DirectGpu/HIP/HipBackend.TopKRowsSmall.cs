namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : ISmallTopKRowsBackend
{
    /// <inheritdoc/>
    public int MaxSmallTopK => SmallTopKRows.MaxK;

    /// <inheritdoc/>
    public unsafe void TopKRowsSmall(IGpuBuffer input, IGpuBuffer values, IGpuBuffer indices, int rows, int cols, int k)
    {
        if (!SmallTopKRows.Validate(input, values, indices, rows, cols, k)) return;
        var kernel = ResolveInstantNgpKernel("resident_topk_rows_small");
        System.IntPtr a0 = input.Handle, a1 = values.Handle, a2 = indices.Handle;
        int a3 = rows, a4 = cols, a5 = k;
        void** args = stackalloc void*[6];
        args[0] = &a0; args[1] = &a1; args[2] = &a2; args[3] = &a3; args[4] = &a4; args[5] = &a5;
        LaunchKernel(kernel, (uint)((rows + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        Synchronize();
    }
}
