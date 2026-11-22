namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : ISmallTopKRowsBackend
{
    /// <inheritdoc/>
    public int MaxSmallTopK => SmallTopKRows.MaxK;

    /// <inheritdoc/>
    public void TopKRowsSmall(IGpuBuffer input, IGpuBuffer values, IGpuBuffer indices, int rows, int cols, int k)
    {
        if (!SmallTopKRows.Validate(input, values, indices, rows, cols, k)) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_topk_rows_small");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(rows);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)values, 1);
        encoder.SetBuffer((MetalGpuBuffer)indices, 2);
        encoder.SetBytes(rows, 3);
        encoder.SetBytes(cols, 4);
        encoder.SetBytes(k, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
