namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    public void CosineSimilarity(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
    {
        if (batchSize <= 0) return;
        ThrowIfDisposed();
        RequireMetal3(a, b, output, out var aBuffer, out var bBuffer, out var outputBuffer);
        var pipeline = GetParity210Pipeline("cosine_similarity");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(outputBuffer, 2);
        encoder.SetBytes(batchSize, 3);
        encoder.SetBytes(dim, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void PairwiseDistance(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => DispatchScientificPairwise("parity210_cdist_l2", a, b, output, m, n, dim);

    public void PairwiseDistanceSquared(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => DispatchScientificPairwise("pairwise_distance_squared", a, b, output, m, n, dim);

    private void DispatchScientificPairwise(
        string kernelName, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output,
        int m, int n, int dim)
    {
        long totalLong = (long)m * n;
        if (totalLong <= 0) return;
        if (totalLong > int.MaxValue)
            throw new OverflowException(
                $"Pairwise-distance work-item count {totalLong} exceeds Int32.MaxValue.");
        int total = (int)totalLong;
        ThrowIfDisposed();
        RequireMetal3(a, b, output, out var aBuffer, out var bBuffer, out var outputBuffer);
        var pipeline = GetParity210Pipeline(kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(outputBuffer, 2);
        encoder.SetBytes(m, 3);
        encoder.SetBytes(n, 4);
        encoder.SetBytes(dim, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
