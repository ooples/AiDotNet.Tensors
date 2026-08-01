#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    public void CosineSimilarity(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
        => DispatchScientificDistanceAsync(
            "CosineSimilarity", WebGpuParity210Kernels.CosineSimilarity,
            a, b, output, batchSize, batchSize, dim).GetAwaiter().GetResult();

    public void PairwiseDistance(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => DispatchScientificDistanceAsync(
            "PairwiseDistance", WebGpuParity210Kernels.CdistL2,
            a, b, output, CheckedPairwiseWorkItems(m, n), m, n, dim).GetAwaiter().GetResult();

    public void PairwiseDistanceSquared(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => DispatchScientificDistanceAsync(
            "PairwiseDistanceSquared", WebGpuParity210Kernels.CdistL2Squared,
            a, b, output, CheckedPairwiseWorkItems(m, n), m, n, dim).GetAwaiter().GetResult();

    private static int CheckedPairwiseWorkItems(int m, int n)
    {
        long total = (long)m * n;
        if (total <= 0) return 0;
        if (total > int.MaxValue)
            throw new OverflowException(
                $"Pairwise-distance work-item count {total} exceeds Int32.MaxValue.");
        return (int)total;
    }

    private async Task DispatchScientificDistanceAsync(
        string operation, string source, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output,
        int workItems, params int[] dimensions)
    {
        if (workItems <= 0) return;
        var pipelineId = await GetOrCreatePipelineAsync(
            $"ScientificDistance:{operation}", source, "main");
        using var uniforms = new WebGpuBuffer(
            UniformInts(dimensions), WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bind = new WebGpuBindGroup(
            pipelineId, AsWgpu(a), AsWgpu(b), AsWgpu(output));
        var (workgroups, _) = _device.CalculateWorkgroups1D(workItems);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bind.BindGroupId, uniforms.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
}
#endif
