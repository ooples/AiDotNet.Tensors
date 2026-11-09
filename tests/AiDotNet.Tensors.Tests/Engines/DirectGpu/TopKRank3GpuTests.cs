using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorTopK on the last axis of a rank-3 tensor (LM logits [B, S, V]) runs the GPU row kernel and keeps the
/// leading shape. Before, it fell through to the CPU, which rejected every rank other than 2.
/// </summary>
public class TopKRank3GpuTests
{
    [SkippableFact]
    public void Rank3_last_axis_topk_matches_the_reference()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        int[] shape = { 2, 3, 500 };
        // Distinct values: the GPU sort's tie order is not part of the contract.
        var data = Enumerable.Range(0, 2 * 3 * 500).Select(i => (float)((i * 7919) % 3000)).ToArray();
        var values = ((IEngine)gpu).TensorTopK(new Tensor<float>(data, shape), 8, -1, out var indices);
        var (ev, ei) = TopKAnyAxisTests.Reference(data, shape, 2, 8, largest: true);
        Assert.Equal(new[] { 2, 3, 8 }, values.Shape.ToArray());
        Assert.Equal(new[] { 2, 3, 8 }, indices.Shape.ToArray());
        Assert.Equal(ev, values.ToArray());
        Assert.Equal(ei, indices.ToArray());
    }
}
