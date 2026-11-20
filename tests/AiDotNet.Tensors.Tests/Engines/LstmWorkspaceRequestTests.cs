using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CpuParallelSettings")]
public sealed class LstmWorkspaceRequestTests
{
    [Fact]
    public void BatchOneForwardCountsAllSevenWorkspaceRequestsIncludingPoolHits()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new int[] { 1, 32, 32 });
        var inputWeight = new Tensor<float>(new int[] { 256, 32 });
        var recurrentWeight = new Tensor<float>(new int[] { 256, 64 });
        // Wx, transposed recurrent weights, four 64-element states and one 256-element recurrent projection.
        const long bytes = (32 * 256 + 64 * 256 + 4 * 64 + 256) * sizeof(float);
        var before = ForwardRequests();
        for (int call = 0; call < 2; call++)
        {
            var result = engine.LstmSequenceForward(input, null, null, inputWeight, recurrentWeight, null, null);
            var after = ForwardRequests();
            Assert.Equal(CpuStorageAllocationDiagnostics.Enabled ? 7L : 0L, after.Count - before.Count);
            Assert.Equal(CpuStorageAllocationDiagnostics.Enabled ? bytes : 0L, after.Bytes - before.Bytes);
            Assert.Equal(new[] { 1, 64 }, result.Shape);
            Assert.All(result.ToVector().ToArray(), value => Assert.Equal(0f, value));
            before = after;
        }
    }

    private static (long Count, long Bytes) ForwardRequests()
    {
        var tags = (Dictionary<string, object>)CpuStorageAllocationDiagnostics.Snapshot()["tags"];
        if (!tags.TryGetValue("workspace.lstm_forward_pool", out object? value)) return (0, 0);
        var totals = (Dictionary<string, object>)value;
        return ((long)totals["count"], (long)totals["bytes"]);
    }
}
