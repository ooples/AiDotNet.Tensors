using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CpuParallelSettings")]
public sealed class LstmWorkspaceRequestTests
{
    [Fact]
    public void BatchEightSharesOneBoundedProjectionAcrossItsCompleteRowBlockAndTail()
    {
        const int batch = 8, steps = 32, inputs = 32, hidden = 64, gates = 4 * hidden, rows = 6;
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { batch, steps, inputs });
        var wi = new Tensor<float>(new[] { gates, inputs });
        var wh = new Tensor<float>(new[] { gates, hidden });
        int originalWorkers = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var before = ForwardRequests();
            var result = engine.LstmSequenceForward(x, null, null, wi, wh, null, null, returnSequences: true);
            var after = ForwardRequests();
            const int bytes = ((inputs + hidden) * gates + rows * (steps * gates + gates + 2 * hidden)) * sizeof(float);
            Assert.Equal(CpuStorageAllocationDiagnostics.Enabled ? 1L : 0L, after.Count - before.Count);
            Assert.Equal(CpuStorageAllocationDiagnostics.Enabled ? (long)bytes : 0L, after.Bytes - before.Bytes);
            Assert.Equal(new[] { batch, steps, hidden }, result.Shape.ToArray());
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = originalWorkers;
        }
    }

    [Theory]
    [InlineData(8, 7, false)]
    [InlineData(8, 8, true)]
    [InlineData(16, 7, true)]
    [InlineData(16, 8, false)]
    public void BatchBlocksKeepEachSequenceAndStateIndependent(int batch, int hidden, bool returnSequences)
    {
        const int steps = 9, inputs = 5;
        int gates = 4 * hidden;
        var engine = new CpuEngine();
        var rng = new Random(3941);
        float[] Values(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
        var x = Values(batch * steps * inputs);
        var h0 = Values(batch * hidden);
        var c0 = Values(batch * hidden);
        var wi = new Tensor<float>(Values(gates * inputs), new[] { gates, inputs });
        var wh = new Tensor<float>(Values(gates * hidden), new[] { gates, hidden });
        var bi = new Tensor<float>(Values(gates), new[] { gates });
        var bh = new Tensor<float>(Values(gates), new[] { gates });
        var output = engine.LstmSequenceForward(new Tensor<float>(x, new[] { batch, steps, inputs }),
            new Tensor<float>(h0, new[] { batch, hidden }), new Tensor<float>(c0, new[] { batch, hidden }),
            wi, wh, bi, bh, out var h, out var c, returnSequences);

        static void CheckBlock(Tensor<float> actual, Tensor<float> expected, int offset)
        {
            var a = actual.AsSpan();
            var e = expected.AsSpan();
            for (int i = 0; i < e.Length; i++)
                Assert.InRange(Math.Abs(a[offset + i] - e[i]), 0f, 1e-5f);
        }
        for (int row = 0; row < batch; row += 2)
        {
            var chunk = engine.LstmSequenceForward(
                new Tensor<float>(x.AsSpan(row * steps * inputs, 2 * steps * inputs).ToArray(), new[] { 2, steps, inputs }),
                new Tensor<float>(h0.AsSpan(row * hidden, 2 * hidden).ToArray(), new[] { 2, hidden }),
                new Tensor<float>(c0.AsSpan(row * hidden, 2 * hidden).ToArray(), new[] { 2, hidden }),
                wi, wh, bi, bh, out var chunkH, out var chunkC, returnSequences);
            CheckBlock(output, chunk, row * hidden * (returnSequences ? steps : 1));
            CheckBlock(h, chunkH, row * hidden);
            CheckBlock(c, chunkC, row * hidden);
        }
    }


    [Fact]
    public void BatchOneForwardBorrowsOneBoundedWorkspaceOnEveryCall()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new int[] { 1, 32, 32 });
        var inputWeight = new Tensor<float>(new int[] { 256, 32 });
        var recurrentWeight = new Tensor<float>(new int[] { 256, 64 });
        // Eight projected input rows, one projected hidden row, and the current hidden and cell vectors.
        const long bytes = (8 * 256 + 256 + 2 * 64) * sizeof(float);
        var before = ForwardRequests();
        for (int call = 0; call < 2; call++)
        {
            var result = engine.LstmSequenceForward(input, null, null, inputWeight, recurrentWeight, null, null);
            var after = ForwardRequests();
            Assert.Equal(CpuStorageAllocationDiagnostics.Enabled ? 1L : 0L, after.Count - before.Count);
            Assert.Equal(CpuStorageAllocationDiagnostics.Enabled ? bytes : 0L, after.Bytes - before.Bytes);
            Assert.Equal(new[] { 1, 64 }, result.Shape.ToArray());
            Assert.All(result.ToVector().ToArray(), value => Assert.Equal(0f, value));
            before = after;
        }
    }

    [Theory]
    [InlineData(3, false)]
    [InlineData(3, true)]
    [InlineData(16, false)]
    [InlineData(16, true)]
    public void BatchOneMatchesIndependentBatchedRecurrenceAcrossTileBoundary(int hidden, bool returnSequences)
    {
        const int steps = 11, inputs = 5;
        var engine = new CpuEngine();
        var random = new Random(827);
        float[] Values(int length) => Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() - .5)).ToArray();
        var x = Values(steps * inputs);
        var h = Values(hidden);
        var c = Values(hidden);
        var wi = new Tensor<float>(Values(4 * hidden * inputs), new[] { 4 * hidden, inputs });
        var wh = new Tensor<float>(Values(4 * hidden * hidden), new[] { 4 * hidden, hidden });
        var bi = new Tensor<float>(Values(4 * hidden), new[] { 4 * hidden });
        var bh = new Tensor<float>(Values(4 * hidden), new[] { 4 * hidden });
        var one = engine.LstmSequenceForward(new Tensor<float>(x, new[] { 1, steps, inputs }),
            new Tensor<float>(h, new[] { 1, hidden }), new Tensor<float>(c, new[] { 1, hidden }),
            wi, wh, bi, bh, out var h1, out var c1, returnSequences);
        var two = engine.LstmSequenceForward(new Tensor<float>(x.Concat(x).ToArray(), new[] { 2, steps, inputs }),
            new Tensor<float>(h.Concat(h).ToArray(), new[] { 2, hidden }),
            new Tensor<float>(c.Concat(c).ToArray(), new[] { 2, hidden }),
            wi, wh, bi, bh, out var h2, out var c2, returnSequences);

        static void CheckFirstRow(Tensor<float> actual, Tensor<float> batched)
        {
            var a = actual.ToVector().ToArray();
            var b = batched.ToVector().ToArray();
            Assert.Equal(2 * a.Length, b.Length);
            for (int i = 0; i < a.Length; i++)
                Assert.InRange(Math.Abs(a[i] - b[i]), 0f, 1e-5f);
        }
        CheckFirstRow(one, two);
        CheckFirstRow(h1, h2);
        CheckFirstRow(c1, c2);
    }

    private static (long Count, long Bytes) ForwardRequests()
    {
        var tags = (Dictionary<string, object>)CpuStorageAllocationDiagnostics.Snapshot()["tags"];
        if (!tags.TryGetValue("workspace.lstm_forward_pool", out object? value)) return (0, 0);
        var totals = (Dictionary<string, object>)value;
        return ((long)totals["count"], (long)totals["bytes"]);
    }
}
