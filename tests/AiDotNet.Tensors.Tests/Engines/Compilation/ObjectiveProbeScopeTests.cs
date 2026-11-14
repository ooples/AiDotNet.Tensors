using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The storage-free probe trace behind compiled closure objectives: it records the objective's graph without
/// renting any output storage or executing a node, and aborts the moment the objective asks for a value.
/// </summary>
public class ObjectiveProbeScopeTests
{
    private const int N = 256;

    private static Tensor<float> Filled(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows * cols; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5));
        return t;
    }

    [Fact]
    public void Probe_records_every_op_without_tensor_storage()
    {
        var engine = new CpuEngine();
        var x = Filled(N, N, 1);
        var w = Filled(N, N, 2);

        int Probe()
        {
            using var scope = GraphMode.EnableProbe();
            engine.ReduceSum(engine.Tanh(engine.TensorMatMul(x, w)));
            return scope.Nodes.Count;
        }

        Probe(); // warm up JIT and field caches
#if NET6_0_OR_GREATER
        long before = GC.GetTotalAllocatedBytes(true);
        int count = Probe();
        long delta = GC.GetTotalAllocatedBytes(true) - before;
#else
        // net471 has no allocation counter; the live-heap delta after a forced collection still shows retained storage.
        long before = GC.GetTotalMemory(forceFullCollection: true);
        int count = Probe();
        long delta = GC.GetTotalMemory(forceFullCollection: false) - before;
#endif

        Assert.Equal(3, count);
        // Two [N,N] activations would cost 2*N*N*4 bytes if the probe rented storage; it must stay under a quarter of one.
        Assert.True(delta < N * N * sizeof(float) / 4,
            $"probe allocated {delta} bytes; one [{N},{N}] float activation is {N * N * sizeof(float)} bytes");
    }

    [Fact]
    public void Host_read_of_a_probe_tensor_aborts_the_probe()
    {
        var engine = new CpuEngine();
        var x = Filled(8, 8, 3);
        var w = Filled(8, 8, 4);

        var ex = Assert.Throws<ObjectiveProbeAbortedException>(() =>
        {
            using var scope = GraphMode.EnableProbe();
            var h = engine.Tanh(engine.TensorMatMul(x, w));
            _ = h.ToArray(); // a host read of an intermediate: the probe cannot know its value
            engine.ReduceSum(h);
        });

        Assert.Equal("Tanh", ex.OpName);
        Assert.False(GraphMode.IsActive);
    }

    [Fact]
    public void Probe_does_not_change_ordinary_graph_mode()
    {
        var engine = new CpuEngine();
        var x = Filled(8, 8, 5);
        var w = Filled(8, 8, 6);
        float expected = engine.ReduceSum(engine.Tanh(engine.TensorMatMul(x, w))).ToArray()[0];

        // A probe opened and closed first must leave no state behind.
        using (var probe = GraphMode.EnableProbe())
        {
            engine.ReduceSum(engine.Tanh(engine.TensorMatMul(x, w)));
            Assert.True(GraphMode.IsProbe);
        }
        Assert.False(GraphMode.IsProbe);

        float traced;
        using (var scope = GraphMode.Enable())
        {
            Assert.False(GraphMode.IsProbe);
            var loss = engine.ReduceSum(engine.Tanh(engine.TensorMatMul(x, w)));
            traced = loss.ToArray()[0];
        }

        Assert.Equal(expected, traced, 5);
    }
}
