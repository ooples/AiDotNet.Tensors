using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public sealed class RebindablePlanCacheCorrectnessTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public async Task FreshTapeReplay_PreservesGradientAcrossMetadataView()
    {
        await Task.Yield();
        RebindablePlanCache<double>.ResetForTests();

        try
        {
            for (int run = 0; run < 2; run++)
            {
                var x = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, new[] { 2, 3 });
                var scale = new Tensor<double>(new[] { 2.0 }, new[] { 1 });

                using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = false });
                // Exercise the direct Tensor API. Engine.Reshape already owns an engine-level
                // tape entry; direct metadata views historically created only a GradNode and were
                // therefore invisible to entry-index replay.
                var view = x.Reshape(new[] { 3, 2 });
                var scaled = _engine.TensorMultiply(view, scale);
                var loss = _engine.ReduceSum(scaled, null);
                var gradients = tape.ComputeGradients(loss, new[] { x, scale });

                Assert.True(gradients.TryGetValue(x, out var xGradient));
                Assert.True(gradients.TryGetValue(scale, out var scaleGradient));
                Assert.Equal(new[] { 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 }, xGradient.ToArray());
                Assert.Equal(21.0, scaleGradient[0], 12);
            }
        }
        finally
        {
            RebindablePlanCache<double>.ResetForTests();
        }
    }

    [Fact]
    public async Task StructureHash_DistinguishesDeadBranchFromConnectedBranch()
    {
        await Task.Yield();

        long deadBranchHash = RecordHash(connectProductToLoss: false);
        long connectedBranchHash = RecordHash(connectProductToLoss: true);

        Assert.NotEqual(deadBranchHash, connectedBranchHash);
    }

    [Fact]
    public async Task FreshTapeReplay_DoesNotReusePlanWithDifferentDagConnectivity()
    {
        await Task.Yield();
        RebindablePlanCache<double>.ResetForTests();

        try
        {
            // Store a plan whose multiply branch is recorded but dead.
            _ = ComputeGradient(connectProductToLoss: false);

            // Same operation names, shapes, and entry count, but the multiply now feeds the loss.
            // A flat structure hash aliases these graphs and replays the incomplete dead-branch plan.
            var gradient = ComputeGradient(connectProductToLoss: true);
            Assert.Equal(new[] { 5.0, 6.0, 7.0 }, gradient.ToArray());
        }
        finally
        {
            RebindablePlanCache<double>.ResetForTests();
        }
    }

    private long RecordHash(bool connectProductToLoss)
    {
        var a = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });
        var b = new Tensor<double>(new[] { 4.0, 5.0, 6.0 }, new[] { 3 });
        using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = false });
        var product = _engine.TensorMultiply(a, b);
        var sum = connectProductToLoss
            ? _engine.TensorAdd(product, a)
            : _engine.TensorAdd(a, b);
        var loss = _engine.ReduceSum(sum, null);
        long hash = AutoTrainingCompiler.ComputeStructureHash(tape.Entries, tape.EntryCount);
        GC.KeepAlive(loss);
        return hash;
    }

    private Tensor<double> ComputeGradient(bool connectProductToLoss)
    {
        var a = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });
        var b = new Tensor<double>(new[] { 4.0, 5.0, 6.0 }, new[] { 3 });
        using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = false });
        var product = _engine.TensorMultiply(a, b);
        var sum = connectProductToLoss
            ? _engine.TensorAdd(product, a)
            : _engine.TensorAdd(a, b);
        var loss = _engine.ReduceSum(sum, null);
        return tape.ComputeGradients(loss, new[] { a })[a];
    }
}
