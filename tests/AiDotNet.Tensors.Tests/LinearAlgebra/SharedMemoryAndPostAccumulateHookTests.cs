using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>share_memory_ / is_shared, and register_post_accumulate_grad_hook.</summary>
public class SharedMemoryAndPostAccumulateHookTests
{
    [Fact]
    public void ShareMemory_KeepsTheValues_AndASecondMappingSeesEveryWrite()
    {
        var t = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        Assert.False(t.IsShared);
        Assert.Same(t, t.ShareMemory());
        Assert.True(t.IsShared);
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, t.ToArray());

        // An independent mapping of the same memory, as another process would open it.
        var other = Tensor<float>.FromSharedMemory(t.SharedMemoryPath!, new[] { 2, 3 });
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, other.ToArray());
        t[1, 2] = 42f;
        Assert.Equal(42f, other[1, 2]);
        other[0, 0] = -7f;
        Assert.Equal(-7f, t[0, 0]);
    }

    [Fact]
    public void ShareMemory_IsANoOpWhenAlreadyShared()
    {
        var t = new Tensor<double>(new double[] { 1, 2 }, new[] { 2 }).ShareMemory();
        string path = t.SharedMemoryPath!;
        t.ShareMemory();
        Assert.Equal(path, t.SharedMemoryPath);
    }

    [Fact]
    public void PostAccumulateHook_SeesTheFinishedGradient_OnceAllContributionsAreIn()
    {
        IEngine engine = new CpuEngine();
        var x = new Tensor<double>(new double[] { 1, 2, 3 }, new[] { 3 });
        var seen = new List<double[]>();
        using var tape = new GradientTape<double>();
        using (tape.RegisterPostAccumulateGradHook(x, (t, g) => { Assert.Same(x, t); seen.Add(g.ToArray()); }))
        {
            // x is used twice: the hook must see the sum of both contributions, once.
            var loss = engine.TensorAdd(engine.ReduceSum(engine.TensorMultiplyScalar(x, 2.0), null), engine.ReduceSum(engine.TensorMultiply(x, x), null));
            tape.ComputeGradients(loss, new[] { x });
        }
        Assert.Single(seen);
        Assert.Equal(new double[] { 4, 6, 8 }, seen[0]);   // 2 + 2x
    }

    [Fact]
    public void PostAccumulateHook_FiresInStreamingBackward_AndStopsWhenRemoved()
    {
        IEngine engine = new CpuEngine();
        var x = new Tensor<double>(new double[] { 1, 2 }, new[] { 2 });
        int calls = 0;
        using (var tape = new GradientTape<double>())
        {
            tape.RegisterPostAccumulateGradHook(x, (_, g) => { calls++; Assert.Equal(new double[] { 3, 3 }, g.ToArray()); });
            var loss = engine.ReduceSum(engine.TensorMultiplyScalar(x, 3.0), null);
            tape.ComputeGradientsStreaming(loss, new[] { x }, (_, _) => { });
        }
        Assert.Equal(1, calls);

        // A streaming backward releases the tape's graph, so the removal is checked on a tape of its own.
        using (var tape = new GradientTape<double>())
        {
            var handle = tape.RegisterPostAccumulateGradHook(x, (_, _) => calls++);
            handle.Dispose();
            var loss = engine.ReduceSum(engine.TensorMultiplyScalar(x, 3.0), null);
            tape.ComputeGradients(loss, new[] { x });
        }
        Assert.Equal(1, calls);
    }
}
