using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A persistent tape caches the leaf set (and, on large tapes, the relevance set) of its last gradient call, and both
/// hold the tape's recorded outputs. Dispose must drop them: an owner that keeps the disposed tape referenced would
/// otherwise pin every output of the step, and on a GPU engine their device buffers with them.
/// </summary>
public class PersistentTapeDisposeReleaseTests
{
    [Theory]
    [InlineData(0)]     // small tape: only the leaf-source cache is built
    [InlineData(120)]   // 100+ entries: the gradient relevance set is cached as well
    public void DisposedPersistentTape_StillReferenced_DoesNotPinItsOutputs(int extraOps)
    {
        var engine = new CpuEngine();
        var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var hidden = RecordAndDifferentiate(engine, tape, extraOps);
        tape.Dispose();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Assert.False(hidden.IsAlive, "a disposed persistent tape still pins its recorded outputs");
        GC.KeepAlive(tape);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference RecordAndDifferentiate(CpuEngine engine, GradientTape<float> tape, int extraOps)
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var w = new Tensor<float>(new float[] { 0.5f, -1f, 2f, 0.25f }, new[] { 2, 2 });
        var hidden = engine.TensorMatMul(x, w);
        var y = engine.TensorMultiply(hidden, hidden);
        for (int i = 0; i < extraOps; i++) y = engine.TensorAdd(y, hidden);
        var loss = engine.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { w });
        Assert.True(grads.ContainsKey(w));
        return new WeakReference(hidden);
    }
}
