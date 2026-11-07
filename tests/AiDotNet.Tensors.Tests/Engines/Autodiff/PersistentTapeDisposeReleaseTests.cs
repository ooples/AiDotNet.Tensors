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
    [Fact]
    public void DisposedPersistentTape_StillReferenced_DoesNotPinItsOutputs()
    {
        var engine = new CpuEngine();
        var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var hidden = RecordAndDifferentiate(engine, tape);
        tape.Dispose();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        Assert.False(hidden.IsAlive, "a disposed persistent tape still pins its recorded outputs");
        GC.KeepAlive(tape);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference RecordAndDifferentiate(CpuEngine engine, GradientTape<float> tape)
    {
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var w = new Tensor<float>(new float[] { 0.5f, -1f, 2f, 0.25f }, new[] { 2, 2 });
        var hidden = engine.TensorMatMul(x, w);
        var loss = engine.ReduceSum(engine.TensorMultiply(hidden, hidden), null);
        var grads = tape.ComputeGradients(loss, new[] { w });
        Assert.True(grads.ContainsKey(w));
        return new WeakReference(hidden);
    }
}
