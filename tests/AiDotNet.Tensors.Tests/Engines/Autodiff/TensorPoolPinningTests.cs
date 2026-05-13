using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Issue #338: validates the tape-pinning flag on <see cref="Tensor{T}"/>
/// that allows per-op backward intermediate pooling without breaking
/// Hvp / SumToScalarTensor aliasing.
///
/// Pre-fix, PR #331 attempted to pool MatMulBackward's bT2 / aT2 scratch
/// tensors but had to revert because pooled tensors could be reissued
/// under aliases before a higher-order backward consumed them. The pin
/// flag is the gate: tensors recorded as tape inputs get pinned, and
/// <see cref="TensorPool{T}.Return"/> refuses to recycle them until the
/// tape's cleanup walk clears the flag.
/// </summary>
public class TensorPoolPinningTests
{
    [Fact]
    public void TensorPool_Return_RefusesPinnedTensor()
    {
        TensorPool<float>.Clear();
        var pinned = new Tensor<float>(new[] { 16 });
        pinned._pinnedByTape = true;

        // Pool must refuse the pinned tensor — Rent on the same length
        // must return a fresh tensor, not the pinned one.
        TensorPool<float>.Return(pinned);
        var rented = TensorPool<float>.Rent(new[] { 16 });
        Assert.NotSame(pinned, rented);
    }

    [Fact]
    public void TensorPool_Return_AcceptsUnpinnedTensor()
    {
        TensorPool<float>.Clear();
        var unpinned = new Tensor<float>(new[] { 16 });
        Assert.False(unpinned._pinnedByTape);

        // Unpinned tensors flow through the pool normally — Rent
        // returns the same instance.
        TensorPool<float>.Return(unpinned);
        var rented = TensorPool<float>.Rent(new[] { 16 });
        Assert.Same(unpinned, rented);
    }

    [Fact]
    public void TapeRecord_SetsPinFlagOnInputs()
    {
        // The forward op records onto an active tape; the recorder pins
        // each input so a downstream caller can't pool-return it before
        // backward consumes it.
        var engine = new CpuEngine();
        var a = new Tensor<float>(new[] { 4, 4 });
        var b = new Tensor<float>(new[] { 4, 4 });

        Assert.False(a._pinnedByTape);
        Assert.False(b._pinnedByTape);

        using (var tape = new GradientTape<float>())
        {
            var c = engine.TensorAdd(a, b);
            // Both inputs must now be pinned — the binary recorder
            // sets the flag on Input0 AND Input1.
            Assert.True(a._pinnedByTape, "Input0 not pinned by tape recorder");
            Assert.True(b._pinnedByTape, "Input1 not pinned by tape recorder");
        }
    }

    [Fact]
    public void TapeCleanup_ClearsPinFlag()
    {
        // After backward completes, the tape's cleanup walk clears the
        // pin so the tensor can be pooled by a subsequent unrelated op.
        // Use a clean tape lifecycle: explicit Clear of any thread-static
        // backward caches that prior tests may have warmed up.
        AiDotNet.Tensors.Engines.Autodiff.RebindablePlanCache<float>.ResetForTests();
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { 4, 4 });
        for (int i = 0; i < w.Length; i++) w.SetFlat(i, 0.1f);
        var x = new Tensor<float>(new[] { 4, 4 });
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, 1.0f);

        using (var tape = new GradientTape<float>())
        {
            var y = engine.TensorMatMul(x, w);
            var loss = engine.ReduceSum(y, null);
            tape.ComputeGradients(loss, sources: new[] { w });
        }
        // Pin must be cleared by the cleanup walk so the tensor is
        // poolable again.
        Assert.False(w._pinnedByTape, "Pin flag not cleared after backward (w)");
        Assert.False(x._pinnedByTape, "Pin flag not cleared after backward (x)");
    }

    [Fact]
    public void HvpScenario_PinnedTensor_NotPooledDuringSecondBackward()
    {
        // Higher-order AD: the inner backward consumes a tensor that the
        // outer tape has also recorded. Without pinning, the inner could
        // pool-return the tensor while the outer's later backward still
        // references it. With pinning, the pool refuses the return —
        // tensor stays valid for the outer's consumption.
        TensorPool<float>.Clear();
        var t = new Tensor<float>(new[] { 8 });
        t._pinnedByTape = true;

        TensorPool<float>.Return(t);
        // Confirm pool is empty after rejected return — Rent must
        // allocate fresh, not hand back the pinned tensor.
        var fresh = TensorPool<float>.Rent(new[] { 8 });
        Assert.NotSame(t, fresh);
    }
}
