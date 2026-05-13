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
        // Higher-order AD via createGraph: true: the first backward records
        // gradient-of-gradient ops back onto the same tape, then the second
        // backward replays. Both backwards reference the parameter tensor
        // `x`. The forward and the inner backward must NOT pool-return `x`
        // while the outer (second) backward still needs to walk through it.
        AiDotNet.Tensors.Engines.Autodiff.RebindablePlanCache<float>.ResetForTests();
        TensorPool<float>.Clear();
        var engine = new CpuEngine();

        var x = new Tensor<float>(new[] { 1 });
        x.SetFlat(0, 3f);

        using (var tape = new GradientTape<float>())
        {
            // y = x · x · x  (y'  = 3x², y'' = 6x)
            var xx = engine.TensorMultiply(x, x);
            var y  = engine.TensorMultiply(xx, x);

            // First backward with createGraph: true so the d/dx ops are
            // themselves recorded onto the tape — that's the moment a
            // naive pool-return on `x` would corrupt the second backward.
            var firstGrads  = tape.ComputeGradients(y, new[] { x }, createGraph: true);
            // y' at x=3 = 27
            Assert.Equal(27f, firstGrads[x][0], precision: 3);

            // Second backward must still see correct math — would be
            // broken if any of {x, xx} got recycled out from under us.
            // The ref-count semantics matter here: the createGraph=true
            // first backward re-pins inputs through its own Record* calls,
            // and a naive blanket `_pinnedByTape = false` cleanup would
            // unpin the originals while the second backward still walks
            // them. The fact that this Assert.Equal holds is the test.
            var secondGrads = tape.ComputeGradients(firstGrads[x], new[] { x });
            // y'' at x=3 = 18
            Assert.Equal(18f, secondGrads[x][0], precision: 3);
        }
    }

    [Fact]
    public void RefCount_OverlappingPins_StayPinnedUntilAllCleared()
    {
        // The ref-counted pin survives one tape's cleanup pass when an
        // outer tape is still pinning the same tensor. This is the
        // scenario the plain-bool pin got wrong (PR #347 review): inner
        // tape's blanket `_pinnedByTape = false` would unpin a tensor
        // the outer tape still depends on.
        TensorPool<float>.Clear();
        var t = new Tensor<float>(new[] { 8 });

        t._pinnedByTape = true;   // outer-tape pin
        t._pinnedByTape = true;   // inner-tape pin (count -> 2)

        // Inner tape cleanup clears its pin once.
        t._pinnedByTape = false;
        Assert.True(t._pinnedByTape, "First clear must NOT fully unpin while outer tape still holds a pin");

        // Pool still refuses while the outer pin is live.
        TensorPool<float>.Return(t);
        var fresh = TensorPool<float>.Rent(new[] { 8 });
        Assert.NotSame(t, fresh);

        // Outer tape cleanup clears the last pin.
        t._pinnedByTape = false;
        Assert.False(t._pinnedByTape, "Final clear must fully unpin");

        // Extra clears are safe no-ops (clamped at 0): a stale cleanup
        // from a different tape must not flip the count negative.
        t._pinnedByTape = false;
        Assert.False(t._pinnedByTape);
    }

    [Fact]
    public void TensorPool_Return_RefusesStridedView()
    {
        // PR #331's pooling attempt broke gradients because
        // TransposeLastTwoDims of a rank-3+ tensor returns a strided
        // permute view, not a fresh allocation — and the pool's
        // Reshape on Rent would let a later caller write through the
        // shared storage into the source tensor. The view-safety
        // guard in TensorPool.Return must refuse such tensors.
        TensorPool<float>.Clear();
        var source = new Tensor<float>(new[] { 2, 3, 4 });
        // Construct a permute view that swaps the last two dims —
        // this is what TransposeLastTwoDims produces.
        var permView = source.Transpose(new[] { 0, 2, 1 });
        Assert.False(permView.IsContiguous,
            "Test setup invalid — permute should produce non-contiguous view.");

        TensorPool<float>.Return(permView);

        // Pool must be empty — the view was refused. A Rent of the
        // same length must allocate fresh, not hand back the view.
        var rented = TensorPool<float>.Rent(new[] { 2, 4, 3 });
        Assert.NotSame(permView, rented);
    }

    [Fact]
    public void TensorPool_Return_RefusesReshapeView()
    {
        // A Reshape on contiguous storage returns a view sharing the
        // backing array. Same pooling hazard as the permute view —
        // the view-safety guard catches both via
        // GetLiveBackingArrayOrNull, which requires _storageOffset == 0
        // AND _storage.Length == Length (i.e. owned-not-aliased).
        TensorPool<float>.Clear();
        var source = new Tensor<float>(new[] { 24 });
        var reshapeView = source.Reshape(new[] { 2, 3, 4 });

        TensorPool<float>.Return(reshapeView);

        var rented = TensorPool<float>.Rent(new[] { 24 });
        Assert.NotSame(reshapeView, rented);
        Assert.NotSame(source, rented);
    }
}
