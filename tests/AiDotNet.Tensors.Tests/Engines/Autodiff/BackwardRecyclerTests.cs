using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Backward buffers going back to the step's arena before the step ends (<see cref="BackwardRecycler{T}"/>): the
/// arena hands a returned buffer out again only within its step, and a training step's gradients are the same with
/// recycling on, off, or poisoning every buffer it calls dead.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class BackwardRecyclerTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<float> Random(int[] shape, Random rng, float scale)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    [Fact]
    public void ReturnedRingTensor_ArrayIsReissuedUnderANewTensor_AndNotTwiceAfterReset()
    {
        using var arena = TensorArena.Create();
        var log = new List<TensorArena.IssuedBuffer>();
        arena.OpenIssueLog(log);
        var first = TensorAllocator.RentUninitialized<float>(new[] { 64, 64 });
        arena.CloseIssueLog(log);
        Assert.Single(log);
        var array = first.GetBackingArrayForCacheLookupUnsafe();

        // The array comes back within the step, but never the wrapper: state on the old object described its first
        // owner, and handing the object on corrupted LayerNorm and BatchNorm gradients.
        arena.ReturnIssued(log[0]);
        var again = TensorAllocator.RentUninitialized<float>(new[] { 32, 128 });
        Assert.NotSame(first, again);
        Assert.Same(array, again.GetBackingArrayForCacheLookupUnsafe());
        Assert.Equal(new[] { 32, 128 }, again.Shape.ToArray());

        // Returned, then the step ends: the cursor hands the wrapper out once, and the stale free entry must not hand
        // its array out a second time.
        arena.ReturnIssued(log[0]);
        arena.Reset();
        var afterReset1 = TensorAllocator.RentUninitialized<float>(new[] { 64, 64 });
        var afterReset2 = TensorAllocator.RentUninitialized<float>(new[] { 64, 64 });
        Assert.Same(first, afterReset1);
        Assert.NotSame(afterReset1.GetBackingArrayForCacheLookupUnsafe(), afterReset2.GetBackingArrayForCacheLookupUnsafe());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NormBackward_AfterARecycledGradient_MatchesTheUnrecycledGradient(bool batchNorm)
    {
        // The loss gradient a Multiply backward consumes is returned and taken by the norm's input gradient. Reissuing
        // the same wrapper object there gave LayerNorm a kernel gradient of 2.0577 against -0.0271.
        var rng = new Random(13);
        var x = Random(new[] { 2, 1, 4, 4 }, rng, 1f);
        var k = Random(new[] { 8, 1, 3, 3 }, rng, 1f);
        var b = Random(new[] { 8 }, rng, 1f);
        var scale = Random(batchNorm ? new[] { 8 } : new[] { 4 }, rng, 1f);
        var shift = Random(batchNorm ? new[] { 8 } : new[] { 4 }, rng, 1f);
        var w = Random(new[] { 2, 8, 4, 4 }, rng, 1f);
        var one = new[] { 1, 1 };
        float[] Gradient(bool recycle)
        {
            var prior = BackwardRecycler<float>.EnabledOverride;
            BackwardRecycler<float>.EnabledOverride = recycle;
            try
            {
                using var tape = new GradientTape<float>();
                var h = _engine.ReLU(_engine.TensorChannelBiasAdd(_engine.Conv2D(x, k, one, one, one), b));
                var y = batchNorm
                    ? _engine.BatchNorm(h, scale, shift, 1e-5, out _, out _)
                    : _engine.LayerNorm(h, scale, shift, 1e-5, out _, out _);
                var loss = _engine.ReduceSum(_engine.TensorMultiply(y, w), null, false);
                return tape.ComputeGradients(loss, new[] { k, b })[k].ToArray();
            }
            finally
            {
                BackwardRecycler<float>.EnabledOverride = prior;
            }
        }

        BackwardRecycler<float>.LastPassReturned = 0;
        var recycled = Gradient(true);
        Assert.True(BackwardRecycler<float>.LastPassReturned > 0, "nothing was recycled");
        Assert.Equal(Gradient(false), recycled);
    }

    // The arena has one issue log. A second owner (a recycler of another element type on the same thread) must not
    // take it over, nor close it for the owner that opened it.
    [Fact]
    public void IssueLog_HasOneOwner()
    {
        using var arena = TensorArena.Create();
        var owner = new List<TensorArena.IssuedBuffer>();
        var other = new List<TensorArena.IssuedBuffer>();
        Assert.True(arena.OpenIssueLog(owner));
        Assert.False(arena.OpenIssueLog(other));
        arena.CloseIssueLog(other);
        TensorAllocator.RentUninitialized<float>(new[] { 64, 64 });
        arena.CloseIssueLog(owner);
        TensorAllocator.RentUninitialized<float>(new[] { 64, 64 });
        Assert.Single(owner);
        Assert.Empty(other);
        Assert.True(arena.OpenIssueLog(other));
        arena.CloseIssueLog(other);
    }

    // ReduceSum's backward hands back a Fill-ed ones tensor, tagged as uniform so ReluBackward can read one scalar. The
    // recycler returned that wrapper, LayerNorm's backward was given it and wrote its gradient through a lease, and the
    // ReLU then read the stale tag: every gradient below it came out as if the upstream gradient were all ones (#1702).
    [Fact]
    public void ReissuedRingTensor_DropsThePreviousOwnersUniformFill()
    {
        var engine = new CpuEngine();
        using var arena = TensorArena.Create();
        var log = new List<TensorArena.IssuedBuffer>();
        arena.OpenIssueLog(log);
        var ones = TensorAllocator.RentUninitialized<float>(new[] { 4, 8 });
        arena.CloseIssueLog(log);
        ones.Fill(1f);
        Assert.Single(log);

        arena.ReturnIssued(log[0]);
        var gradient = TensorAllocator.RentUninitialized<float>(new[] { 4, 8 });
        Assert.Same(ones, gradient);
        using (var lease = gradient.LeaseArray())
            for (int i = 0; i < gradient.Length; i++) lease.Array[i] = 0.25f * (i + 1);

        var input = new Tensor<float>(new[] { 4, 8 });
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, i % 3 == 0 ? -1f : 1f);
        var result = engine.ReluBackward(gradient, input);
        for (int i = 0; i < result.Length; i++)
            Assert.Equal(i % 3 == 0 ? 0f : 0.25f * (i + 1), result.GetFlat(i));
    }

    [Fact]
    public void ReturnedScratchArray_IsReissuedWithinTheStep_AndNotTwiceAfterReset()
    {
        using var arena = TensorArena.Create();
        var log = new List<TensorArena.IssuedBuffer>();
        arena.OpenIssueLog(log);
        var first = arena.TryAllocate<float>(5000)!;
        arena.CloseIssueLog(log);
        Assert.Single(log);
        first[7] = 3f;

        arena.ReturnIssued(log[0]);
        var again = arena.TryAllocate<float>(5000)!;
        Assert.Same(first, again);
        Assert.Equal(0f, again[7]); // a zeroing rent still zeroes a returned array

        arena.ReturnIssued(log[0]);
        arena.Reset();
        var afterReset1 = arena.TryAllocate<float>(5000)!;
        var afterReset2 = arena.TryAllocate<float>(5000)!;
        Assert.NotSame(afterReset1, afterReset2);
    }

    // A 7x7 stem, two same-shape 1x1 convs with a residual fan-out, a reshape, a matmul head and a squared loss:
    // gradients that pass through views, gradients shared by two consumers, and conv backward kernels whose temporaries
    // other caches pool (the transposed-convolution input gradient's flipped kernel went through AutoTensorCache).
    private Dictionary<Tensor<float>, Tensor<float>> Step(Tensor<float> x, Tensor<float> k1, Tensor<float> k2,
        Tensor<float> w, int step, bool releaseActivations = false)
    {
        using var tape = new GradientTape<float>(new GradientTapeOptions { ReleaseActivationsDuringBackward = releaseActivations });
        var h = _engine.ReLU(_engine.Conv2D(x, k1, new[] { 1, 1 }, new[] { 3, 3 }, new[] { 1, 1 }));
        var r = _engine.ReLU(_engine.Conv2D(h, k2, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }));
        r = _engine.Conv2D(r, k2, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var y = _engine.Sigmoid(_engine.TensorAdd(h, r));
        var flat = _engine.Reshape(y, new[] { x.Shape[0], y.Length / x.Shape[0] });
        var logits = _engine.TensorMatMul(flat, w);
        var scaled = _engine.TensorMultiplyScalar(logits, 1f + 0.1f * step);
        var loss = _engine.ReduceMean(_engine.TensorMultiply(scaled, scaled), new[] { 0, 1 }, keepDims: false);
        return tape.ComputeGradients(loss, new[] { k1, k2, w });
    }

    private List<float[][]> Train(bool? recycle, bool stress, bool inArena, out int returned)
        => Train(recycle, stress, inArena, releaseActivations: false, out returned, out _);

    private List<float[][]> Train(bool? recycle, bool stress, bool inArena, bool releaseActivations, out int returned,
        out int forwardReturned)
    {
        var rng = new Random(2453);
        var x = Random(new[] { 2, 3, 40, 40 }, rng, 1f);
        var k1 = Random(new[] { 16, 3, 7, 7 }, rng, 0.1f);
        var k2 = Random(new[] { 16, 16, 1, 1 }, rng, 0.2f);
        var w = Random(new[] { 16 * 40 * 40, 10 }, rng, 0.005f);
        var gradients = new List<float[][]>();
        returned = 0;
        forwardReturned = 0;
        var priorEnabled = BackwardRecycler<float>.EnabledOverride;
        var priorStress = BackwardRecycler<float>.StressOverride;
        BackwardRecycler<float>.EnabledOverride = recycle;
        BackwardRecycler<float>.StressOverride = stress;
        try
        {
            using var arena = inArena ? TensorArena.Create() : null;
            for (int step = 0; step < 5; step++)
            {
                arena?.Reset();
                var g = Step(x, k1, k2, w, step, releaseActivations);
                returned += BackwardRecycler<float>.LastPassReturned;
                forwardReturned += BackwardRecycler<float>.LastPassForwardReturned;
                BackwardRecycler<float>.LastPassReturned = 0;
                BackwardRecycler<float>.LastPassForwardReturned = 0;
                gradients.Add(new[] { g[k1].ToArray(), g[k2].ToArray(), g[w].ToArray() });
                // Apply the update inside the step, as a training loop does, so later steps see new weights.
                _engine.TensorSubtractInPlace(k1, _engine.TensorMultiplyScalar(g[k1], 0.05f));
                _engine.TensorSubtractInPlace(k2, _engine.TensorMultiplyScalar(g[k2], 0.05f));
                _engine.TensorSubtractInPlace(w, _engine.TensorMultiplyScalar(g[w], 0.05f));
            }
        }
        finally
        {
            BackwardRecycler<float>.EnabledOverride = priorEnabled;
            BackwardRecycler<float>.StressOverride = priorStress;
        }
        return gradients;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TrainingStepGradients_AreIdentical_WithRecyclingOnOffAndPoisoned(bool stress)
    {
        var reference = Train(recycle: false, stress: false, inArena: false, out _);
        var withoutRecycling = Train(recycle: false, stress: false, inArena: true, out int returnedWhenOff);
        var recycled = Train(recycle: true, stress: stress, inArena: true, out int returned);

        // The mechanism must have engaged, or the comparison proves nothing; and must stay off when disabled.
        Assert.True(returned > 0, "no backward buffer was returned to the arena");
        Assert.Equal(0, returnedWhenOff);
        for (int step = 0; step < reference.Count; step++)
        {
            for (int p = 0; p < reference[step].Length; p++)
            {
                Assert.Equal(reference[step][p], withoutRecycling[step][p]);
                Assert.Equal(reference[step][p], recycled[step][p]);
            }
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void TrainingStepGradients_AreIdentical_WhenActivationsAreReleasedDuringBackward(bool tapeOwnsArena, bool stress)
    {
        // Poisoning fills every released activation with NaN instead of reusing it: a backward step that still read one
        // after its planned last use would turn the gradients NaN, and a second owner writing it fails at the pass end.
        var reference = Train(recycle: false, stress: false, inArena: false, out _);
        var released = Train(recycle: true, stress, inArena: !tapeOwnsArena, releaseActivations: true, out _,
            out int forwardReturned);

        Assert.True(forwardReturned > 0, "no forward activation was handed back during backward");
        for (int step = 0; step < reference.Count; step++)
            for (int p = 0; p < reference[step].Length; p++)
                Assert.Equal(reference[step][p], released[step][p]);
    }

    [Fact]
    public void ActivationsAreKept_WithoutTheTapeOption()
    {
        Train(recycle: true, stress: false, inArena: true, releaseActivations: false, out int returned, out int forwardReturned);
        Assert.True(returned > 0);
        Assert.Equal(0, forwardReturned);
    }

    [Fact]
    public void SecondBackward_AfterActivationsWereReleased_Throws()
    {
        var rng = new Random(11);
        var x = Random(new[] { 8, 64 }, rng, 1f);
        var w = Random(new[] { 64, 64 }, rng, 0.1f);
        using var tape = new GradientTape<float>(new GradientTapeOptions { ReleaseActivationsDuringBackward = true });
        var h = _engine.Sigmoid(_engine.TensorMatMul(x, w));
        var h2 = _engine.Sigmoid(_engine.TensorMatMul(h, w));
        var loss = _engine.ReduceMean(_engine.TensorMultiply(h2, h2), new[] { 0, 1 }, keepDims: false);
        tape.ComputeGradients(loss, new[] { w });
        Assert.True(BackwardRecycler<float>.LastPassForwardReturned > 0);
        Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(loss, new[] { w }));
    }

    [Fact]
    public void ForwardOutputNoBackwardStepReads_KeepsItsValues()
    {
        // A forward output outside the loss's graph (a prediction the caller reads afterwards) is never planned for
        // release, however much the backward rents.
        var rng = new Random(5);
        var x = Random(new[] { 16, 128 }, rng, 1f);
        var w = Random(new[] { 128, 128 }, rng, 0.1f);
        using var tape = new GradientTape<float>(new GradientTapeOptions { ReleaseActivationsDuringBackward = true });
        var side = _engine.Sigmoid(_engine.TensorMatMul(x, x.Transpose(new[] { 1, 0 })));
        var expected = side.ToArray();
        var h = _engine.Sigmoid(_engine.TensorMatMul(x, w));
        var loss = _engine.ReduceMean(_engine.TensorMultiply(h, h), new[] { 0, 1 }, keepDims: false);
        tape.ComputeGradients(loss, new[] { w });
        for (int i = 0; i < 4; i++) _engine.TensorMultiplyScalar(_engine.TensorMatMul(x, x.Transpose(new[] { 1, 0 })), 2f);
        Assert.Equal(expected, side.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReturnedGradient_IsHeldOneStep_UnlessReleasedAtDispose(bool releaseAtDispose)
    {
        // Each tape owns a fresh arena and hands its buffers to the cross-arena pool when disposed; a held gradient
        // stays out of that pool until the next step ends, so the next step cannot be given it.
        var rng = new Random(3);
        var x = Random(new[] { 8, 512 }, rng, 1f);
        var w = Random(new[] { 512, 512 }, rng, 0.05f);
        float[]? gradientArray;
        using (var tape = new GradientTape<float>(new GradientTapeOptions { ReleaseResultsAtDispose = releaseAtDispose }))
        {
            var h = _engine.TensorMatMul(x, w);
            var loss = _engine.ReduceMean(_engine.TensorMultiply(h, h), new[] { 0, 1 }, keepDims: false);
            gradientArray = tape.ComputeGradients(loss, new[] { w })[w].GetBackingArrayForCacheLookupUnsafe();
        }
        Assert.NotNull(gradientArray);

        bool reused = false;
        using (new GradientTape<float>())
        {
            for (int i = 0; i < 4; i++)
            {
                var rented = TensorAllocator.RentUninitialized<float>(new[] { 512, 512 });
                reused |= ReferenceEquals(rented.GetBackingArrayForCacheLookupUnsafe(), gradientArray);
            }
        }
        Assert.Equal(releaseAtDispose, reused);
    }

    [Fact]
    public void AutoTensorCache_RefusesArenaBuffers()
    {
        // A kernel temporary rented inside an arena and handed to AutoTensorCache (Conv2DBackwardInputInto's flipped
        // kernel) must stay with the arena alone: the cache outlives the step, and the arena reissues the buffer, so
        // both handed the same array to different tensors and FasterRCNN trained on corrupted gradients.
        using var arena = TensorArena.Create();
        var temporary = AutoTensorCache.RentOrAllocate<float>(new[] { 48, 48 });
        AutoTensorCache.Return(temporary);
        arena.Reset();
        var fromArena = TensorAllocator.RentUninitialized<float>(new[] { 48, 48 });
        var fromCache = AutoTensorCache.RentOrAllocate<float>(new[] { 48, 48 });
        Assert.Same(temporary, fromArena);
        Assert.NotSame(fromArena, fromCache);
    }

    [Fact]
    public void OutsideAnyArena_NothingIsRecycled()
    {
        // A top-level tape opens its own arena by default; with that off and none open, there is nowhere to return to.
        bool prior = GradientTape<float>.EnableTapeOwnedArena;
        GradientTape<float>.EnableTapeOwnedArena = false;
        try
        {
            Train(recycle: true, stress: false, inArena: false, out int returned);
            Assert.Equal(0, returned);
        }
        finally
        {
            GradientTape<float>.EnableTapeOwnedArena = prior;
        }
    }

    [Fact]
    public void WholeGradientMap_IsNotRecycled()
    {
        // With no sources the caller receives every intermediate gradient, so none of them may be reused.
        var rng = new Random(7);
        var x = Random(new[] { 4, 16 }, rng, 1f);
        var w = Random(new[] { 16, 16 }, rng, 0.1f);
        using var arena = TensorArena.Create();
        BackwardRecycler<float>.LastPassReturned = 0;
        using var tape = new GradientTape<float>();
        var h = _engine.Sigmoid(_engine.TensorMatMul(x, w));
        var loss = _engine.ReduceMean(_engine.TensorMultiply(h, h), new[] { 0, 1 }, keepDims: false);
        var all = tape.ComputeGradients(loss);
        Assert.Equal(0, BackwardRecycler<float>.LastPassReturned);
        Assert.True(all.ContainsKey(h));
    }
}
