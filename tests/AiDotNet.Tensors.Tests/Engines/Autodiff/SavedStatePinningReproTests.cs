using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Coverage for the "savedState is pinned against pool/arena reuse" contract (Issue #338
/// completion).
///
/// Issue #338 pinned a recorded op's INPUTS so they can't be pool-reissued before the backward
/// consumes them, and <see cref="TensorPool{T}.Return"/> refuses a pinned tensor. But the
/// recorders originally pinned only inputs — the tensors a backward reads out of
/// <c>savedState</c> (RMSNorm/LayerNorm <c>rms</c>/<c>mean</c>/<c>variance</c>, attention weights,
/// dropout masks, RoPE cos/sin, fused pre-activations) were left unpinned. Under buffer reuse
/// those live-for-backward buffers could be reissued and overwritten mid-backward, silently
/// corrupting gradients — the aliasing failure the consumer's <c>Gru_ArenaOnEqualsOff</c> surfaces
/// at model scale.
///
/// The fix pins every <see cref="Tensor{T}"/> in savedState at record time and decrements it in
/// every backward-cleanup walk, exactly mirroring the input pin lifecycle. These tests lock that
/// in: the saved buffers are pinned during the backward, the pool refuses to reissue them, the
/// gradient is unaffected by a reissue attempt, and the pins are fully released afterward (no
/// leak). The pre-fix behavior was the opposite on every count.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class SavedStatePinningReproTests
{
    private static Tensor<float> Fixed(int[] shape, float start, float step)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, start + step * i);
        return t;
    }

    private static int[] ShapeOf(Tensor<float> t)
    {
        var s = new int[t.Rank];
        for (int i = 0; i < t.Rank; i++) s[i] = t.Shape[i];
        return s;
    }

    private static float[] Flat(Tensor<float> t)
    {
        var a = new float[t.Length];
        for (int i = 0; i < t.Length; i++) a[i] = t.GetFlat(i);
        return a;
    }

    private static void PassThroughBackward(
        Tensor<float> gradOutput,
        Tensor<float>[] inputs,
        Tensor<float> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
    }

    private static void AssertReleasedAndPoolable(Tensor<float> saved, string route)
    {
        Assert.False(saved._pinnedByTape, $"{route} leaked a saved-state pin");
        TensorPool<float>.Return(saved);
        var rerented = TensorPool<float>.Rent(ShapeOf(saved));
        Assert.Same(saved, rerented);
    }

    /// <summary>
    /// The saved <c>rms</c> that RMSNormBackward reads is pinned by the recorder, exactly like the
    /// op's inputs. Pre-fix only the inputs were pinned.
    /// </summary>
    [Fact]
    public void RmsNorm_SavedStateTensor_IsPinnedLikeInputs()
    {
        TensorPool<float>.Clear();
        var engine = new CpuEngine();
        var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
        var gamma = Fixed(new[] { 8 }, 1.0f, 0.05f);

        using var tape = new GradientTape<float>();
        var y = engine.RMSNorm(input, gamma, 1e-5, out var rms);

        Assert.True(input._pinnedByTape, "RMSNorm input should be pinned by the tape recorder");
        Assert.True(gamma._pinnedByTape, "RMSNorm gamma should be pinned by the tape recorder");
        Assert.True(rms._pinnedByTape,
            "the saved 'rms' tensor RMSNormBackward depends on must be pinned against pool/arena reuse");
    }

    /// <summary>
    /// LayerNorm saves TWO tensors (mean and variance). Both must be pinned during the backward and
    /// both released afterward — the recorder pins every tensor entry in savedState and the cleanup
    /// walk decrements each, keeping the refcount balanced.
    /// </summary>
    [Fact]
    public void LayerNorm_MultiTensorSavedState_AllPinnedThenReleased()
    {
        TensorPool<float>.Clear();
        var engine = new CpuEngine();
        var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
        var gamma = Fixed(new[] { 8 }, 1.0f, 0.05f);
        var beta = Fixed(new[] { 8 }, 0.0f, 0.02f);

        Tensor<float> mean, variance;
        using (var tape = new GradientTape<float>())
        {
            var y = engine.LayerNorm(input, gamma, beta, 1e-5, out mean, out variance);

            Assert.True(mean._pinnedByTape, "LayerNorm saved 'mean' must be pinned during backward");
            Assert.True(variance._pinnedByTape, "LayerNorm saved 'variance' must be pinned during backward");

            var loss = engine.ReduceSum(y, null);
            tape.ComputeGradients(loss, sources: new[] { input, gamma, beta });

            // Balanced refcount: the cleanup walk released the savedState pins.
            Assert.False(mean._pinnedByTape, "LayerNorm saved 'mean' pin leaked after backward");
            Assert.False(variance._pinnedByTape, "LayerNorm saved 'variance' pin leaked after backward");
        }
    }

    /// <summary>
    /// No leak: the savedState pin set at record time is fully cleared by the backward cleanup, so
    /// the buffer can be pooled again after training. A missed cleanup site would leave this &gt; 0.
    /// </summary>
    [Fact]
    public void RmsNorm_SavedStatePin_ReleasedAfterBackward_NoLeak()
    {
        TensorPool<float>.Clear();
        var engine = new CpuEngine();
        var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
        var gamma = Fixed(new[] { 8 }, 1.0f, 0.05f);

        Tensor<float> rms;
        using (var tape = new GradientTape<float>())
        {
            var y = engine.RMSNorm(input, gamma, 1e-5, out rms);
            Assert.True(rms._pinnedByTape, "precondition: rms pinned during the backward window");
            var loss = engine.ReduceSum(y, null);
            tape.ComputeGradients(loss, sources: new[] { input });
        }

        Assert.False(rms._pinnedByTape, "savedState pin leaked — a cleanup path failed to decrement it");
        // And the released buffer is genuinely poolable again.
        TensorPool<float>.Return(rms);
        var rerented = TensorPool<float>.Rent(ShapeOf(rms));
        Assert.Same(rms, rerented);
    }

    /// <summary>
    /// Empirical protection: while the tape is live, the pool REFUSES to reissue the saved
    /// <c>rms</c> (it is pinned), so an attempt to reuse+overwrite that buffer lands on a fresh
    /// allocation and the RMSNorm gradient is unaffected. Pre-fix the pool accepted the unpinned
    /// rms and the overwrite corrupted the gradient.
    /// </summary>
    [Fact]
    public void RmsNorm_PinnedSavedState_PoolReissueRefused_GradientUnchanged()
    {
        float[] RunGradient(bool attemptReissue)
        {
            TensorPool<float>.Clear();
            var engine = new CpuEngine();
            var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
            var gamma = Fixed(new[] { 8 }, 1.0f, 0.05f);

            using var tape = new GradientTape<float>();
            var y = engine.RMSNorm(input, gamma, 1e-5, out var rms);

            if (attemptReissue)
            {
                TensorPool<float>.Return(rms);
                var scratch = TensorPool<float>.Rent(ShapeOf(rms));
                Assert.NotSame(rms, scratch); // pool refused the pinned saved buffer
                for (int i = 0; i < scratch.Length; i++) scratch.SetFlat(i, 987654f);
            }

            var loss = engine.ReduceSum(y, null);
            var grads = tape.ComputeGradients(loss, sources: new[] { input });
            return Flat(grads[input]);
        }

        var clean = RunGradient(attemptReissue: false);
        var afterAttempt = RunGradient(attemptReissue: true);
        Assert.Equal(clean, afterAttempt);
    }

    /// <summary>
    /// Control: a PINNED input is refused by the pool, so the same reissue attempt cannot corrupt
    /// its gradient. Isolates the discriminator to the pin — savedState now behaves like inputs.
    /// </summary>
    [Fact]
    public void MatMul_PinnedInput_ReissueRefused_NoCorruption()
    {
        float[] RunGradient(bool attemptReissue)
        {
            TensorPool<float>.Clear();
            var engine = new CpuEngine();
            var a = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
            var b = Fixed(new[] { 8, 4 }, 0.2f, 0.07f);

            using var tape = new GradientTape<float>();
            var y = engine.TensorMatMul(a, b);

            if (attemptReissue)
            {
                TensorPool<float>.Return(a);
                var scratch = TensorPool<float>.Rent(ShapeOf(a));
                Assert.NotSame(a, scratch);
                for (int i = 0; i < scratch.Length; i++) scratch.SetFlat(i, 987654f);
            }

            var loss = engine.ReduceSum(y, null);
            var grads = tape.ComputeGradients(loss, sources: new[] { a });
            return Flat(grads[a]);
        }

        var clean = RunGradient(attemptReissue: false);
        var afterAttempt = RunGradient(attemptReissue: true);
        Assert.Equal(clean, afterAttempt);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void BoundedTape_RejectedRecorder_DoesNotCreateNodeOrPins(int arity)
    {
        TensorPool<float>.Clear();
        using var tape = new GradientTape<float>(new GradientTapeOptions
        {
            Persistent = false,
            MaxEntries = 1
        });

        var acceptedInput = Fixed(new[] { 2 }, 1f, 1f);
        var acceptedOutput = Fixed(new[] { 2 }, 2f, 1f);
        var acceptedSaved = Fixed(new[] { 2 }, 3f, 1f);
        DifferentiableOps.RecordUnary(
            "Accepted", acceptedOutput, acceptedInput, PassThroughBackward,
            new object[] { acceptedSaved });
        Assert.Equal(1, tape.EntryCount);
        Assert.True(acceptedSaved._pinnedByTape);

        var rejectedOutput = Fixed(new[] { 2 }, 4f, 1f);
        var rejectedSaved = Fixed(new[] { 2 }, 5f, 1f);
        var rejectedInputs = Enumerable.Range(0, arity)
            .Select(i => Fixed(new[] { 2 }, 10f + i, 1f))
            .ToArray();

        if (arity == 1)
        {
            DifferentiableOps.RecordUnary(
                "RejectedUnary", rejectedOutput, rejectedInputs[0], PassThroughBackward,
                new object[] { rejectedSaved });
        }
        else if (arity == 2)
        {
            DifferentiableOps.RecordBinary(
                "RejectedBinary", rejectedOutput, rejectedInputs[0], rejectedInputs[1],
                PassThroughBackward, new object[] { rejectedSaved });
        }
        else
        {
            DifferentiableOps.RecordIfActive(
                "RejectedVariadic", rejectedOutput, rejectedInputs, PassThroughBackward,
                new object[] { rejectedSaved });
        }

        Assert.Equal(1, tape.EntryCount);
        Assert.Null(rejectedOutput.GradFn);
        Assert.False(rejectedSaved._pinnedByTape);
        Assert.All(rejectedInputs, input => Assert.False(input._pinnedByTape));
    }

    [Fact]
    public void ManualTapeEntry_SavedState_IsPinnedAndReleased()
    {
        TensorPool<float>.Clear();
        var engine = new CpuEngine();
        var input = Fixed(new[] { 2 }, 1f, 1f);
        var output = Fixed(new[] { 2 }, 3f, 1f);
        var saved = Fixed(new[] { 2 }, 5f, 1f);

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = false });
        tape.Record(new TapeEntry<float>
        {
            OperationName = "ManualSavedState",
            Output = output,
            Input0 = input,
            InputCount = 1,
            Backward = PassThroughBackward,
            SavedState = new object[] { saved }
        });

        Assert.True(saved._pinnedByTape);
        Assert.NotNull(output.GradFn);
        var loss = engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { input });
        Assert.True(grads.ContainsKey(input));
        AssertReleasedAndPoolable(saved, "manual TapeEntry cleanup");
    }

    [Fact]
    public void StandardTapeWalk_ReleasesSavedStateForExecutedAndDeadEntries()
    {
        TensorPool<float>.Clear();
        var engine = new CpuEngine();
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = false });
        tape.DetectAnomaly = true; // Forces the standard tape walk instead of the GradFn fast path.

        var deadInput = Fixed(new[] { 2, 8 }, 0.2f, 0.03f);
        var deadGamma = Fixed(new[] { 8 }, 1f, 0.01f);
        _ = engine.RMSNorm(deadInput, deadGamma, 1e-5, out var deadRms);

        var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
        var gamma = Fixed(new[] { 8 }, 1f, 0.05f);
        var output = engine.RMSNorm(input, gamma, 1e-5, out var liveRms);
        var loss = engine.ReduceSum(output, null);
        tape.ComputeGradients(loss, new[] { input });

        Assert.False(liveRms._pinnedByTape);
        AssertReleasedAndPoolable(deadRms, "standard tape-walk dead-entry cleanup");
    }

    [Fact]
    public void CompiledBackward_ReleasesSavedStateForExecutedAndEliminatedEntries()
    {
        TensorPool<float>.Clear();
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableAlgebraicBackward = false });
        try
        {
            var engine = new CpuEngine();
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

            var deadInput = Fixed(new[] { 2, 8 }, 0.2f, 0.03f);
            var deadGamma = Fixed(new[] { 8 }, 1f, 0.01f);
            _ = engine.RMSNorm(deadInput, deadGamma, 1e-5, out var deadRms);

            var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
            var gamma = Fixed(new[] { 8 }, 1f, 0.05f);
            var output = engine.RMSNorm(input, gamma, 1e-5, out var liveRms);
            var loss = engine.ReduceSum(output, null);
            var compiled = tape.CompileBackward(loss, new[] { input, gamma });

            Assert.True(compiled.EliminatedEntryCount > 0);
            var grads = compiled.Execute();
            Assert.True(grads.ContainsKey(input));
            Assert.False(liveRms._pinnedByTape);
            AssertReleasedAndPoolable(deadRms, "compiled eliminated-entry cleanup");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(null);
        }
    }

    [Fact]
    public void AlgebraicOptimizedBackward_ReleasesAllSavedStatePins()
    {
        TensorPool<float>.Clear();
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableAlgebraicBackward = true });
        try
        {
            var engine = new CpuEngine();
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
            var input = Fixed(new[] { 2, 4 }, 0.3f, 0.11f);
            var w1 = Fixed(new[] { 4, 4 }, 0.2f, 0.03f);
            var w2 = Fixed(new[] { 4, 4 }, 0.4f, 0.02f);
            var gamma = Fixed(new[] { 4 }, 1f, 0.05f);
            var hidden = engine.TensorMatMul(input, w1);
            var projected = engine.TensorMatMul(hidden, w2);
            var output = engine.RMSNorm(projected, gamma, 1e-5, out var rms);
            var loss = engine.ReduceSum(output, null);
            var compiled = tape.CompileBackward(loss, new[] { input, w1, w2, gamma });

            var grads = compiled.Execute(); // Two reachable MatMuls force OptimizedBackwardPlan.
            Assert.True(grads.ContainsKey(input));
            AssertReleasedAndPoolable(rms, "algebraic optimized backward cleanup");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(null);
        }
    }

    [Fact]
    public void PersistentReplay_ReleasesEntryOnce_WithoutStealingAnotherTapePin()
    {
        TensorPool<float>.Clear();
        var engine = new CpuEngine();
        var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
        var gamma = Fixed(new[] { 8 }, 1f, 0.05f);
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var output = engine.RMSNorm(input, gamma, 1e-5, out var rms);
        var loss = engine.ReduceSum(output, null);

        var first = tape.ComputeGradients(loss, new[] { input });
        Assert.False(rms._pinnedByTape);

        using (var otherTape = new GradientTape<float>(new GradientTapeOptions { Persistent = false }))
        {
            otherTape.Record(new TapeEntry<float>
            {
                OperationName = "OtherTapeOwner",
                Output = Fixed(new[] { 2 }, 7f, 1f),
                Input0 = Fixed(new[] { 2 }, 9f, 1f),
                InputCount = 1,
                Backward = PassThroughBackward,
                SavedState = new object[] { rms }
            });
            Assert.True(rms._pinnedByTape);

            var second = tape.ComputeGradients(loss, new[] { input });
            Assert.Equal(Flat(first[input]), Flat(second[input]));
            Assert.True(rms._pinnedByTape,
                "persistent replay released the same entry twice and consumed another tape's pin");
        }

        Assert.False(rms._pinnedByTape);
    }

    [Fact]
    public void CachedReplay_ReleasesSavedStateAndKeepsGradientsStable()
    {
        TensorPool<float>.Clear();
        RebindablePlanCache<float>.ResetForTests();

        (float[] Gradient, Tensor<float> Saved) Run(bool expectCachedSignature)
        {
            var engine = new CpuEngine();
            var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
            var gamma = Fixed(new[] { 8 }, 1f, 0.05f);
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = false });
            var output = engine.RMSNorm(input, gamma, 1e-5, out var rms);
            var loss = engine.ReduceSum(output, null);

            if (expectCachedSignature)
            {
                long hash = AutoTrainingCompiler.ComputeStructureHash(tape.Entries, tape.EntryCount);
                Assert.True(RebindablePlanCache<float>.TrySignature(hash, tape.EntryCount));
            }

            var grads = tape.ComputeGradients(loss, new[] { input });
            Assert.False(rms._pinnedByTape);
            return (Flat(grads[input]), rms);
        }

        var first = Run(expectCachedSignature: false);
        Assert.False(RebindablePlanCache<float>.IsEmpty);
        var replay = Run(expectCachedSignature: true);

        Assert.Equal(first.Gradient, replay.Gradient);
        AssertReleasedAndPoolable(replay.Saved, "rebindable cached replay cleanup");
    }

    [Fact]
    public void StreamingBackward_ReleasesSavedStateForExecutedAndDeadEntries()
    {
        TensorPool<float>.Clear();
        bool previousReleaseSetting = GradientTape<float>.ReleaseStreamingActivations;
        GradientTape<float>.ReleaseStreamingActivations = true;
        try
        {
            var engine = new CpuEngine();
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = false });
            var deadInput = Fixed(new[] { 2, 8 }, 0.2f, 0.03f);
            var deadGamma = Fixed(new[] { 8 }, 1f, 0.01f);
            _ = engine.RMSNorm(deadInput, deadGamma, 1e-5, out var deadRms);

            var input = Fixed(new[] { 2, 8 }, 0.3f, 0.11f);
            var gamma = Fixed(new[] { 8 }, 1f, 0.05f);
            var output = engine.RMSNorm(input, gamma, 1e-5, out var liveRms);
            var loss = engine.ReduceSum(output, null);
            bool emitted = false;
            tape.ComputeGradientsStreaming(loss, new[] { input }, (source, gradient) =>
            {
                emitted = true;
                Assert.Same(input, source);
                Assert.Equal(input.Length, gradient.Length);
            });

            Assert.True(emitted);
            Assert.False(liveRms._pinnedByTape);
            AssertReleasedAndPoolable(deadRms, "streaming dead-entry cleanup");
        }
        finally
        {
            GradientTape<float>.ReleaseStreamingActivations = previousReleaseSetting;
        }
    }
}
