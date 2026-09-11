using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[CollectionDefinition("GradientTapeStreamingSerial", DisableParallelization = true)]
public class GradientTapeStreamingSerialCollection { }

/// <summary>
/// Tests for <see cref="GradientTape{T}.ComputeGradientsStreaming"/> — the
/// memory-bounded streaming backward that emits + releases each parameter
/// gradient at its topological last-use. The headline contract is that the
/// streamed gradients are bit-identical to the non-streaming
/// <see cref="GradientTape{T}.ComputeGradients"/> path, that each source is
/// emitted exactly once, and that a source used in multiple ops is only emitted
/// after its FINAL contribution (so partial gradients are never handed out).
/// </summary>
[Collection("GradientTapeStreamingSerial")]
public class GradientTapeStreamingTests
{
    private readonly CpuEngine _engine = new();

    /// <summary>
    /// Builds a small graph where `a` feeds TWO ops (multiply + add) so its
    /// gradient accumulates across two backward steps — the case the last-use
    /// release point must get right. loss = sum(a*b + (a+b)); da = b + 1, db = a + 1.
    /// </summary>
    [Fact]
    public void Streaming_MatchesComputeGradients_BitIdentical()
    {
        var aData = new double[] { 2, 3, 4 };
        var bData = new double[] { 5, 6, 7 };

        // Reference: standard ComputeGradients.
        Tensor<double> refGradA, refGradB;
        {
            var a = new Tensor<double>(new[] { 3 }, new Vector<double>((double[])aData.Clone()));
            var b = new Tensor<double>(new[] { 3 }, new Vector<double>((double[])bData.Clone()));
            using var tape = new GradientTape<double>();
            var c = _engine.TensorMultiply(a, b);
            var d = _engine.TensorAdd(a, b);
            var e = _engine.TensorAdd(c, d);
            var loss = _engine.ReduceSum(e, null);
            var grads = tape.ComputeGradients(loss, new[] { a, b });
            refGradA = grads[a];
            refGradB = grads[b];
        }

        // Streaming: same graph, gradients collected via the callback.
        var streamed = new Dictionary<Tensor<double>, Tensor<double>>(
            ReferenceEqualityComparer<Tensor<double>>.Instance);
        var callbackCount = new Dictionary<Tensor<double>, int>(
            ReferenceEqualityComparer<Tensor<double>>.Instance);
        Tensor<double> sa, sb;
        {
            sa = new Tensor<double>(new[] { 3 }, new Vector<double>((double[])aData.Clone()));
            sb = new Tensor<double>(new[] { 3 }, new Vector<double>((double[])bData.Clone()));
            using var tape = new GradientTape<double>();
            var c = _engine.TensorMultiply(sa, sb);
            var d = _engine.TensorAdd(sa, sb);
            var e = _engine.TensorAdd(c, d);
            var loss = _engine.ReduceSum(e, null);
            tape.ComputeGradientsStreaming(loss, new[] { sa, sb }, (src, grad) =>
            {
                // Copy out of the grad before it is released after the callback.
                var copy = new Tensor<double>(grad._shape, new Vector<double>(grad.ToArray()));
                streamed[src] = copy;
                callbackCount[src] = callbackCount.TryGetValue(src, out var n) ? n + 1 : 1;
            });
        }

        // Each source emitted exactly once (even though `a` feeds two ops).
        Assert.Equal(1, callbackCount[sa]);
        Assert.Equal(1, callbackCount[sb]);

        // PRIMARY CONTRACT: streamed gradients are bit-identical to the
        // non-streaming ComputeGradients result, element for element.
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(refGradA[i], streamed[sa][i]);
            Assert.Equal(refGradB[i], streamed[sb][i]);
        }
    }

    /// <summary>
    /// #1624: releasing each node's activation references as the streaming
    /// backward consumes them (the default policy) must NOT change the gradients —
    /// it only frees memory. Compare both typed policies on a deeper
    /// chain (several intermediates released mid-walk) and assert the gradients
    /// are bit-identical on vs off.
    /// </summary>
    [Fact]
    public void StreamingActivationRelease_OnVsOff_BitIdentical()
    {
        var aData = new double[] { 2, 3, 4, 5 };
        var bData = new double[] { 5, 6, 7, 8 };

        System.Collections.Generic.Dictionary<string, double[]> Run(
            StreamingGraphRetentionMode retention)
        {
            var a = new Tensor<double>(new[] { 4 }, new Vector<double>((double[])aData.Clone()));
            var b = new Tensor<double>(new[] { 4 }, new Vector<double>((double[])bData.Clone()));
            using var tape = new GradientTape<double>(new GradientTapeOptions
            {
                Persistent = true,
                StreamingGraphRetention = retention,
            });
            // Deeper chain so multiple intermediates are released mid-walk.
            var c = _engine.TensorMultiply(a, b);
            var d = _engine.TensorAdd(c, a);
            var e = _engine.TensorMultiply(d, b);
            var f = _engine.TensorAdd(e, c);
            var loss = _engine.ReduceSum(f, null);
            var outp = new System.Collections.Generic.Dictionary<string, double[]>();
            tape.ComputeGradientsStreaming(loss, new[] { a, b }, (src, grad) =>
            {
                outp[ReferenceEquals(src, a) ? "a" : "b"] = grad.ToArray();
            });
            return outp;
        }

        var released = Run(StreamingGraphRetentionMode.ReleaseAfterBackward);
        var retained = Run(StreamingGraphRetentionMode.RetainUntilTapeDisposal);

        Assert.Equal(retained["a"], released["a"]);
        Assert.Equal(retained["b"], released["b"]);
        Assert.Contains(released["a"], g => g != 0.0);
    }

    /// <summary>
    /// Sanity-checks the streaming gradient against the textbook value on a
    /// clean graph where each source is used exactly once:
    /// loss = sum(a*b) → da = b, db = a.
    /// </summary>
    [Fact]
    public void Streaming_SingleUseSources_MatchesTextbook()
    {
        var aData = new double[] { 2, 3, 4 };
        var bData = new double[] { 5, 6, 7 };
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>((double[])aData.Clone()));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>((double[])bData.Clone()));

        using var tape = new GradientTape<double>();
        var c = _engine.TensorMultiply(a, b);
        var loss = _engine.ReduceSum(c, null);

        var streamed = new Dictionary<Tensor<double>, double[]>(
            ReferenceEqualityComparer<Tensor<double>>.Instance);
        tape.ComputeGradientsStreaming(loss, new[] { a, b }, (src, grad) => streamed[src] = grad.ToArray());

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(bData[i], streamed[a][i], 10); // da = b
            Assert.Equal(aData[i], streamed[b][i], 10); // db = a
        }
    }

    /// <summary>
    /// A source that contributes no gradient (not on the loss path) must get no
    /// callback — matching ComputeGradients omitting it from the returned dict.
    /// </summary>
    [Fact]
    public void Streaming_UnusedSource_GetsNoCallback()
    {
        var a = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1, 2 }));
        var b = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 3, 4 }));
        var unused = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 9, 9 }));

        using var tape = new GradientTape<double>();
        var z = _engine.TensorAdd(a, b);
        var loss = _engine.ReduceSum(z, null);

        var emitted = new HashSet<Tensor<double>>(ReferenceEqualityComparer<Tensor<double>>.Instance);
        tape.ComputeGradientsStreaming(loss, new[] { a, b, unused }, (src, _) => emitted.Add(src));

        Assert.Contains(a, emitted);
        Assert.Contains(b, emitted);
        Assert.DoesNotContain(unused, emitted);
    }

    [Fact]
    public void Streaming_RejectsIntermediateSourcesBeforeMutatingTheTape()
    {
        var source = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 2.0, 3.0 }));
        using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });
        Tensor<double> intermediate = _engine.TensorMultiply(source, source);
        Tensor<double> loss = _engine.ReduceSum(intermediate, null);
        int entryCount = tape.EntryCount;

        Assert.Throws<ArgumentException>(() =>
            tape.ComputeGradientsStreaming(
                loss,
                new[] { intermediate, source },
                (_, _) => throw new InvalidOperationException("Callback must not run.")));

        Assert.Equal(entryCount, tape.EntryCount);
        Dictionary<Tensor<double>, Tensor<double>> gradients =
            tape.ComputeGradients(loss, new[] { source });
        Assert.Equal(new[] { 4.0, 6.0 }, gradients[source].ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Streaming_CallbackFailureClearsEverySourceGradientAndRetainedGraphReplays(
        int failingCallback)
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 4, 5, 6 }));
        using var tape = new GradientTape<double>(new GradientTapeOptions
        {
            Persistent = true,
            StreamingGraphRetention = StreamingGraphRetentionMode.RetainUntilTapeDisposal,
        });
        Tensor<double> loss = _engine.ReduceSum(_engine.TensorMultiply(a, b), null);
        var expected = new InvalidOperationException("injected streaming callback failure");
        int callbacks = 0;

        InvalidOperationException actual = Assert.Throws<InvalidOperationException>(() =>
            tape.ComputeGradientsStreaming(loss, new[] { a, b }, (_, _) =>
            {
                callbacks++;
                if (callbacks == failingCallback) throw expected;
            }));

        Assert.Same(expected, actual);
        Assert.Null(a.Grad);
        Assert.Null(b.Grad);

        var replayed = new HashSet<Tensor<double>>(ReferenceEqualityComparer<Tensor<double>>.Instance);
        tape.ComputeGradientsStreaming(loss, new[] { a, b }, (source, _) => replayed.Add(source));
        Assert.Contains(a, replayed);
        Assert.Contains(b, replayed);
        Assert.Null(a.Grad);
        Assert.Null(b.Grad);
    }

    [Fact]
    public void Streaming_CallbackFailureInReleaseMode_ClearsTheConsumedGraph()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 4.0, 5.0, 6.0 }));
        using var tape = new GradientTape<double>(new GradientTapeOptions
        {
            Persistent = true,
            StreamingGraphRetention = StreamingGraphRetentionMode.ReleaseAfterBackward,
        });
        Tensor<double> loss = _engine.ReduceSum(_engine.TensorMultiply(a, b), null);

        Assert.Throws<InvalidOperationException>(() =>
            tape.ComputeGradientsStreaming(
                loss,
                new[] { a, b },
                (_, _) => throw new InvalidOperationException("injected failure")));

        Assert.Equal(0, tape.EntryCount);
        Assert.Null(loss.GradFn?.Backward);
        Assert.Null(a.Grad);
        Assert.Null(b.Grad);
        Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(loss, new[] { a, b }));
    }

    [Fact]
    public void StreamingTyped_EmbeddingGradientStaysSparseAndPreservesRepeatedRows()
    {
        const int vocabulary = 1_000_000;
        const int embeddingDimension = 2;
        var embeddings = new Tensor<float>(new[] { vocabulary, embeddingDimension });
        var indices = new Tensor<int>(new[] { 4 });
        indices[0] = 7;
        indices[1] = 7;
        indices[2] = 42;
        indices[3] = 999_999;

        using var tape = new GradientTape<float>();
        Tensor<float> output = _engine.TensorEmbeddingLookup<float, int>(embeddings, indices);
        Tensor<float> loss = _engine.ReduceSum(output, null);
        StreamingSourceGradientKind? observedKind = null;
        int observedValueCount = 0;
        long[]? observedIndices = null;

        tape.ComputeGradientsStreamingTyped(loss, new[] { embeddings }, (_, gradient) =>
        {
            observedKind = gradient.Kind;
            Assert.False(gradient.HasDense);
            IReadOnlyList<SparseEmbeddingGradient<float>> sparse = gradient.SparseEmbedding;
            SparseEmbeddingGradient<float> contribution = Assert.Single(sparse);
            observedValueCount = contribution.Values.Length;
            observedIndices = contribution.Indices.ToArray();
        });

        Assert.Equal(StreamingSourceGradientKind.SparseEmbedding, observedKind);
        Assert.Equal(indices.Length * embeddingDimension, observedValueCount);
        Assert.Equal(new long[] { 7, 7, 42, 999_999 }, observedIndices);
        Assert.Null(embeddings.Grad);
    }

    [Fact]
    public void Float32AccumulationOverridesOnlyTheNestedAdditionAutocastScope()
    {
        using var backwardScope = new AutocastScope(PrecisionMode.Float16);
        Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);

        using (var accumulation = new GradientAccumulationPrecisionScope(
            GradientAccumulationPrecision.Float32))
        {
            using (IDisposable? additionScope =
                GradientAccumulationPrecisionScope.EnterFloat32AutocastForAddition())
            {
                Assert.NotNull(additionScope);
                Assert.Equal(PrecisionMode.Float32, AutocastScope.ActivePrecision);
            }

            Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);
        }

        Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);
    }

    /// <summary>
    /// Determinism: streaming twice over the same graph yields identical
    /// gradients (the accumulation order is the fixed reverse-topological order).
    /// </summary>
    [Fact]
    public void Streaming_IsDeterministic_AcrossRuns()
    {
        double[] Run()
        {
            var a = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 1.5, -2.0, 3.25, 0.5 }));
            var b = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 2.0, 4.0, -1.0, 6.0 }));
            using var tape = new GradientTape<double>();
            var c = _engine.TensorMultiply(a, b);
            var d = _engine.TensorMultiply(c, a); // a used twice
            var loss = _engine.ReduceSum(d, null);
            double[]? ga = null;
            tape.ComputeGradientsStreaming(loss, new[] { a }, (src, grad) =>
            {
                if (ReferenceEquals(src, a)) ga = grad.ToArray();
            });
            return ga!;
        }

        var r1 = Run();
        var r2 = Run();
        Assert.Equal(r1.Length, r2.Length);
        for (int i = 0; i < r1.Length; i++)
            Assert.Equal(r1[i], r2[i]);
    }

    /// <summary>
    /// Proves the streaming backward drops the persistent node graph's REFERENCES to a mid-chain
    /// activation (the reference-management half of the memory fix). An activation is the output of one
    /// node and the input of its consumers; a PERSISTENT tape keeps the whole node graph, so each
    /// activation stays pinned by a consumer node's Input field for the entire backward unless the
    /// release drops those refs. With the release ON the wrapper becomes collectable while the tape +
    /// sources are still strongly held; with it OFF the persistent node graph keeps it alive — which
    /// isolates the release as the cause. (This A/B FAILS the original implementation, which nulled
    /// outputs but not inputs.)
    ///
    /// SCOPE: this asserts the activation WRAPPER's collectability, not an end-to-end peak-memory number.
    /// Backing arrays may be pooled (returned to the arena rather than GC'd), so a weak-ref cannot by
    /// itself prove the OOM reduction — that is covered by the consumer-side streaming-training
    /// integration tests. Here we only prove the tape stops pinning activations it no longer needs.
    /// </summary>
    [Fact]
    public void StreamingActivationRelease_FreesIntermediateActivationChain()
    {
        GradientTape<double>? tapeOff = null;
        GradientTape<double>? tapeOn = null;
        try
        {
            // release OFF: the persistent node graph still pins the mid-chain activation (control).
            var off = BuildPersistentChainAndStream(release: false);
            tapeOff = off.tape;
            System.GC.Collect(); System.GC.WaitForPendingFinalizers(); System.GC.Collect();
            bool aliveOff = off.weak.IsAlive;
            System.GC.KeepAlive(off.tape); System.GC.KeepAlive(off.src);

            // release ON: the streaming backward drops the node graph's refs to it, so it is collectable
            // even though the persistent tape + sources are still alive.
            var on = BuildPersistentChainAndStream(release: true);
            tapeOn = on.tape;
            System.GC.Collect(); System.GC.WaitForPendingFinalizers(); System.GC.Collect();
            bool aliveOn = on.weak.IsAlive;
            System.GC.KeepAlive(on.tape); System.GC.KeepAlive(on.src);

            Assert.True(aliveOff, "control: with release OFF the persistent node graph must keep the activation pinned");
            Assert.False(aliveOn, "with release ON the streaming backward must drop the node graph's refs to the mid-chain activation");
        }
        finally
        {
            // The persistent tapes were intentionally kept alive across the GC checks above; dispose
            // them now so their graph/arena resources don't leak into later tests.
            tapeOn?.Dispose();
            tapeOff?.Dispose();
        }
    }

    /// <summary>
    /// #636 safety guard (faithful persistent-tape regression): a PERSISTENT tape whose activations were
    /// destructively released by a streaming backward must NOT be silently reusable — recording onto it
    /// or differentiating it again would walk a defaulted graph, so both fail fast with
    /// <see cref="System.InvalidOperationException"/> rather than corrupting the graph.
    /// </summary>
    [Fact]
    public void PersistentTape_AfterStreamingRelease_RejectsReuse()
    {
        var a = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 1, 2, 3, 4 }));
        var b = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 5, 6, 7, 8 }));
        using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });
        var c = _engine.TensorMultiply(a, b);
        var loss = _engine.ReduceSum(c, null);
        tape.ComputeGradientsStreaming(loss, new[] { a, b }, (_, __) => { });

        // The streaming release destroyed the persistent graph: any reuse must throw, not corrupt.
        Assert.Throws<System.InvalidOperationException>(
            () => tape.ComputeGradients(loss, new[] { a, b }));
        Assert.Throws<System.InvalidOperationException>(
            () => tape.Record(default));
    }

    [Fact]
    public void RetainedPersistentTape_ReplaysStreamingBackwardBitIdentically()
    {
        var a = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 1, 2, 3, 4 }));
        var b = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 5, 6, 7, 8 }));
        using var tape = new GradientTape<double>(new GradientTapeOptions
        {
            Persistent = true,
            StreamingGraphRetention = StreamingGraphRetentionMode.RetainUntilTapeDisposal,
        });
        var loss = _engine.ReduceSum(_engine.TensorMultiply(a, b), null);

        var first = new Dictionary<Tensor<double>, double[]>(ReferenceEqualityComparer<Tensor<double>>.Instance);
        var second = new Dictionary<Tensor<double>, double[]>(ReferenceEqualityComparer<Tensor<double>>.Instance);
        tape.ComputeGradientsStreaming(loss, new[] { a, b }, (source, gradient) => first[source] = gradient.ToArray());
        tape.ComputeGradientsStreaming(loss, new[] { a, b }, (source, gradient) => second[source] = gradient.ToArray());

        Assert.Equal(first[a], second[a]);
        Assert.Equal(first[b], second[b]);
    }

    [Fact]
    public void RetainedPersistentTape_KeepsStatefulOperationSavedTensorPinnedAcrossReplay()
    {
        TensorPool<float>.Clear();
        var engine = new CpuEngine();
        var input = new Tensor<float>(
            new float[] { 0.25f, -0.5f, 0.75f, 1.5f, -1.25f, 0.1f, 0.2f, 0.3f },
            new[] { 2, 4 });
        var gamma = new Tensor<float>(new float[] { 1.0f, 0.9f, 1.1f, 0.8f }, new[] { 4 });
        var tape = new GradientTape<float>(new GradientTapeOptions
        {
            Persistent = true,
            StreamingGraphRetention = StreamingGraphRetentionMode.RetainUntilTapeDisposal,
        });
        Tensor<float>? savedRms = null;
        try
        {
            var output = engine.RMSNorm(input, gamma, 1e-5, out var rms);
            savedRms = rms;
            var loss = engine.ReduceSum(output, null);
            float[]? first = null;
            float[]? second = null;

            tape.ComputeGradientsStreaming(loss, new[] { input }, (_, gradient) => first = gradient.ToArray());
            Assert.True(rms._pinnedByTape);

            // A retained pin must keep the saved RMS out of the reusable tensor pool. If streaming
            // cleanup drops only the pin while leaving the graph, this rent can overwrite the state
            // that RMSNormBackward needs on the second pass.
            TensorPool<float>.Return(rms);
            Tensor<float> churn = TensorPool<float>.Rent((int[])rms._shape.Clone());
            Assert.NotSame(rms, churn);
            for (int i = 0; i < churn.Length; i++) churn[i] = 12345.0f + i;

            tape.ComputeGradientsStreaming(loss, new[] { input }, (_, gradient) => second = gradient.ToArray());
            Assert.Equal(first, second);
            Assert.True(rms._pinnedByTape);
            TensorPool<float>.Return(churn);
        }
        finally
        {
            tape.Dispose();
        }

        Tensor<float> releasedRms = Assert.IsType<Tensor<float>>(savedRms);
        Assert.False(releasedRms._pinnedByTape);
        TensorPool<float>.Return(releasedRms);
        Tensor<float> reused = TensorPool<float>.Rent((int[])releasedRms._shape.Clone());
        Assert.Same(releasedRms, reused);
        TensorPool<float>.Return(reused);
    }

    [Fact]
    public void OppositeRetentionPolicies_RunConcurrentlyWithoutInterference()
    {
        using var ready = new CountdownEvent(2);
        using var start = new ManualResetEventSlim(false);

        Task retained = Task.Run(() => RunConcurrentTape(
            StreamingGraphRetentionMode.RetainUntilTapeDisposal,
            expectReplay: true,
            ready,
            start));
        Task released = Task.Run(() => RunConcurrentTape(
            StreamingGraphRetentionMode.ReleaseAfterBackward,
            expectReplay: false,
            ready,
            start));

        ready.Wait();
        start.Set();
        Task.WaitAll(retained, released);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private (System.WeakReference weak, GradientTape<double> tape, Tensor<double> src) BuildPersistentChainAndStream(bool release)
    {
        var aData = new double[32]; var bData = new double[32];
        for (int i = 0; i < 32; i++) { aData[i] = 1.3; bData[i] = 0.7; }
        var a = new Tensor<double>(new[] { 32 }, new Vector<double>(aData));
        var b = new Tensor<double>(new[] { 32 }, new Vector<double>(bData));
        var tape = new GradientTape<double>(new GradientTapeOptions
        {
            Persistent = true,
            StreamingGraphRetention = release
                ? StreamingGraphRetentionMode.ReleaseAfterBackward
                : StreamingGraphRetentionMode.RetainUntilTapeDisposal,
        });
        var t1 = _engine.TensorMultiply(a, b);
        var t2 = _engine.TensorMultiply(t1, b);   // mid-chain activation we weak-ref
        var t3 = _engine.TensorMultiply(t2, b);
        var t4 = _engine.TensorMultiply(t3, b);
        var loss = _engine.ReduceSum(t4, null);
        var weak = new System.WeakReference(t2);  // the tensor object (its backing array may be pooled)
        tape.ComputeGradientsStreaming(loss, new[] { a, b }, (_, __) => { });
        // t1..t4 and loss are locals — not rooted after return; only the persistent tape's node graph
        // could keep t2 alive (via a consumer node's Input field), which the release drops.
        return (weak, tape, a);
    }

    private static void RunConcurrentTape(
        StreamingGraphRetentionMode retention,
        bool expectReplay,
        CountdownEvent ready,
        ManualResetEventSlim start)
    {
        var engine = new CpuEngine();
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 2, 3, 4 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 5, 6, 7 }));
        using var tape = new GradientTape<double>(new GradientTapeOptions
        {
            Persistent = true,
            StreamingGraphRetention = retention,
        });
        var loss = engine.ReduceSum(engine.TensorMultiply(a, b), null);
        ready.Signal();
        start.Wait();
        tape.ComputeGradientsStreaming(loss, new[] { a, b }, (_, __) => { });

        if (expectReplay)
        {
            tape.ComputeGradientsStreaming(loss, new[] { a, b }, (_, __) => { });
        }
        else
        {
            Assert.Throws<System.InvalidOperationException>(
                () => tape.ComputeGradientsStreaming(loss, new[] { a, b }, (_, __) => { }));
        }
    }
}
