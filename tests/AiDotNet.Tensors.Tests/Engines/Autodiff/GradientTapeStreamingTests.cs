using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for <see cref="GradientTape{T}.ComputeGradientsStreaming"/> — the
/// memory-bounded streaming backward that emits + releases each parameter
/// gradient at its topological last-use. The headline contract is that the
/// streamed gradients are bit-identical to the non-streaming
/// <see cref="GradientTape{T}.ComputeGradients"/> path, that each source is
/// emitted exactly once, and that a source used in multiple ops is only emitted
/// after its FINAL contribution (so partial gradients are never handed out).
/// </summary>
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
    /// backward consumes them (ReleaseStreamingActivations, on by default) must
    /// NOT change the gradients — it only frees memory. A/B the flag on a deeper
    /// chain (several intermediates released mid-walk) and assert the gradients
    /// are bit-identical on vs off.
    /// </summary>
    [Fact]
    public void StreamingActivationRelease_OnVsOff_BitIdentical()
    {
        var saved = GradientTape<double>.ReleaseStreamingActivations;
        try
        {
            var aData = new double[] { 2, 3, 4, 5 };
            var bData = new double[] { 5, 6, 7, 8 };

            System.Collections.Generic.Dictionary<string, double[]> Run(bool release)
            {
                GradientTape<double>.ReleaseStreamingActivations = release;
                var a = new Tensor<double>(new[] { 4 }, new Vector<double>((double[])aData.Clone()));
                var b = new Tensor<double>(new[] { 4 }, new Vector<double>((double[])bData.Clone()));
                using var tape = new GradientTape<double>();
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

            var on = Run(release: true);
            var off = Run(release: false);

            Assert.Equal(off["a"], on["a"]);
            Assert.Equal(off["b"], on["b"]);
            Assert.Contains(on["a"], g => g != 0.0);
        }
        finally
        {
            GradientTape<double>.ReleaseStreamingActivations = saved;
        }
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
    /// Proves the streaming backward actually FREES the intermediate-activation chain (the memory
    /// benefit, not just gradient correctness). An activation is the output of one node and the input of
    /// its consumers; a PERSISTENT tape keeps the whole node graph, so each activation stays pinned by a
    /// consumer node's Input field for the entire backward unless the release drops those refs. With the
    /// release ON a mid-chain activation is collectable while the tape + sources are still strongly held;
    /// with it OFF the persistent node graph keeps it alive — which is exactly what proves the release is
    /// the cause. (This A/B would FAIL the original implementation, which nulled outputs but not inputs.)
    /// </summary>
    [Fact]
    public void StreamingActivationRelease_FreesIntermediateActivationChain()
    {
        var saved = GradientTape<double>.ReleaseStreamingActivations;
        try
        {
            // release OFF: the persistent node graph still pins the mid-chain activation (control).
            var (weakOff, tapeOff, srcOff) = BuildPersistentChainAndStream(release: false);
            System.GC.Collect(); System.GC.WaitForPendingFinalizers(); System.GC.Collect();
            bool aliveOff = weakOff.IsAlive;
            System.GC.KeepAlive(tapeOff); System.GC.KeepAlive(srcOff);

            // release ON: the streaming backward drops the node graph's refs to it, so it is collectable
            // even though the persistent tape + sources are still alive.
            var (weakOn, tapeOn, srcOn) = BuildPersistentChainAndStream(release: true);
            System.GC.Collect(); System.GC.WaitForPendingFinalizers(); System.GC.Collect();
            bool aliveOn = weakOn.IsAlive;
            System.GC.KeepAlive(tapeOn); System.GC.KeepAlive(srcOn);

            Assert.True(aliveOff, "control: with release OFF the persistent node graph must keep the activation pinned");
            Assert.False(aliveOn, "with release ON the streaming backward must free the mid-chain activation chain");
        }
        finally { GradientTape<double>.ReleaseStreamingActivations = saved; }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private (System.WeakReference weak, GradientTape<double> tape, Tensor<double> src) BuildPersistentChainAndStream(bool release)
    {
        GradientTape<double>.ReleaseStreamingActivations = release;
        var aData = new double[32]; var bData = new double[32];
        for (int i = 0; i < 32; i++) { aData[i] = 1.3; bData[i] = 0.7; }
        var a = new Tensor<double>(new[] { 32 }, new Vector<double>(aData));
        var b = new Tensor<double>(new[] { 32 }, new Vector<double>(bData));
        var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });
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
}
