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
}
