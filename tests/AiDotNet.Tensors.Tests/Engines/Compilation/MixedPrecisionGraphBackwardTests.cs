using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase-2 gate (Tensors #555, docs/fp16-activation-storage-design.md): a mixed-dtype reverse pass over
/// a UNIFIED lazy graph in a single topological sweep (no Gauss-Seidel — the graph already carries the
/// cross-type edges). Builds the same interleaved chain as the eager-tape gate, but as a lazy graph of
/// LazyNode&lt;float&gt;, LazyNode&lt;Half&gt;, and CrossTypeLazyNode bridges, realizes it, then runs
/// <see cref="MixedPrecisionGraphBackward"/>. FP16-exact values ⇒ the analytic gradient is the exact
/// target, so a defect in the bridge or the walk ordering shows as a real mismatch.
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public class MixedPrecisionGraphBackwardTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private static Tensor<float> Vf(params float[] v) => new(v, new[] { v.Length });
    private static Tensor<Half> Vh(params float[] v)
    {
        var h = new Half[v.Length];
        for (int i = 0; i < v.Length; i++) h[i] = (Half)v[i];
        return new Tensor<Half>(h, new[] { v.Length });
    }

    private static void Acc<T>(Dictionary<Tensor<T>, Tensor<T>> g, Tensor<T> k, Tensor<T> v, IEngine e)
        => g[k] = g.TryGetValue(k, out var ex) ? e.TensorAdd(ex, v) : v;

    // Elementwise-multiply backward: dL/da = grad*b, dL/db = grad*a.
    private BackwardFunction<T> MulBwd<T>() => (gradOut, inputs, output, state, eng, grads) =>
    {
        Acc(grads, inputs[0], eng.TensorMultiply(gradOut, inputs[1]), eng);
        Acc(grads, inputs[1], eng.TensorMultiply(gradOut, inputs[0]), eng);
    };

    // Sum-to-scalar backward: dL/dy_i = gradOut[0] (broadcast).
    private BackwardFunction<float> SumBwd() => (gradOut, inputs, output, state, eng, grads) =>
    {
        var y = inputs[0];
        var go = gradOut.ToArray()[0];
        var data = new float[y.Length];
        for (int i = 0; i < data.Length; i++) data[i] = go;
        Acc(grads, y, new Tensor<float>(data, y._shape), eng);
    };

    [Fact]
    public void Interleaved_MixedDtype_LazyGraph_Backward_MatchesAnalytic()
    {
        // loss = sum_i( x_i * W1_i * c1_i * W2_i )
        var x  = Vf(1f, 2f, 0.5f, -1f);
        var c1 = Vf(1f, 0.5f, 2f, -2f);
        var W1 = Vh(2f, -1f, 0.5f, 1f);
        var W2 = Vh(0.5f, 2f, -1f, 1f);
        int n = 4;

        // d1 = cast16(x)
        var d1 = new Tensor<Half>(new[] { n });
        var nd1 = new CrossTypeLazyNode<float, Half>(LazyNodeType.Custom, "cast16", x, d1,
            (e, o) => MixedPrecisionCast.CastToFp16(x).AsSpan().CopyTo(o.AsWritableSpan()),
            (g, inp, o, s, e) => MixedPrecisionCast.CastToFp16Backward(g));
        d1.LazySource = nd1;

        // h1 = d1 * W1   (FP16)
        var h1 = new Tensor<Half>(new[] { n });
        var nh1 = new LazyNode<Half>(LazyNodeType.Multiply, "mul16", d1, W1, h1,
            (e, o) => e.TensorMultiply(d1, W1).AsSpan().CopyTo(o.AsWritableSpan()), MulBwd<Half>());
        h1.LazySource = nh1;

        // y1 = cast32(h1)
        var y1 = new Tensor<float>(new[] { n });
        var ny1 = new CrossTypeLazyNode<Half, float>(LazyNodeType.Custom, "cast32", h1, y1,
            (e, o) => MixedPrecisionCast.CastToFp32(h1).AsSpan().CopyTo(o.AsWritableSpan()),
            (g, inp, o, s, e) => MixedPrecisionCast.CastToFp32Backward(g));
        y1.LazySource = ny1;

        // a1 = y1 * c1   (FP32)
        var a1 = new Tensor<float>(new[] { n });
        var na1 = new LazyNode<float>(LazyNodeType.Multiply, "mul32", y1, c1, a1,
            (e, o) => e.TensorMultiply(y1, c1).AsSpan().CopyTo(o.AsWritableSpan()), MulBwd<float>());
        a1.LazySource = na1;

        // d2 = cast16(a1)
        var d2 = new Tensor<Half>(new[] { n });
        var nd2 = new CrossTypeLazyNode<float, Half>(LazyNodeType.Custom, "cast16", a1, d2,
            (e, o) => MixedPrecisionCast.CastToFp16(a1).AsSpan().CopyTo(o.AsWritableSpan()),
            (g, inp, o, s, e) => MixedPrecisionCast.CastToFp16Backward(g));
        d2.LazySource = nd2;

        // h2 = d2 * W2   (FP16)
        var h2 = new Tensor<Half>(new[] { n });
        var nh2 = new LazyNode<Half>(LazyNodeType.Multiply, "mul16", d2, W2, h2,
            (e, o) => e.TensorMultiply(d2, W2).AsSpan().CopyTo(o.AsWritableSpan()), MulBwd<Half>());
        h2.LazySource = nh2;

        // y2 = cast32(h2)
        var y2 = new Tensor<float>(new[] { n });
        var ny2 = new CrossTypeLazyNode<Half, float>(LazyNodeType.Custom, "cast32", h2, y2,
            (e, o) => MixedPrecisionCast.CastToFp32(h2).AsSpan().CopyTo(o.AsWritableSpan()),
            (g, inp, o, s, e) => MixedPrecisionCast.CastToFp32Backward(g));
        y2.LazySource = ny2;

        // loss = sum(y2)
        var loss = new Tensor<float>(new[] { 1 });
        var nloss = new LazyNode<float>(LazyNodeType.Sum, "sum", y2, loss,
            (e, o) => { float s = 0; foreach (var v in y2.ToArray()) s += v; o.AsWritableSpan()[0] = s; }, SumBwd());
        loss.LazySource = nloss;

        nloss.Realize(_engine); // realize the whole graph forward

        var grads = MixedPrecisionGraphBackward.Backward(loss, _engine);

        // Analytic targets (FP16-exact ⇒ exact).
        var xa = x.ToArray(); var c1a = c1.ToArray();
        var w1a = W1.ToArray(); var w2a = W2.ToArray();

        Assert.True(grads.Fp32.TryGetValue(x, out var gx), "no FP32 grad for x");
        Assert.True(grads.Fp16.TryGetValue(W1, out var gW1), "no FP16 grad for W1 (first segment)");
        Assert.True(grads.Fp16.TryGetValue(W2, out var gW2), "no FP16 grad for W2");

        var gxa = gx.ToArray(); var gw1 = gW1.ToArray(); var gw2 = gW2.ToArray();
        for (int i = 0; i < n; i++)
        {
            float w1 = (float)w1a[i], w2 = (float)w2a[i];
            Assert.True(Math.Abs(gxa[i] - w1 * c1a[i] * w2) <= 1e-4f + 1e-3f * Math.Abs(w1 * c1a[i] * w2), $"dL/dx[{i}]");
            Assert.True(Math.Abs((float)gw1[i] - xa[i] * c1a[i] * w2) <= 1e-3f + 2e-3f * Math.Abs(xa[i] * c1a[i] * w2), $"dL/dW1[{i}]");
            Assert.True(Math.Abs((float)gw2[i] - xa[i] * w1 * c1a[i]) <= 1e-3f + 2e-3f * Math.Abs(xa[i] * w1 * c1a[i]), $"dL/dW2[{i}]");
        }
    }
}
