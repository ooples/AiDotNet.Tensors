using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Gate for FP16-NATIVE ops (#558, docs/fp16-activation-storage-design.md §"SEVENTH FINDING"): the
/// FP32-between-matmul ops (GELU/norm/softmax/residual) must read a Half activation, run their FP32 math on
/// an up-cast-in-realize copy, and SAVE the Half (not FP32) for backward — so the activation chain stays
/// Half end-to-end and the resident-memory win actually materializes (the matmul-only emission did not,
/// because each up-cast FP32 output was re-saved by the consuming FP32 op).
/// <para>This proves the keystone on CPU (verifiable on any host): (1) the GELU activation node emitted via
/// <see cref="MixedPrecisionEmit.Unary"/> is genuinely <see cref="Tensor{Half}"/>; (2) the next FP16 op
/// REUSES that Half output (the chain never widens); (3) the auto-emitted mixed-dtype graph backprops to
/// gradients matching the all-FP32 reference graph within FP16 tolerance.</para>
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public class Fp16NativeOpTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private static Tensor<float> F(int[] shape, params float[] v) => new(v, shape);

    private static void Acc(Dictionary<Tensor<float>, Tensor<float>> g, Tensor<float> k, Tensor<float> v, IEngine e)
        => g[k] = g.TryGetValue(k, out var ex) ? e.TensorAdd(ex, v) : v;

    /// <summary>Records loss = sum(t) as an FP32 node (broadcasts the scalar grad back over t).</summary>
    private static Tensor<float> SumLoss(LazyTensorScope scope, Tensor<float> t) =>
        scope.RecordUnary<float>(LazyNodeType.Sum, "sum", t, new[] { 1 },
            (e, o) => { float s = 0; foreach (var v in t.ToArray()) s += v; o.AsWritableSpan()[0] = s; },
            (gradOut, inputs, output, state, e, grads) =>
            {
                var yt = inputs[0];
                float go = gradOut.ToArray()[0];
                var data = new float[yt.Length];
                for (int i = 0; i < data.Length; i++) data[i] = go;
                Acc(grads, yt, new Tensor<float>(data, yt._shape), e);
            });

    [Fact]
    public void Unary_Gelu_StoresFp16Activation_ChainStaysHalf_BackpropMatchesFp32Reference()
    {
        // loss = sum( GELU(x @ W) ), x[2,3] @ W[3,2] -> [2,2]; FP16-exact small inputs.
        var x = F(new[] { 2, 3 }, 1f, 2f, 3f, 0.5f, -1f, -2f);
        var W = F(new[] { 3, 2 }, 0.5f, -1f, 1f, 2f, -0.5f, 1.5f);

        // --- FP16-native graph: matmul (Half) -> GELU (Half) -> sum ---
        var scope = new LazyTensorScope(null);
        Tensor<float> y, g, loss;
        using (new AutocastScope(PrecisionMode.Float16))
        {
            var prev = MixedPrecisionEmit.TestOverrideEnabled;
            MixedPrecisionEmit.TestOverrideEnabled = true; // force FP16 activation emission for this trace
            try
            {
                y = MixedPrecisionEmit.MatMul(scope, x, W, new[] { 2, 2 });
                g = MixedPrecisionEmit.Unary(scope, y, new[] { 2, 2 }, "GELU", LazyNodeType.GELU,
                    (e, xf) => e.GELU(xf),
                    (e, gOut, xin, _o, _s) => e.GeluBackward(gOut, xin));
            }
            finally { MixedPrecisionEmit.TestOverrideEnabled = prev; }
            loss = SumLoss(scope, g);
        }

        // (1) GELU activation is genuinely Half.
        Assert.True(g.LazySource is CrossTypeLazyNode<Half, float>, "GELU output should be an FP16->FP32 up-cast");
        var geluUp = (CrossTypeLazyNode<Half, float>)g.LazySource!;
        Tensor<Half> geluAct = geluUp.Input;
        Assert.Equal(new[] { 2, 2 }, geluAct.Shape.ToArray());

        // (2) The chain never widened: GELU's Half node consumes the matmul's Half output DIRECTLY (no
        //     re-down-cast of the matmul's FP32 up-cast). The matmul up-cast that fed GELU is left dead.
        var geluHalfNode = geluAct.LazySource as LazyNode<Half>;
        Assert.NotNull(geluHalfNode);
        var geluInput = geluHalfNode!.GetInputNodes();
        Assert.Contains(geluInput, n => n is LazyNode<Half> hn && hn.OpType == LazyNodeType.MatMul);

        _ = loss.ToArray();
        var got = MixedPrecisionGraphBackward.Backward(loss, _engine);
        Assert.True(got.Fp32.TryGetValue(x, out var gxFp16), "no FP32 grad for x (FP16-native path)");
        Assert.True(got.Fp32.TryGetValue(W, out var gWFp16), "no FP32 grad for W (FP16-native path)");

        // --- FP32 reference graph: matmul -> GELU -> sum, all FP32 ---
        var refScope = new LazyTensorScope(null);
        var yRef = refScope.RecordBinary<float>(LazyNodeType.MatMul, "mm", x, W, new[] { 2, 2 },
            (e, o) => e.TensorMatMul(x, W).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.MatMulBackward);
        var gRef = refScope.RecordUnary<float>(LazyNodeType.GELU, "gelu", yRef, new[] { 2, 2 },
            (e, o) => e.GELU(yRef).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.GELUBackward);
        var lossRef = SumLoss(refScope, gRef);
        _ = lossRef.ToArray();
        var refGrads = MixedPrecisionGraphBackward.Backward(lossRef, _engine);
        Assert.True(refGrads.Fp32.TryGetValue(x, out var gxRef), "no FP32 grad for x (reference)");
        Assert.True(refGrads.Fp32.TryGetValue(W, out var gWRef), "no FP32 grad for W (reference)");

        // (3) FP16-native gradients match the FP32 reference within FP16 tolerance.
        AssertClose(gxRef.ToArray(), gxFp16.ToArray(), "dL/dx");
        AssertClose(gWRef.ToArray(), gWFp16.ToArray(), "dL/dW");
    }

    [Fact]
    public void Binary_ResidualAdd_And_Relu_StayHalf_BackpropMatchesFp32Reference()
    {
        // loss = sum( ReLU(x @ W) + r ) — exercises FP16-native ReLU (Unary) + residual Add (Binary).
        var x = F(new[] { 2, 3 }, 1f, -2f, 3f, 0.5f, 1f, -1.5f);
        var W = F(new[] { 3, 2 }, 0.5f, 1f, -1f, 2f, 1.5f, -0.5f);
        var r = F(new[] { 2, 2 }, 0.25f, -0.5f, 1f, 0.75f);

        var scope = new LazyTensorScope(null);
        Tensor<float> relu, z, loss;
        using (new AutocastScope(PrecisionMode.Float16))
        {
            var prev = MixedPrecisionEmit.TestOverrideEnabled;
            MixedPrecisionEmit.TestOverrideEnabled = true;
            try
            {
                var y = MixedPrecisionEmit.MatMul(scope, x, W, new[] { 2, 2 });
                relu = MixedPrecisionEmit.Unary(scope, y, new[] { 2, 2 }, "ReLU", LazyNodeType.ReLU,
                    (e, xf) => e.ReLU(xf),
                    (e, gOut, xin, _o, _s) => e.ReluBackward(gOut, xin));
                z = MixedPrecisionEmit.Binary(scope, relu, r, new[] { 2, 2 }, "Add", LazyNodeType.Add,
                    (e, af, bf) => e.TensorAdd(af, bf),
                    (e, gOut, af, bf) => (gOut, gOut));
            }
            finally { MixedPrecisionEmit.TestOverrideEnabled = prev; }
            loss = SumLoss(scope, z);
        }

        // Both ReLU and the residual sum are genuinely Half activations.
        Assert.True(relu.LazySource is CrossTypeLazyNode<Half, float>, "ReLU output should be FP16->FP32 up-cast");
        Assert.True(z.LazySource is CrossTypeLazyNode<Half, float>, "residual sum should be FP16->FP32 up-cast");
        var addAct = ((CrossTypeLazyNode<Half, float>)z.LazySource!).Input;
        Assert.IsType<LazyNode<Half>>(addAct.LazySource);

        _ = loss.ToArray();
        var got = MixedPrecisionGraphBackward.Backward(loss, _engine);

        var refScope = new LazyTensorScope(null);
        var yRef = refScope.RecordBinary<float>(LazyNodeType.MatMul, "mm", x, W, new[] { 2, 2 },
            (e, o) => e.TensorMatMul(x, W).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.MatMulBackward);
        var reluRef = refScope.RecordUnary<float>(LazyNodeType.ReLU, "relu", yRef, new[] { 2, 2 },
            (e, o) => e.ReLU(yRef).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.ReLUBackward);
        var zRef = refScope.RecordBinary<float>(LazyNodeType.Add, "add", reluRef, r, new[] { 2, 2 },
            (e, o) => e.TensorAdd(reluRef, r).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.AddBackward);
        var lossRef = SumLoss(refScope, zRef);
        _ = lossRef.ToArray();
        var refGrads = MixedPrecisionGraphBackward.Backward(lossRef, _engine);

        foreach (var (t, name) in new[] { (x, "dL/dx"), (W, "dL/dW"), (r, "dL/dr") })
        {
            Assert.True(got.Fp32.TryGetValue(t, out var g16), $"no FP16-native grad for {name}");
            Assert.True(refGrads.Fp32.TryGetValue(t, out var gRef), $"no reference grad for {name}");
            AssertClose(gRef.ToArray(), g16.ToArray(), name);
        }
    }

    [Fact]
    public void Unary_Softmax_StaysHalf_BackpropMatchesFp32Reference()
    {
        // loss = sum( Softmax(x @ W, axis=-1) ) over [2,2]; softmax backward needs the OUTPUT.
        var x = F(new[] { 2, 3 }, 1f, 2f, -1f, 0.5f, -0.5f, 1.5f);
        var W = F(new[] { 3, 2 }, 0.5f, 1f, -1f, 0.5f, 2f, -0.5f);

        var scope = new LazyTensorScope(null);
        Tensor<float> sm, loss;
        using (new AutocastScope(PrecisionMode.Float16))
        {
            var prev = MixedPrecisionEmit.TestOverrideEnabled;
            MixedPrecisionEmit.TestOverrideEnabled = true;
            try
            {
                var y = MixedPrecisionEmit.MatMul(scope, x, W, new[] { 2, 2 });
                sm = MixedPrecisionEmit.Unary(scope, y, new[] { 2, 2 }, "Softmax", LazyNodeType.Softmax,
                    (e, xf) => e.Softmax(xf, -1),
                    (e, gOut, _in, outF, st) => e.SoftmaxBackward(gOut, outF, (int)st[0]),
                    new object[] { 1 });
            }
            finally { MixedPrecisionEmit.TestOverrideEnabled = prev; }
            loss = SumLoss(scope, sm);
        }
        Assert.True(sm.LazySource is CrossTypeLazyNode<Half, float>, "softmax output should be Half up-cast");

        _ = loss.ToArray();
        var got = MixedPrecisionGraphBackward.Backward(loss, _engine);

        var refScope = new LazyTensorScope(null);
        var yRef = refScope.RecordBinary<float>(LazyNodeType.MatMul, "mm", x, W, new[] { 2, 2 },
            (e, o) => e.TensorMatMul(x, W).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.MatMulBackward);
        var smRef = refScope.RecordUnary<float>(LazyNodeType.Softmax, "sm", yRef, new[] { 2, 2 },
            (e, o) => e.Softmax(yRef, -1).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.SoftmaxBackward, new object[] { 1 });
        var lossRef = SumLoss(refScope, smRef);
        _ = lossRef.ToArray();
        var refGrads = MixedPrecisionGraphBackward.Backward(lossRef, _engine);

        foreach (var (t, name) in new[] { (x, "dL/dx"), (W, "dL/dW") })
        {
            Assert.True(got.Fp32.TryGetValue(t, out var g16), $"no FP16-native grad for {name}");
            Assert.True(refGrads.Fp32.TryGetValue(t, out var gRef), $"no reference grad for {name}");
            AssertClose(gRef.ToArray(), g16.ToArray(), name);
        }
    }

    [Fact]
    public void LayerNorm_StaysHalf_ParamGradsBridgeToFp32_BackpropMatchesReference()
    {
        // loss = sum( LayerNorm(x @ W, gamma, beta) ) — exercises FP16-native affine LayerNorm with
        // gamma/beta param grads bridged back to FP32 via the down-cast nodes.
        var x = F(new[] { 2, 3 }, 1f, 2f, -1f, 0.5f, -0.5f, 1.5f);
        var W = F(new[] { 3, 4 }, 0.5f, 1f, -1f, 0.5f, 2f, -0.5f, 1f, -1f, 0.25f, 0.5f, -0.25f, 1.5f);
        var gamma = F(new[] { 4 }, 1f, 1.5f, 0.5f, 1f);
        var beta = F(new[] { 4 }, 0f, 0.25f, -0.25f, 0.5f);
        const double eps = 1e-5;

        var scope = new LazyTensorScope(null);
        Tensor<float> ln, loss;
        using (new AutocastScope(PrecisionMode.Float16))
        {
            var prev = MixedPrecisionEmit.TestOverrideEnabled;
            MixedPrecisionEmit.TestOverrideEnabled = true;
            try
            {
                var y = MixedPrecisionEmit.MatMul(scope, x, W, new[] { 2, 4 });
                ln = MixedPrecisionEmit.LayerNorm(scope, y, gamma, beta, eps, new[] { 2, 4 });
            }
            finally { MixedPrecisionEmit.TestOverrideEnabled = prev; }
            loss = SumLoss(scope, ln);
        }
        Assert.True(ln.LazySource is CrossTypeLazyNode<Half, float>, "LayerNorm output should be Half up-cast");

        _ = loss.ToArray();
        var got = MixedPrecisionGraphBackward.Backward(loss, _engine);

        var refScope = new LazyTensorScope(null);
        var yRef = refScope.RecordBinary<float>(LazyNodeType.MatMul, "mm", x, W, new[] { 2, 4 },
            (e, o) => e.TensorMatMul(x, W).AsSpan().CopyTo(o.AsWritableSpan()),
            BackwardFunctions<float>.MatMulBackward);
        var lnRef = refScope.RecordVariadic<float>(LazyNodeType.Custom, "ln", new[] { yRef, gamma, beta }, new[] { 2, 4 },
            (e, o) => e.LayerNorm(yRef, gamma, beta, eps, out _, out _).AsSpan().CopyTo(o.AsWritableSpan()),
            (gradOut, inputs, output, state, e, grads) =>
            {
                _ = e.LayerNorm(inputs[0], inputs[1], inputs[2], eps, out var m, out var v);
                var gi = e.LayerNormBackward(gradOut, inputs[0], inputs[1], m, v, eps, out var gg, out var gb);
                Acc(grads, inputs[0], gi, e);
                Acc(grads, inputs[1], gg, e);
                Acc(grads, inputs[2], gb, e);
            });
        var lossRef = SumLoss(refScope, lnRef);
        _ = lossRef.ToArray();
        var refGrads = MixedPrecisionGraphBackward.Backward(lossRef, _engine);

        foreach (var (t, name) in new[] { (x, "dL/dx"), (W, "dL/dW"), (gamma, "dL/dgamma"), (beta, "dL/dbeta") })
        {
            Assert.True(got.Fp32.TryGetValue(t, out var g16), $"no FP16-native grad for {name}");
            Assert.True(refGrads.Fp32.TryGetValue(t, out var gRef), $"no reference grad for {name}");
            AssertClose(gRef.ToArray(), g16.ToArray(), name);
        }
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(IsFinite(actual[i]), $"{label}[{i}] not finite: {actual[i]}");
            double tol = 2e-2 + 5e-2 * Math.Abs(expected[i]); // FP16 activation rounding
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"{label}[{i}] reference {expected[i]} vs FP16-native {actual[i]} (tol {tol})");
        }
    }

    private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
}
