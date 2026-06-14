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
/// Phase-2c gate (Tensors #555, docs/fp16-activation-storage-design.md): the forward emission that
/// actually stores an activation as FP16. <see cref="MixedPrecisionEmit.MatMul"/> records, under an FP16
/// autocast scope, a down-cast → FP16 matmul → up-cast triple, so the matmul OUTPUT buffer is
/// <see cref="Tensor{Half}"/> (2 bytes — the resident-memory win the compute-only autocast can't give).
/// This test proves BOTH halves of the claim on CPU: (1) the emitted activation node is genuinely Half,
/// and (2) <see cref="MixedPrecisionGraphBackward"/> backprops the auto-emitted mixed-dtype graph to the
/// correct FP32 parameter gradients (analytic, FP16-exact values).
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public class MixedPrecisionEmitTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private static Tensor<float> F(int[] shape, params float[] v) => new(v, shape);

    private static void Acc(Dictionary<Tensor<float>, Tensor<float>> g, Tensor<float> k, Tensor<float> v, IEngine e)
        => g[k] = g.TryGetValue(k, out var ex) ? e.TensorAdd(ex, v) : v;

    [Fact]
    public void Emit_StoresFp16Activation_And_BackpropMatchesAnalytic()
    {
        // loss = sum(x @ W), x[2,3] @ W[3,2] -> [2,2]; all values FP16-exact.
        var x = F(new[] { 2, 3 }, 1f, 2f, 3f, 4f, 0.5f, -1f);
        var W = F(new[] { 3, 2 }, 2f, -1f, 0.5f, 1f, -2f, 4f);

        var scope = new LazyTensorScope(null);
        Tensor<float> y, loss;
        using (new AutocastScope(PrecisionMode.Float16))
        {
            Assert.True(MixedPrecisionEmit.ShouldEmitFp16<float>(), "autocast should request FP16 emission");
            y = MixedPrecisionEmit.MatMul(scope, x, W, new[] { 2, 2 });

            // loss = sum(y) recorded as an FP32 node; backward broadcasts the scalar grad.
            loss = scope.RecordUnary<float>(LazyNodeType.Sum, "sum", y, new[] { 1 },
                (e, o) => { float s = 0; foreach (var v in y.ToArray()) s += v; o.AsWritableSpan()[0] = s; },
                (gradOut, inputs, output, state, e, grads) =>
                {
                    var yt = inputs[0];
                    float go = gradOut.ToArray()[0];
                    var data = new float[yt.Length];
                    for (int i = 0; i < data.Length; i++) data[i] = go;
                    Acc(grads, yt, new Tensor<float>(data, yt._shape), e);
                });
        }

        // (1) The matmul activation is FP16: y's producer is the up-cast, whose INPUT is the Half matmul output.
        Assert.True(y.LazySource is CrossTypeLazyNode<Half, float>, "y should be produced by an FP16->FP32 up-cast");
        var upCast = (CrossTypeLazyNode<Half, float>)y.LazySource!;
        Tensor<Half> activation = upCast.Input;
        Assert.Equal(typeof(Half), activation.GetType().GenericTypeArguments.Length > 0
            ? activation.GetType().GenericTypeArguments[0] : typeof(Half)); // it's Tensor<Half> by static type
        Assert.Equal(new[] { 2, 2 }, activation.Shape.ToArray());

        // Realize the forward, then backprop the mixed-dtype graph.
        _ = loss.ToArray();
        var grads = MixedPrecisionGraphBackward.Backward(loss, _engine);

        // Analytic: dL/dx[i,k] = sum_j W[k,j]; dL/dW[k,j] = sum_i x[i,k].
        var xa = x.ToArray(); var wa = W.ToArray();
        var expGx = new float[6];
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < 3; k++)
                expGx[i * 3 + k] = wa[k * 2 + 0] + wa[k * 2 + 1];
        var expGw = new float[6];
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 2; j++)
                expGw[k * 2 + j] = xa[0 * 3 + k] + xa[1 * 3 + k];

        Assert.True(grads.Fp32.TryGetValue(x, out var gx), "no FP32 grad for x");
        Assert.True(grads.Fp32.TryGetValue(W, out var gW), "no FP32 grad for W");
        var gxa = gx.ToArray(); var gwa = gW.ToArray();
        for (int i = 0; i < 6; i++)
        {
            Assert.True(Math.Abs(gxa[i] - expGx[i]) <= 1e-3f + 2e-3f * Math.Abs(expGx[i]), $"dL/dx[{i}] exp {expGx[i]} got {gxa[i]}");
            Assert.True(Math.Abs(gwa[i] - expGw[i]) <= 1e-3f + 2e-3f * Math.Abs(expGw[i]), $"dL/dW[{i}] exp {expGw[i]} got {gwa[i]}");
        }
    }
}
