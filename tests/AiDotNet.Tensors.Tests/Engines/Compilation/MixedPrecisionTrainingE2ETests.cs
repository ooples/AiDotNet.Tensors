using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase-2c end-to-end gate (Tensors #555, docs/fp16-activation-storage-design.md): a full mixed-
/// precision TRAINING loop driven through the production engine seam. With AIDOTNET_FP16_ACTIVATIONS on
/// and an FP16 autocast scope, <see cref="CpuEngine.TensorMatMul{T}"/> recorded under GraphMode auto-
/// emits the down-cast → FP16 matmul → up-cast triple (the matmul activation is a Tensor&lt;Half&gt;),
/// and <see cref="MixedPrecisionGraphBackward"/> supplies the FP32 parameter gradients for an SGD step.
/// The test learns a linear map y = x·W toward a known target and asserts the loss actually descends —
/// proving the activation is half-precision AND the gradients are good enough to train, not just to pass
/// an isolated gradient check.
/// </summary>
public class MixedPrecisionTrainingE2ETests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int rows, int cols, int seed, double scale)
    {
        var rng = new Random(seed);
        var d = new float[rows * cols];
        for (int i = 0; i < d.Length; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(d, new[] { rows, cols });
    }

    [Fact]
    public void MixedPrecision_Training_DescendsLoss_WithHalfActivations()
    {
        const int B = 8, D = 6;
        var x = Rand(B, D, 1, 1.0);
        var wTrue = Rand(D, D, 2, 0.5);
        var t = _engine.TensorMatMul(x, wTrue); // target, computed eagerly in FP32

        // Learnable weight, started away from the solution.
        var W = Rand(D, D, 3, 0.1);

        // Hand-built squared-error loss node: L = sum((y - t)^2), dL/dy = 2(y - t).
        BackwardFunction<float> lossBwd = (gradOut, inputs, output, state, eng, grads) =>
        {
            var y = inputs[0];
            var ya = y.ToArray(); var ta = t.ToArray();
            float go = gradOut.ToArray()[0];
            var g = new float[y.Length];
            for (int i = 0; i < g.Length; i++) g[i] = 2f * (ya[i] - ta[i]) * go;
            grads[y] = grads.TryGetValue(y, out var ex) ? eng.TensorAdd(ex, new Tensor<float>(g, y._shape)) : new Tensor<float>(g, y._shape);
        };

        float LossValue(Tensor<float> y)
        {
            var ya = y.ToArray(); var ta = t.ToArray();
            float s = 0; for (int i = 0; i < ya.Length; i++) { float d = ya[i] - ta[i]; s += d * d; }
            return s;
        }

        float firstLoss = 0, lastLoss = 0;
        const float lr = 0.03f;
        const int steps = 60;

        var prevOverride = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true; // AIDOTNET_FP16_ACTIVATIONS on
        try
        {
            for (int step = 0; step < steps; step++)
            {
                var scope = new LazyTensorScope(null);
                Tensor<float> y, loss;
                using (new AutocastScope(PrecisionMode.Float16))
                {
                    GraphMode.SetCurrent(scope);
                    try
                    {
                        // Engine auto-emits the FP16-activation matmul here.
                        y = _engine.TensorMatMul(x, W);
                        loss = scope.RecordUnary<float>(LazyNodeType.Sum, "sqerr", y, new[] { 1 },
                            (e, o) =>
                            {
                                var ya = y.ToArray(); var ta = t.ToArray();
                                float s = 0; for (int i = 0; i < ya.Length; i++) { float d = ya[i] - ta[i]; s += d * d; }
                                o.AsWritableSpan()[0] = s;
                            },
                            lossBwd);
                    }
                    finally { GraphMode.SetCurrent(null); }
                }

                _ = loss.ToArray(); // realize forward

                // First step: confirm the matmul activation is genuinely FP16.
                if (step == 0)
                {
                    Assert.True(y.LazySource is CrossTypeLazyNode<Half, float>, "matmul activation should be FP16 (up-cast producer)");
                    firstLoss = LossValue(y);
                }

                var grads = MixedPrecisionGraphBackward.Backward(loss, _engine);
                Assert.True(grads.Fp32.TryGetValue(W, out var gW), $"no FP32 gradient for W at step {step}");

                // SGD step on the FP32 master weight.
                var wSpan = W.AsWritableSpan();
                var gwa = gW.ToArray();
                for (int i = 0; i < wSpan.Length; i++) wSpan[i] -= lr * gwa[i];
                W.IncrementVersion();

                if (step == steps - 1)
                {
                    var yFinal = _engine.TensorMatMul(x, W); // eager FP32 eval of the learned map
                    lastLoss = LossValue(yFinal);
                }
            }
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prevOverride; GraphMode.SetCurrent(null); }

        Assert.True(firstLoss > 0, "first loss should be positive");
        Assert.False(float.IsNaN(lastLoss) || float.IsInfinity(lastLoss), "final loss must be finite");
        // The mixed-precision loop must make real progress (FP16 noise notwithstanding).
        Assert.True(lastLoss < 0.5f * firstLoss, $"loss did not descend: first {firstLoss}, last {lastLoss}");
    }
}
