using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The CPU engine's softmax cross-entropy is one recorded node for dense float logits. Its loss and logits gradient
/// must be the composed objective's: -(sum of t * log_softmax(z)) over the supervised rows (rows with target mass),
/// with gradient (target_sum_row * softmax(z) - t) / count.
/// </summary>
public class CpuSoftmaxCrossEntropyFusedTests
{
    private const int Rows = 8, Cols = 10;

    // Soft and partly negative targets, an ignored (all-zero) row 0, and a row of large logits for stability.
    private static (Tensor<float> Logits, Tensor<float> Target) Batch()
    {
        var logits = new Tensor<float>(new[] { Rows, Cols });
        var target = new Tensor<float>(new[] { Rows, Cols });
        for (int i = 0; i < logits.Length; i++)
        {
            int r = i / Cols;
            logits[i] = (float)Math.Sin(i * 0.71) * (r == 3 ? 60f : 2f);
            target[i] = r == 0 ? 0f : (float)(Math.Cos(i * 1.3) * 0.5 + 0.3);
        }
        return (logits, target);
    }

    private static (double Loss, double[] Grad) ClosedForm(Tensor<float> logits, Tensor<float> target)
    {
        var grad = new double[Rows * Cols];
        double total = 0;
        int supervised = 0;
        var lse = new double[Rows];
        var targetSum = new double[Rows];
        for (int r = 0; r < Rows; r++)
        {
            double max = double.NegativeInfinity;
            for (int c = 0; c < Cols; c++) max = Math.Max(max, logits[r * Cols + c]);
            double sum = 0, mass = 0;
            for (int c = 0; c < Cols; c++) sum += Math.Exp(logits[r * Cols + c] - max);
            lse[r] = max + Math.Log(sum);
            for (int c = 0; c < Cols; c++)
            {
                double t = target[r * Cols + c];
                total += t * (logits[r * Cols + c] - lse[r]);
                targetSum[r] += t;
                mass += Math.Abs(t);
            }
            if (mass > 0) supervised++;
        }
        double count = Math.Max(supervised, 1);
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
            {
                int i = r * Cols + c;
                grad[i] = (targetSum[r] * Math.Exp(logits[i] - lse[r]) - target[i]) / count;
            }
        return (-total / count, grad);
    }

    [Fact]
    public void EagerLossAndLogitsGradient_MatchTheClosedForm()
    {
        var engine = new CpuEngine();
        var (logits, target) = Batch();
        var (expectedLoss, expectedGrad) = ClosedForm(logits, target);

        using var tape = new GradientTape<float>();
        var loss = engine.TensorSoftmaxCrossEntropy(logits, target);
        var grads = tape.ComputeGradients(loss, new[] { logits });

        Assert.True(Math.Abs(loss[0] - expectedLoss) <= 1e-5 * Math.Max(1, Math.Abs(expectedLoss)),
            $"loss {loss[0]:G9}, closed form {expectedLoss:G9}");
        Assert.True(grads.TryGetValue(logits, out var dz), "the logits got no gradient");
        for (int i = 0; i < expectedGrad.Length; i++)
            Assert.True(Math.Abs(dz[i] - expectedGrad[i]) <= 1e-5,
                $"dz[{i}] = {dz[i]:G9}, closed form {expectedGrad[i]:G9}");
    }

    /// <summary>
    /// One SGD step of a linear classifier through a compiled plan lands the weights where the eager tape's gradient
    /// puts them, and the plan holds the loss as a single node rather than the composed primitives.
    /// </summary>
    [Fact]
    public void CompiledStep_UpdatesTheWeightsLikeTheEagerGradient_AsOneLossNode()
    {
        const int inputs = 12;
        const float lr = 0.5f;
        var engine = new CpuEngine();
        var (_, target) = Batch();
        var x = new Tensor<float>(new[] { Rows, inputs });
        for (int i = 0; i < x.Length; i++) x[i] = (float)Math.Sin(i * 0.43);
        Tensor<float> Weights()
        {
            var w = new Tensor<float>(new[] { inputs, Cols });
            for (int i = 0; i < w.Length; i++) w[i] = (float)Math.Cos(i * 0.29) * 0.3f;
            return w;
        }

        var wEager = Weights();
        Tensor<float> eagerGrad;
        using (var tape = new GradientTape<float>())
        {
            var loss = engine.TensorSoftmaxCrossEntropy(engine.TensorMatMul(x, wEager), target);
            eagerGrad = tape.ComputeGradients(loss, new[] { wEager })[wEager];
        }

        var w = Weights();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorSoftmaxCrossEntropy(engine.TensorMatMul(x, w), target);
            plan = scope.CompileTraining(new[] { w });
        }
        using (plan)
        {
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            Assert.True(concrete.ForwardStepCount <= 2,
                $"the loss should be one node after the matmul; the plan has {concrete.ForwardStepCount} forward steps");
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: lr);
            plan.Step();
        }

        for (int i = 0; i < w.Length; i++)
        {
            float expected = wEager[i] - lr * eagerGrad[i];
            Assert.True(Math.Abs(w[i] - expected) <= 1e-5,
                $"w[{i}] = {w[i]:G9} after the compiled step, eager SGD gives {expected:G9}");
        }
    }
}
