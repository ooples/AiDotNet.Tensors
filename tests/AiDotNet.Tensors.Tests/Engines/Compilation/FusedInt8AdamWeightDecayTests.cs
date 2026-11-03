using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Int8 block-quantized moments with weight decay on the fused plan: AdamW (decoupled decay, bitsandbytes'
/// AdamW8bit) and Adam with L2 decay. The plan refused both ("supported only for Adam", "do not support weight
/// decay"), so a model trained with AdamW -- the decay every LM recipe uses -- could not halve its optimizer state
/// below bf16.
/// </summary>
public class FusedInt8AdamWeightDecayTests
{
    private const int Rows = 64, Cols = 64;     // 4096 elements: the default int8 minimum, so the moments quantize
    private const int BlockSize = 256;
    private const float Lr = 1e-2f, Beta1 = 0.9f, Beta2 = 0.99f, Eps = 1e-8f, WeightDecay = 0.1f;

    private static float[] Initial()
    {
        var rng = new Random(5);
        return Enumerable.Range(0, Rows * Cols).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
    }

    /// <summary>
    /// Trains <c>sum(w)</c> (gradient all ones) for three steps on the CPU fused plan and returns the weights.
    /// </summary>
    private static float[] TrainOnPlan(OptimizerType optimizer, float weightDecay)
    {
        var weight = new Tensor<float>(new[] { Rows, Cols }, new Vector<float>(Initial()));
        var engine = new CpuEngine();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(weight, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.RequestInt8MomentStorage(true, blockSize: BlockSize);
            plan.ConfigureOptimizer(optimizer, Lr, Beta1, Beta2, Eps, weightDecay);
            for (int i = 0; i < 3; i++) plan.Step();
        }
        return weight.AsSpan().ToArray();
    }

    /// <summary>
    /// The same three steps with the host reference. The decay is applied outside it, the way each optimizer
    /// defines it, so the reference is the decay-free int8 Adam the plan already matched: AdamW scales the weights
    /// by <c>1 - lr * wd</c> before the step (torch.optim.AdamW); Adam adds <c>wd * w</c> to the gradient.
    /// </summary>
    private static float[] Reference(bool decoupled, float weightDecay)
    {
        var w = Initial();
        int n = w.Length, blocks = (n + BlockSize - 1) / BlockSize;
        var mQ = new byte[n]; var vQ = new byte[n];
        var mS = new float[blocks]; var vS = new float[blocks];
        for (int step = 1; step <= 3; step++)
        {
            var g = new float[n];
            for (int i = 0; i < n; i++)
            {
                g[i] = decoupled ? 1f : 1f + weightDecay * w[i];
                if (decoupled) w[i] *= 1f - Lr * weightDecay;
            }
            CompressedMomentHostFallback.Adam8Bit(w, g, mQ, vQ, mS, vS, Lr, Beta1, Beta2, Eps, 1f - Beta1, 1f - Beta2,
                1f - MathF.Pow(Beta1, step), 1f - MathF.Pow(Beta2, step), BlockSize, n, blocks);
        }
        return w;
    }

    [Theory]
    [InlineData(OptimizerType.AdamW, true)]
    [InlineData(OptimizerType.Adam, false)]
    public void Int8Moments_WithWeightDecay_TrainLikeTheDecayFreeKernelWithTheDecayApplied(OptimizerType optimizer, bool decoupled)
    {
        var got = TrainOnPlan(optimizer, WeightDecay);
        var want = Reference(decoupled, WeightDecay);
        float worst = 0f;
        for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(want[i] - got[i]));
        Assert.True(worst < 1e-5f, $"{optimizer} int8 with weight decay: max |plan - reference| = {worst:E3}");

        // The decay really acted: the same run without it lands elsewhere.
        var noDecay = TrainOnPlan(optimizer, 0f);
        float moved = 0f;
        for (int i = 0; i < got.Length; i++) moved = Math.Max(moved, Math.Abs(noDecay[i] - got[i]));
        Assert.True(moved > 1e-4f, $"{optimizer}: weight decay {WeightDecay} changed the weights by only {moved:E3}");
    }

    [Fact]
    public void Int8Moments_StillRefuseOptimizersWithoutAnInt8Kernel()
    {
        var weight = Tensor<float>.CreateRandom([Rows, Cols]);
        var engine = new CpuEngine();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(weight, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.RequestInt8MomentStorage(true, blockSize: BlockSize);
            Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizer(OptimizerType.AMSGrad, Lr));
        }
    }
}
