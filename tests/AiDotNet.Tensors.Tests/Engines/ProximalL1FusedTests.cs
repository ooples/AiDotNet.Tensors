using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Covers the fused ProximalL1 (ISTA) kernel and its plan dispatch.
/// </summary>
/// <remarks>
/// <para>
/// The soft-threshold is the entire method, and it is the part that fails silently if implemented as an L1
/// penalty folded into the gradient instead. A subgradient step only SHRINKS coordinates toward zero and
/// leaves them oscillating around it at a scale set by the learning rate; the prox sets them to EXACTLY
/// zero. Both make the loss go down and both look like they are regularizing, so the tests below assert
/// exact zeros and exact soft-threshold values rather than "the weights got smaller".
/// </para>
/// </remarks>
public class ProximalL1FusedTests
{
    private const float Tolerance = 1e-6f;

    /// <summary>
    /// Deliberately not a multiple of 8: the kernel computes <c>simdLen = length &amp; ~7</c> and handles the
    /// remainder in a separate scalar loop, so a round length would leave that branch unexecuted on AVX2
    /// hardware and the two implementations of the same math would not both be covered.
    /// </summary>
    private const int VectorLength = 67;

    private static unsafe float[] RunKernel(float[] param, float[] grad, float lr, float l1)
    {
        var p = (float[])param.Clone();
        var g = (float[])grad.Clone();
        fixed (float* pp = p, pg = g)
            FusedOptimizer.ProximalL1UpdateSimd(pp, pg, p.Length, lr, l1);
        return p;
    }

    /// <summary>Independent scalar transcription of the ISTA step.</summary>
    private static float[] Reference(float[] param, float[] grad, float lr, float l1)
    {
        var outp = new float[param.Length];
        float threshold = lr * l1;
        for (int i = 0; i < param.Length; i++)
        {
            float z = param[i] - lr * grad[i];
            float mag = Math.Abs(z) - threshold;
            outp[i] = mag <= 0f ? 0f : (z > 0f ? mag : -mag);
        }
        return outp;
    }

    [Fact]
    public unsafe void MatchesTheReferenceSoftThreshold_IncludingTheScalarTail()
    {
        var rng = new Random(4242);
        var param = new float[VectorLength];
        var grad = new float[VectorLength];
        for (int i = 0; i < VectorLength; i++)
        {
            param[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            grad[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        const float lr = 0.1f, l1 = 0.3f;
        var actual = RunKernel(param, grad, lr, l1);
        var expected = Reference(param, grad, lr, l1);

        for (int i = 0; i < VectorLength; i++)
            Assert.Equal(expected[i], actual[i], 5);
    }

    /// <summary>
    /// The defining property: a coordinate whose post-gradient magnitude is below the threshold must become
    /// EXACTLY zero, not merely small.
    /// </summary>
    /// <remarks>
    /// This is what an L1-subgradient step cannot produce, and it is why the method is chosen at all. Asserting
    /// exact equality to 0f is deliberate — a near-zero value would satisfy any "weights shrank" check while
    /// meaning the prox is absent.
    /// </remarks>
    [Fact]
    public unsafe void DrivesSubThresholdCoordinatesToExactlyZero()
    {
        // |param - lr*grad| = 0.02 for every element; threshold = lr*l1 = 0.05 > 0.02.
        var param = new float[VectorLength];
        var grad = new float[VectorLength];
        for (int i = 0; i < VectorLength; i++)
        {
            param[i] = (i % 2 == 0) ? 0.02f : -0.02f;
            grad[i] = 0f;
        }

        var result = RunKernel(param, grad, lr: 0.1f, l1: 0.5f);

        for (int i = 0; i < VectorLength; i++)
            Assert.True(result[i] == 0f, $"element {i} is {result[i]}, not exactly zero — the proximal operator is not being applied.");
    }

    /// <summary>
    /// Above the threshold the coordinate survives, shrunk by exactly the threshold and keeping its sign.
    /// </summary>
    [Fact]
    public unsafe void ShrinksSuprathresholdCoordinatesByExactlyTheThreshold_PreservingSign()
    {
        var param = new[] { 1.0f, -1.0f, 0.5f, -0.5f };
        var grad = new float[param.Length];
        const float lr = 0.1f, l1 = 2.0f;   // threshold = 0.2

        var result = RunKernel(param, grad, lr, l1);

        Assert.Equal(0.8f, result[0], 5);
        Assert.Equal(-0.8f, result[1], 5);
        Assert.Equal(0.3f, result[2], 5);
        Assert.Equal(-0.3f, result[3], 5);
    }

    /// <summary>
    /// With zero L1 strength the step must reduce exactly to plain SGD, so the prox cannot be perturbing
    /// anything when it is switched off.
    /// </summary>
    [Fact]
    public unsafe void WithZeroL1_ReducesToPlainSgd()
    {
        var rng = new Random(7);
        var param = new float[VectorLength];
        var grad = new float[VectorLength];
        for (int i = 0; i < VectorLength; i++)
        {
            param[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            grad[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        const float lr = 0.05f;
        var result = RunKernel(param, grad, lr, l1: 0f);

        for (int i = 0; i < VectorLength; i++)
            Assert.Equal(param[i] - lr * grad[i], result[i], 5);
    }

    /// <summary>
    /// End-to-end through the compiled plan: it must dispatch, stay finite, and actually zero coordinates.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_ProximalL1_DispatchesAndProducesSparsity()
    {
        var engine = new CpuEngine();
        const int n = 32;
        var weight = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) weight[i] = 0.01f * ((i % 5) - 2);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.ProximalL1, LrSchedule.Constant(0.05),
                beta1: 0f, beta2: 0f, eps: 0f, weightDecay: 0f,
                extras: new FusedOptimizerExtras { L1 = 1.0f });
            for (int s = 0; s < 5; s++) plan.Step();
        }

        var post = weight.GetDataArray();
        int zeros = 0;
        for (int i = 0; i < n; i++)
        {
            Assert.True(!float.IsNaN(post[i]) && !float.IsInfinity(post[i]),
                $"ProximalL1 produced a non-finite parameter at {i}.");
            if (post[i] == 0f) zeros++;
        }

        Assert.True(zeros > 0,
            "no coordinate reached exactly zero after 5 proximal steps — the prox is not reaching the kernel.");
    }

    /// <summary>
    /// The grouped plan has a separate optimizer closure and must preserve the same proximal
    /// sparsity contract as the ungrouped path.
    /// </summary>
    [Fact]
    public void ConfigureOptimizerGrouped_ProximalL1_DispatchesAndProducesSparsity()
    {
        var engine = new CpuEngine();
        const int n = 32;
        var weight = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) weight[i] = 0.01f * ((i % 5) - 2);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            plan.ConfigureOptimizerGrouped(
                OptimizerType.ProximalL1,
                new[] { LrSchedule.Constant(0.05) },
                new[] { 0 },
                beta1: 0f, beta2: 0f, eps: 0f, weightDecay: 0f,
                extras: new FusedOptimizerExtras { L1 = 1.0f });
            for (int s = 0; s < 5; s++) plan.Step();
        }

        var post = weight.GetDataArray();
        int zeros = 0;
        for (int i = 0; i < n; i++)
        {
            Assert.True(!float.IsNaN(post[i]) && !float.IsInfinity(post[i]),
                $"Grouped ProximalL1 produced a non-finite parameter at {i}.");
            if (post[i] == 0f) zeros++;
        }

        Assert.True(zeros > 0,
            "no coordinate reached exactly zero through the grouped ProximalL1 dispatch.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ConfigureOptimizer_ProximalL1_RejectsUnsupportedWeightDecay(bool grouped)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new[] { 8 });

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            NotSupportedException error = grouped
                ? Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizerGrouped(
                    OptimizerType.ProximalL1,
                    new[] { LrSchedule.Constant(0.05) },
                    new[] { 0 },
                    weightDecay: 0.01f,
                    extras: new FusedOptimizerExtras { L1 = 1.0f }))
                : Assert.Throws<NotSupportedException>(() => plan.ConfigureOptimizer(
                    OptimizerType.ProximalL1,
                    LrSchedule.Constant(0.05),
                    weightDecay: 0.01f,
                    extras: new FusedOptimizerExtras { L1 = 1.0f }));

            Assert.Contains("does not support weightDecay", error.Message);
        }
    }
}
