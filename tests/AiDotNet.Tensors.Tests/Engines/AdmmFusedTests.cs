using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Covers the fused linearized-ADMM kernel against Boyd et al. (2011).
/// </summary>
/// <remarks>
/// <para>
/// ADMM's three sub-steps are each elementwise, which is what lets it fuse, but two of them are easy to
/// implement in a way that still converges while not being ADMM: dropping the rho coupling from the
/// x-step leaves plain gradient descent with two unused buffers, and applying the prox without the 1/rho
/// scaling changes the effective regularization strength. Both look fine from a loss curve. The tests
/// below compare against an independent transcription across multiple steps, where z and u have
/// accumulated enough for those errors to separate.
/// </para>
/// </remarks>
public class AdmmFusedTests
{
    /// <summary>
    /// Deliberately not a multiple of 8, so the AVX2 body and the scalar tail both run.
    /// </summary>
    private const int VectorLength = 67;

    private static unsafe void RunKernel(
        float[] param, float[] grad, float[] z, float[] u, float lr, float rho, float l1)
    {
        fixed (float* pp = param, pg = grad, pz = z, pu = u)
            FusedOptimizer.AdmmUpdateSimd(pp, pg, pz, pu, param.Length, lr, rho, l1);
    }

    /// <summary>Independent scalar transcription of the linearized ADMM step.</summary>
    private static void Reference(
        float[] param, float[] grad, float[] z, float[] u, float lr, float rho, float l1)
    {
        for (int i = 0; i < param.Length; i++)
        {
            float coupling = (param[i] - z[i]) + u[i];
            param[i] -= lr * (grad[i] + rho * coupling);

            float t = param[i] + u[i];
            float magnitude = Math.Abs(t) - l1;
            z[i] = magnitude <= 0f ? 0f : (t > 0f ? magnitude : -magnitude);

            u[i] += param[i] - z[i];
        }
    }

    private static (float[] Param, float[] Grad) MakeInputs(int seed)
    {
        var rng = new Random(seed);
        var param = new float[VectorLength];
        var grad = new float[VectorLength];
        for (int i = 0; i < VectorLength; i++)
        {
            param[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            grad[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
        return (param, grad);
    }

    [Fact]
    public async Task ConfigureOptimizer_AllocatesBothAdmmStateBuffers_AndStepRuns()
    {
        await Task.Yield();

        var engine = new CpuEngine();
        var parameter = new Tensor<float>(new[] { 1f, -2f, 3f, -4f }, new[] { 4 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(parameter, null);
            plan = scope.CompileTraining(new[] { parameter });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.ADMM,
                learningRate: 0.05f,
                extras: new FusedOptimizerExtras { AdmmRho = 2f, L1 = 0.1f });

            var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            var checkpoint = Assert.IsType<FusedOptimizerCheckpoint>(compiled.CaptureFusedOptimizerCheckpoint());
            Assert.NotNull(checkpoint.Parameters[0].MFloat); // z
            Assert.NotNull(checkpoint.Parameters[0].VFloat); // u

            var loss = plan.Step();
            Assert.True(float.IsFinite(loss[0]));
            Assert.All(parameter.ToArray(), value => Assert.True(float.IsFinite(value)));
        }
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public async Task ConfigureOptimizer_RejectsInvalidRho(float rho)
    {
        await Task.Yield();

        var engine = new CpuEngine();
        var parameter = new Tensor<float>(new[] { 1f }, new[] { 1 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(parameter, null);
            plan = scope.CompileTraining(new[] { parameter });
        }

        using (plan)
        {
            var error = Assert.Throws<ArgumentOutOfRangeException>(() =>
                plan.ConfigureOptimizer(
                    OptimizerType.ADMM,
                    learningRate: 0.05f,
                    extras: new FusedOptimizerExtras { AdmmRho = rho }));
            Assert.Equal("AdmmRho", error.ParamName);
        }
    }

    [Fact]
    public void MatchesTheReferenceAcrossSteps_IncludingTheScalarTail()
    {
        const float lr = 0.05f, rho = 1.5f, l1 = 0.1f;

        var (kernelParam, _) = MakeInputs(seed: 31);
        var referenceParam = (float[])kernelParam.Clone();
        var kernelZ = new float[VectorLength];
        var kernelU = new float[VectorLength];
        var referenceZ = new float[VectorLength];
        var referenceU = new float[VectorLength];

        for (int step = 0; step < 5; step++)
        {
            var (_, grad) = MakeInputs(seed: 200 + step);

            RunKernel(kernelParam, (float[])grad.Clone(), kernelZ, kernelU, lr, rho, l1);
            Reference(referenceParam, grad, referenceZ, referenceU, lr, rho, l1);
        }

        for (int i = 0; i < VectorLength; i++)
        {
            Assert.Equal(referenceParam[i], kernelParam[i], 4);
            Assert.Equal(referenceZ[i], kernelZ[i], 4);
            Assert.Equal(referenceU[i], kernelU[i], 4);
        }
    }

    /// <summary>
    /// The rho coupling has to reach the parameter update, or ADMM is gradient descent with two spectator
    /// buffers.
    /// </summary>
    /// <remarks>
    /// With a zero gradient the entire step is the coupling term, so a kernel that ignored it would leave
    /// the parameters untouched. Both z and u are seeded non-zero, since a first step from zeroed state
    /// cannot tell the two apart.
    /// </remarks>
    [Fact]
    public void RhoCouplingReachesTheParameterUpdate()
    {
        const float lr = 0.1f, rho = 2.0f, l1 = 0f;
        // No value may sit at 0.5, where the coupling (x - z) + u is exactly zero and the step would
        // legitimately not move — which would make the "it moved" assertion below untestable.
        var param = new float[] { 1.0f, -1.0f, 0.6f, 0.25f, -0.75f, 0.125f, 2.0f, -2.0f, 0.0f };
        var grad = new float[param.Length];
        var z = new float[param.Length];
        var u = new float[param.Length];
        for (int i = 0; i < param.Length; i++) { z[i] = 0.3f; u[i] = -0.2f; }

        var expected = (float[])param.Clone();
        for (int i = 0; i < param.Length; i++)
            expected[i] -= lr * rho * ((param[i] - 0.3f) + -0.2f);

        var before = (float[])param.Clone();
        RunKernel(param, grad, z, u, lr, rho, l1);

        for (int i = 0; i < param.Length; i++)
        {
            Assert.Equal(expected[i], param[i], 5);

            // Without the coupling term this step is param -= lr*0, i.e. nothing at all.
            Assert.NotEqual(before[i], param[i], 5);
        }
    }

    /// <summary>
    /// A zero L1 strength makes the prox the identity, which is the correct split for an unregularized
    /// objective.
    /// </summary>
    /// <remarks>
    /// This is what lets one parameter cover both the regularized and unregularized cases without a mode
    /// flag, so it is worth pinning: <c>z</c> must come out as exactly <c>x + u</c>. Note that rho does NOT
    /// scale the argument — Boyd's z-step is prox_{g/rho}(x + u), so rho belongs in the threshold, which
    /// the caller supplies as Strength/rho.
    /// </remarks>
    [Fact]
    public void ZeroL1_LeavesTheSplitVariableUnthresholded()
    {
        const float lr = 0.1f, rho = 2.0f;
        var (param, grad) = MakeInputs(seed: 77);
        var z = new float[VectorLength];
        var u = new float[VectorLength];

        RunKernel(param, (float[])grad.Clone(), z, u, lr, rho, l1: 0f);

        for (int i = 0; i < VectorLength; i++)
            Assert.Equal(param[i] + 0f, z[i], 5);
    }

    /// <summary>
    /// Rho scales the prox STRENGTH, which the caller supplies, and never the prox argument.
    /// </summary>
    /// <remarks>
    /// Boyd's z-step is <c>prox_{g/rho}(x + u)</c>. Thresholding <c>(x + u)/rho</c> at the raw strength
    /// instead is a different function — equal to <c>(1/rho)·soft_threshold(x + u, Strength·rho)</c> — and
    /// the two agree only at rho = 1, which is the default and therefore the value least likely to catch
    /// the mistake. This runs at rho = 4 with the argument fixed, so the kernel's z must not depend on rho
    /// at all.
    /// </remarks>
    [Fact]
    public void RhoDoesNotScaleTheProxArgument()
    {
        const float lr = 0f, l1 = 0.2f;
        var param = new float[] { 1.0f, -1.0f, 0.1f, 0.5f, -0.5f, 0.3f, -0.3f, 0.05f, 0.9f };
        var grad = new float[param.Length];

        var zLowRho = new float[param.Length];
        var uLowRho = new float[param.Length];
        RunKernel((float[])param.Clone(), grad, zLowRho, uLowRho, lr, rho: 1f, l1: l1);

        var zHighRho = new float[param.Length];
        var uHighRho = new float[param.Length];
        RunKernel((float[])param.Clone(), grad, zHighRho, uHighRho, lr, rho: 4f, l1: l1);

        // lr = 0 means x does not move, so with u starting at zero the z-step sees the same argument in
        // both runs and must produce the same z.
        for (int i = 0; i < param.Length; i++)
            Assert.Equal(zLowRho[i], zHighRho[i], 6);

        // And it is the plain soft-threshold of x, not of x/rho.
        for (int i = 0; i < param.Length; i++)
        {
            float mag = Math.Abs(param[i]) - l1;
            float expected = mag <= 0f ? 0f : (param[i] > 0f ? mag : -mag);
            Assert.Equal(expected, zHighRho[i], 6);
        }
    }

    /// <summary>
    /// A non-zero L1 strength drives small split coordinates to exactly zero.
    /// </summary>
    [Fact]
    public void NonZeroL1_ThresholdsTheSplitVariableToExactlyZero()
    {
        const float lr = 0.0f, rho = 1.0f, l1 = 0.5f;
        var param = new float[] { 0.1f, -0.1f, 0.05f, 0.9f, -0.9f, 0.2f, -0.2f, 0.3f, 0.4f };
        var grad = new float[param.Length];
        var z = new float[param.Length];
        var u = new float[param.Length];

        RunKernel(param, grad, z, u, lr, rho, l1);

        // |x| <= 0.5 thresholds to zero; the two at 0.9 keep 0.4 of magnitude.
        Assert.Equal(0f, z[0], 6);
        Assert.Equal(0f, z[1], 6);
        Assert.Equal(0f, z[2], 6);
        Assert.Equal(0.4f, z[3], 5);
        Assert.Equal(-0.4f, z[4], 5);
    }

    /// <summary>
    /// The dual variable accumulates the primal residual, which is what enforces the split constraint.
    /// </summary>
    /// <remarks>
    /// A kernel that computed u from the PRE-update x would still produce a plausible trajectory; this
    /// pins that it uses the post-update value, by making the two differ.
    /// </remarks>
    [Fact]
    public void DualVariableAccumulatesThePostUpdateResidual()
    {
        const float lr = 0.1f, rho = 1.0f, l1 = 0f;
        var (param, grad) = MakeInputs(seed: 99);
        var before = (float[])param.Clone();
        var z = new float[VectorLength];
        var u = new float[VectorLength];

        RunKernel(param, (float[])grad.Clone(), z, u, lr, rho, l1);

        for (int i = 0; i < VectorLength; i++)
        {
            Assert.Equal(param[i] - z[i], u[i], 5);
            Assert.NotEqual(before[i] - z[i], u[i], 5);
        }
    }
}
