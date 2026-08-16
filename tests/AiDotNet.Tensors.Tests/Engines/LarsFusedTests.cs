using System;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Covers the fused LARS kernel against You, Gitman &amp; Ginsburg (2017), Algorithm 1.
/// </summary>
/// <remarks>
/// <para>
/// LARS is easy to implement as "SGD-with-momentum at a rescaled learning rate", which is what it looks like
/// and is not what it is. The trust ratio is recomputed from the current norms on every step, so where the
/// rate is applied matters: scaling the accumulated velocity at write time re-scales the whole gradient
/// history by whatever the ratio is now, while the paper scales each gradient by the rate that was current
/// when it arrived. Both converge, so the difference never shows up as a failure — only as different
/// training. The tests below therefore compare against a transcription of the paper across MULTIPLE steps,
/// since any single step from zero velocity cannot tell the two apart.
/// </para>
/// </remarks>
public class LarsFusedTests
{
    /// <summary>
    /// Deliberately not a multiple of 8: the kernel computes <c>simdLen = length &amp; ~7</c> and runs the
    /// remainder in a scalar loop, so a round length would leave one of the two implementations of the same
    /// math unexecuted on AVX2 hardware.
    /// </summary>
    private const int VectorLength = 67;

    private static unsafe void RunKernel(
        float[] param, float[] grad, float[] velocity,
        float lr, float momentum, float wd, float trust, float eps)
    {
        fixed (float* pp = param, pg = grad, pv = velocity)
            FusedOptimizer.LARSUpdateSimd(pp, pg, pv, param.Length, lr, momentum, wd, trust, eps);
    }

    /// <summary>Independent scalar transcription of Algorithm 1.</summary>
    private static void Reference(
        float[] param, float[] grad, float[] velocity,
        float lr, float momentum, float wd, float trust, float eps)
    {
        double pSq = 0.0, gSq = 0.0;
        for (int i = 0; i < param.Length; i++)
        {
            pSq += (double)param[i] * param[i];
            gSq += (double)grad[i] * grad[i];
        }
        float pNorm = (float)Math.Sqrt(pSq);
        float gNorm = (float)Math.Sqrt(gSq);

        float localLr = (pNorm >= eps && gNorm >= eps)
            ? lr * trust * pNorm / (gNorm + wd * pNorm + eps)
            : lr;

        for (int i = 0; i < param.Length; i++)
        {
            float g = grad[i] + wd * param[i];
            velocity[i] = momentum * velocity[i] + localLr * g;
            param[i] -= velocity[i];
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

    /// <summary>
    /// Runs several steps with a fresh gradient each time, so the trust ratio genuinely moves between steps.
    /// </summary>
    [Fact]
    public void MatchesTheReferenceAcrossSteps_IncludingTheScalarTail()
    {
        const float lr = 0.1f, momentum = 0.9f, wd = 1e-4f, trust = 1e-3f, eps = 1e-8f;

        var (kernelParam, _) = MakeInputs(seed: 7);
        var referenceParam = (float[])kernelParam.Clone();
        var kernelVelocity = new float[VectorLength];
        var referenceVelocity = new float[VectorLength];

        for (int step = 0; step < 5; step++)
        {
            var (_, grad) = MakeInputs(seed: 100 + step);

            RunKernel(kernelParam, (float[])grad.Clone(), kernelVelocity, lr, momentum, wd, trust, eps);
            Reference(referenceParam, grad, referenceVelocity, lr, momentum, wd, trust, eps);
        }

        for (int i = 0; i < VectorLength; i++)
        {
            Assert.Equal(referenceParam[i], kernelParam[i], 4);
            Assert.Equal(referenceVelocity[i], kernelVelocity[i], 4);
        }
    }

    /// <summary>
    /// Pins where the trust ratio is applied: inside the velocity, not at the parameter write.
    /// </summary>
    /// <remarks>
    /// This is the test that fails for the "SGD-with-momentum at local_lr" implementation. It drives the
    /// trust ratio to two very different values on consecutive steps by changing the gradient magnitude by
    /// 100×, then checks the second step's velocity against the paper. Applying the rate at write time
    /// would rescale step 1's contribution by step 2's ratio, which is a visibly different number rather
    /// than a rounding difference.
    /// </remarks>
    [Fact]
    public void AppliesTheTrustRatioWhenTheGradientArrives_NotAtWriteTime()
    {
        const float lr = 1.0f, momentum = 0.9f, wd = 0f, trust = 1.0f, eps = 1e-8f;
        var param = new float[] { 1f, 1f, 1f, 1f };

        var kernelParam = (float[])param.Clone();
        var kernelVelocity = new float[param.Length];
        var referenceParam = (float[])param.Clone();
        var referenceVelocity = new float[param.Length];

        var bigGrad = new float[] { 1f, 1f, 1f, 1f };
        var smallGrad = new float[] { 0.01f, 0.01f, 0.01f, 0.01f };

        RunKernel(kernelParam, (float[])bigGrad.Clone(), kernelVelocity, lr, momentum, wd, trust, eps);
        Reference(referenceParam, bigGrad, referenceVelocity, lr, momentum, wd, trust, eps);

        RunKernel(kernelParam, (float[])smallGrad.Clone(), kernelVelocity, lr, momentum, wd, trust, eps);
        Reference(referenceParam, smallGrad, referenceVelocity, lr, momentum, wd, trust, eps);

        for (int i = 0; i < param.Length; i++)
            Assert.Equal(referenceVelocity[i], kernelVelocity[i], 5);

        // Now the rejected implementation, run over the same two steps: velocity accumulates the raw
        // gradient and the trust ratio is applied at the parameter write.
        var writeTimeParam = (float[])param.Clone();
        var writeTimeVelocity = new float[param.Length];
        foreach (var grad in new[] { bigGrad, smallGrad })
        {
            double pSq = 0.0, gSq = 0.0;
            for (int i = 0; i < grad.Length; i++)
            {
                pSq += (double)writeTimeParam[i] * writeTimeParam[i];
                gSq += (double)grad[i] * grad[i];
            }
            float localLr = lr * trust * (float)Math.Sqrt(pSq) / ((float)Math.Sqrt(gSq) + eps);
            for (int i = 0; i < grad.Length; i++)
            {
                writeTimeVelocity[i] = momentum * writeTimeVelocity[i] + grad[i];
                writeTimeParam[i] -= localLr * writeTimeVelocity[i];
            }
        }

        // If the two were interchangeable this test would be vacuous, so assert they are not.
        Assert.NotEqual(writeTimeParam[0], kernelParam[0], 3);
    }

    /// <summary>
    /// Weight decay has to appear in the velocity as well as in the trust denominator; a kernel that only
    /// used it to bound the step would leave the weights undecayed.
    /// </summary>
    [Fact]
    public void WeightDecayReachesTheParameterUpdate()
    {
        const float lr = 0.1f, momentum = 0f, trust = 1e-3f, eps = 1e-8f;
        var param = new float[] { 0.5f, -0.5f, 0.25f, -0.25f, 1f, -1f, 0.75f, -0.75f, 0.125f };
        var zeroGrad = new float[param.Length];

        var withDecay = (float[])param.Clone();
        var velocity = new float[param.Length];
        RunKernel(withDecay, (float[])zeroGrad.Clone(), velocity, lr, momentum, 0.1f, trust, eps);

        // ||g|| is 0, so the trust ratio falls back to the base lr and the whole update is the decay term.
        for (int i = 0; i < param.Length; i++)
            Assert.Equal(param[i] - lr * 0.1f * param[i], withDecay[i], 5);

        var withoutDecay = (float[])param.Clone();
        var velocity2 = new float[param.Length];
        RunKernel(withoutDecay, (float[])zeroGrad.Clone(), velocity2, lr, momentum, 0f, trust, eps);
        for (int i = 0; i < param.Length; i++)
            Assert.Equal(param[i], withoutDecay[i], 6);
    }

    /// <summary>
    /// A zero-norm layer — a freshly zero-initialised bias vector is the everyday case — must fall back to
    /// the base learning rate instead of dividing by zero.
    /// </summary>
    [Fact]
    public void ZeroWeightNorm_FallsBackToTheBaseLearningRate()
    {
        const float lr = 0.1f, momentum = 0f, wd = 0f, trust = 1e-3f, eps = 1e-8f;
        var param = new float[9];                       // all zeros
        var grad = new float[9];
        for (int i = 0; i < grad.Length; i++) grad[i] = 0.5f;

        var updated = (float[])param.Clone();
        var velocity = new float[param.Length];
        RunKernel(updated, (float[])grad.Clone(), velocity, lr, momentum, wd, trust, eps);

        for (int i = 0; i < param.Length; i++)
        {
            Assert.False(float.IsNaN(updated[i]));
            Assert.Equal(-lr * 0.5f, updated[i], 6);
        }
    }

    /// <summary>
    /// The sparse and dense kernels are the same optimizer and must produce the same numbers when the
    /// sparse gradient happens to touch every index.
    /// </summary>
    /// <remarks>
    /// They diverged before this: the sparse kernel already followed the paper while the dense one applied
    /// the rate at write time and dropped weight decay from the update, so the same parameter trained
    /// differently depending on whether its gradient arrived sparse.
    /// </remarks>
    [Fact]
    public unsafe void SparseAndDenseKernelsAgreeWhenEveryIndexIsTouched()
    {
        const float lr = 0.1f, momentum = 0.9f, wd = 1e-3f, trust = 1e-3f, eps = 1e-8f;
        var (param, grad) = MakeInputs(seed: 11);

        var dense = (float[])param.Clone();
        var denseVelocity = new float[VectorLength];
        RunKernel(dense, (float[])grad.Clone(), denseVelocity, lr, momentum, wd, trust, eps);

        var sparse = (float[])param.Clone();
        var sparseVelocity = new float[VectorLength];
        var indices = new int[VectorLength];
        for (int i = 0; i < VectorLength; i++) indices[i] = i;
        var values = (float[])grad.Clone();
        fixed (float* pp = sparse, pv = sparseVelocity, pVal = values)
        fixed (int* pIdx = indices)
            FusedOptimizer.SparseLARSUpdate(pp, pIdx, pVal, pv, VectorLength, VectorLength,
                lr, momentum, wd, trust, eps);

        for (int i = 0; i < VectorLength; i++)
        {
            Assert.Equal(dense[i], sparse[i], 5);
            Assert.Equal(denseVelocity[i], sparseVelocity[i], 5);
        }
    }
}
