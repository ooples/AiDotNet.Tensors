// #1930 follow-up: SgdMomentumUpdateSimd has always accepted a `nesterov` flag, but
// CompiledTrainingPlan passed a hardcoded `false`, so a Nesterov optimizer routed through the
// fused path silently ran CLASSICAL momentum — a different algorithm, with no error. The flag is
// now surfaced through FusedOptimizerExtras.Nesterov. These tests pin the kernel's two branches
// against scalar references, so "threaded the flag" cannot pass while the kernel ignores it.

using System;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class SgdMomentumNesterovTests
{
    private const float Tolerance = 1e-5f;
    private const float Lr = 0.01f;
    private const float Momentum = 0.9f;

    /// <summary>
    /// Classical momentum: v = mu*v + g; param -= lr*v.
    /// </summary>
    [Fact]
    public unsafe void SgdMomentum_Classical_MatchesScalarReference()
    {
        const int n = 64;
        var (paramSimd, grad, velocitySimd) = MakeInputs(n, seed: 7);
        var paramRef = (float[])paramSimd.Clone();
        var velocityRef = (float[])velocitySimd.Clone();

        fixed (float* p = paramSimd, g = grad, v = velocitySimd)
            FusedOptimizer.SgdMomentumUpdateSimd(p, g, v, n, Lr, Momentum, nesterov: false);

        for (int i = 0; i < n; i++)
        {
            velocityRef[i] = Momentum * velocityRef[i] + grad[i];
            paramRef[i] -= Lr * velocityRef[i];
        }

        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(paramSimd[i] - paramRef[i]) < Tolerance,
                $"param[{i}]: SIMD {paramSimd[i]} vs scalar {paramRef[i]}");
            Assert.True(Math.Abs(velocitySimd[i] - velocityRef[i]) < Tolerance,
                $"velocity[{i}]: SIMD {velocitySimd[i]} vs scalar {velocityRef[i]}");
        }
    }

    /// <summary>
    /// Nesterov: the velocity update is the same, but the step applied to the parameter looks ahead —
    /// param -= lr*(g + mu*v) rather than param -= lr*v.
    /// </summary>
    [Fact]
    public unsafe void SgdMomentum_Nesterov_MatchesScalarReference()
    {
        const int n = 64;
        var (paramSimd, grad, velocitySimd) = MakeInputs(n, seed: 11);
        var paramRef = (float[])paramSimd.Clone();
        var velocityRef = (float[])velocitySimd.Clone();

        fixed (float* p = paramSimd, g = grad, v = velocitySimd)
            FusedOptimizer.SgdMomentumUpdateSimd(p, g, v, n, Lr, Momentum, nesterov: true);

        for (int i = 0; i < n; i++)
        {
            velocityRef[i] = Momentum * velocityRef[i] + grad[i];
            paramRef[i] -= Lr * (grad[i] + Momentum * velocityRef[i]);
        }

        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(paramSimd[i] - paramRef[i]) < Tolerance,
                $"param[{i}]: SIMD {paramSimd[i]} vs scalar {paramRef[i]}");
        }
    }

    /// <summary>
    /// The load-bearing assertion: the two branches must produce DIFFERENT parameters from identical
    /// inputs. Without this, threading the flag through the plan could "pass" while the kernel quietly
    /// ignored it — which is precisely the bug being fixed.
    /// </summary>
    [Fact]
    public unsafe void SgdMomentum_NesterovAndClassical_Differ()
    {
        const int n = 64;
        var (paramClassical, grad, velocityClassical) = MakeInputs(n, seed: 23);
        var paramNesterov = (float[])paramClassical.Clone();
        var velocityNesterov = (float[])velocityClassical.Clone();

        fixed (float* p = paramClassical, g = grad, v = velocityClassical)
            FusedOptimizer.SgdMomentumUpdateSimd(p, g, v, n, Lr, Momentum, nesterov: false);
        fixed (float* p = paramNesterov, g = grad, v = velocityNesterov)
            FusedOptimizer.SgdMomentumUpdateSimd(p, g, v, n, Lr, Momentum, nesterov: true);

        int differing = 0;
        for (int i = 0; i < n; i++)
            if (Math.Abs(paramClassical[i] - paramNesterov[i]) > 1e-7f)
                differing++;

        Assert.True(differing > n / 2,
            $"Only {differing}/{n} elements differ between classical and Nesterov momentum. " +
            "The nesterov flag is being ignored by the kernel.");
    }

    /// <summary>The default must stay classical, so existing callers are unaffected.</summary>
    [Fact]
    public void FusedOptimizerExtras_DefaultsToClassicalMomentum()
    {
        Assert.False(new FusedOptimizerExtras().Nesterov);
    }

    /// <summary>FTRL's beta defaults to 0, preserving the kernel's previous alpha/sqrt(n) behaviour.</summary>
    [Fact]
    public void FusedOptimizerExtras_FtrlBetaDefaultsToZero()
    {
        Assert.Equal(0f, new FusedOptimizerExtras().FtrlBeta);
    }

    /// <summary>
    /// beta = 0 must reproduce the kernel's previous output bit-for-bit, so adding the parameter
    /// cannot change any existing caller's training.
    /// </summary>
    [Fact]
    public unsafe void Ftrl_BetaZero_IsIdenticalToTheDefaultOverload()
    {
        const int n = 32;
        var (paramA, grad, _) = MakeInputs(n, seed: 31);
        var paramB = (float[])paramA.Clone();
        var zA = new float[n]; var nA = new float[n];
        var zB = new float[n]; var nB = new float[n];

        fixed (float* p = paramA, g = grad, z = zA, acc = nA)
            FusedOptimizer.FTRLUpdateSimd(p, g, z, acc, n, 0.005f, 0.1f, 0.1f, -0.5f);
        fixed (float* p = paramB, g = grad, z = zB, acc = nB)
            FusedOptimizer.FTRLUpdateSimd(p, g, z, acc, n, 0.005f, 0.1f, 0.1f, -0.5f, beta: 0f);

        for (int i = 0; i < n; i++)
            Assert.Equal(paramA[i], paramB[i]);
    }

    /// <summary>
    /// beta &gt; 0 must actually change the result. McMahan et al. 2013 use beta = 1; before this the
    /// kernel could only express beta = 0, so such an optimizer could not be fused faithfully.
    /// </summary>
    [Fact]
    public unsafe void Ftrl_NonZeroBeta_ChangesTheResult()
    {
        const int n = 32;
        var (paramZero, grad, _) = MakeInputs(n, seed: 37);
        var paramOne = (float[])paramZero.Clone();
        var z0 = new float[n]; var n0 = new float[n];
        var z1 = new float[n]; var n1 = new float[n];

        fixed (float* p = paramZero, g = grad, z = z0, acc = n0)
            FusedOptimizer.FTRLUpdateSimd(p, g, z, acc, n, 0.005f, 0.1f, 0.1f, -0.5f, beta: 0f);
        fixed (float* p = paramOne, g = grad, z = z1, acc = n1)
            FusedOptimizer.FTRLUpdateSimd(p, g, z, acc, n, 0.005f, 0.1f, 0.1f, -0.5f, beta: 1f);

        int differing = 0;
        for (int i = 0; i < n; i++)
            if (Math.Abs(paramZero[i] - paramOne[i]) > 1e-7f) differing++;

        Assert.True(differing > 0,
            "beta had no effect on the FTRL update — the parameter is not reaching the denominator.");
    }

    private static (float[] Param, float[] Grad, float[] Velocity) MakeInputs(int n, int seed)
    {
        var rng = new Random(seed);
        var param = new float[n];
        var grad = new float[n];
        var velocity = new float[n];
        for (int i = 0; i < n; i++)
        {
            param[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            grad[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            // A non-zero starting velocity matters: with v = 0 the two branches would differ only by
            // the single momentum*grad term, understating the divergence the third test checks for.
            velocity[i] = (float)(rng.NextDouble() * 0.5 - 0.25);
        }
        return (param, grad, velocity);
    }
}
