// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GpuOptimizerHostResidencyAndAuditTests : IDisposable
{
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly CpuEngine _cpu = new();
    private readonly bool _gpuReady;

    public GpuOptimizerHostResidencyAndAuditTests()
    {
        bool required = string.Equals(
            Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal);
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable)
            {
                _gpu = gpu;
                _gpuReady = true;
            }
            else
            {
                gpu.Dispose();
                if (required)
                    throw new InvalidOperationException("DirectGpu backend is required when AIDOTNET_REQUIRE_GPU_TESTS=1.");
            }
        }
        catch (Exception ex) when (!required && (ex is PlatformNotSupportedException or DllNotFoundException))
        {
            _gpuReady = false;
        }
    }

    public void Dispose()
    {
        _gpu?.Dispose();
    }

    private DirectGpuTensorEngine RequireGpuEngine()
    {
        Skip.If(!_gpuReady || _gpu is null, "DirectGpu backend not available.");
        return _gpu ?? throw new InvalidOperationException("DirectGpu engine unavailable after skip guard.");
    }

    private static float[] MakeRandomFloats(int seed, int length, double scale = 1.0)
    {
        var rng = new Random(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return values;
    }

    private static Tensor<float> HostTensor(float[] values, params int[] shape)
    {
        var copy = (float[])values.Clone();
        int[] resolvedShape = shape.Length == 0 ? new[] { copy.Length } : shape;
        return new Tensor<float>(resolvedShape, new Vector<float>(copy));
    }

    private static float[] IdentitySparseIndexFloats(int length)
    {
        var raw = new float[length];
        for (int i = 0; i < length; i++)
            raw[i] = Int32BitsToSingleCompat(i);
        return raw;
    }

    private static float[] EncodeSparseIndexFloats(int[] indices)
    {
        var raw = new float[indices.Length];
        for (int i = 0; i < indices.Length; i++)
            raw[i] = Int32BitsToSingleCompat(indices[i]);
        return raw;
    }

#if !NETCOREAPP
    private static unsafe float Int32BitsToSingleCompat(int value) => *(float*)&value;
#else
    private static float Int32BitsToSingleCompat(int value) => BitConverter.Int32BitsToSingle(value);
#endif

    private static void AssertClose(float[] expected, float[] actual, float tol, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxDiff = 0f;
        int worstIdx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            if (diff > maxDiff)
            {
                maxDiff = diff;
                worstIdx = i;
            }
        }

        Assert.True(
            maxDiff <= tol,
            $"{label} mismatch at [{worstIdx}]: expected {expected[worstIdx]:R}, actual {actual[worstIdx]:R}, maxDiff={maxDiff:E3} > tol={tol:E3}");
    }

    private static void WithEngine(IEngine engine, Action action)
    {
        var prev = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            action();
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }
    }

    private static void SimulateLars(
        float[] param, float[] grad, float[] velocity,
        float lr, float momentum, float weightDecay)
    {
        for (int i = 0; i < param.Length; i++)
        {
            float g = grad[i];
            float p = param[i];
            if (weightDecay > 0f)
                g += weightDecay * p;
            float v = momentum * velocity[i] + g;
            velocity[i] = v;
            param[i] = p - lr * v;
        }
    }

    private static void SimulateLamb(
        float[] param, float[] grad, float[] m, float[] v,
        float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        float bc1 = 1f - (float)Math.Pow(beta1, step);
        float bc2 = 1f - (float)Math.Pow(beta2, step);
        var update = new float[param.Length];
        double pNormSq = 0.0;
        double uNormSq = 0.0;
        for (int i = 0; i < param.Length; i++)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1f - beta1) * g;
            v[i] = beta2 * v[i] + (1f - beta2) * g * g;
            float r = (m[i] / bc1) / ((float)Math.Sqrt(v[i] / bc2) + eps);
            update[i] = r + weightDecay * param[i];
            pNormSq += (double)param[i] * param[i];
            uNormSq += (double)update[i] * update[i];
        }

        float pNorm = (float)Math.Sqrt(pNormSq);
        float uNorm = (float)Math.Sqrt(uNormSq);
        float ratio = pNorm > 0f && uNorm > 0f ? pNorm / uNorm : 1f;
        for (int i = 0; i < param.Length; i++)
            param[i] -= lr * ratio * update[i];
    }

    [SkippableFact]
    public void AllDenseTryStepMethods_StepFromStep1OnHostAllocatedTensors_AndMatchCpuReference()
    {
        var gpu = RequireGpuEngine();
        const int n = 67;
        float[] idxAll = IdentitySparseIndexFloats(n);
        float[] pInit = MakeRandomFloats(1001, n, 1.25);
        float[] g1Raw = MakeRandomFloats(1002, n, 0.5);
        float[] g2Raw = MakeRandomFloats(1003, n, 0.5);

        WithEngine(gpu, () =>
        {
            // 1. TryAdamStep
            {
                var p = HostTensor(pInit);
                var m = new Tensor<float>(new[] { n });
                var v = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryAdamStep(p, HostTensor(g1Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 1));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryAdamStep(p, HostTensor(g2Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 2));
                var refP = (float[])pInit.Clone(); var refM = new float[n]; var refV = new float[n];
                SparseOptimizerReference.SparseAdamUpdate(refP, refM, refV, idxAll, g1Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 1);
                SparseOptimizerReference.SparseAdamUpdate(refP, refM, refV, idxAll, g2Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 2);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryAdamStep.param");
                AssertClose(refM, m.ToArray(), 1e-5f, "TryAdamStep.m");
                AssertClose(refV, v.ToArray(), 1e-5f, "TryAdamStep.v");
            }

            // 2. TryAdamWStep
            {
                var p = HostTensor(pInit);
                var m = new Tensor<float>(new[] { n });
                var v = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryAdamWStep(p, HostTensor(g1Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.05f, 1));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryAdamWStep(p, HostTensor(g2Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.05f, 2));
                var refP = (float[])pInit.Clone(); var refM = new float[n]; var refV = new float[n];
                SparseOptimizerReference.SparseAdamWUpdate(refP, refM, refV, idxAll, g1Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.05f, 1);
                SparseOptimizerReference.SparseAdamWUpdate(refP, refM, refV, idxAll, g2Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.05f, 2);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryAdamWStep.param");
                AssertClose(refM, m.ToArray(), 1e-5f, "TryAdamWStep.m");
                AssertClose(refV, v.ToArray(), 1e-5f, "TryAdamWStep.v");
            }

            // 3. TrySgdStep
            {
                var p = HostTensor(pInit);
                Assert.True(GpuOptimizer.TrySgdStep(p, HostTensor(g1Raw), 0.025f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TrySgdStep(p, HostTensor(g2Raw), 0.025f));
                var refP = (float[])pInit.Clone();
                SparseOptimizerReference.SparseSgdUpdate(refP, idxAll, g1Raw, n, 0.025f, 0f);
                SparseOptimizerReference.SparseSgdUpdate(refP, idxAll, g2Raw, n, 0.025f, 0f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TrySgdStep.param");
            }

            // 4. TryProximalL1Step
            {
                var p = HostTensor(pInit);
                Assert.True(GpuOptimizer.TryProximalL1Step(p, HostTensor(g1Raw), 0.05f, 0.25f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryProximalL1Step(p, HostTensor(g2Raw), 0.05f, 0.25f));
                var refP = (float[])pInit.Clone();
                SparseOptimizerReference.SparseProximalL1Update(refP, idxAll, g1Raw, n, 0.05f, 0.25f);
                SparseOptimizerReference.SparseProximalL1Update(refP, idxAll, g2Raw, n, 0.05f, 0.25f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryProximalL1Step.param");
            }

            // 5. TrySgdMomentumStep
            {
                var p = HostTensor(pInit);
                var vel = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TrySgdMomentumStep(p, HostTensor(g1Raw), vel, 0.01f, 0.9f, 0.01f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TrySgdMomentumStep(p, HostTensor(g2Raw), vel, 0.01f, 0.9f, 0.01f));
                var refP = (float[])pInit.Clone(); var refVel = new float[n];
                SparseOptimizerReference.SparseSgdMomentumUpdate(refP, refVel, idxAll, g1Raw, n, 0.01f, 0.9f, 0.01f);
                SparseOptimizerReference.SparseSgdMomentumUpdate(refP, refVel, idxAll, g2Raw, n, 0.01f, 0.9f, 0.01f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TrySgdMomentumStep.param");
                AssertClose(refVel, vel.ToArray(), 1e-5f, "TrySgdMomentumStep.vel");
            }

            // 6. TryRmspropStep
            {
                var p = HostTensor(pInit);
                var sq = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryRmspropStep(p, HostTensor(g1Raw), sq, 0.01f, 0.95f, 1e-6f, 0.01f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryRmspropStep(p, HostTensor(g2Raw), sq, 0.01f, 0.95f, 1e-6f, 0.01f));
                var refP = (float[])pInit.Clone(); var refSq = new float[n];
                SparseOptimizerReference.SparseRmspropUpdate(refP, refSq, idxAll, g1Raw, n, 0.01f, 0.95f, 1e-6f, 0.01f);
                SparseOptimizerReference.SparseRmspropUpdate(refP, refSq, idxAll, g2Raw, n, 0.01f, 0.95f, 1e-6f, 0.01f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryRmspropStep.param");
                AssertClose(refSq, sq.ToArray(), 1e-5f, "TryRmspropStep.sq");
            }

            // 7. TryAdagradStep
            {
                var p = HostTensor(pInit);
                var accum = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryAdagradStep(p, HostTensor(g1Raw), accum, 0.02f, 1e-6f, 0.01f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryAdagradStep(p, HostTensor(g2Raw), accum, 0.02f, 1e-6f, 0.01f));
                var refP = (float[])pInit.Clone(); var refAccum = new float[n];
                SparseOptimizerReference.SparseAdagradUpdate(refP, refAccum, idxAll, g1Raw, n, 0.02f, 1e-6f, 0.01f);
                SparseOptimizerReference.SparseAdagradUpdate(refP, refAccum, idxAll, g2Raw, n, 0.02f, 1e-6f, 0.01f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryAdagradStep.param");
                AssertClose(refAccum, accum.ToArray(), 1e-5f, "TryAdagradStep.accum");
            }

            // 8. TryNagStep
            {
                var p = HostTensor(pInit);
                var vel = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryNagStep(p, HostTensor(g1Raw), vel, 0.01f, 0.9f, 0.01f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryNagStep(p, HostTensor(g2Raw), vel, 0.01f, 0.9f, 0.01f));
                var refP = (float[])pInit.Clone(); var refVel = new float[n];
                SparseOptimizerReference.SparseNagUpdate(refP, refVel, idxAll, g1Raw, n, 0.01f, 0.9f, 0.01f);
                SparseOptimizerReference.SparseNagUpdate(refP, refVel, idxAll, g2Raw, n, 0.01f, 0.9f, 0.01f);
                var expectedDenseVel = new float[n];
                for (int i = 0; i < n; i++) expectedDenseVel[i] = -0.01f * refVel[i];
                AssertClose(refP, p.ToArray(), 1e-5f, "TryNagStep.param");
                AssertClose(expectedDenseVel, vel.ToArray(), 1e-5f, "TryNagStep.vel");
            }

            // 9. TryLarsStep
            {
                var p = HostTensor(pInit);
                var vel = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryLarsStep(p, HostTensor(g1Raw), vel, 0.01f, 0.9f, 0.01f, 0.001f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryLarsStep(p, HostTensor(g2Raw), vel, 0.01f, 0.9f, 0.01f, 0.001f));
                var refP = (float[])pInit.Clone(); var refVel = new float[n];
                SimulateLars(refP, g1Raw, refVel, 0.01f, 0.9f, 0.01f);
                SimulateLars(refP, g2Raw, refVel, 0.01f, 0.9f, 0.01f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryLarsStep.param");
                AssertClose(refVel, vel.ToArray(), 1e-5f, "TryLarsStep.vel");
            }

            // 10. TryLambStep
            {
                var p = HostTensor(pInit);
                var m = new Tensor<float>(new[] { n });
                var v = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryLambStep(p, HostTensor(g1Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-6f, 0.02f, 1));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryLambStep(p, HostTensor(g2Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-6f, 0.02f, 2));
                var refP = (float[])pInit.Clone(); var refM = new float[n]; var refV = new float[n];
                SimulateLamb(refP, g1Raw, refM, refV, 0.01f, 0.9f, 0.999f, 1e-6f, 0.02f, 1);
                SimulateLamb(refP, g2Raw, refM, refV, 0.01f, 0.9f, 0.999f, 1e-6f, 0.02f, 2);
                AssertClose(refP, p.ToArray(), 2e-5f, "TryLambStep.param");
                AssertClose(refM, m.ToArray(), 1e-5f, "TryLambStep.m");
                AssertClose(refV, v.ToArray(), 1e-5f, "TryLambStep.v");
            }

            // 11. TryAdadeltaStep
            {
                var p = HostTensor(pInit);
                var ag = new Tensor<float>(new[] { n });
                var au = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryAdadeltaStep(p, HostTensor(g1Raw), ag, au, 0.95f, 1e-5f, 0.01f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryAdadeltaStep(p, HostTensor(g2Raw), ag, au, 0.95f, 1e-5f, 0.01f));
                var refP = (float[])pInit.Clone(); var refAg = new float[n]; var refAu = new float[n];
                SparseOptimizerReference.SparseAdadeltaUpdate(refP, refAg, refAu, idxAll, g1Raw, n, 0.95f, 1e-5f, 0.01f);
                SparseOptimizerReference.SparseAdadeltaUpdate(refP, refAg, refAu, idxAll, g2Raw, n, 0.95f, 1e-5f, 0.01f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryAdadeltaStep.param");
                AssertClose(refAg, ag.ToArray(), 1e-5f, "TryAdadeltaStep.ag");
                AssertClose(refAu, au.ToArray(), 1e-5f, "TryAdadeltaStep.au");
            }

            // 12. TryAmsgradStep
            {
                var p = HostTensor(pInit);
                var m = new Tensor<float>(new[] { n });
                var v = new Tensor<float>(new[] { n });
                var vMax = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryAmsgradStep(p, HostTensor(g1Raw), m, v, vMax, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryAmsgradStep(p, HostTensor(g2Raw), m, v, vMax, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 2));
                var refP = (float[])pInit.Clone(); var refM = new float[n]; var refV = new float[n]; var refVm = new float[n];
                SparseOptimizerReference.SparseAmsgradUpdate(refP, refM, refV, refVm, idxAll, g1Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
                SparseOptimizerReference.SparseAmsgradUpdate(refP, refM, refV, refVm, idxAll, g2Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 2);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryAmsgradStep.param");
                AssertClose(refVm, vMax.ToArray(), 1e-5f, "TryAmsgradStep.vMax");
            }

            // 13. TryAdamaxStep
            {
                var p = HostTensor(pInit);
                var m = new Tensor<float>(new[] { n });
                var u = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryAdamaxStep(p, HostTensor(g1Raw), m, u, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryAdamaxStep(p, HostTensor(g2Raw), m, u, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 2));
                var refP = (float[])pInit.Clone(); var refM = new float[n]; var refU = new float[n];
                SparseOptimizerReference.SparseAdamaxUpdate(refP, refM, refU, idxAll, g1Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
                SparseOptimizerReference.SparseAdamaxUpdate(refP, refM, refU, idxAll, g2Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 2);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryAdamaxStep.param");
                AssertClose(refU, u.ToArray(), 1e-5f, "TryAdamaxStep.u");
            }

            // 14. TryLionStep
            {
                var p = HostTensor(pInit);
                var m = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryLionStep(p, HostTensor(g1Raw), m, 0.005f, 0.9f, 0.99f, 0.01f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryLionStep(p, HostTensor(g2Raw), m, 0.005f, 0.9f, 0.99f, 0.01f));
                var refP = (float[])pInit.Clone(); var refM = new float[n];
                SparseOptimizerReference.SparseLionUpdate(refP, refM, idxAll, g1Raw, n, 0.005f, 0.9f, 0.99f, 0.01f);
                SparseOptimizerReference.SparseLionUpdate(refP, refM, idxAll, g2Raw, n, 0.005f, 0.9f, 0.99f, 0.01f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryLionStep.param");
                AssertClose(refM, m.ToArray(), 1e-5f, "TryLionStep.m");
            }

            // 15. TryNadamStep
            {
                var p = HostTensor(pInit);
                var m = new Tensor<float>(new[] { n });
                var v = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryNadamStep(p, HostTensor(g1Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryNadamStep(p, HostTensor(g2Raw), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 2));
                var refP = (float[])pInit.Clone(); var refM = new float[n]; var refV = new float[n];
                SparseOptimizerReference.SparseNadamUpdate(refP, refM, refV, idxAll, g1Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
                SparseOptimizerReference.SparseNadamUpdate(refP, refM, refV, idxAll, g2Raw, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 2);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryNadamStep.param");
                AssertClose(refM, m.ToArray(), 1e-5f, "TryNadamStep.m");
            }

            // 16. TryFtrlStep
            {
                var p = HostTensor(pInit);
                var z = new Tensor<float>(new[] { n });
                var nState = new Tensor<float>(new[] { n });
                Assert.True(GpuOptimizer.TryFtrlStep(p, HostTensor(g1Raw), z, nState, 0.1f, 0.01f, 0.05f, 1f));
                Assert.True(gpu.IsDeviceAuthoritative(p));
                Assert.True(GpuOptimizer.TryFtrlStep(p, HostTensor(g2Raw), z, nState, 0.1f, 0.01f, 0.05f, 1f));
                var refP = (float[])pInit.Clone(); var refZ = new float[n]; var refN = new float[n];
                SparseOptimizerReference.SparseFtrlUpdate(refP, refZ, refN, idxAll, g1Raw, n, 0.1f, 0.01f, 0.05f, 1f);
                SparseOptimizerReference.SparseFtrlUpdate(refP, refZ, refN, idxAll, g2Raw, n, 0.1f, 0.01f, 0.05f, 1f);
                AssertClose(refP, p.ToArray(), 1e-5f, "TryFtrlStep.param");
                AssertClose(refZ, z.ToArray(), 1e-5f, "TryFtrlStep.z");
                AssertClose(refN, nState.ToArray(), 1e-5f, "TryFtrlStep.n");
            }
        });
    }

    [SkippableFact]
    public void CompressedTryAdamBf16And8BitSteps_StepFromStep1OnUnpromotedHostTensors()
    {
        var gpu = RequireGpuEngine();
        const int n = 128;
        const int blockSize = 64;
        float[] pInit = MakeRandomFloats(2001, n, 1.0);
        float[] g1Raw = MakeRandomFloats(2002, n, 0.5);

        WithEngine(gpu, () =>
        {
            // BF16 moment step on plain host param + plain host grad (no pre-call to AcquireParameterBuffer)
            var pBf16 = HostTensor(pInit);
            var gBf16 = HostTensor(g1Raw);
            IGpuBuffer? mBf16 = null;
            IGpuBuffer? vBf16 = null;
            try
            {
                Assert.True(GpuOptimizer.TryAllocBf16MomentState(n, out mBf16, out vBf16));
                Assert.True(GpuOptimizer.TryAdamBf16Step(pBf16, gBf16, mBf16, vBf16, 0.01f, 0.9f, 0.99f, 1e-8f, 0.1f, 1, true));
                Assert.True(gpu.IsDeviceAuthoritative(pBf16));
                var refBf16 = (float[])pInit.Clone();
                HostCompressedMomentAdam.Bf16Step(refBf16, g1Raw, new byte[n * 2], new byte[n * 2], n, 0.01f, 0.9f, 0.99f, 1e-8f, 0.1f, 1, true);
                AssertClose(refBf16, pBf16.ToArray(), 1e-5f, "TryAdamBf16Step.hostParam");
            }
            finally
            {
                GpuOptimizer.FreeGpuBuffer(mBf16);
                GpuOptimizer.FreeGpuBuffer(vBf16);
            }

            // 8-bit Adam (13-arg decay-free overload + 15-arg decay overload) on plain host param + plain host grad
            foreach (bool useDecayOverload in new[] { false, true })
            {
                var p8 = HostTensor(pInit);
                var g8 = HostTensor(g1Raw);
                IGpuBuffer? mQ = null;
                IGpuBuffer? vQ = null;
                IGpuBuffer? mS = null;
                IGpuBuffer? vS = null;
                try
                {
                    Assert.True(GpuOptimizer.TryAllocAdam8BitState(n, blockSize, out mQ, out vQ, out mS, out vS));
                    bool stepped = useDecayOverload
                        ? GpuOptimizer.TryAdam8BitStep(p8, g8, mQ, vQ, mS, vS, 0.01f, 0.9f, 0.99f, 1e-8f, 0.1f, 0.01f, blockSize, 0.1f, true)
                        : GpuOptimizer.TryAdam8BitStep(p8, g8, mQ, vQ, mS, vS, 0.01f, 0.9f, 0.99f, 1e-8f, 0.1f, 0.01f, blockSize);
                    Assert.True(stepped);
                    Assert.True(gpu.IsDeviceAuthoritative(p8));
                    var ref8 = (float[])pInit.Clone();
                    int blocks = (n + blockSize - 1) / blockSize;
                    HostCompressedMomentAdam.Int8Step(
                        ref8, (float[])g1Raw.Clone(), new byte[n], new byte[n], new float[blocks], new float[blocks],
                        n, blockSize, 0.01f, 0.9f, 0.99f, 1e-8f, useDecayOverload ? 0.1f : 0f, 1, true);
                    AssertClose(ref8, p8.ToArray(), 1e-5f, $"TryAdam8BitStep(decay={useDecayOverload}).hostParam");
                }
                finally
                {
                    GpuOptimizer.FreeGpuBuffer(mQ);
                    GpuOptimizer.FreeGpuBuffer(vQ);
                    GpuOptimizer.FreeGpuBuffer(mS);
                    GpuOptimizer.FreeGpuBuffer(vS);
                }
            }

            // Multi-tensor SGD on resident parameters + gradients: refuses host-only or stale host-mutated inputs
            // without modifying any parameter, and commits resident updates across sequential host checkpoints.
            var pMulti0 = gpu.UploadToGpu(HostTensor(pInit), GpuTensorRole.Weight);
            float[] pMulti1Init = MakeRandomFloats(2003, 48, 1.0);
            var pMulti1 = gpu.UploadToGpu(HostTensor(pMulti1Init), GpuTensorRole.Weight);
            float[] gMulti1Raw = MakeRandomFloats(2004, 48, 0.5);
            var gMulti0 = gpu.UploadToGpu(HostTensor(g1Raw), GpuTensorRole.Gradient);
            var gMulti1 = gpu.UploadToGpu(HostTensor(gMulti1Raw), GpuTensorRole.Gradient);

            Assert.False(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMulti0, HostTensor(pMulti1Init) }, new[] { gMulti0, gMulti1 }, 0.05f));
            AssertClose(pInit, pMulti0.ToArray(), 0f, "TrySgdStepMultiTensor.hostRefusalLeavesFirstParamUnchanged");

            Assert.True(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMulti0, pMulti1 }, new[] { gMulti0, gMulti1 }, 0.05f));
            var expectedMulti0 = (float[])pInit.Clone();
            var expectedMulti1 = (float[])pMulti1Init.Clone();
            for (int i = 0; i < expectedMulti0.Length; i++) expectedMulti0[i] -= 0.05f * g1Raw[i];
            for (int i = 0; i < expectedMulti1.Length; i++) expectedMulti1[i] -= 0.05f * gMulti1Raw[i];
            AssertClose(expectedMulti0, pMulti0.ToArray(), 1e-5f, "TrySgdStepMultiTensor.step1.param0");
            AssertClose(expectedMulti1, pMulti1.ToArray(), 1e-5f, "TrySgdStepMultiTensor.step1.param1");

            // Step 2 after host read must re-arm HostSync and produce step-2 values (never serving stale step-1 host cache).
            Assert.True(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMulti0, pMulti1 }, new[] { gMulti0, gMulti1 }, 0.05f));
            for (int i = 0; i < expectedMulti0.Length; i++) expectedMulti0[i] -= 0.05f * g1Raw[i];
            for (int i = 0; i < expectedMulti1.Length; i++) expectedMulti1[i] -= 0.05f * gMulti1Raw[i];
            AssertClose(expectedMulti0, pMulti0.ToArray(), 1e-5f, "TrySgdStepMultiTensor.step2.param0");
            AssertClose(expectedMulti1, pMulti1.ToArray(), 1e-5f, "TrySgdStepMultiTensor.step2.param1");

            // Host-mutating gMulti1 invalidates its device buffer version, so TrySgdStepMultiTensor must refuse cleanly.
            gMulti1.AsWritableSpan()[0] = 99f;
            gMulti1.IncrementVersion();
            Assert.False(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMulti0, pMulti1 }, new[] { gMulti0, gMulti1 }, 0.05f));
        });
    }

    [SkippableFact]
    public void AllSparseTryStepMethods_StepFromStep1OnHostAllocatedTensors_AndMatchCpuReference()
    {
        var gpu = RequireGpuEngine();
        const int tableLen = 32;
        int[] indicesRaw = { 1, 7, 15, 29 };
        int nnz = indicesRaw.Length;
        float[] encodedIdx = EncodeSparseIndexFloats(indicesRaw);
        float[] pInit = MakeRandomFloats(3001, tableLen, 1.5);
        float[] valsRaw = MakeRandomFloats(3002, nnz, 0.75);

        WithEngine(gpu, () =>
        {
            var idxTensor = new Tensor<int>(new[] { nnz }, new Vector<int>((int[])indicesRaw.Clone()));
            var valTensor = HostTensor(valsRaw);
            int idxVersionBefore = idxTensor.Version;
            int valVersionBefore = valTensor.Version;

            void Check(string name, Func<Tensor<float>, bool> stepGpu, Action<float[]> stepCpu)
            {
                var p = HostTensor(pInit);
                Assert.True(stepGpu(p), $"{name} should execute on GPU from host-allocated tensors.");
                Assert.True(gpu.IsDeviceAuthoritative(p), $"{name} should leave param device-authoritative.");
                var expected = (float[])pInit.Clone();
                stepCpu(expected);
                AssertClose(expected, p.ToArray(), 1e-5f, name);
            }

            Check("TryAdamStepSparse",
                p => GpuOptimizer.TryAdamStepSparse(p, new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 1),
                ep => SparseOptimizerReference.SparseAdamUpdate(ep, new float[tableLen], new float[tableLen], encodedIdx, valsRaw, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 1));

            Check("TryAdamWStepSparse",
                p => GpuOptimizer.TryAdamWStepSparse(p, new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.05f, 1),
                ep => SparseOptimizerReference.SparseAdamWUpdate(ep, new float[tableLen], new float[tableLen], encodedIdx, valsRaw, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.05f, 1));

            Check("TrySgdStepSparse",
                p => GpuOptimizer.TrySgdStepSparse(p, idxTensor, valTensor, nnz, 0.03f, 0.01f),
                ep => SparseOptimizerReference.SparseSgdUpdate(ep, encodedIdx, valsRaw, nnz, 0.03f, 0.01f));

            Check("TrySgdMomentumStepSparse",
                p => GpuOptimizer.TrySgdMomentumStepSparse(p, new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.02f, 0.9f, 0.01f),
                ep => SparseOptimizerReference.SparseSgdMomentumUpdate(ep, new float[tableLen], encodedIdx, valsRaw, nnz, 0.02f, 0.9f, 0.01f));

            Check("TryRmspropStepSparse",
                p => GpuOptimizer.TryRmspropStepSparse(p, new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.01f, 0.95f, 1e-6f, 0.01f),
                ep => SparseOptimizerReference.SparseRmspropUpdate(ep, new float[tableLen], encodedIdx, valsRaw, nnz, 0.01f, 0.95f, 1e-6f, 0.01f));

            Check("TryAdagradStepSparse",
                p => GpuOptimizer.TryAdagradStepSparse(p, new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.02f, 1e-6f, 0.01f),
                ep => SparseOptimizerReference.SparseAdagradUpdate(ep, new float[tableLen], encodedIdx, valsRaw, nnz, 0.02f, 1e-6f, 0.01f));

            Check("TryNagStepSparse",
                p => GpuOptimizer.TryNagStepSparse(p, new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.01f, 0.9f, 0.01f),
                ep => SparseOptimizerReference.SparseNagUpdate(ep, new float[tableLen], encodedIdx, valsRaw, nnz, 0.01f, 0.9f, 0.01f));

            Check("TryAdadeltaStepSparse",
                p => GpuOptimizer.TryAdadeltaStepSparse(p, new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.95f, 1e-5f, 0.01f),
                ep => SparseOptimizerReference.SparseAdadeltaUpdate(ep, new float[tableLen], new float[tableLen], encodedIdx, valsRaw, nnz, 0.95f, 1e-5f, 0.01f));

            Check("TryAmsgradStepSparse",
                p => GpuOptimizer.TryAmsgradStepSparse(p, new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1),
                ep => SparseOptimizerReference.SparseAmsgradUpdate(ep, new float[tableLen], new float[tableLen], new float[tableLen], encodedIdx, valsRaw, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));

            Check("TryAdamaxStepSparse",
                p => GpuOptimizer.TryAdamaxStepSparse(p, new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1),
                ep => SparseOptimizerReference.SparseAdamaxUpdate(ep, new float[tableLen], new float[tableLen], encodedIdx, valsRaw, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));

            Check("TryLionStepSparse",
                p => GpuOptimizer.TryLionStepSparse(p, new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.005f, 0.9f, 0.99f, 0.01f),
                ep => SparseOptimizerReference.SparseLionUpdate(ep, new float[tableLen], encodedIdx, valsRaw, nnz, 0.005f, 0.9f, 0.99f, 0.01f));

            Check("TryNadamStepSparse",
                p => GpuOptimizer.TryNadamStepSparse(p, new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1),
                ep => SparseOptimizerReference.SparseNadamUpdate(ep, new float[tableLen], new float[tableLen], encodedIdx, valsRaw, nnz, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));

            Check("TryFtrlStepSparse",
                p => GpuOptimizer.TryFtrlStepSparse(p, new Tensor<float>(new[] { tableLen }), new Tensor<float>(new[] { tableLen }), idxTensor, valTensor, nnz, 0.1f, 0.01f, 0.05f, 1f),
                ep => SparseOptimizerReference.SparseFtrlUpdate(ep, new float[tableLen], new float[tableLen], encodedIdx, valsRaw, nnz, 0.1f, 0.01f, 0.05f, 1f));

            Check("TryProximalL1StepSparse",
                p => GpuOptimizer.TryProximalL1StepSparse(p, idxTensor, valTensor, nnz, 0.05f, 0.2f),
                ep => SparseOptimizerReference.SparseProximalL1Update(ep, encodedIdx, valsRaw, nnz, 0.05f, 0.2f));

            Assert.Equal(idxVersionBefore, idxTensor.Version);
            Assert.Equal(valVersionBefore, valTensor.Version);
        });
    }

    [SkippableFact]
    public void SequentialHostCheckpointsAndHostRestore_RearmHostSyncWithZeroStaleReads()
    {
        var gpu = RequireGpuEngine();
        const int n = 96;
        float[] idxAll = IdentitySparseIndexFloats(n);
        float[] pInit = MakeRandomFloats(4001, n, 1.0);
        float[] g1 = MakeRandomFloats(4002, n, 0.5);
        float[] g2 = MakeRandomFloats(4003, n, 0.5);
        float[] g3 = MakeRandomFloats(4004, n, 0.5);

        WithEngine(gpu, () =>
        {
            var p = HostTensor(pInit);
            using var m = GpuOptimizer.CreateStateTensor(new[] { n });
            using var v = GpuOptimizer.CreateStateTensor(new[] { n });

            // CreateStateTensor sets _ownsGpuBuffer=true and isDeviceModified=false: reading before step 1 triggers zero D2H readbacks.
            Assert.True(m._ownsGpuBuffer);
            Assert.False(m.HasPendingGpuData);
            GpuLaunchProbe.Reset();
            Assert.All(m.ToArray(), x => Assert.Equal(0f, x));
            Assert.Equal(0L, GpuLaunchProbe.ReadbackBytes);

            var refP = (float[])pInit.Clone();
            var refM = new float[n];
            var refV = new float[n];

            // Step 1 -> Checkpoint #1
            Assert.True(GpuOptimizer.TryAdamWStep(p, HostTensor(g1), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 1));
            SparseOptimizerReference.SparseAdamWUpdate(refP, refM, refV, idxAll, g1, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 1);
            AssertClose(refP, p.ToArray(), 1e-5f, "Checkpoint1.param");
            AssertClose(refM, m.ToArray(), 1e-5f, "Checkpoint1.m");
            AssertClose(refV, v.ToArray(), 1e-5f, "Checkpoint1.v");

            // Step 2 AFTER Checkpoint #1 (HostSync was cleared by Checkpoint #1 and MUST re-arm on param, m, and v!) -> Checkpoint #2
            Assert.True(GpuOptimizer.TryAdamWStep(p, HostTensor(g2), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 2));
            Assert.True(gpu.IsDeviceAuthoritative(p));
            Assert.True(p.HasPendingGpuData);
            Assert.True(m.HasPendingGpuData);
            Assert.True(v.HasPendingGpuData);
            SparseOptimizerReference.SparseAdamWUpdate(refP, refM, refV, idxAll, g2, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 2);
            AssertClose(refP, p.ToArray(), 1e-5f, "Checkpoint2.param");
            AssertClose(refM, m.ToArray(), 1e-5f, "Checkpoint2.m");
            AssertClose(refV, v.ToArray(), 1e-5f, "Checkpoint2.v");

            // Simulate mid-training checkpoint restore on host (mutating p, m, v via AsWritableSpan) -> Step 3 -> Checkpoint #3
            float[] restoredP = MakeRandomFloats(4010, n, 0.8);
            float[] restoredM = MakeRandomFloats(4011, n, 0.1);
            float[] restoredV = MakeRandomFloats(4012, n, 0.05);
            for (int i = 0; i < n; i++) restoredV[i] = Math.Abs(restoredV[i]) + 1e-4f;

            restoredP.AsSpan().CopyTo(p.AsWritableSpan());
            restoredM.AsSpan().CopyTo(m.AsWritableSpan());
            restoredV.AsSpan().CopyTo(v.AsWritableSpan());
            p.IncrementVersion();
            m.IncrementVersion();
            v.IncrementVersion();

            Assert.True(GpuOptimizer.TryAdamWStep(p, HostTensor(g3), m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 3));
            SparseOptimizerReference.SparseAdamWUpdate(restoredP, restoredM, restoredV, idxAll, g3, n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.02f, 3);
            AssertClose(restoredP, p.ToArray(), 1e-5f, "Checkpoint3AfterRestore.param");
            AssertClose(restoredM, m.ToArray(), 1e-5f, "Checkpoint3AfterRestore.m");
            AssertClose(restoredV, v.ToArray(), 1e-5f, "Checkpoint3AfterRestore.v");
        });
    }

    [SkippableFact]
    public void ForwardThenOptimizerThenForward_PromotesActivationCachedBias_WithZeroReadbacksAndHostClippedGradFreshness()
    {
        var gpu = RequireGpuEngine();
        float[] xRaw = MakeRandomFloats(5001, 4 * 8, 1.0);
        float[] wRaw = MakeRandomFloats(5002, 8 * 6, 0.5);
        float[] bRaw = MakeRandomFloats(5003, 4 * 6, 0.25);
        float[] gwRaw = MakeRandomFloats(5004, 8 * 6, 0.3);
        float[] gbRaw = MakeRandomFloats(5005, 4 * 6, 0.3);

        WithEngine(gpu, () =>
        {
            var x = HostTensor(xRaw, 4, 8);
            var w = HostTensor(wRaw, 8, 6);
            var b = HostTensor(bRaw, 4, 6);
            using var mw = GpuOptimizer.CreateStateTensor(new[] { w.Length });
            using var vw = GpuOptimizer.CreateStateTensor(new[] { w.Length });
            using var mb = GpuOptimizer.CreateStateTensor(new[] { b.Length });
            using var vb = GpuOptimizer.CreateStateTensor(new[] { b.Length });

            // Step 1 forward: b goes through TensorAdd BEFORE any optimizer step (landing in _activationCache first).
            IEngine eng = gpu;
            var y1 = eng.TensorAdd(eng.TensorMatMul(x, w), b);
            Assert.Equal(new[] { 4, 6 }, y1.Shape.ToArray());

            var gwDev = eng.TensorMultiplyScalar(HostTensor(gwRaw, 8, 6), 1f);
            var gbDev = eng.TensorMultiplyScalar(HostTensor(gbRaw, 4, 6), 1f);

            Assert.True(GpuOptimizer.TryAdamWStep(w, gwDev, mw, vw, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryAdamWStep(b, gbDev, mb, vb, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));

            // Both w AND b (promoted from activation cache to persistent cache) must now be device-authoritative!
            Assert.True(gpu.IsDeviceAuthoritative(w));
            Assert.True(gpu.IsDeviceAuthoritative(b));

            // Pre-stage x on device, then verify Step 2 forward + optimizer step executes with ZERO D2H readbacks.
            var xDev = eng.TensorMultiplyScalar(x, 1f);
            GpuLaunchProbe.Reset();
            var y2 = eng.TensorAdd(eng.TensorMatMul(xDev, w), b);
            Assert.True(GpuOptimizer.TryAdamWStep(w, gwDev, mw, vw, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 2));
            Assert.True(GpuOptimizer.TryAdamWStep(b, gbDev, mb, vb, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 2));
            Assert.Equal(0L, GpuLaunchProbe.ReadbackBytes);

            // Verify y2 (computed BEFORE step 2 weight update) and step-2 weights match CPU reference!
            float[] idxW = IdentitySparseIndexFloats(w.Length);
            float[] idxB = IdentitySparseIndexFloats(b.Length);
            var refW = (float[])wRaw.Clone(); var refMw = new float[w.Length]; var refVw = new float[w.Length];
            var refB = (float[])bRaw.Clone(); var refMb = new float[b.Length]; var refVb = new float[b.Length];
            SparseOptimizerReference.SparseAdamWUpdate(refW, refMw, refVw, idxW, gwRaw, w.Length, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
            SparseOptimizerReference.SparseAdamWUpdate(refB, refMb, refVb, idxB, gbRaw, b.Length, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
            var expectedY2 = _cpu.TensorAdd(_cpu.TensorMatMul(HostTensor(xRaw, 4, 8), HostTensor(refW, 8, 6)), HostTensor(refB, 4, 6));
            AssertClose(expectedY2.ToArray(), y2.ToArray(), 2e-4f, "Step2ForwardWithResidentWeightsAndBias");

            SparseOptimizerReference.SparseAdamWUpdate(refW, refMw, refVw, idxW, gwRaw, w.Length, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 2);
            SparseOptimizerReference.SparseAdamWUpdate(refB, refMb, refVb, idxB, gbRaw, b.Length, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 2);
            AssertClose(refW, w.ToArray(), 1e-5f, "Step2Weight");
            AssertClose(refB, b.ToArray(), 1e-5f, "Step2Bias");

            // 1. Host-clipped gradient freshness on a CPU tensor with a bound GPU buffer:
            // gClip has a valid device buffer, then FusedOptimizer.TryScaleHost mutates it on host -> TrySgdStep must re-upload scaled grad!
            var backend = gpu.GetBackend() ?? throw new InvalidOperationException("Expected GPU backend.");
            var pClip = HostTensor(new[] { 10f, 20f, 30f, 40f });
            var gClip = HostTensor(new[] { 4f, 8f, 12f, 16f });
            using var preClipBuf = backend.AllocateBuffer(new[] { 4f, 8f, 12f, 16f });
            gpu.BindResidentBuffer(gClip, preClipBuf, backend, isDeviceModified: false);
            Assert.NotNull(gClip.TryGetGpuBuffer());
            Assert.True(FusedOptimizer.TryScaleHost(gClip, 0.25)); // gClip on host is now [1, 2, 3, 4]; TryGetGpuBuffer stays non-null while _gpuBufferVersion becomes -1
            Assert.NotNull(gClip.TryGetGpuBuffer());
            Assert.Equal(-1, gClip._gpuBufferVersion);
            Assert.True(GpuOptimizer.TrySgdStep(pClip, gClip, 1.0f));
            AssertClose(new[] { 9f, 18f, 27f, 36f }, pClip.ToArray(), 1e-5f, "HostClippedGradientFreshness.CpuCached");

            // 2. Host-mutated gradient freshness on a device-produced tensor:
            var pGpuClip = HostTensor(new[] { 10f, 20f, 30f, 40f });
            var gGpuClip = eng.TensorMultiplyScalar(HostTensor(new[] { 4f, 8f, 12f, 16f }), 1f);
            Assert.NotNull(gGpuClip.TryGetGpuBuffer());
            var gSpan = gGpuClip.AsWritableSpan();
            for (int i = 0; i < gSpan.Length; i++) gSpan[i] *= 0.25f;
            gGpuClip.IncrementVersion();
            Assert.True(GpuOptimizer.TrySgdStep(pGpuClip, gGpuClip, 1.0f));
            AssertClose(new[] { 9f, 18f, 27f, 36f }, pGpuClip.ToArray(), 1e-5f, "HostClippedGradientFreshness.GpuProduced");
        });
    }

    [Fact]
    public void BindResidentBuffer_And_MockTryAccumulateSumOfSquares_EnforceSyncAndSingleDispose()
    {
        var prevEngine = AiDotNetEngine.Current;
        var prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");
        DirectGpuTensorEngine? mockEngine = null;
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "none");
            var directGpu = new DirectGpuEngine();
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", prevEnv);

            var state = new MockBackendState();
            var backend = MockDirectGpuBackend.Create(state);
            var backendField = typeof(DirectGpuEngine).GetField("_backend", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("Field _backend not found.");
            var availField = typeof(DirectGpuEngine).GetField("_isAvailable", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("Field _isAvailable not found.");
            backendField.SetValue(directGpu, backend);
            availField.SetValue(directGpu, true);

            mockEngine = new DirectGpuTensorEngine(directGpu);
            AiDotNetEngine.Current = mockEngine;

            // 1. BindResidentBuffer(..., isDeviceModified: false) avoids redundant D2H download when host already matches.
            var hostTensor = HostTensor(new[] { 1f, 2f, 3f, 4f });
            var devBuf = new MockGpuBuffer(new[] { 1f, 2f, 3f, 4f });
            mockEngine.BindResidentBuffer(hostTensor, devBuf, backend, isDeviceModified: false);
            Assert.False(hostTensor.HasPendingGpuData);
            Assert.Equal(hostTensor.GpuCacheVersion, hostTensor._gpuBufferVersion);
            Assert.Equal(new[] { 1f, 2f, 3f, 4f }, hostTensor.AsSpan().ToArray());
            Assert.Equal(0, state.DownloadBufferCalls);

            // Once modified on device, ResidentHostMirror downloads in place via 2-arg DownloadBuffer(buf, dst).
            devBuf.Data[0] = 42f;
            hostTensor.MarkModified(GpuSyncPoint.CreateComplete());
            hostTensor._gpuBufferVersion = hostTensor.GpuCacheVersion;
            Assert.True(hostTensor.HasPendingGpuData);
            Assert.Equal(new[] { 42f, 2f, 3f, 4f }, hostTensor.AsSpan().ToArray());
            Assert.Equal(1, state.DownloadBufferCalls);

            // 2. TryAccumulateSumOfSquares rejects non-scalar accumulator and disposes FromGpuBuffer(ownsBuffer: true) exactly once!
            Tensor<float>? invalidAcc = HostTensor(new[] { 1f, 2f });
            Assert.Throws<ArgumentException>(() => mockEngine.TryAccumulateSumOfSquares(HostTensor(new[] { 3f, 4f }), ref invalidAcc));

            var ownedAccBuffer = new MockGpuBuffer(new[] { 10f });
            Tensor<float>? acc = Tensor<float>.FromGpuBuffer(backend, ownedAccBuffer, new[] { 1 }, ownsBuffer: true);
            Assert.True(mockEngine.TryAccumulateSumOfSquares(HostTensor(new[] { 3f, 4f }), ref acc)); // 10 + (9 + 16) = 35
            Assert.Equal(1, ownedAccBuffer.DisposeCount); // Must be 1, NOT 2 (double-free bug fix!)
            Assert.True(mockEngine.TryAccumulateSumOfSquares(HostTensor(new[] { 2f, 1f }), ref acc)); // 35 + (4 + 1) = 40
            if (acc is null) throw new InvalidOperationException("Accumulator became null.");
            AssertClose(new[] { 40f }, acc.ToArray(), 1e-5f, "MockTryAccumulateSumOfSquares");
        }
        finally
        {
            AiDotNetEngine.Current = prevEngine;
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", prevEnv);
            mockEngine?.Dispose();
        }
    }

    [SkippableFact]
    public void BackendUploadBufferInPlace_DownloadBuffer_And_RealGpuAccumulateSumOfSquares_VerifyBoundsAndPointerStability()
    {
        var gpu = RequireGpuEngine();
        var backend = gpu.GetBackend() ?? throw new InvalidOperationException("Expected active GPU backend.");

        using (var buf = backend.AllocateBuffer(new[] { 1f, 2f, 3f, 4f }))
        {
            IntPtr originalHandle = buf.Handle;
            backend.UploadBufferInPlace(new[] { 10f, 20f, 30f, 40f }, buf);
            Assert.Equal(originalHandle, buf.Handle);

            // Oversized destination buffer must fill [0..Size) and preserve trailing sentinel values!
            var oversizedDest = new[] { -1f, -1f, -1f, -1f, 999f, 888f };
            backend.DownloadBuffer(buf, oversizedDest);
            AssertClose(new[] { 10f, 20f, 30f, 40f, 999f, 888f }, oversizedDest, 1e-6f, "DownloadBuffer.OversizedDest");

            // Bounds and null guards
            ThrowNullChecksWithoutForgivingOperator(backend, buf);
            Assert.Throws<ArgumentException>(() => backend.UploadBufferInPlace(new float[5], buf));
            Assert.Throws<ArgumentException>(() => backend.DownloadBuffer(buf, new float[3]));
        }

        // Real GPU TryAccumulateSumOfSquares starting from UploadToGpu / FromGpuBuffer(..., ownsBuffer: true)
        var seedBuf = backend.AllocateBuffer(new[] { 5f });
        Tensor<float>? acc = Tensor<float>.FromGpuBuffer(backend, seedBuf, new[] { 1 }, ownsBuffer: true);
        Assert.True(gpu.TryAccumulateSumOfSquares(HostTensor(new[] { 3f, 4f }), ref acc));   // 5 + 25 = 30
        Assert.True(gpu.TryAccumulateSumOfSquares(HostTensor(new[] { 1f, 2f, 2f }), ref acc)); // 30 + 9 = 39
        if (acc is null) throw new InvalidOperationException("GPU accumulator became null.");
        AssertClose(new[] { 39f }, acc.ToArray(), 1e-4f, "RealGpuTryAccumulateSumOfSquares");
        acc.Dispose();
    }

    private static void ThrowNullChecksWithoutForgivingOperator(IDirectGpuBackend backend, IGpuBuffer buf)
    {
        MethodInfo upload = typeof(IDirectGpuBackend).GetMethod(nameof(IDirectGpuBackend.UploadBufferInPlace))
            ?? throw new InvalidOperationException("UploadBufferInPlace not found.");
        var ex1 = Assert.Throws<TargetInvocationException>(() => upload.Invoke(backend, new object?[] { null, buf }));
        Assert.IsType<ArgumentNullException>(ex1.InnerException);

        var ex2 = Assert.Throws<TargetInvocationException>(() => upload.Invoke(backend, new object?[] { new float[1], null }));
        Assert.IsType<ArgumentNullException>(ex2.InnerException);

        MethodInfo download = typeof(IDirectGpuBackend).GetMethod(
            nameof(IDirectGpuBackend.DownloadBuffer), new[] { typeof(IGpuBuffer), typeof(float[]) })
            ?? throw new InvalidOperationException("DownloadBuffer(IGpuBuffer, float[]) not found.");
        var ex3 = Assert.Throws<TargetInvocationException>(() => download.Invoke(backend, new object?[] { null, new float[1] }));
        Assert.IsType<ArgumentNullException>(ex3.InnerException);

        var ex4 = Assert.Throws<TargetInvocationException>(() => download.Invoke(backend, new object?[] { buf, null }));
        Assert.IsType<ArgumentNullException>(ex4.InnerException);
    }

    [SkippableFact]
    public void CudaMultiContextAndMultiDevicePinnedStaging_SwitchesContextsAndDisposesCleanly()
    {
        var gpu = RequireGpuEngine();
        if (gpu.GetBackend() is not CudaBackend)
            return; // Active lane is a non-CUDA backend (e.g. OpenCL/POCL); runs with 0 skips.

        int deviceCount = 1;
        if (CuBlasNative.cuInit(0) == CudaResult.Success && CuBlasNative.cuDeviceGetCount(out int detected) == CudaResult.Success && detected > 0)
            deviceCount = detected;

        int secondOrdinal = deviceCount >= 2 ? 1 : 0;
        var backendA = new CudaBackend(0);
        using var backendB = new CudaBackend(secondOrdinal);

        try
        {
            using (var checkBuf = backendA.AllocateBuffer(4))
                ThrowNullChecksWithoutForgivingOperator(backendA, checkBuf);

            float[] srcA = MakeRandomFloats(7001, 2048, 2.0);
            float[] srcB = MakeRandomFloats(7002, 2048, 3.0);

            var bufA = backendA.AllocateBuffer(srcA.Length);
            var bufB = backendB.AllocateBuffer(srcB.Length);
            try
            {
                // Interleave staged pinned H2D uploads across CUDA contexts on the same host thread!
                backendA.UploadBufferInPlace(srcA, bufA);
                backendB.UploadBufferInPlace(srcB, bufB);
                backendA.Scale(bufA, bufA, 2.0f, srcA.Length);
                backendB.Scale(bufB, bufB, 0.5f, srcB.Length);

                var dstA = new float[srcA.Length];
                var dstB = new float[srcB.Length];
                backendA.DownloadBuffer(bufA, dstA);
                backendB.DownloadBuffer(bufB, dstB);

                var expA = new float[srcA.Length];
                var expB = new float[srcB.Length];
                for (int i = 0; i < srcA.Length; i++)
                {
                    expA[i] = srcA[i] * 2.0f;
                    expB[i] = srcB[i] * 0.5f;
                }

                AssertClose(expA, dstA, 1e-5f, $"CudaMultiContext(devCount={deviceCount}).A");
                AssertClose(expB, dstB, 1e-5f, $"CudaMultiContext(devCount={deviceCount}).B");
            }
            finally
            {
                bufA.Dispose();
                bufB.Dispose();
            }

            // Dispose backendA from a fresh OS thread (where current CUDA context is guaranteed IntPtr.Zero) while backendB stays live!
            Exception? disposeError = null;
            var disposeThread = new System.Threading.Thread(() =>
            {
                try { backendA.Dispose(); }
                catch (Exception ex) { disposeError = ex; }
            });
            disposeThread.Start();
            disposeThread.Join();
            Assert.Null(disposeError);
            Assert.False(backendA.IsAvailable);
            Assert.Throws<GpuCapabilityUnavailableException>(() => backendA.AllocateBuffer(4));

            // Verify backendB remains healthy after cross-context teardown of backendA.
            using var postBufB = backendB.AllocateBuffer(new[] { 4f, 6f, 8f, 10f });
            Assert.Throws<ObjectDisposedException>(() => backendA.UploadBufferInPlace(new float[4], postBufB));
            Assert.Throws<ObjectDisposedException>(() => backendA.DownloadBuffer(postBufB, new float[4]));
            var postDstB = new float[4];
            backendB.DownloadBuffer(postBufB, postDstB);
            AssertClose(new[] { 4f, 6f, 8f, 10f }, postDstB, 1e-6f, "CudaMultiContext.PostDisposeB");
        }
        finally
        {
            backendA.Dispose();
        }
    }

    [Fact]
    public void TensorScopeSuspend_IsIdempotent_AndNeverLeaksSuspendedDepthAcrossNestedScope()
    {
        Assert.False(TensorScope.HasActiveOrSuspendedScope);

        using (var outer = TensorScope.Begin())
        {
            Assert.True(TensorScope.HasActiveOrSuspendedScope);
            var suspend = TensorScope.Suspend();
            Assert.Null(TensorScope.Current);
            Assert.True(TensorScope.HasActiveOrSuspendedScope);

            using (var inner = TensorScope.Begin())
            {
                Assert.Same(inner, TensorScope.Current);
                Assert.True(TensorScope.HasActiveOrSuspendedScope);
            }

            // Resume suspended outer scope cleanly; repeated Dispose calls must be safe idempotent no-ops!
            suspend.Dispose();
            Assert.Same(outer, TensorScope.Current);
            suspend.Dispose();
            Assert.Same(outer, TensorScope.Current);
        }

        Assert.False(TensorScope.HasActiveOrSuspendedScope);
        Assert.Null(TensorScope.Current);

        // Disposing SuspendHandle while an inner scope is still active must still decrement s_suspendedDepth without corrupting Current!
        var outerUnclosed = TensorScope.Begin();
        var leakCheckSuspend = TensorScope.Suspend();
        var innerActive = TensorScope.Begin();
        leakCheckSuspend.Dispose(); // innerActive is still Current, so s_suspendedDepth decrements without overwriting s_current
        leakCheckSuspend.Dispose();
        innerActive.Dispose();
        outerUnclosed.Dispose();
        Assert.False(TensorScope.HasActiveOrSuspendedScope);
        Assert.Null(TensorScope.Current);

        // Verify double-dispose of SuspendHandle cannot resurrect an already-disposed scope.
        var scope = TensorScope.Begin();
        var handle = TensorScope.Suspend();
        handle.Dispose();
        scope.Dispose();
        handle.Dispose();
        Assert.Null(TensorScope.Current);
        Assert.False(TensorScope.HasActiveOrSuspendedScope);
    }

    [SkippableFact]
    public void RebindablePlanCache_WithDeadIntermediateBranch_ReplaysAndMatchesCpuGradients()
    {
        var gpu = RequireGpuEngine();
        float[] xRaw = MakeRandomFloats(8001, 16 * 16, 1.0);
        float[] wRaw = MakeRandomFloats(8002, 16 * 8, 0.5);

        Tensor<float> RunTwoSteps(IEngine engine)
        {
            Tensor<float>? finalGrad = null;
            WithEngine(engine, () =>
            {
                var x = HostTensor(xRaw, 16, 16);
                var w = HostTensor(wRaw, 16, 8);
                for (int step = 0; step < 2; step++)
                {
                    using var tape = new GradientTape<float>();
                    var deadBranch = engine.TensorMultiplyScalar(engine.TensorAdd(x, x), 3.0f);
                    Assert.Equal(256, deadBranch.Length);
                    var h = engine.TensorMatMul(x, w);
                    var loss = engine.ReduceSum(engine.TensorMultiply(h, h), new[] { 0, 1 }, keepDims: false);
                    var grads = tape.ComputeGradients(loss, new[] { w });
                    finalGrad = grads[w];
                }
            });
            return finalGrad ?? throw new InvalidOperationException("Expected gradient for w.");
        }

        var cpuGrad = RunTwoSteps(_cpu);
        var gpuGrad = RunTwoSteps(gpu);
        AssertClose(cpuGrad.ToArray(), gpuGrad.ToArray(), 2e-4f, "RebindablePlanCache.DeadBranchReplay");
    }

    [Fact]
    public void FusedOptimizerHostSimdFastPaths_MatchDoublePrecisionReference_WithZeroSmallTensorAllocations()
    {
        // 1. Sub-million float tensor with large values (1e20f^2 = 1e40 > float.MaxValue, proving fp64 accumulation!)
        var largeVals = new[] { 1e20f, -1e20f, 3e19f, -4e19f };
        var largeFloat = HostTensor(largeVals);
        Assert.True(FusedOptimizer.TrySumOfSquaresHost(largeFloat, out double largeSq));
        double expectedLargeSq = 0.0;
        for (int i = 0; i < largeVals.Length; i++) expectedLargeSq += (double)largeVals[i] * largeVals[i];
        Assert.True(largeSq > (double)float.MaxValue);
        Assert.True(Math.Abs(expectedLargeSq - largeSq) / expectedLargeSq < 1e-12);

        // 2. Non-finite propagation (NaN and +Infinity)
        var nanTensor = HostTensor(new[] { 1f, 2f, float.NaN, 4f });
        Assert.True(FusedOptimizer.TrySumOfSquaresHost(nanTensor, out double nanSq));
        Assert.True(double.IsNaN(nanSq));

        var infTensor = HostTensor(new[] { 1f, float.PositiveInfinity, 3f });
        Assert.True(FusedOptimizer.TrySumOfSquaresHost(infTensor, out double infSq));
        Assert.True(double.IsPositiveInfinity(infSq));

        // 3. Sub-million multi-chunk (70,003 elements) and >1M parallel threshold (1,100,017 elements) parity for float & double
        foreach (int len in new[] { 4096, 70_003, 1_100_017 })
        {
            float[] fData = MakeRandomFloats(9000 + len, len, 2.5);
            var fTensor = HostTensor(fData);
            double expectedF = 0.0;
            for (int i = 0; i < len; i++) expectedF += (double)fData[i] * fData[i];

            Assert.True(FusedOptimizer.TrySumOfSquaresHost(fTensor, out double actualF));
            Assert.True(Math.Abs(expectedF - actualF) <= Math.Max(1e-9, Math.Abs(expectedF) * 1e-11),
                $"Float SumOfSquares mismatch at len={len}: expected {expectedF:R}, got {actualF:R}");

            var dData = new double[len];
            double expectedD = 0.0;
            for (int i = 0; i < len; i++)
            {
                dData[i] = fData[i] * 1.125;
                expectedD += dData[i] * dData[i];
            }

            var dTensor = new Tensor<double>(new[] { len }, new Vector<double>((double[])dData.Clone()));
            Assert.True(FusedOptimizer.TrySumOfSquaresHost(dTensor, out double actualD));
            Assert.True(Math.Abs(expectedD - actualD) <= Math.Max(1e-9, Math.Abs(expectedD) * 1e-12),
                $"Double SumOfSquares mismatch at len={len}: expected {expectedD:R}, got {actualD:R}");

            int dVerBefore = dTensor.Version;
            Assert.True(FusedOptimizer.TryScaleHost(dTensor, 0.375));
            Assert.True(dTensor.Version > dVerBefore);
            var dScaled = dTensor.AsSpan();
            Assert.True(Math.Abs(dData[0] * 0.375 - dScaled[0]) < 1e-12);
            Assert.True(Math.Abs(dData[len - 1] * 0.375 - dScaled[len - 1]) < 1e-12);
        }

#if NETCOREAPP
        // 4. Warm up JIT + GC.GetAllocatedBytesForCurrentThread, then verify 32 hot-loop passes allocate ZERO managed bytes!
        var warmF = HostTensor(MakeRandomFloats(9500, 8192, 1.0));
        _ = MeasureHotPathAllocations(warmF, 64);
        long allocated = MeasureHotPathAllocations(warmF, 32);
        Assert.Equal(0L, allocated);
#endif
    }

#if NETCOREAPP
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static long MeasureHotPathAllocations(Tensor<float> tensor, int iterations)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < iterations; i++)
        {
            if (!FusedOptimizer.TrySumOfSquaresHost(tensor, out _))
                throw new InvalidOperationException("TrySumOfSquaresHost failed.");
            if (!FusedOptimizer.TryScaleHost(tensor, 1.0))
                throw new InvalidOperationException("TryScaleHost failed.");
        }
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }
#endif

    [Fact]
    public void MockBackend_HostAllocatedDenseCompressedAndSparseSteps_AndCpuPlanCache_CoverHostPromotionPaths()
    {
        // 1. CPU RebindablePlanCache replay with dead intermediate branch (covers RebindablePlanCache on GPU-less CI).
        WithEngine(_cpu, () =>
        {
            var x = HostTensor(MakeRandomFloats(9601, 8 * 8, 1.0), 8, 8);
            var w = HostTensor(MakeRandomFloats(9602, 8 * 4, 0.5), 8, 4);
            for (int step = 0; step < 2; step++)
            {
                using var tape = new GradientTape<float>();
                var dead = _cpu.TensorAdd(x, x);
                Assert.Equal(64, dead.Length);
                var h = _cpu.TensorMatMul(x, w);
                var loss = _cpu.ReduceSum(_cpu.TensorMultiply(h, h), new[] { 0, 1 }, keepDims: false);
                var grads = tape.ComputeGradients(loss, new[] { w });
                Assert.Equal(32, grads[w].Length);
            }
        });

        // 2. MockDirectGpuBackend host-allocated promotion across all dense, compressed, and sparse GpuOptimizer entry points!
        var prevEngine = AiDotNetEngine.Current;
        var prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");
        DirectGpuTensorEngine? mockEngine = null;
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "none");
            var directGpu = new DirectGpuEngine();
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", prevEnv);

            var state = new MockBackendState();
            var backend = MockDirectGpuBackend.Create(state);
            var backendField = typeof(DirectGpuEngine).GetField("_backend", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("Field _backend not found.");
            var availField = typeof(DirectGpuEngine).GetField("_isAvailable", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("Field _isAvailable not found.");
            backendField.SetValue(directGpu, backend);
            availField.SetValue(directGpu, true);

            mockEngine = new DirectGpuTensorEngine(directGpu);
            AiDotNetEngine.Current = mockEngine;

            const int n = 8;
            var gHost = HostTensor(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
            using var createdState = GpuOptimizer.CreateStateTensor(new[] { n });
            Assert.True(createdState._ownsGpuBuffer);

            Tensor<float> FreshHost() => HostTensor(new[] { 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f });

            var mShared = FreshHost();
            var vShared = FreshHost();
            var pShared = FreshHost();
            Assert.True(GpuOptimizer.TryAdamStep(pShared, gHost, mShared, vShared, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(mockEngine.IsDeviceAuthoritative(pShared));
            // Mutate mShared on host to exercise the in-place UploadBufferInPlace state-refresh branch in TryResolveStateBuffer!
            mShared.AsWritableSpan()[0] = 0.25f;
            mShared.IncrementVersion();
            Assert.True(GpuOptimizer.TryAdamWStep(pShared, gHost, mShared, vShared, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 2));
            Assert.NotNull(GpuOptimizer.TryDownload(pShared));
            // Exercise ResidentHostMirror's in-place 2-arg DownloadBuffer fast path on MockDirectGpuBackend:
            Assert.Equal(n, pShared.ToArray().Length);

            // Multi-tensor SGD refuses host-only or host-mutated (stale GPU) tensors, and commits resident updates cleanly:
            var pMultiA = mockEngine.UploadToGpu(FreshHost(), GpuTensorRole.Weight);
            var pMultiB = mockEngine.UploadToGpu(FreshHost(), GpuTensorRole.Weight);
            var gMultiA = mockEngine.UploadToGpu(gHost, GpuTensorRole.Gradient);
            var gMultiB = mockEngine.UploadToGpu(gHost, GpuTensorRole.Gradient);
            Assert.False(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMultiA, FreshHost() }, new[] { gMultiA, gMultiB }, 0.01f));
            Assert.True(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMultiA, pMultiB }, new[] { gMultiA, gMultiB }, 0.01f));
            Assert.Equal(n, pMultiA.ToArray().Length);
            Assert.True(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMultiA, pMultiB }, new[] { gMultiA, gMultiB }, 0.01f));
            Assert.Equal(n, pMultiA.ToArray().Length);
            gMultiB.AsWritableSpan()[0] = 0.5f;
            gMultiB.IncrementVersion();
            Assert.False(GpuOptimizer.TrySgdStepMultiTensor(new[] { pMultiA, pMultiB }, new[] { gMultiA, gMultiB }, 0.01f));

            // Flat ParameterBuffer staging via AcquireParameterBuffer / TryStageFlatParameter on per-tensor step:
            var flatBuf = new ParameterBuffer<float>(new[] { new[] { n }, new[] { n } });
            var flatViews = flatBuf.CreateAllViews();
            flatBuf.CopyFrom(new[] { FreshHost(), FreshHost() });
            Assert.True(GpuOptimizer.TrySgdStep(flatViews[0], gHost, 0.01f));
            Assert.True(GpuOptimizer.TrySgdStep(flatViews[1], gHost, 0.01f));
            Assert.Equal(n, flatViews[0].ToArray().Length);
            Assert.Equal(n, flatViews[1].ToArray().Length);

            Assert.True(GpuOptimizer.TrySgdStep(FreshHost(), gHost, 0.01f));
            Assert.True(GpuOptimizer.TryProximalL1Step(FreshHost(), gHost, 0.01f, 0.05f));
            Assert.True(GpuOptimizer.TrySgdMomentumStep(FreshHost(), gHost, FreshHost(), 0.01f, 0.9f, 0.01f));
            Assert.True(GpuOptimizer.TryRmspropStep(FreshHost(), gHost, FreshHost(), 0.01f, 0.99f, 1e-8f, 0.01f));
            Assert.True(GpuOptimizer.TryAdagradStep(FreshHost(), gHost, FreshHost(), 0.01f, 1e-8f, 0.01f));
            Assert.True(GpuOptimizer.TryNagStep(FreshHost(), gHost, FreshHost(), 0.01f, 0.9f, 0.01f));
            Assert.True(GpuOptimizer.TryLarsStep(FreshHost(), gHost, FreshHost(), 0.01f, 0.9f, 0.01f, 0.001f));
            Assert.True(GpuOptimizer.TryLambStep(FreshHost(), gHost, FreshHost(), FreshHost(), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryAdadeltaStep(FreshHost(), gHost, FreshHost(), FreshHost(), 0.95f, 1e-6f, 0.01f));
            Assert.True(GpuOptimizer.TryAmsgradStep(FreshHost(), gHost, FreshHost(), FreshHost(), FreshHost(), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryAdamaxStep(FreshHost(), gHost, FreshHost(), FreshHost(), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryLionStep(FreshHost(), gHost, FreshHost(), 0.01f, 0.9f, 0.99f, 0.01f));
            Assert.True(GpuOptimizer.TryNadamStep(FreshHost(), gHost, FreshHost(), FreshHost(), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryFtrlStep(FreshHost(), gHost, FreshHost(), FreshHost(), 0.1f, 0.01f, 0.01f, 1f));

            // Compressed BF16 + 8-bit steps with host-allocated param & grad on mock backend
            if (GpuOptimizer.TryAllocBf16MomentState(n, out var mBf, out var vBf))
            {
                try { Assert.True(GpuOptimizer.TryAdamBf16Step(FreshHost(), gHost, mBf, vBf, 0.01f, 0.9f, 0.99f, 1e-8f, 0.01f, 1, true)); }
                finally { GpuOptimizer.FreeGpuBuffer(mBf); GpuOptimizer.FreeGpuBuffer(vBf); }
            }

            if (GpuOptimizer.TryAllocAdam8BitState(n, 64, out var mQ, out var vQ, out var mS, out var vS))
            {
                try
                {
                    Assert.True(GpuOptimizer.TryAdam8BitStep(FreshHost(), gHost, mQ, vQ, mS, vS, 0.01f, 0.9f, 0.99f, 1e-8f, 0.1f, 0.01f, 64));
                    Assert.True(GpuOptimizer.TryAdam8BitStep(FreshHost(), gHost, mQ, vQ, mS, vS, 0.01f, 0.9f, 0.99f, 1e-8f, 0.1f, 0.01f, 64, 0.05f, true));
                }
                finally
                {
                    GpuOptimizer.FreeGpuBuffer(mQ); GpuOptimizer.FreeGpuBuffer(vQ);
                    GpuOptimizer.FreeGpuBuffer(mS); GpuOptimizer.FreeGpuBuffer(vS);
                }
            }

            // Sparse steps with host-allocated indices, values, states, and params on mock backend
            var spIdx = new Tensor<int>(new[] { 3 }, new Vector<int>(new[] { 0, 3, 6 }));
            var spVal = HostTensor(new[] { 0.5f, -0.25f, 0.75f });
            Assert.True(GpuOptimizer.TryAdamStepSparse(FreshHost(), FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryAdamWStepSparse(FreshHost(), FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TrySgdStepSparse(FreshHost(), spIdx, spVal, 3, 0.01f, 0.01f));
            Assert.True(GpuOptimizer.TrySgdMomentumStepSparse(FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.01f));
            Assert.True(GpuOptimizer.TryRmspropStepSparse(FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.99f, 1e-8f, 0.01f));
            Assert.True(GpuOptimizer.TryAdagradStepSparse(FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 1e-8f, 0.01f));
            Assert.True(GpuOptimizer.TryNagStepSparse(FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.01f));
            Assert.True(GpuOptimizer.TryAdadeltaStepSparse(FreshHost(), FreshHost(), FreshHost(), spIdx, spVal, 3, 0.95f, 1e-6f, 0.01f));
            Assert.True(GpuOptimizer.TryAmsgradStepSparse(FreshHost(), FreshHost(), FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryAdamaxStepSparse(FreshHost(), FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryLionStepSparse(FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.99f, 0.01f));
            Assert.True(GpuOptimizer.TryNadamStepSparse(FreshHost(), FreshHost(), FreshHost(), spIdx, spVal, 3, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1));
            Assert.True(GpuOptimizer.TryFtrlStepSparse(FreshHost(), FreshHost(), FreshHost(), spIdx, spVal, 3, 0.1f, 0.01f, 0.01f, 1f));
            Assert.True(GpuOptimizer.TryProximalL1StepSparse(FreshHost(), spIdx, spVal, 3, 0.01f, 0.05f));
        }
        finally
        {
            AiDotNetEngine.Current = prevEngine;
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", prevEnv);
            mockEngine?.Dispose();
        }
    }
}
