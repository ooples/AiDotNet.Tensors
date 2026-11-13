using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Half mean-type reductions, losses, norms and their backward on <see cref="DirectGpuTensorEngine"/> over 70,000+
/// elements. GPU buffers are float32: Half operands are widened on upload, the kernels divide by an int count in
/// float32, and the result is rounded to Half once on download, so no count or 1/N factor is ever held in Half on the
/// device. Ops whose GPU override takes only float (BatchNorm, MSE, BCE-with-logits, ReduceMean, ReduceVariance,
/// TensorVarMean, and the device-scaled loss/mean/pool backward) fall back to CpuEngine; those must match the CPU
/// engine bit for bit. Ops with a float32 kernel used for Half (L1, Huber, KL-div, AvgPool1D, LayerNorm, GroupNorm,
/// ReduceStd) must be within one Half ulp of the double reference.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class LowPrecisionGpuReductionTests
{
    private const int N = 70_000;

    private static Half[] HalfData(int n, Func<int, double> f)
    {
        var data = new Half[n];
        for (int i = 0; i < n; i++) data[i] = (Half)f(i);
        return data;
    }

    private static readonly Half[] P = HalfData(N, i => 1.0 + 0.5 * Math.Sin(0.37 * i));
    private static readonly Half[] Tg = HalfData(N, i => 0.5 + 0.25 * Math.Cos(0.11 * i));

    private static DirectGpuTensorEngine? CreateGpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (InvalidOperationException) { return null; }
        catch (NotSupportedException) { return null; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); return null; }
        return gpu;
    }

    private static Tensor<Half> T(Half[] data, params int[] shape) => new Tensor<Half>(data, shape);

    private static void AssertHalf(double expected, Half actual, string what)
    {
        double a = (double)actual;
        double ulp = Math.Max(Math.Pow(2, -24), expected == 0 ? 0 : Math.Pow(2, Math.Floor(Math.Log(Math.Abs(expected), 2)) - 10));
        Assert.True(!double.IsNaN(a) && Math.Abs(a - expected) <= ulp, $"{what}: got {a}, expected {expected} (one ulp = {ulp})");
    }

    private static void AssertBitEqual(Half[] cpu, Half[] gpu, string what)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(cpu[i].Equals(gpu[i]), $"{what}[{i}]: GPU engine {(double)gpu[i]} differs from CPU engine {(double)cpu[i]}");
    }

    [SkippableFact]
    public void FloatOnlyGpuOverrides_FallBackToCpu_ForHalf()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        var cpu = new CpuEngine();
        var p = T(P, N);
        var t = T(Tg, N);

        AssertBitEqual(cpu.TensorMSELoss(p, t).ToArray(), gpu.TensorMSELoss(p, t).ToArray(), "TensorMSELoss");
        AssertBitEqual(cpu.TensorBCEWithLogitsLoss(p, t).ToArray(), gpu.TensorBCEWithLogitsLoss(p, t).ToArray(), "TensorBCEWithLogitsLoss");
        AssertBitEqual(cpu.ReduceMean(p, new[] { 0 }, false).ToArray(), gpu.ReduceMean(p, new[] { 0 }, false).ToArray(), "ReduceMean");
        AssertBitEqual(cpu.ReduceVariance(p, new[] { 0 }, false).ToArray(), ((IEngine)gpu).ReduceVariance(p, new[] { 0 }, false).ToArray(), "ReduceVariance");
        var (cv, cm) = cpu.TensorVarMean(p);
        var (gv, gm) = gpu.TensorVarMean(p);
        AssertBitEqual(cv.ToArray(), gv.ToArray(), "TensorVarMean var");
        AssertBitEqual(cm.ToArray(), gm.ToArray(), "TensorVarMean mean");

        var bnIn = T(P, 1, 1, 280, 250);
        var gamma = T(HalfData(1, _ => 1.0), 1);
        var beta = T(HalfData(1, _ => 0.0), 1);
        var cy = cpu.BatchNorm(bnIn, gamma, beta, 1e-5, out var cMean, out var cVar);
        var gy = gpu.BatchNorm(bnIn, gamma, beta, 1e-5, out var gMean, out var gVar);
        AssertBitEqual(cy.ToArray(), gy.ToArray(), "BatchNorm output");
        AssertBitEqual(cMean.ToArray(), gMean.ToArray(), "BatchNorm mean");
        AssertBitEqual(cVar.ToArray(), gVar.ToArray(), "BatchNorm variance");
    }

    [SkippableFact]
    public void DeviceScaledBackward_FallsBackToCpu_ForHalf()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        var cpu = new CpuEngine();
        var g = T(HalfData(1, _ => 7.0), 1);

        Half[] Run(IEngine engine, BackwardFunction<Half> backward, Tensor<Half> upstream, Tensor<Half>[] inputs, object[] state)
        {
            var grads = new Dictionary<Tensor<Half>, Tensor<Half>>();
            backward(upstream, inputs, upstream, state, engine, grads);
            return grads[inputs[0]].ToArray();
        }

        var pair = new[] { T(P, N), T(Tg, N) };
        AssertBitEqual(Run(cpu, BackwardFunctions<Half>.MSELossBackward, g, pair, Array.Empty<object>()),
            Run(gpu, BackwardFunctions<Half>.MSELossBackward, g, pair, Array.Empty<object>()), "MSELossBackward");
        var single = new[] { T(P, N) };
        AssertBitEqual(Run(cpu, BackwardFunctions<Half>.MeanBackward, g, single, Array.Empty<object>()),
            Run(gpu, BackwardFunctions<Half>.MeanBackward, g, single, Array.Empty<object>()), "MeanBackward");
        var pool = new[] { T(P, 1, 1, 280, 250) };
        var g4 = T(HalfData(1, _ => 7.0), 1, 1, 1, 1);
        AssertBitEqual(Run(cpu, BackwardFunctions<Half>.AdaptiveAvgPool2DBackward, g4, pool, new object[] { 1, 1 }),
            Run(gpu, BackwardFunctions<Half>.AdaptiveAvgPool2DBackward, g4, pool, new object[] { 1, 1 }), "AdaptiveAvgPool2DBackward");
    }

    [SkippableFact]
    public void Float32KernelsUsedForHalf_AreWithinOneUlp()
    {
        using var gpu = CreateGpu();
        Skip.If(gpu is null, "No GPU available");
        double l1 = 0, huber = 0, kl = 0, sum = 0;
        for (int i = 0; i < N; i++)
        {
            double p = (double)P[i], t = (double)Tg[i], d = p - t;
            l1 += Math.Abs(d);
            huber += Math.Abs(d) <= 1.0 ? 0.5 * d * d : Math.Abs(d) - 0.5;
            kl += t * (Math.Log(t) - p);
            sum += p;
        }
        double mean = sum / N, sq = 0;
        for (int i = 0; i < N; i++) { double d = (double)P[i] - mean; sq += d * d; }

        var pt = T(P, N);
        var tt = T(Tg, N);
        AssertHalf(l1 / N, gpu.TensorL1Loss(pt, tt).ToArray()[0], "TensorL1Loss");
        AssertHalf(huber / N, gpu.TensorHuberLoss(pt, tt, 1.0).ToArray()[0], "TensorHuberLoss");
        AssertHalf(kl / N, gpu.TensorKLDivLoss(pt, tt).ToArray()[0], "TensorKLDivLoss");
        AssertHalf(mean, gpu.TensorAvgPool1D(T(P, 1, 1, N), N, N).ToArray()[0], "TensorAvgPool1D");
        AssertHalf(Math.Sqrt(sq / N), gpu.ReduceStd(pt, new[] { 0 }, false).ToArray()[0], "ReduceStd");

        var ones = T(HalfData(N, _ => 1.0), N);
        var zeros = T(HalfData(N, _ => 0.0), N);
        gpu.LayerNorm(T(P, 1, N), ones, zeros, 1e-5, out var lnMean, out var lnVar);
        AssertHalf(mean, lnMean.ToArray()[0], "LayerNorm mean");
        AssertHalf(sq / N, lnVar.ToArray()[0], "LayerNorm variance");

        var gnGamma = T(HalfData(2, _ => 1.0), 2);
        var gnBeta = T(HalfData(2, _ => 0.0), 2);
        gpu.GroupNorm(T(P, 1, 2, N / 2), 1, gnGamma, gnBeta, 1e-5, out var gnMean, out var gnVar);
        AssertHalf(mean, gnMean.ToArray()[0], "GroupNorm mean");
        AssertHalf(sq / N, gnVar.ToArray()[0], "GroupNorm variance");
    }
}
