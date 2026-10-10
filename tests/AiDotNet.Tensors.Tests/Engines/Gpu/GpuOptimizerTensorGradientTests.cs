// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// <see cref="OptimizerBase.Step(IReadOnlyDictionary{Tensor{float}, Tensor{float}})"/> on GPU-resident parameters and
/// gradients: an optimizer with a device kernel for its configuration updates the parameter and its state on the
/// device (no host step), one without runs a host step and writes the result back, and either way the result
/// matches the same optimizer on the CPU.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GpuOptimizerTensorGradientTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IDisposable
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public GpuOptimizerTensorGradientTests(DirectGpuTensorEngineTestFixture fixture)
    {
        _fixture = fixture;
        if (fixture.IsAvailable) AiDotNetEngine.Current = fixture.Engine!;
    }

    public void Dispose() => AiDotNetEngine.Current = _prior;

    public enum Config
    {
        Sgd, SgdMomentum, Adam, AmsGrad, AdamW, NAdam, Adamax, Adagrad, RmsProp, AdaDelta, Lion, Ftrl,
        // No device kernel for these configurations: a host step, written back.
        SgdNesterov, RAdam, Lamb, RmsPropCentered,
    }

    private static bool HasDeviceKernel(Config c) => c < Config.SgdNesterov;

    private static (OptimizerBase Optimizer, Dictionary<string, double> Options) Create(Config c)
    {
        var o = new Dictionary<string, double> { ["lr"] = 1e-2, ["weight_decay"] = 0.01, ["maximize"] = 1.0 };
        OptimizerBase optimizer;
        switch (c)
        {
            case Config.Sgd: optimizer = new SgdOptimizer(); break;
            case Config.SgdMomentum: optimizer = new SgdOptimizer(); o["momentum"] = 0.9; break;
            case Config.SgdNesterov: optimizer = new SgdOptimizer(); o["momentum"] = 0.9; o["nesterov"] = 1.0; break;
            case Config.Adam: optimizer = new AdamOptimizer(); break;
            case Config.AmsGrad: optimizer = new AdamOptimizer(); o["amsgrad"] = 1.0; break;
            case Config.AdamW: optimizer = new AdamWOptimizer(); break;
            case Config.NAdam: optimizer = new NAdamOptimizer(); break;
            case Config.Adamax: optimizer = new AdamaxOptimizer(); break;
            case Config.Adagrad: optimizer = new AdagradOptimizer(); o["lr_decay"] = 0.1; break;
            case Config.RmsProp: optimizer = new RmsPropOptimizer(); break;
            case Config.RmsPropCentered: optimizer = new RmsPropOptimizer(); o["centered"] = 1.0; break;
            case Config.AdaDelta: optimizer = new AdaDeltaOptimizer(); o["lr"] = 1.0; break;
            case Config.Lion: optimizer = new LionOptimizer(); break;
            case Config.Ftrl: optimizer = new FtrlOptimizer(); o.Remove("maximize"); o.Remove("weight_decay"); o["l1_reg"] = 1e-3; break;
            case Config.RAdam: optimizer = new RAdamOptimizer(); break;
            case Config.Lamb: optimizer = new LambOptimizer(); break;
            default: throw new ArgumentOutOfRangeException(nameof(c));
        }
        return (optimizer, o);
    }

    private static float[] RandomArray(int length, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var a = new float[length];
        for (int i = 0; i < length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static string[] HostSteps(OptimizerBase optimizer)
        => GpuLaunchProbe.Fallbacks.Where(f => f.Contains($"{optimizer.GetType().Name}-host-step")).ToArray();

    [SkippableTheory]
    [InlineData(Config.Sgd)] [InlineData(Config.SgdMomentum)] [InlineData(Config.Adam)] [InlineData(Config.AmsGrad)]
    [InlineData(Config.AdamW)] [InlineData(Config.NAdam)] [InlineData(Config.Adamax)] [InlineData(Config.Adagrad)]
    [InlineData(Config.RmsProp)] [InlineData(Config.AdaDelta)] [InlineData(Config.Lion)] [InlineData(Config.Ftrl)]
    [InlineData(Config.SgdNesterov)] [InlineData(Config.RAdam)] [InlineData(Config.Lamb)] [InlineData(Config.RmsPropCentered)]
    public void DeviceParameters_MatchTheCpu_AndStayOnTheDeviceWhenAKernelExists(Config config)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[] shape = { 40, 25 };
        int n = shape[0] * shape[1];
        var initial = RandomArray(n, 1);

        var (cpuOptimizer, options) = Create(config);
        var cpuParameter = new Tensor<float>((float[])initial.Clone(), shape);
        cpuOptimizer.AddParamGroup(options).AddParameter(cpuParameter);

        var (gpuOptimizer, gpuOptions) = Create(config);
        var gpuParameter = gpu.UploadToGpu<float>((float[])initial.Clone(), shape, GpuTensorRole.General);
        gpuOptimizer.AddParamGroup(gpuOptions).AddParameter(gpuParameter);
        var hostStepsBefore = HostSteps(gpuOptimizer);

        for (int step = 0; step < 3; step++)
        {
            var g = RandomArray(n, 100 + step);
            var expected = (float[])g.Clone();
            cpuOptimizer.Step(new Dictionary<Tensor<float>, Tensor<float>> { [cpuParameter] = new Tensor<float>((float[])g.Clone(), shape) });
            var gpuGradient = gpu.UploadToGpu<float>(g, shape, GpuTensorRole.General);
            gpuOptimizer.Step(new Dictionary<Tensor<float>, Tensor<float>> { [gpuParameter] = gpuGradient });
            Assert.Equal(expected, gpuGradient.ToArray());   // the device gradient is never written either
        }

        Assert.True(gpuParameter.IsGpuResident, "the parameter left the device");
        var actual = gpuParameter.ToArray();
        for (int i = 0; i < n; i++)
        {
            float e = cpuParameter.GetFlat(i);
            Assert.True(Math.Abs(e - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e)),
                $"{config}: parameter[{i}] cpu {e} vs gpu {actual[i]}");
        }

        bool tookHostStep = !HostSteps(gpuOptimizer).SequenceEqual(hostStepsBefore);
        Assert.True(tookHostStep != HasDeviceKernel(config),
            HasDeviceKernel(config) ? $"{config} has a device kernel but ran a host step" : $"{config} has no device kernel, yet no host step was recorded");

        // The saved state is host data either way, and matches the CPU optimizer's.
        // (Plain SGD keeps no state.)
        if (!cpuOptimizer.StateDict().State.TryGetValue(0, out var cpuState)) return;
        var gpuState = gpuOptimizer.StateDict().State[0];
        foreach (var kv in cpuState)
        {
            if (kv.Value.Tensor is not { } reference) continue;
            if (gpuState[kv.Key].Tensor is not { } mine)
            {
                Assert.Fail($"{config}: state '{kv.Key}' was not saved");
                return;
            }
            for (int i = 0; i < n; i++)
                Assert.True(Math.Abs(reference[i] - mine[i]) <= 1e-4f * Math.Max(1f, Math.Abs(reference[i])),
                    $"{config}: state '{kv.Key}'[{i}] cpu {reference[i]} vs gpu {mine[i]}");
        }
    }
}
