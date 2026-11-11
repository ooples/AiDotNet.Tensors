using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The 17 model-level ops of <c>CpuEngine.TorchModelOps</c> on the GPU engine, GPU against CPU: the forward and, under a
/// tape, every input's gradient, on each backend this machine has. Every op must launch device kernels, record no host
/// fallback, and (with device-resident inputs) keep its forward and backward on the device: no device-to-host
/// download inside the step. The activations are fed the edge values their kernels special-case (signed zeros,
/// magnitudes past float's square root, infinities, NaN) alongside random values.
/// </summary>
public abstract class TorchModelOpsGpuTestsBase
{
    private readonly IDirectGpuTensorEngineFixture _fixture;
    protected TorchModelOpsGpuTestsBase(IDirectGpuTensorEngineFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchModelOpsBackend && gpu.GetBackend() is ITorchParityElementwiseBackend,
            "This backend has no model-op device kernels.");
        return gpu;
    }

    private static Tensor<float> Rand(int seed, double scale, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static Tensor<float> Copy(Tensor<float> t) => new Tensor<float>(t.ToArray(), (int[])t._shape.Clone());

    private static void AssertClose(string what, float[] cpu, float[] gpu, double tolerance)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            float c = cpu[i], g = gpu[i];
            if (float.IsNaN(c) || float.IsNaN(g)) { Assert.True(float.IsNaN(c) && float.IsNaN(g), $"{what}[{i}]: cpu {c:R} gpu {g:R}"); continue; }
            if (float.IsInfinity(c) || float.IsInfinity(g)) { Assert.True(c == g, $"{what}[{i}]: cpu {c:R} gpu {g:R}"); continue; }
            Assert.True(Math.Abs(c - g) <= tolerance * Math.Max(1.0, Math.Abs(c)), $"{what}[{i}]: cpu {c:R} gpu {g:R}");
        }
    }

    /// <summary>
    /// Runs <paramref name="op"/> on the CPU and on the GPU (inputs device-resident there), each under its own tape with
    /// loss = sum(y * r), and compares y and every input's gradient. On the GPU the step must launch kernels, record no
    /// fallback and download nothing before the results are read.
    /// </summary>
    private static void CheckForwardAndGradients(DirectGpuTensorEngine gpu, string name, Func<IEngine, Tensor<float>[], Tensor<float>> op,
        Tensor<float>[] inputs, double forwardTolerance, double gradientTolerance, bool differentiable = true)
    {
        var cpu = new CpuEngine();
        var cpuInputs = inputs.Select(Copy).ToArray();
        float[] cpuY;
        float[][] cpuGrads;
        using (var tape = new GradientTape<float>())
        {
            var y = op(cpu, cpuInputs);
            cpuY = y.ToArray();
            var r = Rand(97, 1.0, (int[])y._shape.Clone());
            var loss = cpu.ReduceSum(cpu.TensorMultiply(y, r), Enumerable.Range(0, y.Rank).ToArray(), keepDims: false);
            if (differentiable)
            {
                var grads = tape.ComputeGradients(loss, cpuInputs);
                cpuGrads = cpuInputs.Select(x => grads[x].ToArray()).ToArray();
            }
            else cpuGrads = Array.Empty<float[]>();
        }

        // Device-resident inputs: the product with 1 leaves each input's values on the device.
        var gpuInputs = inputs.Select(x => gpu.TensorMultiplyScalar(Copy(x), 1f)).ToArray();
        long launches = GpuLaunchProbe.Count;
        long fallbacks = TorchParityElementwiseDeviceTestsBase.FallbackCount(name);
        Tensor<float> gy;
        Dictionary<Tensor<float>, Tensor<float>>? gpuGrads = null;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
        using (var tape = new GradientTape<float>())
        {
            gy = op(gpu, gpuInputs);
            Assert.True(gy.IsGpuResident || gy.HasPendingGpuData, $"{name} ran on the host");
            var r = Rand(97, 1.0, (int[])gy._shape.Clone());
            var loss = gpu.ReduceSum(gpu.TensorMultiply(gy, r), Enumerable.Range(0, gy.Rank).ToArray(), keepDims: false);
            if (differentiable) gpuGrads = tape.ComputeGradients(loss, gpuInputs);
        }
        Assert.True(GpuLaunchProbe.Count > launches, $"{name}: no kernel launched");
        Assert.Equal(fallbacks, TorchParityElementwiseDeviceTestsBase.FallbackCount(name));
        Assert.True(scope.Downloads == 0,
            $"{name} read device data back to the host: " + string.Join(", ", scope.Events
                .Where(e => e.Kind == GpuTransferKind.DeviceToHost).Select(e => $"{e.Bytes} B {e.Operation}")));

        AssertClose(name, cpuY, gy.ToArray(), forwardTolerance);
        if (!differentiable) return;
        for (int i = 0; i < gpuInputs.Length; i++)
        {
            Assert.True(gpuGrads!.ContainsKey(gpuInputs[i]), $"{name}: no gradient for input {i}");
            AssertClose($"{name} d input {i}", cpuGrads[i], gpuGrads[gpuInputs[i]].ToArray(), gradientTolerance);
        }
    }

    private static readonly float[] Edges =
    {
        float.NegativeInfinity, -1e30f, -1e20f, -88f, -20f, -5f, -1.5f, -1f, -0.7f, -0.3f, -1e-7f, -0f, 0f, 1e-7f, 0.29999998f,
        0.3f, 0.30000004f, 0.5f, 0.62f, 0.63f, 1f, 1.5f, 5f, 8.9f, 9.1f, 20f, 88f, 1e20f, 1e30f, float.PositiveInfinity, float.NaN,
    };

    private static Tensor<float> ActivationInput(int seed, bool withEdges)
    {
        var rng = new Random(seed);
        int n = (withEdges ? Edges.Length : 0) + 1500;
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++)
            t[i] = withEdges && i < Edges.Length ? Edges[i] : (float)((rng.NextDouble() - 0.5) * 12);
        return t;
    }

    public static TheoryData<string> Activations => new()
    {
        "TensorLisht", "TensorBentIdentity", "TensorGaussian", "TensorScaledTanh", "TensorThresholdedRelu", "TensorIsru",
        "TensorSqrbf", "TensorSurrogateSpike",
    };

    private static Tensor<float> Activation(IEngine e, string name, Tensor<float> x) => name switch
    {
        "TensorLisht" => e.TensorLisht(x),
        "TensorBentIdentity" => e.TensorBentIdentity(x),
        "TensorGaussian" => e.TensorGaussian(x),
        "TensorScaledTanh" => e.TensorScaledTanh(x, 1.7, 0.6),
        "TensorThresholdedRelu" => e.TensorThresholdedRelu(x, 0.5),
        "TensorIsru" => e.TensorIsru(x, 0.8),
        "TensorSqrbf" => e.TensorSqrbf(x, 1.3),
        "TensorSurrogateSpike" => e.TensorSurrogateSpike(x, 0.3, 4.0),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [SkippableTheory]
    [MemberData(nameof(Activations))]
    public void Activation_EdgeValues_MatchCpu(string name)
    {
        var gpu = GpuOrSkip();
        var x = ActivationInput(5, withEdges: true);
        var cpuY = Activation(new CpuEngine(), name, Copy(x)).ToArray();
        long fallbacks = TorchParityElementwiseDeviceTestsBase.FallbackCount(name);
        var gy = Activation(gpu, name, Copy(x));
        AssertClose(name, cpuY, gy.ToArray(), 2e-6);
        Assert.Equal(fallbacks, TorchParityElementwiseDeviceTestsBase.FallbackCount(name));
    }

    [SkippableTheory]
    [MemberData(nameof(Activations))]
    public void Activation_ForwardAndGradient_MatchCpu_OnDevice(string name)
    {
        var gpu = GpuOrSkip();
        CheckForwardAndGradients(gpu, name, (e, v) => Activation(e, name, v[0]), new[] { ActivationInput(6, withEdges: false) }, 2e-6, 1e-5);
    }

    [SkippableFact]
    public void FakeQuantize_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        var x = Rand(11, 3.0, 4, 37);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0]), new[] { x }, 1e-6, 0);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 4, null, null, symmetric: false), new[] { x }, 1e-6, 0);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 3, 0.25, 1, symmetric: false), new[] { x }, 1e-6, 0);
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 5, null, 2, symmetric: true), new[] { x }, 1e-6, 0);
        // Over 1024 values, so the min / max reduction folds several values per slot.
        CheckForwardAndGradients(gpu, "TensorFakeQuantize", (e, v) => e.TensorFakeQuantize(v[0], 6, null, null, symmetric: false),
            new[] { Rand(12, 2.0, 3000) }, 1e-6, 0);
    }

    [SkippableFact]
    public void TopKSoftmax_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 3), new[] { Rand(21, 4.0, 5, 8) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 1), new[] { Rand(22, 4.0, 2, 3, 6) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorTopKSoftmax", (e, v) => e.TensorTopKSoftmax(v[0], 64), new[] { Rand(23, 4.0, 3, 64) }, 2e-6, 1e-5);

        // Ties go to the lower index; NaN sorts below every number (and is selected only when k reaches it).
        var tied = new Tensor<float>(new[] { 1f, 3f, 3f, 2f, 3f, float.NaN, 0f, 5f, 2f, 2f, float.NaN, 1f }, new[] { 3, 4 });
        foreach (int k in new[] { 1, 2, 3, 4 })
        {
            var cpuY = new CpuEngine().TensorTopKSoftmax(Copy(tied), k).ToArray();
            AssertClose($"TensorTopKSoftmax ties k={k}", cpuY, gpu.TensorTopKSoftmax(Copy(tied), k).ToArray(), 2e-6);
        }
    }

    [SkippableFact]
    public void SinusoidalTimeEmbedding_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        // Diffusion-style integer timesteps up to 1000 (angles up to 1000 rad) and fractional ones; an odd dim pads a 0.
        var t = new Tensor<float>(new[] { 0f, 1f, 7f, 49.5f, 250f, 999f, 1000f, 0.001f, 123.456f }, new[] { 9 });
        CheckForwardAndGradients(gpu, "TensorSinusoidalTimeEmbedding", (e, v) => e.TensorSinusoidalTimeEmbedding(v[0], 64), new[] { t }, 2e-6, 2e-4);
        CheckForwardAndGradients(gpu, "TensorSinusoidalTimeEmbedding", (e, v) => e.TensorSinusoidalTimeEmbedding(v[0], 7),
            new[] { new Tensor<float>(new[] { 0.5f, 1.7f, 3.2f }, new[] { 3, 1 }) }, 2e-6, 1e-5);
    }

    [SkippableFact]
    public void CrfForward_MatchesCpu_OnDevice()
    {
        var gpu = GpuOrSkip();
        int b = 3, s = 6, k = 5;
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1], v[2], v[3]),
            new[] { Rand(31, 2.0, b, s, k), Rand(32, 1.0, k, k), Rand(33, 1.0, k), Rand(34, 1.0, k) }, 2e-5, 2e-5);
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1]),
            new[] { Rand(35, 2.0, 7, 4), Rand(36, 1.0, 4, 4) }, 2e-5, 2e-5);
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1], v[2]),
            new[] { Rand(37, 2.0, 2, 1, 3), Rand(38, 1.0, 3, 3), Rand(39, 1.0, 3) }, 2e-5, 2e-5);

        // Forbidden transitions (-inf, a constraint mask): log-sum-exp over -inf terms and a fully forbidden target.
        var masked = Rand(40, 1.0, 4, 4);
        masked[0, 1] = float.NegativeInfinity;
        masked[2, 3] = float.NegativeInfinity;
        for (int i = 0; i < 4; i++) masked[i, 2] = float.NegativeInfinity;
        CheckForwardAndGradients(gpu, "TensorCrfForward", (e, v) => e.TensorCrfForward(v[0], v[1]),
            new[] { Rand(41, 2.0, 2, 5, 4), masked }, 2e-5, 2e-5);
    }

    [SkippableFact]
    public void ComposedOps_MatchCpu_AndStayOnDevice()
    {
        var gpu = GpuOrSkip();
        CheckForwardAndGradients(gpu, "TensorLogSoftmin", (e, v) => e.TensorLogSoftmin(v[0], 1), new[] { Rand(51, 3.0, 4, 9) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorLogSoftmin", (e, v) => e.TensorLogSoftmin(v[0], 0), new[] { Rand(52, 3.0, 4, 9) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorHyperbolicLinear", (e, v) => e.TensorHyperbolicLinear(v[0], v[1], v[2], 0.8),
            new[] { Rand(53, 0.3, 5, 6), Rand(54, 0.5, 4, 6), Rand(55, 0.2, 4) }, 2e-5, 1e-4);
        CheckForwardAndGradients(gpu, "TensorHyperbolicLinear", (e, v) => e.TensorHyperbolicLinear(v[0], v[1]),
            new[] { Rand(56, 0.3, 2, 3, 6), Rand(57, 0.5, 4, 6) }, 2e-5, 1e-4);
        CheckForwardAndGradients(gpu, "TensorSoftSplit", (e, v) => e.TensorSoftSplit(v[0], v[1], v[2], 2, 0.1, 0.7),
            new[] { Rand(58, 1.0, 5), Rand(59, 1.0, 3, 2), Rand(60, 1.0, 3, 2) }, 2e-6, 1e-5);
        CheckForwardAndGradients(gpu, "TensorSoftKnn", (e, v) => e.TensorSoftKnn(v[0], v[1], v[2], 0.9),
            new[] { Rand(61, 1.0, 3, 4), Rand(62, 1.0, 6, 4), Rand(63, 1.0, 6, 2) }, 2e-5, 1e-4);
        CheckForwardAndGradients(gpu, "TensorSoftLocallyWeighted", (e, v) => e.TensorSoftLocallyWeighted(v[0], v[1], v[2], 1.5),
            new[] { Rand(64, 1.0, 4), Rand(65, 1.0, 5, 4), Rand(66, 1.0, 5) }, 2e-5, 1e-4);
    }
}

/// <summary>An engine over the OpenCL backend, which the default order (CUDA first) does not pick on an NVIDIA machine.</summary>
public sealed class OpenClTensorEngineTestFixture : IDirectGpuTensorEngineFixture, IDisposable
{
    private readonly OpenClBackend? _backend;
    public DirectGpuTensorEngine? Engine { get; }
    public bool IsAvailable => Engine?.IsGpuAvailable == true;

    public OpenClTensorEngineTestFixture()
    {
        try
        {
            _backend = new OpenClBackend();
            if (!_backend.IsAvailable)
            {
                _backend.Dispose();
                _backend = null;
                return;
            }
            Engine = new DirectGpuTensorEngine(new DirectGpuEngine(_backend));
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException)
        {
            _backend?.Dispose();
            _backend = null;
            Engine = null;
        }
    }

    public void Dispose()
    {
        Engine?.Dispose();
        _backend?.Dispose();
    }
}

/// <summary>The cases above on the machine's default DirectGpu backend.</summary>
[Collection("VulkanGlobalState")]
public class TorchModelOpsGpuTests : TorchModelOpsGpuTestsBase, IClassFixture<DirectGpuTensorEngineTestFixture>
{
    public TorchModelOpsGpuTests(DirectGpuTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("DirectGpuSerial")]
public class TorchModelOpsOpenClTests : TorchModelOpsGpuTestsBase, IClassFixture<OpenClTensorEngineTestFixture>
{
    public TorchModelOpsOpenClTests(OpenClTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("VulkanGlobalState")]
public class TorchModelOpsVulkanTests : TorchModelOpsGpuTestsBase, IClassFixture<VulkanTensorEngineTestFixture>
{
    public TorchModelOpsVulkanTests(VulkanTensorEngineTestFixture fixture) : base(fixture) { }
}

[Collection("DirectGpuSerial")]
public class TorchModelOpsHipTests : TorchModelOpsGpuTestsBase, IClassFixture<HipTensorEngineTestFixture>
{
    public TorchModelOpsHipTests(HipTensorEngineTestFixture fixture) : base(fixture) { }
}

#if NET7_0_OR_GREATER
/// <summary>The FP32 kernels Metal and WebGPU run, as WGSL on Dawn.</summary>
[Collection("DirectGpuSerial")]
public class TorchModelOpsDawnTests : TorchModelOpsGpuTestsBase, IClassFixture<DawnTorchParityFixture>
{
    public TorchModelOpsDawnTests(DawnTorchParityFixture fixture) : base(fixture) { }
}
#endif

/// <summary>Source-level checks of the shared model-op kernels on every dialect (no device needed).</summary>
public sealed class TorchModelOpsKernelSourceTests
{
    [Fact]
    public void EveryKernel_IsTranslated_ForOpenClAndGlsl()
    {
        foreach (string name in TorchModelOpsKernels.KernelNames)
        {
            Assert.Contains("__kernel void " + name + "(", TorchModelOpsTranslation.OpenCl);
            Assert.True(TorchModelOpsTranslation.Glsl.ContainsKey(name), $"no GLSL shader for {name}");
        }
        Assert.Equal(TorchModelOpsKernels.KernelNames.Length, TorchModelOpsTranslation.Glsl.Count);
    }

    [Fact]
    public void EveryKernel_IsInThePortableSource_AndTranslatesToWgsl()
    {
        var modules = AiDotNet.Tensors.Engines.DirectGpu.WebGpu.CudaDialectToWgsl.Translate(TorchParityF32Kernels.GetSource());
        foreach (string name in TorchModelOpsKernels.KernelNames) Assert.True(modules.ContainsKey(name), $"no WGSL module for {name}");
    }

    [Fact]
    public void FrequencySplit_IsExact_AndThresholdRoundsDown()
    {
        var into = new float[3];
        foreach (double f in new[] { 1.0, 0.7498942093324559, 1e-4, 0.36517412725483767 })
        {
            TorchModelOpsCodes.SplitFrequency(f, into, 0);
            Assert.True(Math.Abs((double)into[0] + into[1] + into[2] - f) <= 1e-15 * f, $"split of {f}");
            Assert.True((BitConverter.ToInt32(BitConverter.GetBytes(into[0]), 0) & 0xFFF) == 0);
            Assert.True((BitConverter.ToInt32(BitConverter.GetBytes(into[1]), 0) & 0xFFF) == 0);
        }
        Assert.Equal(0.29999998f, TorchModelOpsCodes.FloatThresholdBelow(0.3));
        Assert.Equal(0.5f, TorchModelOpsCodes.FloatThresholdBelow(0.5));
        Assert.Equal(-0.3f, TorchModelOpsCodes.FloatThresholdBelow(-0.3));   // -0.3f lies below -0.3
        Assert.Equal(float.MaxValue, TorchModelOpsCodes.FloatThresholdBelow(1e300));
    }
}
