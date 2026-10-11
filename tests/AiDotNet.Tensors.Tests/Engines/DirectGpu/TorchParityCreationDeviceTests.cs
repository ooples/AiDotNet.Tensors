using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1112 tier 5: creation ops -- arange / range / logspace, the five windows, tril / triu indices and the *_like fills
/// (plus isreal, an all-ones fill) -- build their result on the device. Each case must match the CPU element by
/// element (shape included), launch device work, and record no host fallback; invalid arguments must still raise the
/// CPU's exceptions.
/// </summary>
[Collection("VulkanGlobalState")]
public class TorchParityCreationDeviceTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public TorchParityCreationDeviceTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine GpuOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is ITorchParityElementwiseBackend, "These device kernels are CUDA-only.");
        return gpu;
    }

    private static Tensor<float> Like()
    {
        var t = new Tensor<float>(new[] { 3, 4, 5 });
        for (int i = 0; i < t.Length; i++) t[i] = i % 7 == 0 ? float.NaN : i - 20.5f;
        return t;
    }

    public static TheoryData<string> Cases() => new()
    {
        "arange", "arangeNeg", "arangeFrac", "range", "rangeTight", "logspace", "logspace2", "logspace1",
        "hann", "hannSym", "hamming", "blackman", "bartlett", "bartlettSym", "kaiser", "kaiserSym", "window1",
        "tril", "trilNeg", "triu", "triuWide", "triuEmptyRows",
        "zerosLike", "onesLike", "fullLike", "emptyLike", "isReal",
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Creation_MatchesCpu_OnDevice(string op)
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        var like = Like();
        (string Probe, Func<IEngine, Tensor<float>> Run) c = op switch
        {
            "arange" => ("TensorArange", e => e.TensorArange<float>(0, 10)),
            "arangeNeg" => ("TensorArange", e => e.TensorArange<float>(5, -7.5, -0.75)),
            "arangeFrac" => ("TensorArange", e => e.TensorArange<float>(0, 1, 0.1)),
            "range" => ("TensorRange", e => e.TensorRange<float>(-3, 3, 0.5)),
            "rangeTight" => ("TensorRange", e => e.TensorRange<float>(0, 0.3, 0.1)),   // 3 elements: 0.3 / 0.1 < 3
            "logspace" => ("TensorLogspace", e => e.TensorLogspace<float>(-2, 3, 11)),
            "logspace2" => ("TensorLogspace", e => e.TensorLogspace<float>(0, 10, 21, 2)),
            "logspace1" => ("TensorLogspace", e => e.TensorLogspace<float>(1.5, 9, 1)),
            "hann" => ("TensorHannWindow", e => e.TensorHannWindow<float>(64)),
            "hannSym" => ("TensorHannWindow", e => e.TensorHannWindow<float>(33, periodic: false)),
            "hamming" => ("TensorHammingWindow", e => e.TensorHammingWindow<float>(50, true, 0.6, 0.3)),
            "blackman" => ("TensorBlackmanWindow", e => e.TensorBlackmanWindow<float>(40)),
            "bartlett" => ("TensorBartlettWindow", e => e.TensorBartlettWindow<float>(31)),
            "bartlettSym" => ("TensorBartlettWindow", e => e.TensorBartlettWindow<float>(31, periodic: false)),
            "kaiser" => ("TensorKaiserWindow", e => e.TensorKaiserWindow<float>(45)),
            "kaiserSym" => ("TensorKaiserWindow", e => e.TensorKaiserWindow<float>(20, periodic: false, beta: 5)),
            "window1" => ("TensorHannWindow", e => e.TensorHannWindow<float>(1)),
            "tril" => ("TensorTrilIndices", e => e.TensorTrilIndices<float>(7, 5)),
            "trilNeg" => ("TensorTrilIndices", e => e.TensorTrilIndices<float>(9, 6, -2)),
            "triu" => ("TensorTriuIndices", e => e.TensorTriuIndices<float>(6, 6, 1)),
            "triuWide" => ("TensorTriuIndices", e => e.TensorTriuIndices<float>(4, 300, -1)),
            "triuEmptyRows" => ("TensorTriuIndices", e => e.TensorTriuIndices<float>(8, 3, 2)),   // rows past col - offset keep none
            "zerosLike" => ("TensorZerosLike", e => e.TensorZerosLike(like)),
            "onesLike" => ("TensorOnesLike", e => e.TensorOnesLike(like)),
            "fullLike" => ("TensorFullLike", e => e.TensorFullLike(like, -2.75f)),
            "emptyLike" => ("TensorEmptyLike", e => e.TensorEmptyLike(like)),
            _ => ("TensorIsReal", e => e.TensorIsReal(like)),
        };

        var expected = c.Run(cpu);
        long launches = GpuLaunchProbe.Count;
        long fallbacks = TorchParityElementwiseDeviceTests.FallbackCount(c.Probe);
        var actual = c.Run(gpu);
        _ = actual[0];
        Assert.True(GpuLaunchProbe.Count > launches, $"{op}: no device work launched");
        Assert.Equal(fallbacks, TorchParityElementwiseDeviceTests.FallbackCount(c.Probe));
        Assert.Equal(expected._shape, actual._shape);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            Assert.True(Math.Abs(e - a) <= 1e-6 * Math.Max(1.0, Math.Abs(e)), $"{op}[{i}]: cpu {e:R} gpu {a:R}");
        }
    }

    [SkippableFact]
    public void Creation_InvalidArguments_ThrowLikeCpu()
    {
        var gpu = GpuOrSkip();
        Assert.Throws<ArgumentException>(() => gpu.TensorArange<float>(0, 5, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorRange<float>(0, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorLogspace<float>(0, 1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorHannWindow<float>(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.TensorTrilIndices<float>(-1, 3));
        Assert.Throws<ArgumentNullException>(() => gpu.TensorZerosLike<float>(null!));
    }

    [SkippableFact]
    public void Creation_EmptyResults_MatchCpuShapes()
    {
        var gpu = GpuOrSkip();
        var cpu = new CpuEngine();
        Assert.Equal(cpu.TensorLogspace<float>(0, 1, 0)._shape, gpu.TensorLogspace<float>(0, 1, 0)._shape);
        Assert.Equal(cpu.TensorHannWindow<float>(0)._shape, gpu.TensorHannWindow<float>(0)._shape);
        Assert.Equal(cpu.TensorTriuIndices<float>(3, 2, 5)._shape, gpu.TensorTriuIndices<float>(3, 2, 5)._shape);
    }
}
