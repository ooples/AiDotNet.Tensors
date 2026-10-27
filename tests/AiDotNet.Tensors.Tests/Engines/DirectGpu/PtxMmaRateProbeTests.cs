#if NETCOREAPP
using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1549: the tensor-core accumulate-rate probe. On any Ampere+ device both loops must run and report a plausible rate
/// (FP16 accumulation never slower than FP32 accumulation beyond noise); the measured class is printed for the record.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class PtxMmaRateProbeTests
{
    private readonly ITestOutputHelper _output;

    public PtxMmaRateProbeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Ptx_DeclaresBothEntries_WithEightChains()
    {
        string ptx = PtxMmaRateProbe.EmitPtx(8, 6);
        Assert.Contains(".entry " + PtxMmaRateProbe.Fp32AccumulateEntry, ptx);
        Assert.Contains(".entry " + PtxMmaRateProbe.Fp16AccumulateEntry, ptx);
        Assert.Equal(8, CountOf(ptx, "f32.f16.f16.f32"));
        Assert.Equal(8, CountOf(ptx, "f16.f16.f16.f16"));
    }

    [Fact]
    public void RateClass_SplitsFullRateFromDividedRate()
    {
        Assert.Equal("fp32acc-full", new PtxMmaRateProbe.Result(100, 105).RateClass);
        Assert.Equal("fp32acc-1/2", new PtxMmaRateProbe.Result(60, 121).RateClass);
        Assert.Equal("fp32acc-1/8", new PtxMmaRateProbe.Result(50, 395).RateClass);
        Assert.True(new PtxMmaRateProbe.Result(60, 121).Ratio > 2);
    }

    [SkippableFact]
    public void Measure_ReportsBothRates_OnThisDevice()
    {
        DirectPtxRuntime runtime;
        try
        {
            runtime = new DirectPtxRuntime();
        }
        catch (Exception ex) when (ex is DllNotFoundException or InvalidOperationException or NotSupportedException)
        {
            Skip.If(true, $"No CUDA device: {ex.Message}");
            return;
        }
        using (runtime)
        {
            Skip.If(runtime.ComputeCapabilityMajor < 8, "Needs compute capability 8.0+.");
            var result = PtxMmaRateProbe.Measure(runtime);
            _output.WriteLine($"{runtime.DeviceName}: FP32 accumulate {result.Fp32AccumulateTflops:F1} TFLOP/s, " +
                $"FP16 accumulate {result.Fp16AccumulateTflops:F1} TFLOP/s, ratio {result.Ratio:F2} -> {result.RateClass}");
            Assert.InRange(result.Fp32AccumulateTflops, 1, 10_000);
            Assert.InRange(result.Ratio, 0.9, 16);
        }
    }

    private static int CountOf(string text, string value)
    {
        int count = 0;
        for (int i = text.IndexOf(value, StringComparison.Ordinal); i >= 0; i = text.IndexOf(value, i + 1, StringComparison.Ordinal)) count++;
        return count;
    }
}
#endif