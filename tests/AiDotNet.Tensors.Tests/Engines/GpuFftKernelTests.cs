using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Verifies the GPU FFT kernel at FP32 (the precision the kernel actually
/// implements) produces correct results vs the CPU reference. Regression
/// guard for an OpenCL bug where the host loop passed `stride` (2,4,8,…,n)
/// but the kernel reinterpreted it via `halfSize = 1 &lt;&lt; stage`, yielding
/// completely garbage spectra.
/// </summary>
public class GpuFftKernelTests
{
    [Fact]
    public void GpuFft_Fp32_CosWave_MatchesExpectedSpectrum()
    {
        int n = 32;
        var input = new Tensor<float>([n]);
        for (int i = 0; i < n; i++)
            input[i] = MathF.Cos(2f * MathF.PI * 3f * i / n);

        var spectrum = AiDotNetEngine.Current.NativeComplexFFT(input);

        // For cos(2π·3·n/32), expect peaks at k=3 and k=29 each with magnitude n/2 = 16.
        for (int k = 0; k < n; k++)
        {
            float mag = MathF.Sqrt(spectrum[k].Real * spectrum[k].Real
                                 + spectrum[k].Imaginary * spectrum[k].Imaginary);
            float expected = (k == 3 || k == 29) ? 16f : 0f;
            Assert.True(MathF.Abs(mag - expected) < 1e-3f,
                $"spectrum[{k}] magnitude={mag}, expected {expected}");
        }
    }

    [Fact]
    public void GpuFft_Fp32_RoundTrip_RecoverOriginalSignal()
    {
        int n = 64;
        var input = new Tensor<float>([n]);
        for (int i = 0; i < n; i++)
            input[i] = MathF.Sin(2f * MathF.PI * 5f * i / n);

        var spectrum = AiDotNetEngine.Current.NativeComplexFFT(input);
        var recovered = AiDotNetEngine.Current.NativeComplexIFFTReal(spectrum);

        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(input[i] - recovered[i]) < 1e-4f,
                $"i={i}: input={input[i]}, recovered={recovered[i]}");
    }
}
