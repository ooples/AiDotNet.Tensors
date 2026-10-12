#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IGenericFftBackend"/> on every backend (#962, #961, #964): radix-2 and Bluestein lengths, forward and
/// inverse, against a double-precision DFT; and the 16-bit storage conversions bit-exact against round-to-nearest-even.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GenericFftBackendParityTests
{
    private static readonly int[] Lengths = { 1, 2, 5, 8, 12, 17, 64, 100, 257 };
    private const int Batch = 3;

    private static float[] Values(int count, int salt)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = ((i * 7919 + salt * 31) % 61 - 30) / 16f;
        return v;
    }

    private static void Dft(float[] re, float[] im, int n, bool inverse, out double[] outRe, out double[] outIm)
    {
        outRe = new double[re.Length];
        outIm = new double[re.Length];
        double sign = inverse ? 1 : -1;
        for (int b = 0; b < re.Length / n; b++)
            for (int k = 0; k < n; k++)
            {
                double sr = 0, si = 0;
                for (int j = 0; j < n; j++)
                {
                    double a = sign * 2 * Math.PI * ((long)j * k % n) / n;
                    sr += re[b * n + j] * Math.Cos(a) - im[b * n + j] * Math.Sin(a);
                    si += re[b * n + j] * Math.Sin(a) + im[b * n + j] * Math.Cos(a);
                }
                outRe[b * n + k] = inverse ? sr / n : sr;
                outIm[b * n + k] = inverse ? si / n : si;
            }
    }

    private static void CheckTransforms(IDirectGpuBackend backend)
    {
        var fft = Assert.IsAssignableFrom<IGenericFftBackend>(backend);
        Assert.True(fft.SupportsFftElementType(FftElementType.Float32));
        foreach (int n in Lengths)
            foreach (bool inverse in new[] { false, true })
            {
                var re = Values(Batch * n, n);
                var im = Values(Batch * n, n + 7);
                Dft(re, im, n, inverse, out var eRe, out var eIm);
                using var bRe = backend.AllocateBuffer(re);
                using var bIm = backend.AllocateBuffer(im);
                fft.LaunchFftGeneric(bRe, bIm, Batch, n, inverse, FftElementType.Float32);
                var aRe = backend.DownloadBuffer(bRe);
                var aIm = backend.DownloadBuffer(bIm);
                double tol = 2e-5 * n * (inverse ? 1.0 / n : 1.0) * 2 + 1e-5;
                for (int i = 0; i < Batch * n; i++)
                    Assert.True(Math.Abs(aRe[i] - eRe[i]) <= tol && Math.Abs(aIm[i] - eIm[i]) <= tol,
                        $"n={n} {(inverse ? "inverse" : "forward")} [{i}]: got ({aRe[i]}, {aIm[i]}), expected ({eRe[i]}, {eIm[i]})");
            }
    }

    private static ushort BFloat16(float x)
    {
        uint f = BitConverter.SingleToUInt32Bits(x);
        if ((f & 0x7fffffffu) > 0x7f800000u) return (ushort)((f >> 16) | 0x40u);
        return (ushort)((f + 0x7fffu + ((f >> 16) & 1u)) >> 16);
    }

    private static void CheckStorage(IDirectGpuBackend backend)
    {
        var fft = Assert.IsAssignableFrom<IGenericFftBackend>(backend);
        var values = new[] { 0f, -0f, 1f, -2.5f, 65504f, 70000f, 1e-8f, 6.1e-5f, 3.0e-7f, 0.1f, float.PositiveInfinity,
            float.NegativeInfinity, 1234.567f, -0.0009765625f, 5.9604645e-8f };
        foreach (var type in new[] { FftElementType.Float16, FftElementType.BFloat16 })
        {
            Assert.True(fft.SupportsFftElementType(type));
            int count = values.Length;
            using var source = backend.AllocateBuffer(values);
            using var storage = backend.AllocateByteBuffer((count * 2 + 3) & ~3);
            using var back = backend.AllocateBuffer(count);
            fft.ConvertFloatToFftStorage(source, storage, count, type);
            fft.ConvertFftStorageToFloat(storage, back, count, type);
            var actual = backend.DownloadBuffer(back);
            for (int i = 0; i < count; i++)
            {
                float expected = type == FftElementType.Float16
                    ? (float)(Half)values[i]
                    : BitConverter.UInt32BitsToSingle((uint)BFloat16(values[i]) << 16);
                Assert.True(BitConverter.SingleToUInt32Bits(expected) == BitConverter.SingleToUInt32Bits(actual[i]),
                    $"{type} round trip of {values[i]:R}: got {actual[i]:R}, expected {expected:R}");
            }
        }
    }

    private static void CheckInterleaved(IDirectGpuBackend backend)
    {
        var fft = Assert.IsAssignableFrom<IFftBackend>(backend);
        foreach (int n in new[] { 2, 8, 64 })
            foreach (bool inverse in new[] { false, true })
            {
                var re = Values(Batch * n, n + 3);
                var im = Values(Batch * n, n + 11);
                Dft(re, im, n, inverse, out var eRe, out var eIm);
                var interleaved = new float[2 * Batch * n];
                for (int i = 0; i < Batch * n; i++) { interleaved[2 * i] = re[i]; interleaved[2 * i + 1] = im[i]; }
                using var buffer = backend.AllocateBuffer(interleaved);
                fft.LaunchFft(buffer, Batch, n, inverse);
                var actual = backend.DownloadBuffer(buffer);
                double tol = 2e-5 * n * (inverse ? 1.0 / n : 1.0) * 2 + 1e-5;
                for (int i = 0; i < Batch * n; i++)
                    Assert.True(Math.Abs(actual[2 * i] - eRe[i]) <= tol && Math.Abs(actual[2 * i + 1] - eIm[i]) <= tol,
                        $"LaunchFft n={n} {(inverse ? "inverse" : "forward")} [{i}]: got ({actual[2 * i]}, {actual[2 * i + 1]}), expected ({eRe[i]}, {eIm[i]})");
            }
    }

    private static void CheckAll(IDirectGpuBackend backend)
    {
        CheckTransforms(backend);
        CheckStorage(backend);
        CheckInterleaved(backend);
    }

    [SkippableFact]
    public void OpenCl_GenericFft_MatchesDft()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Vulkan_GenericFft_MatchesDft()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckAll(backend);
    }

    [SkippableFact]
    public void Cuda_GenericFft_MatchesDft()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        CheckTransforms(backend);
    }

    [SkippableFact]
    public void Hip_GenericFft_MatchesDft()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        CheckAll(backend);
    }
}
#endif
