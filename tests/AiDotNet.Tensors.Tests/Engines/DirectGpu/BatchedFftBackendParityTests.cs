#if NET6_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IDirectGpuBackend.BatchedFFT"/> of every backend against a float64 direct DFT. The lengths cover the
/// smallest transform, the single-launch on-chip path (up to 4096 on CUDA/HIP, the device's local memory on OpenCL)
/// and a length past it, which takes the multi-pass path. Every row of the batch is checked, so a kernel that
/// transforms only row 0 or mixes rows fails.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class BatchedFftBackendParityTests
{
    public static TheoryData<int, int, bool> Shapes => new()
    {
        { 3, 2, false },
        { 3, 8, true },
        { 5, 1024, false },
        { 5, 1024, true },
        { 4, 2048, false },
        { 4, 2048, true },
        { 3, 4096, false },
        { 3, 4096, true },
        { 2, 8192, false },
        { 2, 8192, true },
    };

    private static (float[] re, float[] im) Signal(int batch, int n)
    {
        var rng = new Random(1000 + n);
        var re = new float[batch * n];
        var im = new float[batch * n];
        for (int i = 0; i < re.Length; i++)
        {
            re[i] = (float)(rng.NextDouble() * 2 - 1);
            im[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        return (re, im);
    }

    // Direct DFT in float64; the inverse carries 1/n, matching BatchedFFT's convention.
    private static (double[] re, double[] im) Reference(float[] xr, float[] xi, int batch, int n, bool inverse)
    {
        var yr = new double[batch * n];
        var yi = new double[batch * n];
        double sign = inverse ? 2.0 : -2.0;
        for (int b = 0; b < batch; b++)
            for (int k = 0; k < n; k++)
            {
                double sr = 0, si = 0;
                for (int t = 0; t < n; t++)
                {
                    // Reduce k*t mod n before the angle so the argument stays exact.
                    double a = sign * Math.PI * ((long)k * t % n) / n;
                    double c = Math.Cos(a), s = Math.Sin(a);
                    double xre = xr[b * n + t], xim = xi[b * n + t];
                    sr += xre * c - xim * s;
                    si += xre * s + xim * c;
                }
                double scale = inverse ? 1.0 / n : 1.0;
                yr[b * n + k] = sr * scale;
                yi[b * n + k] = si * scale;
            }
        return (yr, yi);
    }

    private static void AssertClose(double[] er, double[] ei, float[] ar, float[] ai, int n, string what)
    {
        double scale = 1e-12;
        for (int i = 0; i < er.Length; i++) scale = Math.Max(scale, Math.Max(Math.Abs(er[i]), Math.Abs(ei[i])));
        // A float radix-2 transform's error grows like eps * log2(n) relative to the spectrum's magnitude.
        double tol = 2e-6 * Math.Max(1, Math.Log(n, 2)) * scale;
        for (int i = 0; i < er.Length; i++)
        {
            Assert.True(Math.Abs(er[i] - ar[i]) <= tol, $"{what} re[{i}] (row {i / n}, bin {i % n}): ref {er[i]}, gpu {ar[i]} (tol {tol})");
            Assert.True(Math.Abs(ei[i] - ai[i]) <= tol, $"{what} im[{i}] (row {i / n}, bin {i % n}): ref {ei[i]}, gpu {ai[i]} (tol {tol})");
        }
    }

    private static void Check(IDirectGpuBackend backend, int batch, int n, bool inverse)
    {
        var (xr, xi) = Signal(batch, n);
        var (er, ei) = Reference(xr, xi, batch, n, inverse);
        using var inRe = backend.AllocateBuffer(xr);
        using var inIm = backend.AllocateBuffer(xi);
        using var outRe = backend.AllocateBuffer(batch * n);
        using var outIm = backend.AllocateBuffer(batch * n);
        backend.BatchedFFT(inRe, inIm, outRe, outIm, batch, n, inverse);
        backend.Synchronize();
        AssertClose(er, ei, backend.DownloadBuffer(outRe), backend.DownloadBuffer(outIm), n, "out-of-place");
        // The input is untouched by an out-of-place transform.
        Assert.Equal(xr, backend.DownloadBuffer(inRe));
        Assert.Equal(xi, backend.DownloadBuffer(inIm));
    }

    // Input and output the same buffers: the CUDA and HIP transforms are defined for this (each row is read whole
    // before any of it is written, or staged through scratch on the multi-pass path).
    private static void CheckInPlace(IDirectGpuBackend backend, int batch, int n, bool inverse)
    {
        var (xr, xi) = Signal(batch, n);
        var (er, ei) = Reference(xr, xi, batch, n, inverse);
        using var re = backend.AllocateBuffer(xr);
        using var im = backend.AllocateBuffer(xi);
        backend.BatchedFFT(re, im, re, im, batch, n, inverse);
        backend.Synchronize();
        AssertClose(er, ei, backend.DownloadBuffer(re), backend.DownloadBuffer(im), n, "in-place");
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_BatchedFft_MatchesDft(int batch, int n, bool inverse)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend, batch, n, inverse);
        CheckInPlace(backend, batch, n, inverse);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Hip_BatchedFft_MatchesDft(int batch, int n, bool inverse)
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend, batch, n, inverse);
        CheckInPlace(backend, batch, n, inverse);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_BatchedFft_MatchesDft(int batch, int n, bool inverse)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend, batch, n, inverse);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_BatchedFft_MatchesDft(int batch, int n, bool inverse)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, batch, n, inverse);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_BatchedFft_MatchesDft(int batch, int n, bool inverse)
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, batch, n, inverse);
        CheckInPlace(backend, batch, n, inverse);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_BatchedFft_MatchesDft(int batch, int n, bool inverse)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        Check(backend, batch, n, inverse);
        CheckInPlace(backend, batch, n, inverse);
    }
#endif
}
#endif
