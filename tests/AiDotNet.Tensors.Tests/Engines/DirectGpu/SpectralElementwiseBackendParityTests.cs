#if NET6_0_OR_GREATER
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
/// #941: the elementwise spectral ops of the #850 family, called directly on each backend and compared with a
/// double-precision CPU reference. Every backend defines these, and none of them had a test that called them on any
/// backend. Each op runs at sizes of one element, a few, and more than one work-group, into an output pre-filled with
/// NaN so an element the kernel does not write fails.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class SpectralElementwiseBackendParityTests
{
    private static readonly int[] Sizes = { 1, 7, 257, 1000 };

    private static float[] Values(int count, int seed, double lo = -3, double hi = 3)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(lo + (hi - lo) * rng.NextDouble());
        return a;
    }

    private static IGpuBuffer Blank(IDirectGpuBackend backend, int n)
    {
        var buffer = backend.AllocateBuffer(n);
        backend.Fill(buffer, float.NaN, n);
        return buffer;
    }

    // Mismatches are collected per op (the first element that differs) and reported together, so one run shows
    // every op that disagrees with the reference instead of only the first.
    [ThreadStatic] private static System.Collections.Generic.List<string>? t_mismatches;

    private static void AssertClose(double[] expected, IDirectGpuBackend backend, IGpuBuffer actualBuffer, double tolerance, string what)
    {
        float[] actual = backend.DownloadBuffer(actualBuffer);
        if (actual.Length < expected.Length)
        {
            t_mismatches!.Add($"{what}: {actual.Length} outputs for {expected.Length} expected");
            return;
        }
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1.0, Math.Abs(expected[i]))) continue;
            t_mismatches!.Add($"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
            return;
        }
    }

    private static double[] Map(int n, Func<int, double> f)
    {
        var y = new double[n];
        for (int i = 0; i < n; i++) y[i] = f(i);
        return y;
    }

    // Arithmetic only: the result is a few FP32 operations from the reference.
    private const double Arithmetic = 2e-6;
    // sqrt, atan2, sin/cos, log, pow: device math libraries differ in the last few ulp.
    private const double Transcendental = 2e-5;

    private static void Check(IDirectGpuBackend backend)
    {
        t_mismatches = new System.Collections.Generic.List<string>();
        Run(backend);
        Assert.True(t_mismatches.Count == 0, string.Join(Environment.NewLine, t_mismatches));
    }

    private static void Run(IDirectGpuBackend backend)
    {
        foreach (int n in Sizes)
        {
            float[] aR = Values(n, 1 + n), aI = Values(n, 2 + n), bR = Values(n, 3 + n), bI = Values(n, 4 + n);
            using var daR = backend.AllocateBuffer(aR);
            using var daI = backend.AllocateBuffer(aI);
            using var dbR = backend.AllocateBuffer(bR);
            using var dbI = backend.AllocateBuffer(bI);

            using (var oR = Blank(backend, n))
            using (var oI = Blank(backend, n))
            {
                backend.SplitComplexMultiply(daR, daI, dbR, dbI, oR, oI, n);
                AssertClose(Map(n, i => (double)aR[i] * bR[i] - (double)aI[i] * bI[i]), backend, oR, Arithmetic, $"SplitComplexMultiply re n={n}");
                AssertClose(Map(n, i => (double)aR[i] * bI[i] + (double)aI[i] * bR[i]), backend, oI, Arithmetic, $"SplitComplexMultiply im n={n}");
            }

            using (var oR = Blank(backend, n))
            using (var oI = Blank(backend, n))
            {
                backend.SplitComplexAdd(daR, daI, dbR, dbI, oR, oI, n);
                AssertClose(Map(n, i => (double)aR[i] + bR[i]), backend, oR, Arithmetic, $"SplitComplexAdd re n={n}");
                AssertClose(Map(n, i => (double)aI[i] + bI[i]), backend, oI, Arithmetic, $"SplitComplexAdd im n={n}");
            }

            using (var oR = Blank(backend, n))
            using (var oI = Blank(backend, n))
            {
                // X * conj(Y)
                backend.SplitComplexCrossSpectral(daR, daI, dbR, dbI, oR, oI, n);
                AssertClose(Map(n, i => (double)aR[i] * bR[i] + (double)aI[i] * bI[i]), backend, oR, Arithmetic, $"SplitComplexCrossSpectral re n={n}");
                AssertClose(Map(n, i => (double)aI[i] * bR[i] - (double)aR[i] * bI[i]), backend, oI, Arithmetic, $"SplitComplexCrossSpectral im n={n}");
            }

            using (var oR = Blank(backend, n))
            using (var oI = Blank(backend, n))
            {
                backend.SplitComplexConjugate(daR, daI, oR, oI, n);
                AssertClose(Map(n, i => aR[i]), backend, oR, 0, $"SplitComplexConjugate re n={n}");
                AssertClose(Map(n, i => -(double)aI[i]), backend, oI, 0, $"SplitComplexConjugate im n={n}");
            }

            using (var oR = Blank(backend, n))
            using (var oI = Blank(backend, n))
            {
                const float scalar = -1.75f;
                backend.SplitComplexScale(daR, daI, oR, oI, scalar, n);
                AssertClose(Map(n, i => (double)aR[i] * scalar), backend, oR, Arithmetic, $"SplitComplexScale re n={n}");
                AssertClose(Map(n, i => (double)aI[i] * scalar), backend, oI, Arithmetic, $"SplitComplexScale im n={n}");
            }

            using (var o = Blank(backend, n))
            {
                backend.SplitComplexMagnitudeSquared(daR, daI, o, n);
                AssertClose(Map(n, i => (double)aR[i] * aR[i] + (double)aI[i] * aI[i]), backend, o, Arithmetic, $"SplitComplexMagnitudeSquared n={n}");
            }

            using (var o = Blank(backend, n))
            {
                backend.SplitComplexMagnitude(daR, daI, o, n);
                AssertClose(Map(n, i => Math.Sqrt((double)aR[i] * aR[i] + (double)aI[i] * aI[i])), backend, o, Transcendental, $"SplitComplexMagnitude n={n}");
            }

            using (var o = Blank(backend, n))
            {
                backend.SplitComplexPhase(daR, daI, o, n);
                AssertClose(Map(n, i => Math.Atan2(aI[i], aR[i])), backend, o, Transcendental, $"SplitComplexPhase n={n}");
            }

            using (var o = Blank(backend, n))
            {
                // (real, imag) in, atan2(imag, real) out.
                backend.Atan2Elementwise(daR, daI, o, n);
                AssertClose(Map(n, i => Math.Atan2(aI[i], aR[i])), backend, o, Transcendental, $"Atan2Elementwise n={n}");
            }

            // Polar: magnitudes in [0, 4], phases across several turns.
            float[] mag = Values(n, 5 + n, 0, 4), phase = Values(n, 6 + n, -7, 7);
            using var dMag = backend.AllocateBuffer(mag);
            using var dPhase = backend.AllocateBuffer(phase);
            using (var oR = Blank(backend, n))
            using (var oI = Blank(backend, n))
            {
                backend.PolarToComplex(dMag, dPhase, oR, oI, n);
                AssertClose(Map(n, i => mag[i] * Math.Cos(phase[i])), backend, oR, Transcendental, $"PolarToComplex re n={n}");
                AssertClose(Map(n, i => mag[i] * Math.Sin(phase[i])), backend, oI, Transcendental, $"PolarToComplex im n={n}");
            }
            using (var oR = Blank(backend, n))
            using (var oI = Blank(backend, n))
            {
                backend.SplitComplexFromPolar(dMag, dPhase, oR, oI, n);
                AssertClose(Map(n, i => mag[i] * Math.Cos(phase[i])), backend, oR, Transcendental, $"SplitComplexFromPolar re n={n}");
                AssertClose(Map(n, i => mag[i] * Math.Sin(phase[i])), backend, oI, Transcendental, $"SplitComplexFromPolar im n={n}");
            }

            using (var o = Blank(backend, n))
            {
                backend.ApplyWindow(daR, dbR, o, n);
                AssertClose(Map(n, i => (double)aR[i] * bR[i]), backend, o, Arithmetic, $"ApplyWindow n={n}");
            }

            // Power to dB and back. refValue is a power reference (10 log10(power / refValue)), and is not 1 here
            // so that a backend which squares it fails: CUDA and HIP did until #1236. The power values include zero
            // and one below the 1e-10 floor.
            float[] power = Values(n, 7 + n, 0, 50);
            power[0] = 0f;
            if (n > 1) power[1] = 1e-12f;
            if (n > 2) power[2] = 1e-6f;       // far below the reference: exercises the minDb floor
            const float refValue = 1.5f, minDb = -40f;
            using var dPower = backend.AllocateBuffer(power);
            using (var o = Blank(backend, n))
            {
                backend.PowerToDb(dPower, o, n, refValue, minDb);
                AssertClose(Map(n, i => Math.Max(10.0 * Math.Log10(Math.Max((double)power[i], 1e-10) / refValue), minDb)),
                    backend, o, Transcendental, $"PowerToDb n={n}");
            }
            float[] db = Values(n, 8 + n, -60, 30);
            using var dDb = backend.AllocateBuffer(db);
            using (var o = Blank(backend, n))
            {
                backend.DbToPower(dDb, o, n, refValue);
                AssertClose(Map(n, i => Math.Pow(10.0, db[i] / 10.0) * refValue), backend, o, Transcendental, $"DbToPower n={n}");
            }

            // log(1 + e) over mel energies: non-negative, from tiny to large.
            float[] energy = Values(n, 9 + n, 0, 200);
            energy[0] = 0f;
            if (n > 1) energy[1] = 1e-6f;
            using var dEnergy = backend.AllocateBuffer(energy);
            using (var o = Blank(backend, n))
            {
                backend.MfccLog1p(dEnergy, o, n);
                AssertClose(Map(n, i => Math.Log(1.0 + energy[i])), backend, o, Transcendental, $"MfccLog1p n={n}");
            }
        }

        // Per-row L2 normalization: a single element, short rows, rows longer than a work-group, and an all-zero
        // row, which normalizes to zeros.
        foreach (var (rows, cols) in new[] { (1, 1), (3, 7), (5, 300), (4, 1000), (300, 3) })
        {
            float[] x = Values(rows * cols, 31 + rows * cols);
            if (rows > 1) for (int c = 0; c < cols; c++) x[cols + c] = 0f;
            var expected = new double[rows * cols];
            for (int r = 0; r < rows; r++)
            {
                double sum = 0;
                for (int c = 0; c < cols; c++) sum += (double)x[r * cols + c] * x[r * cols + c];
                double inv = sum > 0 ? 1.0 / Math.Sqrt(sum) : 0.0;
                for (int c = 0; c < cols; c++) expected[r * cols + c] = x[r * cols + c] * inv;
            }
            using var dx = backend.AllocateBuffer(x);
            using var o = Blank(backend, rows * cols);
            backend.NormalizeRowsFused(dx, o, rows, cols);
            AssertClose(expected, backend, o, Transcendental, $"NormalizeRowsFused {rows}x{cols}");
        }
    }

    [SkippableFact]
    public void OpenCl_SpectralElementwise_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_SpectralElementwise_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_SpectralElementwise_MatchesCpu()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_SpectralElementwise_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_SpectralElementwise_MatchesCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_SpectralElementwise_MatchesCpu()
    {
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend();
        Skip.IfNot(backend.IsAvailable, "WebGPU is not available.");
        Check(backend);
    }
#endif
}
#endif
