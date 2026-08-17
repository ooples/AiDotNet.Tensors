// Copyright (c) AiDotNet. All rights reserved.
//
// Tests for the generic-precision, arbitrary-length GPU FFT path.
//
// The CUDA kernels themselves need a device, so these tests lock down the two things that can be verified
// without one and that are where the bugs actually live:
//
//   1. The ALGORITHM the kernels implement, executed here in managed code with the same twiddle-table indexing
//      and the same Bluestein construction, checked against a direct DFT. A transcription error in the kernel
//      is still possible; an error in the algorithm is not, because it is pinned here.
//   2. The SOURCE GENERATION - that each element type produces the right storage type, conversion intrinsics
//      and include, and that every declared entry point is actually present in the emitted source.
//
// The negative control in BluesteinRequiresSymmetricChirpTail is deliberate: it asserts the symmetric extension
// is load-bearing, so a future simplification that drops it fails here rather than silently returning a
// transform that is correct only at DC.

using System;
using System.Numerics;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class CudaFftGenericKernelTests
{
    private const double Tol = 1e-9;

    // ── reference DFT ───────────────────────────────────────────────────────
    private static Complex[] Dft(Complex[] x, bool inverse)
    {
        int n = x.Length;
        var outp = new Complex[n];
        double sign = inverse ? 2.0 : -2.0;
        for (int k = 0; k < n; k++)
        {
            Complex acc = Complex.Zero;
            for (int j = 0; j < n; j++)
            {
                double ang = sign * Math.PI * ((long)j * k % n) / n;
                acc += x[j] * new Complex(Math.Cos(ang), Math.Sin(ang));
            }

            outp[k] = inverse ? acc / n : acc;
        }

        return outp;
    }

    private static Complex[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new Complex[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = new Complex(rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1);
        }

        return x;
    }

    private static double RelErr(Complex[] a, Complex[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            num += (a[i] - b[i]).Magnitude * (a[i] - b[i]).Magnitude;
            den += b[i].Magnitude * b[i].Magnitude;
        }

        return Math.Sqrt(num) / Math.Max(Math.Sqrt(den), 1e-300);
    }

    /// <summary>
    /// Radix-2 with the kernel's exact table indexing: the table holds w[j] = exp(-2*pi*i*j/n) for j &lt; n/2,
    /// and a stage of stride s reads tw[wing * (n/s)]. If that indexing is wrong the transform is wrong for
    /// every stage past the first.
    /// </summary>
    private static Complex[] Radix2WithTable(Complex[] x, bool inverse)
    {
        int n = x.Length;
        int log2n = 0;
        while ((1 << log2n) < n)
        {
            log2n++;
        }

        double sign = inverse ? 2.0 : -2.0;
        var tw = new Complex[Math.Max(n / 2, 1)];
        for (int j = 0; j < tw.Length; j++)
        {
            double ang = sign * Math.PI * j / n;
            tw[j] = new Complex(Math.Cos(ang), Math.Sin(ang));
        }

        var y = new Complex[n];
        for (int i = 0; i < n; i++)
        {
            int rev = 0, t = i;
            for (int b = 0; b < log2n; b++)
            {
                rev = (rev << 1) | (t & 1);
                t >>= 1;
            }

            y[i] = x[rev];
        }

        for (int stride = 2; stride <= n; stride <<= 1)
        {
            int half = stride >> 1;
            int step = n / stride;
            for (int bf = 0; bf < n / stride; bf++)
            {
                for (int wing = 0; wing < half; wing++)
                {
                    Complex w = tw[wing * step];
                    int top = bf * stride + wing;
                    int bot = top + half;
                    Complex a = y[top];
                    Complex bb = y[bot] * w;
                    y[top] = a + bb;
                    y[bot] = a - bb;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                y[i] /= n;
            }
        }

        return y;
    }

    /// <summary>
    /// Bluestein exactly as the kernel chain performs it. <paramref name="symmetricTail"/> exists so the
    /// negative control can run the same code path with the extension omitted.
    /// </summary>
    private static Complex[] Bluestein(Complex[] x, bool inverse, bool symmetricTail = true)
    {
        int n = x.Length;
        int m = CudaFFTGenericKernels.BluesteinLength(n);
        double sgn = inverse ? -1.0 : 1.0;

        var c = new Complex[n];
        for (int k = 0; k < n; k++)
        {
            long kk = (long)k * k % (2L * n);       // 64-bit square then reduce, as fftg_build_chirp does
            double ang = sgn * Math.PI * kk / n;
            c[k] = new Complex(Math.Cos(ang), Math.Sin(ang));
        }

        var a = new Complex[m];
        for (int j = 0; j < n; j++)
        {
            a[j] = x[j] * Complex.Conjugate(c[j]);
        }

        var b = new Complex[m];
        for (int j = 0; j < n; j++)
        {
            b[j] = c[j];
        }

        if (symmetricTail)
        {
            for (int j = 1; j < n; j++)
            {
                b[m - j] = c[j];
            }
        }

        Complex[] fa = Radix2WithTable(a, false);
        Complex[] fb = Radix2WithTable(b, false);
        var prod = new Complex[m];
        for (int i = 0; i < m; i++)
        {
            prod[i] = fa[i] * fb[i];
        }

        Complex[] conv = Radix2WithTable(prod, true);

        var y = new Complex[n];
        for (int k = 0; k < n; k++)
        {
            y[k] = Complex.Conjugate(c[k]) * conv[k];
            if (inverse)
            {
                y[k] /= n;
            }
        }

        return y;
    }

    // ── algorithm ───────────────────────────────────────────────────────────

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(256)]
    public async Task TwiddleTableIndexingMatchesDirectDft(int n)
    {
        await Task.Yield();
        Complex[] x = Random(n, 1234 + n);
        Assert.True(RelErr(Radix2WithTable(x, false), Dft(x, false)) < Tol);
        Assert.True(RelErr(Radix2WithTable(x, true), Dft(x, true)) < Tol);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(12)]
    [InlineData(100)]
    [InlineData(896)]   // Qwen2.5-0.5B hidden width, 2^7 * 7 - the size that forced this path onto the CPU
    [InlineData(900)]
    [InlineData(1023)]
    public async Task BluesteinMatchesDirectDftForArbitraryLengths(int n)
    {
        await Task.Yield();
        Complex[] x = Random(n, 99 + n);
        Assert.True(RelErr(Bluestein(x, false), Dft(x, false)) < Tol, $"forward n={n}");
        Assert.True(RelErr(Bluestein(x, true), Dft(x, true)) < Tol, $"inverse n={n}");
    }

    [Theory]
    [InlineData(5)]
    [InlineData(100)]
    [InlineData(896)]
    public async Task BluesteinRequiresSymmetricChirpTail(int n)
    {
        await Task.Yield();
        // NEGATIVE CONTROL. An m-point FFT computes a CYCLIC convolution; Bluestein needs a LINEAR one, and the
        // two coincide only when the chirp kernel is mirrored into the upper tail. Dropping the mirror leaves
        // the transform correct at k=0 and wrong elsewhere, so a test that only checked DC would pass.
        Complex[] x = Random(n, 7 + n);
        double err = RelErr(Bluestein(x, false, symmetricTail: false), Dft(x, false));
        Assert.True(err > 1e-3, $"omitting the symmetric tail should break the transform, got rel-err {err:E3}");
    }

    [Fact]
    public async Task BluesteinLengthIsSmallestPowerOfTwoAtLeastTwoNMinusOne()
    {
        await Task.Yield();
        Assert.Equal(1, CudaFFTGenericKernels.BluesteinLength(1));
        Assert.Equal(8, CudaFFTGenericKernels.BluesteinLength(3));      // 2*3-1 = 5  -> 8
        Assert.Equal(16, CudaFFTGenericKernels.BluesteinLength(8));     // 2*8-1 = 15 -> 16
        Assert.Equal(2048, CudaFFTGenericKernels.BluesteinLength(896)); // 2*896-1 = 1791 -> 2048
        Assert.Equal(1 << 30, CudaFFTGenericKernels.BluesteinLength(1 << 29));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData((1 << 29) + 1)]
    [InlineData(int.MaxValue)]
    public async Task BluesteinLengthRejectsNonPositiveAndOverflowingLengths(int n)
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaFFTGenericKernels.BluesteinLength(n));
    }

    [Theory]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(1024, true)]
    [InlineData(896, false)]
    [InlineData(0, false)]
    public async Task IsPowerOfTwoClassifiesLengths(int n, bool expected)
    {
        await Task.Yield();
        Assert.Equal(expected, CudaFFTGenericKernels.IsPowerOfTwo(n));
    }

    // ── source generation ───────────────────────────────────────────────────

    [Theory]
    [InlineData(FftElementType.Float32, "float", "_f32")]
    [InlineData(FftElementType.Float16, "__half", "_f16")]
    [InlineData(FftElementType.BFloat16, "__nv_bfloat16", "_bf16")]
    public async Task SourceUsesTheRequestedStorageType(FftElementType type, string storeType, string suffix)
    {
        await Task.Yield();
        string src = CudaFFTGenericKernels.GetSource(type);
        Assert.Contains($"#define STORE_T {storeType}", src);
        Assert.Contains($"fftg_batched_butterfly{suffix}", src);
        Assert.Equal(suffix, type.KernelSuffix());
    }

    [Theory]
    [InlineData(FftElementType.Float16, "cuda_fp16.h")]
    [InlineData(FftElementType.BFloat16, "cuda_bf16.h")]
    public async Task NarrowTypesIncludeTheirHeader(FftElementType type, string header)
    {
        await Task.Yield();
        Assert.Contains(header, CudaFFTGenericKernels.GetSource(type));
    }

    [Fact]
    public async Task Float32SourceNeedsNoNarrowHeaders()
    {
        await Task.Yield();
        string src = CudaFFTGenericKernels.GetSource(FftElementType.Float32);
        Assert.DoesNotContain("cuda_fp16.h", src);
        Assert.DoesNotContain("cuda_bf16.h", src);
    }

    [Theory]
    [InlineData(FftElementType.Float32)]
    [InlineData(FftElementType.Float16)]
    [InlineData(FftElementType.BFloat16)]
    public async Task EveryDeclaredKernelIsPresentInTheSource(FftElementType type)
    {
        await Task.Yield();
        string src = CudaFFTGenericKernels.GetSource(type);
        foreach (string name in CudaFFTGenericKernels.GetKernelNames(type))
        {
            Assert.True(src.Contains("void " + name + "("), $"missing entry point {name}");
        }
    }

    [Fact]
    public async Task ArithmeticIsFloat32ForEveryElementType()
    {
        await Task.Yield();
        // The accumulator must stay float32 whatever the storage width: an n-point transform is log2(n)
        // accumulation stages, and a 7-bit mantissa cannot survive them. The narrow type appears only at the
        // load/store boundary, which is where the bandwidth saving lives.
        foreach (FftElementType type in new[] { FftElementType.Float16, FftElementType.BFloat16 })
        {
            string src = CudaFFTGenericKernels.GetSource(type);
            Assert.Contains("float tR = aidn_ld(re[top])", src);
            Assert.Contains("float xR = bR * twR - bI * twI;", src);
        }
    }

    [Theory]
    [InlineData(FftElementType.Float32, 4)]
    [InlineData(FftElementType.Float16, 2)]
    [InlineData(FftElementType.BFloat16, 2)]
    public async Task ByteSizeMatchesStorageWidth(FftElementType type, int expected)
    {
        await Task.Yield();
        Assert.Equal(expected, type.ByteSize());
    }

    [Fact]
    public async Task UnknownElementTypeIsRejectedByStorageContracts()
    {
        await Task.Yield();
        var unknown = (FftElementType)int.MaxValue;
        Assert.Throws<ArgumentOutOfRangeException>(() => unknown.ByteSize());
        Assert.Throws<ArgumentOutOfRangeException>(() => unknown.KernelSuffix());
    }

    [Theory]
    [InlineData(FftElementType.Float32)]
    [InlineData(FftElementType.Float16)]
    [InlineData(FftElementType.BFloat16)]
    public async Task LaunchBoundsUsesOneNamedSourceConstant(FftElementType type)
    {
        await Task.Yield();
        string source = CudaFFTGenericKernels.GetSource(type);
        Assert.Contains("#define AIDN_FFTG_THREADS_PER_BLOCK 256", source);
        Assert.DoesNotContain("__launch_bounds__(256)", source);
    }

    [Fact]
    public async Task BFloat16RequiresAmpere()
    {
        await Task.Yield();
        // Reporting the requirement lets a backend refuse cleanly; the alternative is an NVRTC compile failure
        // inside a launch, which is far harder to attribute to the element type that caused it.
        Assert.Equal(80, FftElementType.BFloat16.MinComputeCapabilityX10());
        Assert.Equal(53, FftElementType.Float16.MinComputeCapabilityX10());
        Assert.True(FftElementType.Float32.MinComputeCapabilityX10() < 53);
    }

    [SkippableTheory]
    [InlineData(FftElementType.Float32, 8)]
    [InlineData(FftElementType.Float16, 8)]
    [InlineData(FftElementType.BFloat16, 8)]
    [InlineData(FftElementType.Float32, 5)]
    [InlineData(FftElementType.Float16, 5)]
    [InlineData(FftElementType.BFloat16, 5)]
    public async Task CudaExecutionMatchesReferenceAndRoundTrips(
        FftElementType type,
        int n)
    {
        await Task.Yield();
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend failed to initialize.");
        Skip.IfNot(backend.SupportsFftElementType(type), $"CUDA {type} FFT storage is unavailable.");

        const int batch = 2;
        int count = batch * n;
        Complex[] source = Random(count, 4200 + n + (int)type);
        float[] sourceReal = new float[count];
        float[] sourceImaginary = new float[count];
        for (int i = 0; i < count; i++)
        {
            sourceReal[i] = (float)source[i].Real;
            sourceImaginary[i] = (float)source[i].Imaginary;
        }

        IGpuBuffer real = type == FftElementType.Float32
            ? backend.AllocateBuffer(sourceReal)
            : backend.AllocateByteBuffer(checked(count * type.ByteSize()));
        IGpuBuffer imaginary = type == FftElementType.Float32
            ? backend.AllocateBuffer(sourceImaginary)
            : backend.AllocateByteBuffer(checked(count * type.ByteSize()));
        using (real)
        using (imaginary)
        {
            if (type != FftElementType.Float32)
            {
                using IGpuBuffer sourceRealBuffer = backend.AllocateBuffer(sourceReal);
                using IGpuBuffer sourceImaginaryBuffer = backend.AllocateBuffer(sourceImaginary);
                backend.ConvertFloatToFftStorage(sourceRealBuffer, real, count, type);
                backend.ConvertFloatToFftStorage(sourceImaginaryBuffer, imaginary, count, type);
            }

            Complex[] quantizedInput = DownloadComplex(backend, real, imaginary, count, type);
            backend.LaunchFftGeneric(real, imaginary, batch, n, inverse: false, type);
            Complex[] actualForward = DownloadComplex(backend, real, imaginary, count, type);
            Complex[] expectedForward = DftBatched(quantizedInput, batch, n, inverse: false);
            double tolerance = type switch
            {
                FftElementType.Float32 => 2e-5,
                FftElementType.Float16 => 3e-2,
                FftElementType.BFloat16 => 8e-2,
                _ => throw new ArgumentOutOfRangeException(nameof(type)),
            };
            Assert.True(
                RelErr(actualForward, expectedForward) < tolerance,
                $"{type} forward n={n}: rel-err {RelErr(actualForward, expectedForward):E3}");

            backend.LaunchFftGeneric(real, imaginary, batch, n, inverse: true, type);
            Complex[] roundTrip = DownloadComplex(backend, real, imaginary, count, type);
            Assert.True(
                RelErr(roundTrip, quantizedInput) < tolerance,
                $"{type} round-trip n={n}: rel-err {RelErr(roundTrip, quantizedInput):E3}");
        }
    }

    [SkippableFact]
    public async Task IEngineContractDispatchesArbitraryLengthCudaWithoutHostFallback()
    {
        await Task.Yield();
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available.");
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.GetBackend() is CudaBackend, "Active DirectGpu backend is not CUDA.");
        Skip.IfNot(engine.SupportsFftElementType(FftElementType.Float32), "CUDA Float32 generic FFT is unavailable.");

        var input = new Tensor<float>(
            new float[] { 1, 0, 2, -1, -3, 0.5f, 4, 2, -2, -0.25f },
            new[] { 10 });
        Tensor<float> output = ((IEngine)engine).FftGeneric(input);
        Complex[] expected = Dft(ToComplex(input.GetFlattenedData()), inverse: false);
        Complex[] actual = ToComplex(output.GetFlattenedData());
        Assert.True(RelErr(actual, expected) < 2e-5, $"IEngine CUDA path rel-err {RelErr(actual, expected):E3}");
    }

    [SkippableFact]
    public async Task ExplicitPeerBackendRequestThrowsInsteadOfCopyingToHost()
    {
        await Task.Yield();
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "No DirectGpu backend is available.");
        Skip.If(engine.GetBackend() is CudaBackend, "Requires a non-CUDA DirectGpu backend.");
        var input = new Tensor<float>(new float[] { 1, 0, 2, -1, 3, 0.5f }, new[] { 6 });

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => ((IEngine)engine).FftGeneric(input));
        Assert.Contains("No host fallback was performed", error.Message);
    }

    [Fact]
    public async Task DirectGpuContractOverridesManagedCpuFftGeneric()
    {
        await Task.Yield();
        var method = typeof(DirectGpuTensorEngine).GetMethod(
            nameof(IEngine.FftGeneric),
            new[] { typeof(Tensor<float>), typeof(bool), typeof(FftElementType) });

        Assert.NotNull(method);
        Assert.Equal(typeof(DirectGpuTensorEngine), method.DeclaringType);
    }

    [Fact]
    public async Task IEngineContractRecordsAutogradOnCpu()
    {
        await Task.Yield();
        IEngine engine = new CpuEngine();
        var input = new Tensor<float>(new float[] { 1, 0, 2, -1, 3, 0.5f }, new[] { 6 });
        using var tape = new GradientTape<float>();

        Tensor<float> result = engine.FftGeneric(input);
        Assert.NotNull(result.GradFn);
        Tensor<float> loss = engine.ReduceSum(result, new[] { 0 }, keepDims: false);
        var gradients = tape.ComputeGradients(loss);
        Assert.True(gradients.ContainsKey(input), "FftGeneric must keep the active tape connected to its input.");
    }

    [SkippableFact]
    public async Task FftGenericRecordsAutogradOnCuda()
    {
        await Task.Yield();
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available.");
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.GetBackend() is CudaBackend, "Active DirectGpu backend is not CUDA.");
        Assert.True(
            engine.SupportsFftElementType(FftElementType.Float32),
            "CUDA Float32 generic FFT module must be initialized.");

        var input = new Tensor<float>(new float[] { 1, 0, 2, -1, 3, 0.5f }, new[] { 6 });
        using var tape = new GradientTape<float>();

        Tensor<float> result = ((IEngine)engine).FftGeneric(input);
        Assert.NotNull(result.GradFn);
        Tensor<float> loss = engine.ReduceSum(result, new[] { 0 }, keepDims: false);
        var gradients = tape.ComputeGradients(loss);
        Assert.True(
            gradients.ContainsKey(input),
            "CUDA FftGeneric must keep the active tape connected to its input.");
    }

    private static Complex[] DownloadComplex(
        CudaBackend backend,
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int count,
        FftElementType type)
    {
        float[] realValues = new float[count];
        float[] imaginaryValues = new float[count];
        if (type == FftElementType.Float32)
        {
            backend.DownloadBuffer(real, realValues);
            backend.DownloadBuffer(imaginary, imaginaryValues);
        }
        else
        {
            using IGpuBuffer realFloat = backend.AllocateBuffer(count);
            using IGpuBuffer imaginaryFloat = backend.AllocateBuffer(count);
            backend.ConvertFftStorageToFloat(real, realFloat, count, type);
            backend.ConvertFftStorageToFloat(imaginary, imaginaryFloat, count, type);
            backend.DownloadBuffer(realFloat, realValues);
            backend.DownloadBuffer(imaginaryFloat, imaginaryValues);
        }

        var values = new Complex[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = new Complex(realValues[i], imaginaryValues[i]);
        }

        return values;
    }

    private static Complex[] ToComplex(float[] interleaved)
    {
        var values = new Complex[interleaved.Length / 2];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = new Complex(interleaved[2 * i], interleaved[(2 * i) + 1]);
        }

        return values;
    }

    private static Complex[] DftBatched(Complex[] values, int batch, int n, bool inverse)
    {
        var result = new Complex[values.Length];
        for (int b = 0; b < batch; b++)
        {
            var row = new Complex[n];
            Array.Copy(values, b * n, row, 0, n);
            Complex[] transformed = Dft(row, inverse);
            Array.Copy(transformed, 0, result, b * n, n);
        }

        return result;
    }
}
