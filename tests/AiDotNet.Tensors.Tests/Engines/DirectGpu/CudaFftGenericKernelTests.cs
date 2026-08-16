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
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
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
    public void TwiddleTableIndexingMatchesDirectDft(int n)
    {
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
    public void BluesteinMatchesDirectDftForArbitraryLengths(int n)
    {
        Complex[] x = Random(n, 99 + n);
        Assert.True(RelErr(Bluestein(x, false), Dft(x, false)) < Tol, $"forward n={n}");
        Assert.True(RelErr(Bluestein(x, true), Dft(x, true)) < Tol, $"inverse n={n}");
    }

    [Theory]
    [InlineData(5)]
    [InlineData(100)]
    [InlineData(896)]
    public void BluesteinRequiresSymmetricChirpTail(int n)
    {
        // NEGATIVE CONTROL. An m-point FFT computes a CYCLIC convolution; Bluestein needs a LINEAR one, and the
        // two coincide only when the chirp kernel is mirrored into the upper tail. Dropping the mirror leaves
        // the transform correct at k=0 and wrong elsewhere, so a test that only checked DC would pass.
        Complex[] x = Random(n, 7 + n);
        double err = RelErr(Bluestein(x, false, symmetricTail: false), Dft(x, false));
        Assert.True(err > 1e-3, $"omitting the symmetric tail should break the transform, got rel-err {err:E3}");
    }

    [Fact]
    public void BluesteinLengthIsSmallestPowerOfTwoAtLeastTwoNMinusOne()
    {
        Assert.Equal(1, CudaFFTGenericKernels.BluesteinLength(1));
        Assert.Equal(8, CudaFFTGenericKernels.BluesteinLength(3));      // 2*3-1 = 5  -> 8
        Assert.Equal(16, CudaFFTGenericKernels.BluesteinLength(8));     // 2*8-1 = 15 -> 16
        Assert.Equal(2048, CudaFFTGenericKernels.BluesteinLength(896)); // 2*896-1 = 1791 -> 2048
    }

    [Theory]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(1024, true)]
    [InlineData(896, false)]
    [InlineData(0, false)]
    public void IsPowerOfTwoClassifiesLengths(int n, bool expected)
        => Assert.Equal(expected, CudaFFTGenericKernels.IsPowerOfTwo(n));

    // ── source generation ───────────────────────────────────────────────────

    [Theory]
    [InlineData(FftElementType.Float32, "float", "_f32")]
    [InlineData(FftElementType.Float16, "__half", "_f16")]
    [InlineData(FftElementType.BFloat16, "__nv_bfloat16", "_bf16")]
    public void SourceUsesTheRequestedStorageType(FftElementType type, string storeType, string suffix)
    {
        string src = CudaFFTGenericKernels.GetSource(type);
        Assert.Contains($"#define STORE_T {storeType}", src);
        Assert.Contains($"fftg_batched_butterfly{suffix}", src);
        Assert.Equal(suffix, type.KernelSuffix());
    }

    [Theory]
    [InlineData(FftElementType.Float16, "cuda_fp16.h")]
    [InlineData(FftElementType.BFloat16, "cuda_bf16.h")]
    public void NarrowTypesIncludeTheirHeader(FftElementType type, string header)
        => Assert.Contains(header, CudaFFTGenericKernels.GetSource(type));

    [Fact]
    public void Float32SourceNeedsNoNarrowHeaders()
    {
        string src = CudaFFTGenericKernels.GetSource(FftElementType.Float32);
        Assert.DoesNotContain("cuda_fp16.h", src);
        Assert.DoesNotContain("cuda_bf16.h", src);
    }

    [Theory]
    [InlineData(FftElementType.Float32)]
    [InlineData(FftElementType.Float16)]
    [InlineData(FftElementType.BFloat16)]
    public void EveryDeclaredKernelIsPresentInTheSource(FftElementType type)
    {
        string src = CudaFFTGenericKernels.GetSource(type);
        foreach (string name in CudaFFTGenericKernels.GetKernelNames(type))
        {
            Assert.True(src.Contains("void " + name + "("), $"missing entry point {name}");
        }
    }

    [Fact]
    public void ArithmeticIsFloat32ForEveryElementType()
    {
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
    public void ByteSizeMatchesStorageWidth(FftElementType type, int expected)
        => Assert.Equal(expected, type.ByteSize());

    [Fact]
    public void BFloat16RequiresAmpere()
    {
        // Reporting the requirement lets a backend refuse cleanly; the alternative is an NVRTC compile failure
        // inside a launch, which is far harder to attribute to the element type that caused it.
        Assert.Equal(80, FftElementType.BFloat16.MinComputeCapabilityX10());
        Assert.Equal(53, FftElementType.Float16.MinComputeCapabilityX10());
        Assert.True(FftElementType.Float32.MinComputeCapabilityX10() < 53);
    }
}
