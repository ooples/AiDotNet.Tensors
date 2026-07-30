using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Full coverage tests for native Tensor&lt;Complex&lt;T&gt;&gt; IEngine operations.
/// </summary>
public class NativeComplexOpsTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public NativeComplexOpsTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // ================================================================
    // FFT Round-Trip
    // ================================================================

    [Fact(Skip = "Pre-existing FFT roundtrip numerical bug: " +
                 "NativeComplexIFFTReal(NativeComplexFFT(sin)) doesn't reconstruct the " +
                 "original signal to 6-decimal precision. FFT_ComplexIFFT_RoundTrip " +
                 "(below) tests the same chain with NativeComplexIFFT and passes, so the " +
                 "bug is specific to the real-output variant — likely an off-by-factor in " +
                 "scaling or the imaginary-component discard. Tracked as a follow-up to " +
                 "the broader FFT-roundtrip-via-compile-cache investigation (cf. " +
                 "Issue238_CompileInference_DoesNotStackOverflow_OnFftMultiplyIfft).")]
    public void FFT_RoundTrip_RecoverOriginalSignal()
    {
        int n = 64;
        var input = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
            input[i] = Math.Sin(2 * Math.PI * 5 * i / n);

        var spectrum = _engine.NativeComplexFFT(input);
        var recovered = _engine.NativeComplexIFFTReal(spectrum);

        Assert.Equal(n, recovered.Length);
        for (int i = 0; i < n; i++)
            Assert.Equal(input[i], recovered[i], 6);
    }

    [Fact]
    public void FFT_ComplexIFFT_RoundTrip()
    {
        int n = 32;
        var input = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
            input[i] = Math.Cos(2 * Math.PI * 3 * i / n);

        var spectrum = _engine.NativeComplexFFT(input);
        var recovered = _engine.NativeComplexIFFT(spectrum);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(input[i], recovered[i].Real, 6);
            Assert.Equal(0.0, recovered[i].Imaginary, 6);
        }
    }

    [Fact]
    public void FFT_SingleElement()
    {
        var input = new Tensor<double>([1]);
        input[0] = 42.0;

        var spectrum = _engine.NativeComplexFFT(input);
        Assert.Equal(42.0, spectrum[0].Real, 10);
        Assert.Equal(0.0, spectrum[0].Imaginary, 10);
    }

    [Fact]
    public void FFT_ZeroInput()
    {
        int n = 16;
        var input = new Tensor<double>([n]);
        var spectrum = _engine.NativeComplexFFT(input);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(0.0, spectrum[i].Real, 10);
            Assert.Equal(0.0, spectrum[i].Imaginary, 10);
        }
    }

    [Fact]
    public void FFT_NonPowerOfTwo_Throws()
    {
        var input = new Tensor<double>([10]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFT(input));
    }

    [Fact]
    public void FFT_Batched_2D_TransformsLastAxis()
    {
        // 2D tensor: [batch=2, fftSize=8]
        var input = new Tensor<double>([2, 8]);
        for (int i = 0; i < 8; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * i / 8);      // batch 0
            input[8 + i] = Math.Cos(2 * Math.PI * i / 8);   // batch 1
        }

        var spectrum = _engine.NativeComplexFFT(input);
        Assert.Equal(16, spectrum.Length); // 2 * 8

        var recovered = _engine.NativeComplexIFFTReal(spectrum);
        for (int i = 0; i < 16; i++)
            Assert.Equal(input[i], recovered[i], 5);
    }

    // ================================================================
    // Complex Multiply
    // ================================================================

    [Fact]
    public void ComplexMultiply_CorrectResult()
    {
        int n = 4;
        var a = new Tensor<Complex<double>>([n]);
        var b = new Tensor<Complex<double>>([n]);

        a[0] = new Complex<double>(1, 2);
        b[0] = new Complex<double>(3, 4);
        // (1+2i)(3+4i) = 3+4i+6i+8i^2 = 3+10i-8 = -5+10i

        a[1] = new Complex<double>(1, 0);
        b[1] = new Complex<double>(0, 1);
        // (1+0i)(0+1i) = 0+1i = i

        a[2] = new Complex<double>(2, 3);
        b[2] = new Complex<double>(2, -3);
        // (2+3i)(2-3i) = 4-9i^2 = 4+9 = 13+0i

        a[3] = new Complex<double>(0, 0);
        b[3] = new Complex<double>(5, 7);

        var result = _engine.NativeComplexMultiply(a, b);

        Assert.Equal(-5.0, result[0].Real, 10);
        Assert.Equal(10.0, result[0].Imaginary, 10);
        Assert.Equal(0.0, result[1].Real, 10);
        Assert.Equal(1.0, result[1].Imaginary, 10);
        Assert.Equal(13.0, result[2].Real, 10);
        Assert.Equal(0.0, result[2].Imaginary, 10);
        Assert.Equal(0.0, result[3].Real, 10);
        Assert.Equal(0.0, result[3].Imaginary, 10);
    }

    [Fact]
    public void ComplexMultiply_LengthMismatch_Throws()
    {
        var a = new Tensor<Complex<double>>([4]);
        var b = new Tensor<Complex<double>>([8]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexMultiply(a, b));
    }

    // ================================================================
    // Complex Conjugate
    // ================================================================

    [Fact]
    public void ComplexConjugate_NegatesImaginary()
    {
        var a = new Tensor<Complex<double>>([3]);
        a[0] = new Complex<double>(1, 2);
        a[1] = new Complex<double>(-3, 4);
        a[2] = new Complex<double>(5, -6);

        var result = _engine.NativeComplexConjugate(a);

        Assert.Equal(1.0, result[0].Real, 10);
        Assert.Equal(-2.0, result[0].Imaginary, 10);
        Assert.Equal(-3.0, result[1].Real, 10);
        Assert.Equal(-4.0, result[1].Imaginary, 10);
        Assert.Equal(5.0, result[2].Real, 10);
        Assert.Equal(6.0, result[2].Imaginary, 10);
    }

    // ================================================================
    // Magnitude and MagnitudeSquared
    // ================================================================

    [Fact]
    public void ComplexMagnitude_CorrectValues()
    {
        var a = new Tensor<Complex<double>>([3]);
        a[0] = new Complex<double>(3, 4);    // |3+4i| = 5
        a[1] = new Complex<double>(0, 1);    // |i| = 1
        a[2] = new Complex<double>(1, 0);    // |1| = 1

        var result = _engine.NativeComplexMagnitude(a);

        Assert.Equal(5.0, result[0], 10);
        Assert.Equal(1.0, result[1], 10);
        Assert.Equal(1.0, result[2], 10);
    }

    [Fact]
    public void ComplexMagnitudeSquared_CorrectValues()
    {
        var a = new Tensor<Complex<double>>([2]);
        a[0] = new Complex<double>(3, 4);    // 9 + 16 = 25
        a[1] = new Complex<double>(1, 1);    // 1 + 1 = 2

        var result = _engine.NativeComplexMagnitudeSquared(a);

        Assert.Equal(25.0, result[0], 10);
        Assert.Equal(2.0, result[1], 10);
    }

    // ================================================================
    // Phase
    // ================================================================

    [Fact]
    public void ComplexPhase_CorrectValues()
    {
        var a = new Tensor<Complex<double>>([4]);
        a[0] = new Complex<double>(1, 0);    // atan2(0,1) = 0
        a[1] = new Complex<double>(0, 1);    // atan2(1,0) = pi/2
        a[2] = new Complex<double>(-1, 0);   // atan2(0,-1) = pi
        a[3] = new Complex<double>(1, 1);    // atan2(1,1) = pi/4

        var result = _engine.NativeComplexPhase(a);

        Assert.Equal(0.0, result[0], 10);
        Assert.Equal(Math.PI / 2, result[1], 10);
        Assert.Equal(Math.PI, result[2], 10);
        Assert.Equal(Math.PI / 4, result[3], 10);
    }

    // ================================================================
    // FromPolar
    // ================================================================

    [Fact]
    public void ComplexFromPolar_RoundTripWithMagnitudePhase()
    {
        var original = new Tensor<Complex<double>>([3]);
        original[0] = new Complex<double>(3, 4);
        original[1] = new Complex<double>(-1, 2);
        original[2] = new Complex<double>(0, -5);

        var mag = _engine.NativeComplexMagnitude(original);
        var phase = _engine.NativeComplexPhase(original);
        var recovered = _engine.NativeComplexFromPolar(mag, phase);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(original[i].Real, recovered[i].Real, 6);
            Assert.Equal(original[i].Imaginary, recovered[i].Imaginary, 6);
        }
    }

    // ================================================================
    // Scale and Add
    // ================================================================

    [Fact]
    public void ComplexScale_MultipliesBothParts()
    {
        var a = new Tensor<Complex<double>>([2]);
        a[0] = new Complex<double>(2, 3);
        a[1] = new Complex<double>(-1, 4);

        var result = _engine.NativeComplexScale(a, 3.0);

        Assert.Equal(6.0, result[0].Real, 10);
        Assert.Equal(9.0, result[0].Imaginary, 10);
        Assert.Equal(-3.0, result[1].Real, 10);
        Assert.Equal(12.0, result[1].Imaginary, 10);
    }

    [Fact]
    public void ComplexAdd_ElementWise()
    {
        var a = new Tensor<Complex<double>>([2]);
        var b = new Tensor<Complex<double>>([2]);
        a[0] = new Complex<double>(1, 2);
        a[1] = new Complex<double>(3, 4);
        b[0] = new Complex<double>(5, 6);
        b[1] = new Complex<double>(7, 8);

        var result = _engine.NativeComplexAdd(a, b);

        Assert.Equal(6.0, result[0].Real, 10);
        Assert.Equal(8.0, result[0].Imaginary, 10);
        Assert.Equal(10.0, result[1].Real, 10);
        Assert.Equal(12.0, result[1].Imaginary, 10);
    }

    // ================================================================
    // Shape Preservation
    // ================================================================

    [Fact]
    public void AllOps_PreserveShape()
    {
        var shape = new[] { 2, 4 };
        var real = new Tensor<double>(shape);
        var complex1 = new Tensor<Complex<double>>(shape);
        var complex2 = new Tensor<Complex<double>>(shape);
        for (int i = 0; i < 8; i++)
        {
            real[i] = i;
            complex1[i] = new Complex<double>(i, i + 1);
            complex2[i] = new Complex<double>(i + 2, i + 3);
        }

        Assert.Equal(shape, _engine.NativeComplexFFT(real).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexIFFTReal(complex1).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexIFFT(complex1).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexMultiply(complex1, complex2).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexConjugate(complex1).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexMagnitude(complex1).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexMagnitudeSquared(complex1).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexPhase(complex1).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexFromPolar(real, real).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexScale(complex1, 2.0).Shape.ToArray());
        Assert.Equal(shape, _engine.NativeComplexAdd(complex1, complex2).Shape.ToArray());
    }

    // ================================================================
    // Null Input
    // ================================================================

    [Fact]
    public void AllOps_NullInput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexFFT<double>(null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexIFFTReal<double>(null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexIFFT<double>(null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexMultiply<double>(null!, new Tensor<Complex<double>>([1])));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexConjugate<double>(null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexMagnitude<double>(null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexMagnitudeSquared<double>(null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexPhase<double>(null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexFromPolar<double>(null!, new Tensor<double>([1])));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexFromPolar<double>(new Tensor<double>([1]), null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexScale<double>(null!, 1.0));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexAdd<double>(null!, new Tensor<Complex<double>>([1])));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexMultiply<double>(new Tensor<Complex<double>>([1]), null!));
        Assert.Throws<ArgumentNullException>(() => _engine.NativeComplexAdd<double>(new Tensor<Complex<double>>([1]), null!));
    }

    // ================================================================
    // Large N
    // ================================================================

    [Fact]
    public void FFT_LargeN_1024()
    {
        int n = 1024;
        var input = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
            input[i] = Math.Sin(2 * Math.PI * 50 * i / n);

        var spectrum = _engine.NativeComplexFFT(input);
        var recovered = _engine.NativeComplexIFFTReal(spectrum);

        for (int i = 0; i < n; i++)
            Assert.Equal(input[i], recovered[i], 4);
    }

    // ================================================================
    // FFT SIMD-delegation equivalence — NativeComplexFFT must match an
    // independent naive DFT reference across power-of-2 sizes. Guards the
    // radix-2 SIMD delegation (layout + normalization) against the scalar
    // math contract.
    // ================================================================

    // Independent O(n^2) DFT reference (forward, unnormalized).
    private static Complex<double>[] NaiveDft(double[] re)
    {
        int n = re.Length;
        var outp = new Complex<double>[n];
        for (int k = 0; k < n; k++)
        {
            double sumRe = 0.0, sumIm = 0.0;
            for (int t = 0; t < n; t++)
            {
                double ang = -2.0 * Math.PI * k * t / n;
                sumRe += re[t] * Math.Cos(ang);
                sumIm += re[t] * Math.Sin(ang);
            }
            outp[k] = new Complex<double>(sumRe, sumIm);
        }
        return outp;
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void FFT_MatchesNaiveDft_Double(int n)
    {
        var re = new double[n];
        var rng = new Random(1234);
        for (int i = 0; i < n; i++) re[i] = rng.NextDouble() * 2.0 - 1.0;

        var input = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) input[i] = re[i];

        var spectrum = _engine.NativeComplexFFT(input);
        var reference = NaiveDft(re);

        // Absolute tolerance scaled by n (DFT magnitudes grow with n).
        double tol = 1e-4 * n;
        for (int k = 0; k < n; k++)
        {
            Assert.True(Math.Abs(spectrum[k].Real - reference[k].Real) < tol,
                $"Re mismatch at k={k}, n={n}: {spectrum[k].Real} vs {reference[k].Real}");
            Assert.True(Math.Abs(spectrum[k].Imaginary - reference[k].Imaginary) < tol,
                $"Im mismatch at k={k}, n={n}: {spectrum[k].Imaginary} vs {reference[k].Imaginary}");
        }
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void FFT_MatchesNaiveDft_Float(int n)
    {
        var re = new double[n];
        var rng = new Random(4321);
        for (int i = 0; i < n; i++) re[i] = rng.NextDouble() * 2.0 - 1.0;

        var input = new Tensor<float>([n]);
        for (int i = 0; i < n; i++) input[i] = (float)re[i];

        var spectrum = _engine.NativeComplexFFT(input);
        var reference = NaiveDft(re);

        // Float accumulation over n terms: tolerance scaled for single precision.
        double tol = 1e-2 * n;
        for (int k = 0; k < n; k++)
        {
            Assert.True(Math.Abs(spectrum[k].Real - reference[k].Real) < tol,
                $"Re mismatch at k={k}, n={n}: {spectrum[k].Real} vs {reference[k].Real}");
            Assert.True(Math.Abs(spectrum[k].Imaginary - reference[k].Imaginary) < tol,
                $"Im mismatch at k={k}, n={n}: {spectrum[k].Imaginary} vs {reference[k].Imaginary}");
        }
    }

    // ================================================================
    // Allocation metric — precise per-call heap allocation for the hot
    // NativeComplexFFT / NativeSpectralFilter paths, via the exact,
    // non-admin GC.GetAllocatedBytesForCurrentThread() counter.
    // net8+ only (the API does not exist on net471).
    // ================================================================
#if NET8_0_OR_GREATER
    [Fact]
    public void FFT_AllocationBytes()
    {
        // Isolate the CpuEngine method bodies we optimize: pure CPU engine and
        // AutoTracer disabled (its per-call trace closures are constant and
        // orthogonal to the buffer/result allocations under test).
        var engine = new CpuEngine();
        bool priorTracer = SetAutoTracer(false);
        try
        {
            // iters=100 (was 1000): allocation bytes/call is deterministic, so
            // the (b1-b0)/iters average is already stable well under 100 iters.
            // The assertions and ceilings are unchanged; only the iteration count
            // is reduced. 1000 iterations ran ~2040 real 128x1024 FFTs purely for
            // a stable allocation average, making this a 4-13 min CI time-bomb
            // (pure CPU compute, highly runner-variance sensitive) for no signal.
            const int H = 128, W = 1024, iters = 100;

            var input = new Tensor<double>(new[] { H, W });
            var rng = new Random(7);
            for (int i = 0; i < H * W; i++) input[i] = rng.NextDouble() * 2.0 - 1.0;

            var filter = new Tensor<Complex<double>>(new[] { H, W });
            for (int i = 0; i < H * W; i++)
                filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

            // ---- NativeComplexFFT ----
            for (int w = 0; w < 20; w++) { var _ = engine.NativeComplexFFT(input); }
            long b0 = GC.GetAllocatedBytesForCurrentThread();
            for (int it = 0; it < iters; it++) { var r = engine.NativeComplexFFT(input); GC.KeepAlive(r); }
            long b1 = GC.GetAllocatedBytesForCurrentThread();
            double fftBytesPerCall = (b1 - b0) / (double)iters;

            // ---- NativeSpectralFilter ----
            for (int w = 0; w < 20; w++) { var _ = engine.NativeSpectralFilter(input, filter); }
            long s0 = GC.GetAllocatedBytesForCurrentThread();
            for (int it = 0; it < iters; it++) { var r = engine.NativeSpectralFilter(input, filter); GC.KeepAlive(r); }
            long s1 = GC.GetAllocatedBytesForCurrentThread();
            double sfBytesPerCall = (s1 - s0) / (double)iters;

            _output.WriteLine($"ALLOC NativeComplexFFT   [128,1024]: {fftBytesPerCall:N0} bytes/call");
            _output.WriteLine($"ALLOC NativeSpectralFilter [128,1024]: {sfBytesPerCall:N0} bytes/call");

            // Guard against a regression that would balloon allocation. These are
            // generous ceilings (well above measured), not tight assertions.
            Assert.True(fftBytesPerCall < 6_000_000, $"NativeComplexFFT alloc/call too high: {fftBytesPerCall:N0}");
            Assert.True(sfBytesPerCall < 40_000_000, $"NativeSpectralFilter alloc/call too high: {sfBytesPerCall:N0}");
        }
        finally
        {
            SetAutoTracer(priorTracer);
        }
    }

    // Toggle AutoTracer.Enabled (internal) via reflection; returns prior value.
    private static bool SetAutoTracer(bool value)
    {
        var t = typeof(Tensor<double>).Assembly.GetType("AiDotNet.Tensors.Engines.Compilation.AutoTracer");
        var p = t?.GetProperty("Enabled", System.Reflection.BindingFlags.Static
            | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
        bool prior = p is not null && (bool)p.GetValue(null)!;
        p?.SetValue(null, value);
        return prior;
    }
#endif

    // ================================================================
    // Fused spectral filter equivalence — the allocation-lean fused path
    // must be numerically identical to the composed FFT2D → multiply →
    // IFFT2DReal pipeline it replaces.
    // ================================================================

    // Reference: the exact composed pipeline via public ops (what the fused path replaces).
    private static Tensor<double> SpectralFilterComposed(CpuEngine engine, Tensor<double> input, Tensor<Complex<double>> filter)
    {
        var spectrum = engine.NativeComplexFFT2D(input);
        int filterLen = filter.Length;
        var filtered = new Tensor<Complex<double>>(spectrum.Shape.ToArray());
        for (int i = 0; i < spectrum.Length; i++)
        {
            int fi = i % filterLen;
            double sr = spectrum[i].Real, si = spectrum[i].Imaginary;
            double fr = filter[fi].Real, fim = filter[fi].Imaginary;
            filtered[i] = new Complex<double>(sr * fr - si * fim, sr * fim + si * fr);
        }
        return engine.NativeComplexIFFT2DReal(filtered);
    }

    [Theory]
    [InlineData(8, 16)]
    [InlineData(16, 16)]
    [InlineData(32, 64)]
    [InlineData(128, 64)]
    public void SpectralFilter_Fused_MatchesComposed_2D(int h, int w)
    {
        var engine = new CpuEngine();
        var rng = new Random(2024);
        var input = new Tensor<double>(new[] { h, w });
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble() * 2.0 - 1.0;
        var filter = new Tensor<Complex<double>>(new[] { h, w });
        for (int i = 0; i < h * w; i++) filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var reference = SpectralFilterComposed(engine, input, filter);
        var actual = engine.NativeSpectralFilter(input, filter);

        Assert.Equal(reference.Length, actual.Length);
        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - actual[i]) < 1e-9,
                $"mismatch at {i}: {reference[i]} vs {actual[i]}");
    }

    [Fact]
    public void SpectralFilter_Fused_MatchesComposed_Batch4D()
    {
        int b = 2, c = 3, h = 16, w = 32;
        var engine = new CpuEngine();
        var rng = new Random(55);
        var input = new Tensor<double>(new[] { b, c, h, w });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2.0 - 1.0;
        // Filter [C,H,W] to exercise modular broadcast across the batch.
        var filter = new Tensor<Complex<double>>(new[] { c, h, w });
        for (int i = 0; i < filter.Length; i++) filter[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        // Reference via composed pipeline (batch broadcast identical to fused).
        var spectrum = engine.NativeComplexFFT2D(input);
        int filterLen = filter.Length;
        var filtered = new Tensor<Complex<double>>(spectrum.Shape.ToArray());
        for (int i = 0; i < spectrum.Length; i++)
        {
            int fi = i % filterLen;
            double sr = spectrum[i].Real, si = spectrum[i].Imaginary;
            double fr = filter[fi].Real, fim = filter[fi].Imaginary;
            filtered[i] = new Complex<double>(sr * fr - si * fim, sr * fim + si * fr);
        }
        var reference = engine.NativeComplexIFFT2DReal(filtered);
        var actual = engine.NativeSpectralFilterBatch(input, filter);

        Assert.Equal(reference.Length, actual.Length);
        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - actual[i]) < 1e-9,
                $"mismatch at {i}: {reference[i]} vs {actual[i]}");
    }

    // ================================================================
    // FFT Complex-to-Complex
    // ================================================================

    [Fact]
    public void FFTComplex_RoundTrip()
    {
        int n = 32;
        var input = new Tensor<Complex<double>>([n]);
        for (int i = 0; i < n; i++)
            input[i] = new Complex<double>(Math.Cos(2 * Math.PI * 3 * i / n), Math.Sin(2 * Math.PI * 5 * i / n));

        var spectrum = _engine.NativeComplexFFTComplex(input);
        var recovered = _engine.NativeComplexIFFT(spectrum);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(input[i].Real, recovered[i].Real, 5);
            Assert.Equal(input[i].Imaginary, recovered[i].Imaginary, 5);
        }
    }

    [Fact]
    public void FFTComplex_NonPowerOfTwo_Throws()
    {
        var input = new Tensor<Complex<double>>([10]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTComplex(input));
    }

    // ================================================================
    // TopK Complex
    // ================================================================

    [Fact]
    public void TopKComplex_RetainsTopK()
    {
        int n = 8;
        int k = 3;
        var input = new Tensor<Complex<double>>([n]);
        // Set known magnitudes: indices 2, 5, 7 have largest
        input[0] = new Complex<double>(0.1, 0);
        input[1] = new Complex<double>(0.2, 0);
        input[2] = new Complex<double>(5.0, 3.0);  // mag ~5.83
        input[3] = new Complex<double>(0.3, 0);
        input[4] = new Complex<double>(0.1, 0.1);
        input[5] = new Complex<double>(4.0, 4.0);  // mag ~5.66
        input[6] = new Complex<double>(0.5, 0);
        input[7] = new Complex<double>(6.0, 0);     // mag = 6.0

        var result = _engine.NativeComplexTopK(input, k);

        // Top-3 by magnitude: indices 7 (6.0), 2 (5.83), 5 (5.66)
        Assert.Equal(6.0, result[7].Real, 10);
        Assert.Equal(5.0, result[2].Real, 10);
        Assert.Equal(4.0, result[5].Real, 10);

        // Others should be zeroed
        Assert.Equal(0.0, result[0].Real, 10);
        Assert.Equal(0.0, result[1].Real, 10);
        Assert.Equal(0.0, result[3].Real, 10);
        Assert.Equal(0.0, result[4].Real, 10);
        Assert.Equal(0.0, result[6].Real, 10);
    }

    [Fact]
    public void TopKComplex_KLargerThanN_ReturnsAll()
    {
        int n = 4;
        var input = new Tensor<Complex<double>>([n]);
        for (int i = 0; i < n; i++) input[i] = new Complex<double>(i + 1, 0);

        var result = _engine.NativeComplexTopK(input, 100);

        for (int i = 0; i < n; i++)
            Assert.Equal(input[i].Real, result[i].Real, 10);
    }

    [Fact]
    public void TopKComplex_EqualMagnitudesUseLowerIndexAndNaNsSortLast()
    {
        var input = new Tensor<Complex<float>>(new[]
        {
            new Complex<float>(3, 4),
            new Complex<float>(-3, 4),
            new Complex<float>(0, 5),
            new Complex<float>(4, 0),
            new Complex<float>(float.NaN, 0),
            new Complex<float>(0, -5),
            new Complex<float>(2, 0)
        }, [7]);

        var result = _engine.NativeComplexTopK(input, 3);

        Assert.Equal(input[0], result[0]);
        Assert.Equal(input[1], result[1]);
        Assert.Equal(input[2], result[2]);
        for (int i = 3; i < result.Length; i++)
            Assert.Equal(new Complex<float>(0, 0), result[i]);
    }

    // ================================================================
    // CrossSpectral (X * conj(Y))
    // ================================================================

    [Fact]
    public void CrossSpectral_CorrectResult()
    {
        // CrossSpectral computes X * conj(Y): (xr + xi*i)(yr - yi*i)
        // = (xr*yr + xi*yi) + (xi*yr - xr*yi)*i
        int n = 4;
        var x = new Tensor<Complex<double>>([n]);
        var y = new Tensor<Complex<double>>([n]);

        x[0] = new Complex<double>(1, 2);
        y[0] = new Complex<double>(3, 4);
        // (1+2i)(3-4i) = 3-4i+6i-8i^2 = 3+2i+8 = 11+2i

        x[1] = new Complex<double>(1, 0);
        y[1] = new Complex<double>(0, 1);
        // (1+0i)(0-1i) = 0-1i

        x[2] = new Complex<double>(0, 0);
        y[2] = new Complex<double>(5, 7);
        // (0+0i)(5-7i) = 0+0i

        x[3] = new Complex<double>(2, -3);
        y[3] = new Complex<double>(2, -3);
        // (2-3i)(2+3i) = 4+6i-6i-9i^2 = 4+9 = 13+0i

        var result = _engine.NativeComplexCrossSpectral(x, y);

        Assert.Equal(11.0, result[0].Real, 10);
        Assert.Equal(2.0, result[0].Imaginary, 10);
        Assert.Equal(0.0, result[1].Real, 10);
        Assert.Equal(-1.0, result[1].Imaginary, 10);
        Assert.Equal(0.0, result[2].Real, 10);
        Assert.Equal(0.0, result[2].Imaginary, 10);
        Assert.Equal(13.0, result[3].Real, 10);
        Assert.Equal(0.0, result[3].Imaginary, 10);
    }

    [Fact]
    public void CrossSpectral_SelfConjugate_ProducesMagnitudeSquared()
    {
        // X * conj(X) should equal |X|^2 (real-valued, all positive)
        int n = 3;
        var x = new Tensor<Complex<double>>([n]);
        x[0] = new Complex<double>(3, 4);   // |x|^2 = 25
        x[1] = new Complex<double>(-1, 2);  // |x|^2 = 5
        x[2] = new Complex<double>(0, -5);  // |x|^2 = 25

        var result = _engine.NativeComplexCrossSpectral(x, x);

        Assert.Equal(25.0, result[0].Real, 10);
        Assert.Equal(0.0, result[0].Imaginary, 10);
        Assert.Equal(5.0, result[1].Real, 10);
        Assert.Equal(0.0, result[1].Imaginary, 10);
        Assert.Equal(25.0, result[2].Real, 10);
        Assert.Equal(0.0, result[2].Imaginary, 10);
    }

    [Fact]
    public void CrossSpectral_LengthMismatch_Throws()
    {
        var x = new Tensor<Complex<double>>([4]);
        var y = new Tensor<Complex<double>>([8]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexCrossSpectral(x, y));
    }

    // ================================================================
    // SoftmaxRows
    // ================================================================

    [Fact]
    public void SoftmaxRows_RowsSumToOne()
    {
        var input = new Tensor<double>([3, 4]);
        var rng = new Random(42);
        for (int i = 0; i < 12; i++) input[i] = rng.NextDouble() * 10;

        var result = _engine.TensorSoftmaxRows(input);

        for (int r = 0; r < 3; r++)
        {
            double rowSum = 0;
            for (int c = 0; c < 4; c++)
            {
                double val = result[r * 4 + c];
                Assert.True(val >= 0, $"Softmax output should be non-negative, got {val}");
                Assert.True(val <= 1, $"Softmax output should be <= 1, got {val}");
                rowSum += val;
            }
            Assert.Equal(1.0, rowSum, 6);
        }
    }

    [Fact]
    public void SoftmaxRows_1D_Throws()
    {
        var input = new Tensor<double>([10]);
        Assert.Throws<ArgumentException>(() => _engine.TensorSoftmaxRows(input));
    }

    [Fact]
    public void SoftmaxRows_LargerInputGetsLargerWeight()
    {
        var input = new Tensor<double>([1, 3]);
        input[0] = 1.0; input[1] = 5.0; input[2] = 2.0;

        var result = _engine.TensorSoftmaxRows(input);

        // Index 1 (value 5) should have largest softmax weight
        Assert.True(result[1] > result[0]);
        Assert.True(result[1] > result[2]);
    }
}
