using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests for new spectral/audio perf ops: span-based FFT entry points,
/// NativeAnalyticSignal (Hilbert transform), and NativeNormalizeRows.
/// Covers Issue #160 P0 items.
/// </summary>
public class SpectralPerfOpsTests
{
    private readonly IEngine _engine = new CpuEngine();

    // ================================================================
    // Span-based FFT entry points
    // ================================================================

    [Theory]
    [InlineData(8)]
    [InlineData(256)]
    [InlineData(1024)]
    public void FFTSpan_Double_MatchesTensorFFT(int n)
    {
        var rng = new Random(42);
        var input = new double[n];
        for (int i = 0; i < n; i++) input[i] = rng.NextDouble() * 2 - 1;

        // Span-based path
        var spanOutput = new Complex<double>[n];
        _engine.NativeComplexFFTSpan<double>(input, spanOutput);

        // Tensor-based path
        var inputTensor = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) inputTensor[i] = input[i];
        var tensorOutput = _engine.NativeComplexFFT(inputTensor);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(tensorOutput[i].Real, spanOutput[i].Real, 10);
            Assert.Equal(tensorOutput[i].Imaginary, spanOutput[i].Imaginary, 10);
        }
    }

    [Theory]
    [InlineData(8)]
    [InlineData(256)]
    public void FFTSpan_Float_MatchesTensorFFT(int n)
    {
        var rng = new Random(123);
        var input = new float[n];
        for (int i = 0; i < n; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var spanOutput = new Complex<float>[n];
        _engine.NativeComplexFFTSpan<float>(input, spanOutput);

        var inputTensor = new Tensor<float>([n]);
        for (int i = 0; i < n; i++) inputTensor[i] = input[i];
        var tensorOutput = _engine.NativeComplexFFT(inputTensor);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(tensorOutput[i].Real, spanOutput[i].Real, 3);
            Assert.Equal(tensorOutput[i].Imaginary, spanOutput[i].Imaginary, 3);
        }
    }

    [Fact]
    public void FFTSpan_IFFT_RoundTripRecoversOriginal()
    {
        int n = 256;
        var rng = new Random(77);
        var input = new double[n];
        for (int i = 0; i < n; i++) input[i] = rng.NextDouble();

        var spectrum = new Complex<double>[n];
        _engine.NativeComplexFFTSpan<double>(input, spectrum);

        var recovered = new Complex<double>[n];
        _engine.NativeComplexIFFTSpan<double>(spectrum, recovered);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(input[i], recovered[i].Real, 10);
            Assert.Equal(0.0, recovered[i].Imaginary, 10);
        }
    }

    [Fact]
    public void FFTSpan_ComplexToComplexFFT_MatchesExpected()
    {
        int n = 32;
        var rng = new Random(55);
        var input = new Complex<double>[n];
        for (int i = 0; i < n; i++)
            input[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var output = new Complex<double>[n];
        _engine.NativeComplexFFTComplexSpan<double>(input, output);

        // Compare to tensor-based complex FFT
        var inputTensor = new Tensor<Complex<double>>([n]);
        for (int i = 0; i < n; i++) inputTensor[i] = input[i];
        var tensorOutput = _engine.NativeComplexFFTComplex(inputTensor);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(tensorOutput[i].Real, output[i].Real, 10);
            Assert.Equal(tensorOutput[i].Imaginary, output[i].Imaginary, 10);
        }
    }

    [Fact]
    public void FFTSpan_NonPowerOfTwo_Throws()
    {
        var input = new double[15];
        var output = new Complex<double>[15];
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTSpan<double>(input, output));
    }

    [Fact]
    public void FFTSpan_LengthMismatch_Throws()
    {
        var input = new double[8];
        var output = new Complex<double>[16];
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTSpan<double>(input, output));
    }

    // ================================================================
    // NativeAnalyticSignal (Hilbert transform)
    // ================================================================

    [Fact]
    public void AnalyticSignal_RealPart_MatchesInput()
    {
        // The analytic signal z(t) = x(t) + iH{x}(t) has Re{z} = x
        int n = 128;
        var input = new Tensor<double>([n]);
        var rng = new Random(42);
        for (int i = 0; i < n; i++) input[i] = rng.NextDouble() * 2 - 1;

        var analytic = _engine.NativeAnalyticSignal(input);

        for (int i = 0; i < n; i++)
            Assert.Equal(input[i], analytic[i].Real, 8);
    }

    [Fact]
    public void AnalyticSignal_Cosine_ProducesSineInImag()
    {
        // Hilbert transform of cos(ω t) is sin(ω t) — so analytic of cos is cos + i*sin
        int n = 256;
        double freq = 4; // cycles per period
        var input = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
            input[i] = Math.Cos(2.0 * Math.PI * freq * i / n);

        var analytic = _engine.NativeAnalyticSignal(input);

        // Check a few interior points (avoid edge effects from discrete FFT)
        for (int i = 32; i < n - 32; i += 8)
        {
            double expectedSin = Math.Sin(2.0 * Math.PI * freq * i / n);
            Assert.Equal(expectedSin, analytic[i].Imaginary, 8);
        }
    }

    [Fact]
    public void AnalyticSignal_BandLimited_ZerosOutsideBand()
    {
        // Pass a two-frequency cosine and keep only one of them — verify magnitude drops at other freq
        int n = 512;
        double sr = 1000.0; // 1 kHz
        double f1 = 50.0, f2 = 200.0;
        var input = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
            input[i] = Math.Cos(2.0 * Math.PI * f1 * i / sr) + Math.Cos(2.0 * Math.PI * f2 * i / sr);

        // Keep only f1 band
        var analytic = _engine.NativeAnalyticSignal(input, freqLow: 30.0, freqHigh: 100.0, sampleRate: sr);

        // Envelope magnitude should be roughly 1 (for a single cosine at f1) at interior points
        double meanMag = 0.0;
        int sampleCount = 0;
        for (int i = 100; i < n - 100; i += 10)
        {
            double mag = Math.Sqrt(analytic[i].Real * analytic[i].Real + analytic[i].Imaginary * analytic[i].Imaginary);
            meanMag += mag;
            sampleCount++;
        }
        meanMag /= sampleCount;
        // Should be near 1.0 — only f1 survives
        Assert.InRange(meanMag, 0.5, 1.5);
    }

    // ================================================================
    // NativeNormalizeRows
    // ================================================================

    [Fact]
    public void NormalizeRows_Double_EachRowHasUnitNorm()
    {
        int rows = 8, cols = 32;
        var input = new Tensor<double>([rows, cols]);
        var rng = new Random(42);
        for (int i = 0; i < rows * cols; i++) input[i] = rng.NextDouble() * 10 - 5;

        var result = _engine.NativeNormalizeRows(input);

        for (int r = 0; r < rows; r++)
        {
            double sumSq = 0.0;
            for (int c = 0; c < cols; c++)
            {
                double v = result[r * cols + c];
                sumSq += v * v;
            }
            Assert.Equal(1.0, sumSq, 10);
        }
    }

    [Fact]
    public void NormalizeRows_Float_EachRowHasUnitNorm()
    {
        int rows = 4, cols = 64;
        var input = new Tensor<float>([rows, cols]);
        var rng = new Random(123);
        for (int i = 0; i < rows * cols; i++) input[i] = (float)(rng.NextDouble() * 10 - 5);

        var result = _engine.NativeNormalizeRows(input);

        for (int r = 0; r < rows; r++)
        {
            float sumSq = 0f;
            for (int c = 0; c < cols; c++)
            {
                float v = result[r * cols + c];
                sumSq += v * v;
            }
            Assert.Equal(1f, sumSq, 5);
        }
    }

    [Fact]
    public void NormalizeRows_ZeroRow_StaysZero()
    {
        int rows = 3, cols = 16;
        var input = new Tensor<double>([rows, cols]);
        // Row 0 is all zeros, rows 1 and 2 have values
        for (int c = 0; c < cols; c++)
        {
            input[cols + c] = c + 1;       // row 1
            input[2 * cols + c] = c + 10;  // row 2
        }

        var result = _engine.NativeNormalizeRows(input);

        for (int c = 0; c < cols; c++)
            Assert.Equal(0.0, result[c]);
    }

    [Fact]
    public void NormalizeRows_ThrowsOnNon2D()
    {
        var input = new Tensor<double>([8]);
        Assert.Throws<ArgumentException>(() => _engine.NativeNormalizeRows(input));
    }

    [Fact]
    public void NormalizeRows_PreservesDirection()
    {
        // A vector scaled by any positive constant has the same direction after normalization
        int cols = 32;
        var input1 = new Tensor<double>([1, cols]);
        var input2 = new Tensor<double>([1, cols]);
        var rng = new Random(77);
        for (int c = 0; c < cols; c++)
        {
            double v = rng.NextDouble();
            input1[c] = v;
            input2[c] = v * 5.0;  // scaled
        }

        var r1 = _engine.NativeNormalizeRows(input1);
        var r2 = _engine.NativeNormalizeRows(input2);

        for (int c = 0; c < cols; c++)
            Assert.Equal(r1[c], r2[c], 10);
    }
}
