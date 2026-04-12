using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests for NativeComplexFFT2D, NativeComplexIFFT2DReal, NativeComplexFFTND, NativeComplexIFFTNDReal.
/// Covers Issues #135 (N-D FFT), #137 (batched 2D FFT), #139 (multi-channel batched).
/// Tests verify mathematical invariants, not just round-trip smoke tests.
/// </summary>
public class FFT2DAndNDTests
{
    private readonly IEngine _engine = new CpuEngine();

    // ================================================================
    // 2D FFT: round-trip correctness
    // ================================================================

    [Theory]
    [InlineData(4, 4)]
    [InlineData(8, 8)]
    [InlineData(4, 16)]
    [InlineData(16, 4)]
    public void FFT2D_RoundTrip_RecoverOriginal(int h, int w)
    {
        var input = new Tensor<double>([h, w]);
        var rng = new Random(42);
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble();

        var spectrum = _engine.NativeComplexFFT2D(input);
        Assert.Equal(new[] { h, w }, spectrum.Shape.ToArray());

        var recovered = _engine.NativeComplexIFFT2DReal(spectrum);
        for (int i = 0; i < h * w; i++)
            Assert.Equal(input[i], recovered[i], 6);
    }

    // ================================================================
    // 2D FFT: Parseval's theorem (energy conservation)
    // sum(|x|^2) == sum(|X|^2) / N  where N = H*W
    // ================================================================

    [Fact]
    public void FFT2D_ParsevalsTheorem_EnergyConserved()
    {
        int h = 8, w = 8;
        var input = new Tensor<double>([h, w]);
        var rng = new Random(123);
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble() * 2 - 1;

        double spatialEnergy = 0;
        for (int i = 0; i < h * w; i++)
            spatialEnergy += input[i] * input[i];

        var spectrum = _engine.NativeComplexFFT2D(input);
        double spectralEnergy = 0;
        for (int i = 0; i < h * w; i++)
        {
            double re = spectrum[i].Real;
            double im = spectrum[i].Imaginary;
            spectralEnergy += re * re + im * im;
        }
        spectralEnergy /= (h * w);

        Assert.Equal(spatialEnergy, spectralEnergy, 4);
    }

    // ================================================================
    // 2D FFT: linearity — FFT(a*x + b*y) == a*FFT(x) + b*FFT(y)
    // ================================================================

    [Fact]
    public void FFT2D_Linearity()
    {
        int h = 4, w = 8;
        var rng = new Random(456);
        var x = new Tensor<double>([h, w]);
        var y = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) { x[i] = rng.NextDouble(); y[i] = rng.NextDouble(); }

        double a = 2.5, b = -1.3;
        var combined = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) combined[i] = a * x[i] + b * y[i];

        var fftCombined = _engine.NativeComplexFFT2D(combined);
        var fftX = _engine.NativeComplexFFT2D(x);
        var fftY = _engine.NativeComplexFFT2D(y);

        for (int i = 0; i < h * w; i++)
        {
            double expectedRe = a * fftX[i].Real + b * fftY[i].Real;
            double expectedIm = a * fftX[i].Imaginary + b * fftY[i].Imaginary;
            Assert.Equal(expectedRe, fftCombined[i].Real, 5);
            Assert.Equal(expectedIm, fftCombined[i].Imaginary, 5);
        }
    }

    // ================================================================
    // 2D FFT: DC component == sum of all elements
    // ================================================================

    [Fact]
    public void FFT2D_DCComponent_EqualsSum()
    {
        int h = 8, w = 8;
        var input = new Tensor<double>([h, w]);
        var rng = new Random(789);
        double sum = 0;
        for (int i = 0; i < h * w; i++) { input[i] = rng.NextDouble(); sum += input[i]; }

        var spectrum = _engine.NativeComplexFFT2D(input);

        Assert.Equal(sum, spectrum[0].Real, 5);
        Assert.Equal(0.0, spectrum[0].Imaginary, 5);
    }

    // ================================================================
    // 2D FFT: constant input => only DC component non-zero
    // ================================================================

    [Fact]
    public void FFT2D_ConstantInput_OnlyDCNonZero()
    {
        int h = 4, w = 4;
        double c = 3.7;
        var input = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) input[i] = c;

        var spectrum = _engine.NativeComplexFFT2D(input);

        Assert.Equal(c * h * w, spectrum[0].Real, 5);
        Assert.Equal(0.0, spectrum[0].Imaginary, 5);

        for (int i = 1; i < h * w; i++)
        {
            Assert.Equal(0.0, spectrum[i].Real, 5);
            Assert.Equal(0.0, spectrum[i].Imaginary, 5);
        }
    }

    // ================================================================
    // Batched 2D FFT (Issue #137/#139)
    // ================================================================

    [Fact]
    public void FFT2D_Batched_RoundTrip()
    {
        int b = 2, h = 4, w = 8;
        var input = new Tensor<double>([b, h, w]);
        var rng = new Random(101);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var spectrum = _engine.NativeComplexFFT2D(input);
        Assert.Equal(new[] { b, h, w }, spectrum.Shape.ToArray());

        var recovered = _engine.NativeComplexIFFT2DReal(spectrum);
        for (int i = 0; i < input.Length; i++)
            Assert.Equal(input[i], recovered[i], 5);
    }

    [Fact]
    public void FFT2D_Batched_EachSliceIndependent()
    {
        // FFT2D of [2, 4, 4] should equal independent FFT2D of each [4, 4] slice
        int h = 4, w = 4;
        var rng = new Random(202);
        var slice0 = new Tensor<double>([h, w]);
        var slice1 = new Tensor<double>([h, w]);
        for (int i = 0; i < h * w; i++) { slice0[i] = rng.NextDouble(); slice1[i] = rng.NextDouble(); }

        var batched = new Tensor<double>([2, h, w]);
        for (int i = 0; i < h * w; i++) { batched[i] = slice0[i]; batched[h * w + i] = slice1[i]; }

        var batchedFFT = _engine.NativeComplexFFT2D(batched);
        var singleFFT0 = _engine.NativeComplexFFT2D(slice0);
        var singleFFT1 = _engine.NativeComplexFFT2D(slice1);

        for (int i = 0; i < h * w; i++)
        {
            Assert.Equal(singleFFT0[i].Real, batchedFFT[i].Real, 6);
            Assert.Equal(singleFFT0[i].Imaginary, batchedFFT[i].Imaginary, 6);
            Assert.Equal(singleFFT1[i].Real, batchedFFT[h * w + i].Real, 6);
            Assert.Equal(singleFFT1[i].Imaginary, batchedFFT[h * w + i].Imaginary, 6);
        }
    }

    [Fact]
    public void FFT2D_MultiChannelBatched_RoundTrip()
    {
        // [B=2, C=3, H=4, W=4] — per Issue #139 (vision model shape)
        var input = new Tensor<double>([2, 3, 4, 4]);
        var rng = new Random(303);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var spectrum = _engine.NativeComplexFFT2D(input);
        Assert.Equal(new[] { 2, 3, 4, 4 }, spectrum.Shape.ToArray());

        var recovered = _engine.NativeComplexIFFT2DReal(spectrum);
        for (int i = 0; i < input.Length; i++)
            Assert.Equal(input[i], recovered[i], 5);
    }

    // ================================================================
    // N-D FFT (Issue #135)
    // ================================================================

    [Fact]
    public void FFTND_1D_MatchesNativeComplexFFT()
    {
        int n = 16;
        var input = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) input[i] = Math.Sin(2 * Math.PI * 3 * i / n);

        var spectrumND = _engine.NativeComplexFFTND(input, new[] { -1 });
        var spectrum1D = _engine.NativeComplexFFT(input);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(spectrum1D[i].Real, spectrumND[i].Real, 6);
            Assert.Equal(spectrum1D[i].Imaginary, spectrumND[i].Imaginary, 6);
        }
    }

    [Fact]
    public void FFTND_2D_MatchesFFT2D()
    {
        int h = 4, w = 8;
        var input = new Tensor<double>([h, w]);
        var rng = new Random(404);
        for (int i = 0; i < h * w; i++) input[i] = rng.NextDouble();

        var spectrum2D = _engine.NativeComplexFFT2D(input);
        var spectrumND = _engine.NativeComplexFFTND(input, new[] { -2, -1 });

        for (int i = 0; i < h * w; i++)
        {
            Assert.Equal(spectrum2D[i].Real, spectrumND[i].Real, 5);
            Assert.Equal(spectrum2D[i].Imaginary, spectrumND[i].Imaginary, 5);
        }
    }

    [Fact]
    public void FFTND_3D_RoundTrip()
    {
        var input = new Tensor<double>([4, 4, 4]);
        var rng = new Random(505);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var spectrum = _engine.NativeComplexFFTND(input, new[] { 0, 1, 2 });
        var recovered = _engine.NativeComplexIFFTNDReal(spectrum, new[] { 0, 1, 2 });

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(input[i], recovered[i], 4);
    }

    [Fact]
    public void FFTND_3D_ParsevalsTheorem()
    {
        var input = new Tensor<double>([4, 4, 4]);
        var rng = new Random(606);
        double spatialEnergy = 0;
        for (int i = 0; i < input.Length; i++) { input[i] = rng.NextDouble(); spatialEnergy += input[i] * input[i]; }

        var spectrum = _engine.NativeComplexFFTND(input, new[] { 0, 1, 2 });
        double spectralEnergy = 0;
        for (int i = 0; i < input.Length; i++)
        {
            spectralEnergy += spectrum[i].Real * spectrum[i].Real +
                              spectrum[i].Imaginary * spectrum[i].Imaginary;
        }
        spectralEnergy /= input.Length;

        Assert.Equal(spatialEnergy, spectralEnergy, 3);
    }

    [Fact]
    public void FFTND_PartialAxes_RoundTrip()
    {
        // FFT over axes 0 and 2, skip axis 1 (axis 1 length doesn't need to be power of 2)
        var input = new Tensor<double>([4, 3, 8]);
        var rng = new Random(707);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var spectrum = _engine.NativeComplexFFTND(input, new[] { 0, 2 });
        var recovered = _engine.NativeComplexIFFTNDReal(spectrum, new[] { 0, 2 });

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(input[i], recovered[i], 4);
    }

    // ================================================================
    // Edge cases
    // ================================================================

    [Fact]
    public void FFT2D_NonPowerOf2Height_Throws()
    {
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFT2D(new Tensor<double>([3, 4])));
    }

    [Fact]
    public void FFT2D_NonPowerOf2Width_Throws()
    {
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFT2D(new Tensor<double>([4, 5])));
    }

    [Fact]
    public void FFT2D_1DInput_Throws()
    {
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFT2D(new Tensor<double>([8])));
    }

    [Fact]
    public void FFTND_EmptyAxes_Throws()
    {
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTND(new Tensor<double>([4, 4]), Array.Empty<int>()));
    }

    [Fact]
    public void FFTND_InvalidAxis_Throws()
    {
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTND(new Tensor<double>([4, 4]), new[] { 5 }));
    }

    [Fact]
    public void FFTND_NegativeInvalidAxis_Throws()
    {
        // -3 is out of range for a rank-2 tensor
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTND(new Tensor<double>([4, 4]), new[] { -3 }));
    }

    [Fact]
    public void IFFTND_EmptyAxes_Throws()
    {
        var spectrum = new Tensor<Complex<double>>([4, 4]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexIFFTNDReal(spectrum, Array.Empty<int>()));
    }

    [Fact]
    public void IFFTND_InvalidAxis_Throws()
    {
        var spectrum = new Tensor<Complex<double>>([4, 4]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexIFFTNDReal(spectrum, new[] { 5 }));
    }

    [Fact]
    public void IFFTND_NegativeInvalidAxis_Throws()
    {
        var spectrum = new Tensor<Complex<double>>([4, 4]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexIFFTNDReal(spectrum, new[] { -3 }));
    }

    [Fact]
    public void FFTND_DuplicateAxes_Throws()
    {
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTND(new Tensor<double>([4, 4]), new[] { 0, 0 }));
    }

    [Fact]
    public void FFTND_DuplicateAxesNegative_Throws()
    {
        // -1 and 1 are the same axis for rank-2
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexFFTND(new Tensor<double>([4, 4]), new[] { -1, 1 }));
    }

    [Fact]
    public void IFFTND_DuplicateAxes_Throws()
    {
        var spectrum = new Tensor<Complex<double>>([4, 4]);
        Assert.Throws<ArgumentException>(() => _engine.NativeComplexIFFTNDReal(spectrum, new[] { 1, 1 }));
    }

    [Fact]
    public void FFTND_NegativeAndPositiveEquivalent()
    {
        var input = new Tensor<double>([4, 8]);
        var rng = new Random(808);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var resultPos = _engine.NativeComplexFFTND(input, new[] { 1 });
        var resultNeg = _engine.NativeComplexFFTND(input, new[] { -1 });

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(resultPos[i].Real, resultNeg[i].Real, 10);
            Assert.Equal(resultPos[i].Imaginary, resultNeg[i].Imaginary, 10);
        }
    }
}
