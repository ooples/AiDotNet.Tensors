using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Full coverage tests for native Tensor&lt;Complex&lt;T&gt;&gt; IEngine operations.
/// </summary>
public class NativeComplexOpsTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // ================================================================
    // FFT Round-Trip
    // ================================================================

    [Fact]
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
