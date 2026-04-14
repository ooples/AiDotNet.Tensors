using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests for P1-P3 spectral/audio perf ops: transcendentals, bispectrum/trispectrum,
/// batched cavity forward, and fused MFCC/Wideband/PAC feature pipelines.
/// Covers Issue #160 P1-P3 items.
/// </summary>
public class SpectralPerfOpsExtendedTests
{
    private readonly IEngine _engine = new CpuEngine();

    // ================================================================
    // NativeTanh / NativeExp / NativeAtan2 / NativeMagnitudeAndPhase
    // ================================================================

    [Fact]
    public void NativeTanh_Double_MatchesMathTanh()
    {
        var input = new Tensor<double>([16]);
        for (int i = 0; i < 16; i++) input[i] = i * 0.1 - 0.8;

        var result = _engine.NativeTanh(input);

        for (int i = 0; i < 16; i++)
            Assert.Equal(Math.Tanh(input[i]), result[i], 5);
    }

    [Fact]
    public void NativeTanh_Float_MatchesMathTanh()
    {
        var input = new Tensor<float>([16]);
        for (int i = 0; i < 16; i++) input[i] = i * 0.1f - 0.8f;

        var result = _engine.NativeTanh(input);

        for (int i = 0; i < 16; i++)
            Assert.Equal(MathF.Tanh(input[i]), result[i], 3);
    }

    [Fact]
    public void NativeExp_Double_MatchesMathExp()
    {
        var input = new Tensor<double>([16]);
        for (int i = 0; i < 16; i++) input[i] = i * 0.1 - 0.8;

        var result = _engine.NativeExp(input);

        for (int i = 0; i < 16; i++)
            Assert.Equal(Math.Exp(input[i]), result[i], 5);
    }

    [Fact]
    public void NativeAtan2_Double_MatchesMathAtan2()
    {
        int n = 16;
        var imag = new Tensor<double>([n]);
        var real = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
        {
            imag[i] = Math.Sin(i * 0.3);
            real[i] = Math.Cos(i * 0.3);
        }

        var result = _engine.NativeAtan2(imag, real);

        for (int i = 0; i < n; i++)
            Assert.Equal(Math.Atan2(imag[i], real[i]), result[i], 10);
    }

    [Fact]
    public void NativeMagnitudeAndPhase_DecomposesComplex()
    {
        int n = 8;
        var input = new Tensor<Complex<double>>([n]);
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
            input[i] = new Complex<double>(rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1);

        var mag = _engine.NativeMagnitudeAndPhase(input, out var phase);

        for (int i = 0; i < n; i++)
        {
            double expectedMag = Math.Sqrt(input[i].Real * input[i].Real + input[i].Imaginary * input[i].Imaginary);
            double expectedPhase = Math.Atan2(input[i].Imaginary, input[i].Real);
            Assert.Equal(expectedMag, mag[i], 10);
            Assert.Equal(expectedPhase, phase[i], 10);
        }
    }

    // ================================================================
    // NativeBispectrum / NativeTrispectrum
    // ================================================================

    [Fact]
    public void NativeBispectrum_Shape_IsCorrect()
    {
        int n = 32;
        var spec = new Tensor<Complex<double>>([n]);
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
            spec[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        int maxF1 = 8, maxF2 = 8;
        var result = _engine.NativeBispectrum(spec, maxF1, maxF2);

        Assert.Equal(new[] { maxF1, maxF2 }, result.Shape.ToArray());
    }

    [Fact]
    public void NativeBispectrum_MatchesManualFormula()
    {
        int n = 16;
        var spec = new Tensor<Complex<double>>([n]);
        var rng = new Random(77);
        for (int i = 0; i < n; i++)
            spec[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        int maxF1 = 4, maxF2 = 4;
        var result = _engine.NativeBispectrum(spec, maxF1, maxF2);

        // Verify a few entries against the formula B(f1,f2) = X(f1) * X(f2) * conj(X(f1+f2))
        for (int f1 = 0; f1 < maxF1; f1++)
        {
            for (int f2 = 0; f2 < maxF2; f2++)
            {
                var x1 = spec[f1];
                var x2 = spec[f2];
                var x12 = spec[f1 + f2];
                // x1 * x2
                double t1r = x1.Real * x2.Real - x1.Imaginary * x2.Imaginary;
                double t1i = x1.Real * x2.Imaginary + x1.Imaginary * x2.Real;
                // * conj(x12)
                double br = t1r * x12.Real + t1i * x12.Imaginary;
                double bi = -t1r * x12.Imaginary + t1i * x12.Real;

                var b = result[f1 * maxF2 + f2];
                Assert.Equal(br, b.Real, 10);
                Assert.Equal(bi, b.Imaginary, 10);
            }
        }
    }

    [Fact]
    public void NativeTrispectrum_Shape_IsCorrect()
    {
        int n = 32;
        var spec = new Tensor<Complex<double>>([n]);
        var rng = new Random(55);
        for (int i = 0; i < n; i++)
            spec[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        int maxF1 = 4, maxF2 = 4, maxF3 = 4;
        var result = _engine.NativeTrispectrum(spec, maxF1, maxF2, maxF3);
        Assert.Equal(new[] { maxF1, maxF2, maxF3 }, result.Shape.ToArray());
    }

    [Fact]
    public void NativeBispectrum_ThrowsOnOutOfRange()
    {
        int n = 16;
        var spec = new Tensor<Complex<double>>([n]);
        Assert.Throws<ArgumentException>(() => _engine.NativeBispectrum(spec, 10, 10));
    }

    // ================================================================
    // NativeBatchedCavityForward
    // ================================================================

    [Fact]
    public void NativeBatchedCavityForward_Shape_IsCorrect()
    {
        int batch = 2, n = 16, numCavities = 3, numBounces = 2;
        var input = new Tensor<double>([batch, n]);
        var filters = new Tensor<Complex<double>>([numCavities, n]);
        var rng = new Random(42);
        for (int i = 0; i < batch * n; i++) input[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < numCavities * n; i++)
            filters[i] = new Complex<double>(rng.NextDouble(), rng.NextDouble());

        var result = _engine.NativeBatchedCavityForward(input, filters, numBounces);

        Assert.Equal(new[] { batch, numCavities, n }, result.Shape.ToArray());
        // Sanity check output values are finite
        for (int i = 0; i < result.Length; i++)
            Assert.True(!double.IsNaN(result[i]) && !double.IsInfinity(result[i]));
    }

    [Fact]
    public void NativeBatchedCavityForward_ThrowsOnInvalidInput()
    {
        var input1D = new Tensor<double>([16]);
        var filters = new Tensor<Complex<double>>([2, 16]);
        Assert.Throws<ArgumentException>(() => _engine.NativeBatchedCavityForward(input1D, filters, 1));

        var input = new Tensor<double>([1, 16]);
        Assert.Throws<ArgumentException>(() => _engine.NativeBatchedCavityForward(input, filters, 0));
    }

    // ================================================================
    // NativeMfccFeatures / NativeWidebandFeatures / NativePacFeatures
    // ================================================================

    [Fact]
    public void NativeMfccFeatures_Shape_Batched()
    {
        int batch = 2, numSamples = 512, numSegments = 4, numMfcc = 13, paddedDim = 256;
        var waveforms = new Tensor<double>([batch, numSamples]);
        var rng = new Random(42);
        for (int i = 0; i < batch * numSamples; i++) waveforms[i] = rng.NextDouble() * 2 - 1;

        var result = _engine.NativeMfccFeatures(waveforms, numSegments, numMfcc, paddedDim);

        Assert.Equal(new[] { batch, numSegments * numMfcc }, result.Shape.ToArray());
        // Check output contains finite values
        for (int i = 0; i < result.Length; i++)
            Assert.True(!double.IsNaN(result[i]) && !double.IsInfinity(result[i]));
    }

    [Fact]
    public void NativeMfccFeatures_Shape_Single()
    {
        int numSamples = 256, numSegments = 2, numMfcc = 8, paddedDim = 128;
        var waveform = new Tensor<double>([numSamples]);
        for (int i = 0; i < numSamples; i++) waveform[i] = Math.Sin(i * 0.1);

        var result = _engine.NativeMfccFeatures(waveform, numSegments, numMfcc, paddedDim);

        Assert.Equal(new[] { numSegments * numMfcc }, result.Shape.ToArray());
    }

    [Fact]
    public void NativeWidebandFeatures_Shape_Batched()
    {
        int batch = 2, numSamples = 512, numSegments = 4, numBins = 20;
        var waveforms = new Tensor<double>([batch, numSamples]);
        var rng = new Random(123);
        for (int i = 0; i < batch * numSamples; i++) waveforms[i] = rng.NextDouble() * 2 - 1;

        var result = _engine.NativeWidebandFeatures(waveforms, numSegments, numBins);

        Assert.Equal(new[] { batch, numSegments * numBins }, result.Shape.ToArray());
        for (int i = 0; i < result.Length; i++)
            Assert.True(!double.IsNaN(result[i]) && !double.IsInfinity(result[i]));
    }

    [Fact]
    public void NativePacFeatures_Shape_IsCorrect()
    {
        int batch = 2, numSamples = 512;
        var waveforms = new Tensor<double>([batch, numSamples]);
        var rng = new Random(77);
        for (int i = 0; i < batch * numSamples; i++) waveforms[i] = rng.NextDouble() * 2 - 1;

        var gammaBands = new[] { (30.0, 60.0), (60.0, 100.0), (100.0, 150.0) };
        var result = _engine.NativePacFeatures(waveforms, sampleRate: 500, envelopeRate: 100,
            thetaLow: 4.0, thetaHigh: 8.0, gammaBands: gammaBands);

        Assert.Equal(new[] { batch, gammaBands.Length }, result.Shape.ToArray());
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(!double.IsNaN(result[i]) && !double.IsInfinity(result[i]));
            // PAC MI is in [0, 1]
            Assert.InRange(result[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void NativePacFeatures_StructuredSignal_HasHigherPac()
    {
        // A signal that actually has theta-gamma coupling should have higher PAC than an
        // uncoupled control — not just > 0, which any signal with nonzero gamma amplitude
        // would satisfy.
        int numSamples = 2048;
        int sampleRate = 1000;
        double thetaFreq = 6.0;  // Hz
        double gammaFreq = 80.0; // Hz

        var coupled = new Tensor<double>([numSamples]);
        var uncoupled = new Tensor<double>([numSamples]);
        var rng = new Random(42);
        for (int i = 0; i < numSamples; i++)
        {
            double t = i / (double)sampleRate;
            double theta = Math.Cos(2 * Math.PI * thetaFreq * t);
            // Coupled: gamma amplitude modulated by theta phase
            double gammaAmpCoupled = 0.5 + 0.5 * theta;
            coupled[i] = theta + gammaAmpCoupled * Math.Cos(2 * Math.PI * gammaFreq * t);
            // Uncoupled control: gamma with constant amplitude (no phase coupling) + same theta
            uncoupled[i] = theta + Math.Cos(2 * Math.PI * gammaFreq * t) + 0.05 * (rng.NextDouble() - 0.5);
        }

        var gammaBands = new[] { (60.0, 100.0) };
        var resCoupled = _engine.NativePacFeatures(coupled, sampleRate, envelopeRate: 200,
            thetaLow: 4.0, thetaHigh: 8.0, gammaBands: gammaBands);
        var resUncoupled = _engine.NativePacFeatures(uncoupled, sampleRate, envelopeRate: 200,
            thetaLow: 4.0, thetaHigh: 8.0, gammaBands: gammaBands);

        // Coupled signal must have strictly higher PAC MI than the uncoupled control.
        Assert.True(resCoupled[0] > resUncoupled[0],
            $"Expected coupled PAC > uncoupled PAC, but got coupled={resCoupled[0]}, uncoupled={resUncoupled[0]}");
    }
}
