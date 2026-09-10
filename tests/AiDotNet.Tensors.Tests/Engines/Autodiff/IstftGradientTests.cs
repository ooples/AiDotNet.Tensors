using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Gradient guards for <c>ISTFT</c>, which is now on the tape (issue #905 item 2).
/// </summary>
/// <remarks>
/// <para>
/// Synthesis used to be classified non-differentiable on the grounds that it is a reconstruction
/// operator rather than a training-path op. STFT-consistency objectives are defined across the round
/// trip <c>|| S - STFT(ISTFT(S)) ||</c>, so that classification left half of such a loss untrainable
/// while the training loop still ran and reported a number.
/// </para>
/// <para>
/// These tests deliberately construct the engine as <c>new CpuEngine()</c> rather than taking
/// <c>AiDotNetEngine.Current</c>. Where a GPU is present the current engine evaluates a
/// <c>Tensor&lt;double&gt;</c> in single precision on a device without native fp64, and central
/// differences are worthless at that precision - the step size needed to beat fp32 noise is larger
/// than the region over which the derivative is constant. On the CPU engine the same comparison has
/// eight orders of magnitude of headroom.
/// </para>
/// </remarks>
public class IstftGradientTests
{
    private readonly IEngine _engine = new CpuEngine();

    private const int NFft = 8;
    private const int HopLength = 4;
    private const int NumFreqs = NFft / 2 + 1;
    private const int NumFrames = 3;
    /// <summary>
    /// Output length for a given centring, matching how ISTFT derives it when length is null.
    /// </summary>
    /// <remarks>
    /// Centred synthesis drops the nFft of analysis padding, which is also what moves each frame
    /// write offset back by nFft/2 and so makes the first frames write partly before sample zero.
    /// That trimming is a distinct branch in both the forward and its adjoint, which is why both
    /// settings are gradchecked below rather than only the simpler one.
    /// </remarks>
    private static int OutputLengthFor(bool center)
    {
        int length = ((NumFrames - 1) * HopLength) + NFft;
        return center ? length - NFft : length;
    }

    private const int OutputLength = ((NumFrames - 1) * HopLength) + NFft;

    /// <summary>
    /// A deliberately asymmetric window with no zero taps.
    /// </summary>
    /// <remarks>
    /// Asymmetric so that any transposed index error shows up as a wrong value rather than
    /// cancelling, and non-zero throughout so every output sample takes the normalised branch.
    /// </remarks>
    private Tensor<double> Window()
    {
        var w = new double[NFft];
        for (int i = 0; i < NFft; i++)
        {
            w[i] = 0.3 + (0.1 * i);
        }

        return new Tensor<double>(w, new[] { NFft });
    }

    private static Tensor<double> Magnitudes()
    {
        var m = new double[NumFreqs * NumFrames];
        for (int i = 0; i < m.Length; i++)
        {
            m[i] = 0.4 + (0.17 * ((i * 7) % 5));
        }

        return new Tensor<double>(m, new[] { NumFreqs, NumFrames });
    }

    private static Tensor<double> Phases()
    {
        var p = new double[NumFreqs * NumFrames];
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = -2.0 + (0.31 * ((i * 3) % 11));
        }

        return new Tensor<double>(p, new[] { NumFreqs, NumFrames });
    }

    /// <summary>Fixed, varied loss weights, so the seed gradient is not a uniform vector.</summary>
    private static Tensor<double> LossWeights(bool center)
    {
        int length = OutputLengthFor(center);
        var c = new double[length];
        for (int i = 0; i < c.Length; i++)
        {
            c[i] = 0.25 + (0.11 * ((i * 5) % 7));
        }

        return new Tensor<double>(c, new[] { length });
    }

    private double Loss(Tensor<double> magnitude, Tensor<double> phase, Tensor<double> weights, bool center)
    {
        var reconstructed = _engine.ISTFT(magnitude, phase, NFft, HopLength, Window(), center);
        double total = 0;
        for (int i = 0; i < OutputLengthFor(center); i++)
        {
            total += reconstructed[i] * weights[i];
        }

        return total;
    }

    private (Tensor<double> Magnitude, Tensor<double> Phase) AnalyticGradients(bool center)
    {
        var magnitude = Magnitudes();
        var phase = Phases();
        var weights = LossWeights(center);

        using var tape = new GradientTape<double>();
        var reconstructed = _engine.ISTFT(magnitude, phase, NFft, HopLength, Window(), center);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(reconstructed, weights), null);
        var grads = tape.ComputeGradients(loss, new[] { magnitude, phase });

        Assert.True(grads.ContainsKey(magnitude), "ISTFT recorded no gradient for magnitude");
        Assert.True(grads.ContainsKey(phase), "ISTFT recorded no gradient for phase");
        return (grads[magnitude], grads[phase]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Istft_MagnitudeGradient_MatchesFiniteDifferences(bool center)
    {
        var (analyticMagnitude, _) = AnalyticGradients(center);
        var weights = LossWeights(center);
        const double h = 1e-6;

        for (int j = 0; j < NumFreqs * NumFrames; j++)
        {
            var plus = Magnitudes();
            var minus = Magnitudes();
            plus[j] += h;
            minus[j] -= h;

            var numeric = (Loss(plus, Phases(), weights, center) - Loss(minus, Phases(), weights, center)) / (2 * h);
            Assert.Equal(numeric, analyticMagnitude[j], 1e-7);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Istft_PhaseGradient_MatchesFiniteDifferences(bool center)
    {
        var (_, analyticPhase) = AnalyticGradients(center);
        var weights = LossWeights(center);
        const double h = 1e-6;

        for (int j = 0; j < NumFreqs * NumFrames; j++)
        {
            var plus = Phases();
            var minus = Phases();
            plus[j] += h;
            minus[j] -= h;

            var numeric = (Loss(Magnitudes(), plus, weights, center) - Loss(Magnitudes(), minus, weights, center)) / (2 * h);
            Assert.Equal(numeric, analyticPhase[j], 1e-7);
        }
    }

    [Fact]
    public void Istft_GradientReachesBothInputs_NotSilentlyZero()
    {
        // The regression that matters. Before this change ISTFT sat in
        // OpRegistry.NonDifferentiableOps, so a loss downstream of it produced no gradient at all
        // and training proceeded quietly. A forward-only test would not have noticed.
        var (analyticMagnitude, analyticPhase) = AnalyticGradients(center: false);

        var magnitudeMoved = false;
        var phaseMoved = false;
        for (int j = 0; j < NumFreqs * NumFrames; j++)
        {
            if (analyticMagnitude[j] != 0.0) { magnitudeMoved = true; }
            if (analyticPhase[j] != 0.0) { phaseMoved = true; }
        }

        Assert.True(magnitudeMoved, "every magnitude gradient is zero - the tape connection is severed");
        Assert.True(phaseMoved, "every phase gradient is zero - the tape connection is severed");
    }

    [Fact]
    public void Istft_ForwardIsUnchanged()
    {
        // Recording on the tape must not perturb the value. Overlap-add with a window-sum
        // normalisation is exactly reconstructible for a constant-magnitude, zero-phase spectrum
        // only in the interior, so this pins the whole vector against a direct recomputation
        // instead of against a closed form.
        var magnitude = Magnitudes();
        var phase = Phases();
        var window = Window();

        var withoutTape = _engine.ISTFT(magnitude, phase, NFft, HopLength, window, center: false);

        double[] expected;
        using (var tape = new GradientTape<double>())
        {
            var withTape = _engine.ISTFT(magnitude, phase, NFft, HopLength, window, center: false);
            expected = new double[OutputLength];
            for (int i = 0; i < OutputLength; i++)
            {
                expected[i] = withTape[i];
            }
        }

        for (int i = 0; i < OutputLength; i++)
        {
            Assert.Equal(expected[i], withoutTape[i], 1e-12);
        }
    }

    [Theory]
    [InlineData(5)]
    [InlineData(7)]
    public void Istft_GradientMatchesFiniteDifferences_AtOddNFft(int oddNFft)
    {
        // The Hermitian extension is where an odd length could diverge: with nFft even, bin
        // numFreqs-1 is Nyquist and is deliberately not mirrored, whereas with nFft odd there is no
        // Nyquist bin and the forward still stops mirroring at numFreqs-2. The adjoint folds over
        // exactly the range the forward mirrors, so it stays the transpose either way - but that is
        // an argument, and every other gradcheck here runs at nFft 8, so nothing had tested it.
        //
        // Odd lengths also exercise the Bluestein path, since they are not powers of two. Before
        // the transform fix in this branch the forward zero-padded to the next power of two while
        // the adjoint did not, and these cases would have failed outright.
        const int hop = 2;
        const int frames = 3;
        int numFreqs = (oddNFft / 2) + 1;
        int outputLength = ((frames - 1) * hop) + oddNFft;

        var windowData = new double[oddNFft];
        for (int i = 0; i < oddNFft; i++)
        {
            windowData[i] = 0.3 + (0.1 * i);
        }

        var window = new Tensor<double>(windowData, new[] { oddNFft });

        Tensor<double> BuildMagnitudes()
        {
            var m = new double[numFreqs * frames];
            for (int i = 0; i < m.Length; i++)
            {
                m[i] = 0.4 + (0.17 * ((i * 7) % 5));
            }

            return new Tensor<double>(m, new[] { numFreqs, frames });
        }

        Tensor<double> BuildPhases()
        {
            var p = new double[numFreqs * frames];
            for (int i = 0; i < p.Length; i++)
            {
                p[i] = -2.0 + (0.31 * ((i * 3) % 11));
            }

            return new Tensor<double>(p, new[] { numFreqs, frames });
        }

        var weightData = new double[outputLength];
        for (int i = 0; i < outputLength; i++)
        {
            weightData[i] = 0.25 + (0.11 * ((i * 5) % 7));
        }

        var weights = new Tensor<double>(weightData, new[] { outputLength });

        double Loss(Tensor<double> magnitude, Tensor<double> phase)
        {
            var reconstructed = _engine.ISTFT(magnitude, phase, oddNFft, hop, window, center: false);
            double total = 0;
            for (int i = 0; i < outputLength; i++)
            {
                total += reconstructed[i] * weights[i];
            }

            return total;
        }

        var baseMagnitude = BuildMagnitudes();
        var basePhase = BuildPhases();

        using var tape = new GradientTape<double>();
        var output = _engine.ISTFT(baseMagnitude, basePhase, oddNFft, hop, window, center: false);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(output, weights), null);
        var grads = tape.ComputeGradients(loss, new[] { baseMagnitude, basePhase });

        const double h = 1e-6;
        for (int j = 0; j < numFreqs * frames; j++)
        {
            var magPlus = BuildMagnitudes();
            var magMinus = BuildMagnitudes();
            magPlus[j] += h;
            magMinus[j] -= h;
            var magNumeric = (Loss(magPlus, BuildPhases()) - Loss(magMinus, BuildPhases())) / (2 * h);
            Assert.Equal(magNumeric, grads[baseMagnitude][j], 1e-7);

            var phasePlus = BuildPhases();
            var phaseMinus = BuildPhases();
            phasePlus[j] += h;
            phaseMinus[j] -= h;
            var phaseNumeric = (Loss(BuildMagnitudes(), phasePlus) - Loss(BuildMagnitudes(), phaseMinus)) / (2 * h);
            Assert.Equal(phaseNumeric, grads[basePhase][j], 1e-7);
        }
    }

    [Fact]
    public void GriffinLim_StaysOffTheTape()
    {
        // GriffinLim calls ISTFT internally and is classified non-differentiable. Now that ISTFT
        // records, only the NoGradScope around those calls keeps that classification true - and a
        // scope that silently stopped working would show up nowhere else, because the forward value
        // is unaffected either way.
        var magnitude = Magnitudes();

        using var tape = new GradientTape<double>();
        var audio = _engine.GriffinLim(magnitude, NFft, HopLength, Window(), iterations: 2, momentum: 0.0);

        Assert.Null(audio.GradFn);
    }

    [Fact]
    public void Istft_GradientFlowsThroughAConsistencyRoundTrip()
    {
        // The shape of the objective from issue #905: synthesise, re-analyse, and define the loss
        // on the re-analysed magnitude. This only trains if synthesis is differentiable.
        var magnitude = Magnitudes();
        var phase = Phases();

        using var tape = new GradientTape<double>();
        var reconstructed = _engine.ISTFT(magnitude, phase, NFft, HopLength, Window(), center: false);
        var reanalysed = _engine.Spectrogram(reconstructed, NFft, HopLength, NFft, Window());
        var loss = _engine.ReduceSum(_engine.TensorMultiply(reanalysed, reanalysed), null);
        var grads = tape.ComputeGradients(loss, new[] { magnitude, phase });

        Assert.True(grads.ContainsKey(magnitude), "no gradient reached magnitude through the round trip");

        var moved = false;
        for (int j = 0; j < NumFreqs * NumFrames; j++)
        {
            if (grads[magnitude][j] != 0.0) { moved = true; }
        }

        Assert.True(moved, "the round-trip gradient is identically zero");
    }
}
