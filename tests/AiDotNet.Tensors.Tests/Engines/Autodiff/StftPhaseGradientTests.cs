using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Gradient guards for <c>StftPhase</c>, the phase counterpart of <c>Spectrogram</c>.
/// </summary>
/// <remarks>
/// <para>
/// Issue #905 wants a differentiable path to an angle. <c>STFT</c> emits phase through an
/// <c>out</c> parameter and is deliberately unrecorded, so the only differentiable analysis output
/// was magnitude, via <c>Spectrogram</c>. An objective defined on phase had nothing to attach to.
/// </para>
/// <para>
/// Built on <c>new CpuEngine()</c> for the same reason as the ISTFT gradchecks: where a GPU is
/// present the current engine evaluates a <c>Tensor&lt;double&gt;</c> in single precision on a
/// device without native fp64, and central differences are worthless at that precision.
/// </para>
/// </remarks>
public class StftPhaseGradientTests
{
    private readonly IEngine _engine = new CpuEngine();

    private const int NFft = 16;
    private const int HopLength = 4;
    private const int SignalLength = 64;

    private static Tensor<double> Window()
    {
        var w = new double[NFft];
        for (int i = 0; i < NFft; i++)
        {
            w[i] = 0.5 - (0.5 * Math.Cos(2.0 * Math.PI * i / (NFft - 1)));
        }

        return new Tensor<double>(w, new[] { NFft });
    }

    /// <summary>
    /// A deterministic waveform with no near-silent bins.
    /// </summary>
    /// <remarks>
    /// The phase derivative carries a 1/|C| factor, so a bin whose magnitude approaches zero has an
    /// unbounded true derivative and finite differences there measure nothing meaningful. Two
    /// incommensurate tones plus an offset keep every bin comfortably excited.
    /// </remarks>
    private static Tensor<double> Waveform()
    {
        var x = new double[SignalLength];
        for (int i = 0; i < SignalLength; i++)
        {
            x[i] = 0.6 + Math.Sin(0.37 * i) + (0.4 * Math.Cos(1.13 * i));
        }

        return new Tensor<double>(x, new[] { SignalLength });
    }

    private double Loss(Tensor<double> waveform)
    {
        var phase = _engine.StftPhase(waveform, NFft, HopLength, NFft, Window());
        double total = 0;
        for (int i = 0; i < phase.Length; i++)
        {
            total += phase[i];
        }

        return total;
    }

    private Tensor<double> AnalyticGradient()
    {
        var waveform = Waveform();
        using var tape = new GradientTape<double>();
        var phase = _engine.StftPhase(waveform, NFft, HopLength, NFft, Window());
        var loss = _engine.ReduceSum(phase, null);
        var grads = tape.ComputeGradients(loss, new[] { waveform });

        Assert.True(grads.ContainsKey(waveform), "StftPhase recorded no gradient for the waveform");
        return grads[waveform];
    }

    [Fact]
    public void StftPhase_MatchesTheStftPhaseOutput()
    {
        var waveform = Waveform();
        _engine.STFT(waveform, NFft, HopLength, Window(), center: true, out _, out var expected);

        var actual = _engine.StftPhase(waveform, NFft, HopLength, NFft, Window());

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 1e-12);
        }
    }

    [Fact]
    public void StftPhase_ShapeMatchesSpectrogram()
    {
        // The two must be usable together bin for bin - that is the whole point of adding this one.
        var waveform = Waveform();
        var magnitude = _engine.Spectrogram(waveform, NFft, HopLength, NFft, Window());
        var phase = _engine.StftPhase(waveform, NFft, HopLength, NFft, Window());

        Assert.Equal(magnitude.Shape.ToArray(), phase.Shape.ToArray());
    }

    /// <summary>
    /// Central difference at one step size.
    /// </summary>
    private double CentralDifference(int j, double h)
    {
        var plus = Waveform();
        var minus = Waveform();
        plus[j] += h;
        minus[j] -= h;
        return (Loss(plus) - Loss(minus)) / (2 * h);
    }

    [Fact]
    public void StftPhase_Gradient_MatchesFiniteDifferences()
    {
        var analytic = AnalyticGradient();
        const double h = 1e-6;

        for (int j = 0; j < SignalLength; j++)
        {
            // Richardson extrapolation rather than a bare central difference. A central difference
            // is O(h^2) accurate, and wrapped phase is nonlinear enough that at h = 1e-6 that error
            // reached 1.2e-6 in absolute terms - larger than the disagreement worth detecting.
            // Combining two step sizes as (4 D(h/2) - D(h)) / 3 cancels the h^2 term and leaves
            // O(h^4). Measured over all 64 samples, the worst relative disagreement falls from
            // 1.07e-6 to 1.41e-7 - so the threshold below sits about 7x above the observed noise
            // rather than being widened until the test happened to pass.
            //
            // What remains is subtraction roundoff, not a systematic error. Two things say so: the
            // discrepancy SHRANK with the step size, which a missing term would not do, and its
            // sign is balanced across the samples (31 positive, 33 negative) where a missing term
            // would bias one way. A wrong derivative misses by O(1) relative, six orders above this.
            var coarse = CentralDifference(j, h);
            var fine = CentralDifference(j, h / 2);
            var numeric = ((4 * fine) - coarse) / 3;

            var scale = Math.Max(1.0, Math.Max(Math.Abs(numeric), Math.Abs(analytic[j])));
            Assert.True(
                Math.Abs(numeric - analytic[j]) / scale < 1e-6,
                $"waveform[{j}]: analytic {analytic[j]:G10} vs numeric {numeric:G10}");
        }
    }

    [Fact]
    public void StftPhase_GradientReachesTheWaveform_NotSilentlyZero()
    {
        // The regression that matters: before this op existed the only way to reach phase was the
        // unrecorded STFT out-parameter, which produced no gradient at all while training ran.
        var analytic = AnalyticGradient();

        var moved = false;
        for (int j = 0; j < SignalLength; j++)
        {
            if (analytic[j] != 0.0) { moved = true; }
        }

        Assert.True(moved, "every waveform gradient is zero - the tape connection is severed");
    }

    /// <summary>
    /// Fixed, varied weights, so the seed gradient is not uniform.
    /// </summary>
    private static Tensor<double> RoundTripWeights(int length)
    {
        var w = new double[length];
        for (int i = 0; i < length; i++)
        {
            w[i] = 0.25 + (0.11 * ((i * 5) % 7));
        }

        return new Tensor<double>(w, new[] { length });
    }

    [Fact]
    public void StftPhase_AndIstft_CloseTheRoundTrip()
    {
        // Both inputs of ISTFT are now differentiable functions of the waveform, so a consistency
        // objective defined across analysis and synthesis trains end to end.
        //
        // The loss is built from the reconstruction ALONE. An obvious spelling of this test uses the
        // residual (reconstructed - waveform), which is wrong here: that subtraction gives the
        // waveform a direct edge to the loss, so its gradient stays non-zero even with the whole
        // round trip severed. Measured with StopGradient across the chain, the residual form still
        // reported a non-zero waveform gradient - it was passing for a reason unrelated to what it
        // claimed. Asserting on the two intermediates as well pins each leg of the chain.
        var waveform = Waveform();
        var weights = RoundTripWeights(SignalLength);

        using var tape = new GradientTape<double>();
        var magnitude = _engine.Spectrogram(waveform, NFft, HopLength, NFft, Window());
        var phase = _engine.StftPhase(waveform, NFft, HopLength, NFft, Window());
        var reconstructed = _engine.ISTFT(magnitude, phase, NFft, HopLength, Window(), center: true, SignalLength);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(reconstructed, weights), null);
        var grads = tape.ComputeGradients(loss, new[] { waveform, magnitude, phase });

        AssertMoved(grads, waveform, "waveform", SignalLength);
        AssertMoved(grads, magnitude, "magnitude", magnitude.Length);
        AssertMoved(grads, phase, "phase", phase.Length);
    }

    private static void AssertMoved(
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads,
        Tensor<double> tensor,
        string name,
        int length)
    {
        Assert.True(grads.ContainsKey(tensor), $"no gradient reached {name} through the round trip");

        for (int j = 0; j < length; j++)
        {
            if (grads[tensor][j] != 0.0)
            {
                return;
            }
        }

        Assert.Fail($"every {name} gradient is zero - that leg of the round trip is severed");
    }
}
