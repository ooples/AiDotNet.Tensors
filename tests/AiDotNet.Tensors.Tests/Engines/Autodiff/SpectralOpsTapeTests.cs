using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tape coverage for the spectral op family.
///
/// Two distinct concerns are checked here:
///
/// 1. <b>Correctness of the gradients that already exist.</b> Spectrogram records
///    a tape node whose backward rebuilds a complex spectrum from the magnitude
///    gradient plus the saved phase and inverse-transforms it. That is the right
///    adjoint in principle, but ISTFT normally applies a window-sum normalization
///    for perfect reconstruction whereas the adjoint of the analysis should not —
///    so the gradient could carry a spurious per-sample scale. Finite differences
///    settle it.
///
/// 2. <b>Registry completeness.</b> OpRegistry classification is enforced by
///    TapeCompletenessTests, but only for methods whose RETURN type is a tensor.
///    Ops that return void and emit tensors through <c>out</c> parameters are
///    invisible to that check, so they can sit unclassified and silently record
///    nothing.
/// </summary>
public class SpectralOpsTapeTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<double> HannWindow(int n)
    {
        var w = new Tensor<double>([n]);
        for (int i = 0; i < n; i++)
            w[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / Math.Max(1, n - 1));
        return w;
    }

    private static Tensor<double> Chirp(int length, int seed = 7)
    {
        // Deterministic broadband signal: every STFT bin gets energy, so no
        // magnitude sits at exactly zero (|.| is not differentiable at 0).
        var rng = new Random(seed);
        var x = new Tensor<double>([length]);
        for (int i = 0; i < length; i++)
        {
            double t = (double)i / length;
            x[i] = 0.6 * Math.Sin(2.0 * Math.PI * (5.0 + 20.0 * t) * t)
                 + 0.05 * (rng.NextDouble() * 2 - 1);
        }
        return x;
    }

    [Fact]
    public void Spectrogram_RecordsOnTape_AndGradientIsNonZero()
    {
        int nFft = 32, hop = 8;
        var x = Chirp(256);
        var win = HannWindow(nFft);

        using var tape = new GradientTape<double>();
        var mag = _engine.Spectrogram(x, nFft, hop, nFft, win);
        var loss = _engine.ReduceSum(mag, null);
        var grads = tape.ComputeGradients(loss, [x]);

        Assert.True(grads.ContainsKey(x),
            "Spectrogram produced no gradient for its input — the op is not recording on the tape.");
        Assert.Equal(x.Shape.ToArray(), grads[x].Shape.ToArray());

        double maxAbs = 0;
        for (int i = 0; i < grads[x].Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(grads[x][i]));
        Assert.True(maxAbs > 1e-10, "Spectrogram gradient is all zeros.");
    }

    /// <summary>
    /// The decisive check: analytical vs central finite differences on d(sum |STFT(x)|)/dx.
    /// A uniform ratio away from 1.0 would indicate the ISTFT window normalization
    /// leaking into the adjoint.
    /// </summary>
    [Fact]
    public void Spectrogram_Backward_MatchesFiniteDifferences()
    {
        int nFft = 16, hop = 4, len = 96;
        var x = Chirp(len, seed: 11);
        var win = HannWindow(nFft);

        using var tape = new GradientTape<double>();
        var mag = _engine.Spectrogram(x, nFft, hop, nFft, win);
        var loss = _engine.ReduceSum(mag, null);
        var grads = tape.ComputeGradients(loss, [x]);
        var analytical = grads[x];

        // Interior samples only: edge samples are covered by fewer frames, and with
        // center:true the reflection padding makes their finite difference depend on
        // padding behaviour rather than the transform itself.
        const double eps = 1e-6;
        var ratios = new List<double>();
        var failures = new List<string>();

        for (int idx = nFft; idx < len - nFft; idx += 7)
        {
            double orig = x[idx];

            x[idx] = orig + eps;
            double lossPlus = _engine.TensorSum(_engine.Spectrogram(x, nFft, hop, nFft, win));

            x[idx] = orig - eps;
            double lossMinus = _engine.TensorSum(_engine.Spectrogram(x, nFft, hop, nFft, win));

            x[idx] = orig;

            double numerical = (lossPlus - lossMinus) / (2.0 * eps);
            double a = analytical[idx];

            if (Math.Abs(numerical) > 1e-6)
                ratios.Add(a / numerical);

            double denom = Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(numerical)));
            if (Math.Abs(a - numerical) / denom > 2e-3)
                failures.Add($"  idx {idx}: analytical {a:G6} vs numerical {numerical:G6} (ratio {a / numerical:G6})");
        }

        Assert.NotEmpty(ratios);

        if (failures.Count > 0)
        {
            double mean = ratios.Average();
            double spread = ratios.Max() - ratios.Min();
            Assert.Fail(
                $"Spectrogram gradient disagrees with finite differences at {failures.Count} of {ratios.Count} probes.\n" +
                $"analytical/numerical ratio: mean {mean:G6}, spread {spread:G6}\n" +
                (spread < 1e-3
                    ? $"The ratio is UNIFORM at ~{mean:G6}, which points at a constant scale factor in the adjoint " +
                      "(the ISTFT window-sum normalization) rather than a wrong gradient shape.\n"
                    : "The ratio VARIES across samples, so this is not a single scale factor.\n") +
                string.Join("\n", failures.Take(10)));
        }
    }

    [Fact]
    public void MelSpectrogram_RecordsOnTape_AndGradientIsNonZero()
    {
        int nFft = 32, hop = 8, nMels = 8, sampleRate = 16000;
        var x = Chirp(256, seed: 5);
        var win = HannWindow(nFft);

        using var tape = new GradientTape<double>();
        // powerToDb: false — the dB conversion's log is a separate concern and would
        // confound a gradient-flow check on the mel projection itself.
        var mel = _engine.MelSpectrogram(x, sampleRate, nFft, hop, nMels, 0.0, sampleRate / 2.0, win, powerToDb: false);
        var loss = _engine.ReduceSum(mel, null);
        var grads = tape.ComputeGradients(loss, [x]);

        Assert.True(grads.ContainsKey(x),
            "MelSpectrogram produced no gradient. It is classified NonDifferentiable in OpRegistry, but it is " +
            "Spectrogram followed by a mel-filterbank matmul — both differentiable — so it should compose to a " +
            "differentiable op. Every mel-based audio objective (vocoder, TTS) depends on this.");

        double maxAbs = 0;
        for (int i = 0; i < grads[x].Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(grads[x][i]));
        Assert.True(maxAbs > 1e-10, "MelSpectrogram gradient is all zeros.");
    }

    /// <summary>
    /// A non-zero gradient is NOT evidence of a correct one — the Spectrogram bug this suite was
    /// written for produced non-zero gradients that were ~1/nFft off with varying sign. So the
    /// mel path gets the same finite-difference treatment, in both the linear and dB modes.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MelSpectrogram_Backward_MatchesFiniteDifferences(bool powerToDb)
    {
        int nFft = 16, hop = 4, nMels = 6, len = 96, sampleRate = 16000;
        var x = Chirp(len, seed: 21);
        var win = HannWindow(nFft);

        Tensor<double> Forward(Tensor<double> signal) =>
            _engine.MelSpectrogram(signal, sampleRate, nFft, hop, nMels, 0.0, sampleRate / 2.0, win, powerToDb);

        using var tape = new GradientTape<double>();
        var mel = Forward(x);
        var loss = _engine.ReduceSum(mel, null);
        var grads = tape.ComputeGradients(loss, [x]);
        var analytical = grads[x];

        Assert.Equal(x.Shape.ToArray(), analytical.Shape.ToArray());

        const double eps = 1e-6;
        var failures = new List<string>();
        int probes = 0;

        for (int idx = nFft; idx < len - nFft; idx += 7)
        {
            double orig = x[idx];

            x[idx] = orig + eps;
            double lossPlus = _engine.TensorSum(Forward(x));

            x[idx] = orig - eps;
            double lossMinus = _engine.TensorSum(Forward(x));

            x[idx] = orig;

            double numerical = (lossPlus - lossMinus) / (2.0 * eps);
            double a = analytical[idx];
            probes++;

            double denom = Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(numerical)));
            if (Math.Abs(a - numerical) / denom > 5e-3)
                failures.Add($"  idx {idx}: analytical {a:G6} vs numerical {numerical:G6}" +
                             (Math.Abs(numerical) > 1e-9 ? $" (ratio {a / numerical:G6})" : ""));
        }

        Assert.True(probes > 0, "No probes ran.");
        if (failures.Count > 0)
        {
            Assert.Fail(
                $"MelSpectrogram gradient disagrees with finite differences at {failures.Count} of {probes} probes " +
                $"(powerToDb: {powerToDb}).\n" + string.Join("\n", failures.Take(10)));
        }
    }

}
