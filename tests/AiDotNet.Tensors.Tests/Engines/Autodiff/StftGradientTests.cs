using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// IEngine.STFT is differentiable in both outputs, as torch.stft is: audio front ends (CLAP, AST, Whisper-style
/// encoders) train through it when fine-tuned end to end. It recorded no tape edge, so a loss over its magnitude or
/// phase gave the waveform no gradient, and inside a compiled training plan it was frozen as a constant.
/// </summary>
[Collection("CompilationGlobalState")]
public class StftGradientTests
{
    private const int Batch = 2, Length = 64, NFft = 16, Hop = 4;

    private static Tensor<double> Signal()
    {
        var x = new Tensor<double>(new[] { Batch, Length });
        for (int i = 0; i < x.Length; i++)
            x[i] = Math.Sin(i * 0.37) + 0.5 * Math.Cos(i * 1.13) + 0.1 * (i % 7);
        return x;
    }

    private static Tensor<double> Hann()
    {
        var w = new Tensor<double>(new[] { NFft });
        for (int i = 0; i < NFft; i++) w[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / NFft);
        return w;
    }

    // Loss weights over the spectrogram. The phase is not differentiable where its bin is empty or where it wraps at
    // ±π, so the phase weights are zero there; everywhere else both outputs carry a nonzero weight.
    private static (Tensor<double> A, Tensor<double> B) Weights(Tensor<double> magnitude, Tensor<double> phase)
    {
        var a = new Tensor<double>((int[])magnitude._shape.Clone());
        var b = new Tensor<double>((int[])phase._shape.Clone());
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = 0.3 + Math.Sin(i * 0.71);
            bool smooth = magnitude[i] > 1e-2 && Math.Abs(phase[i]) < Math.PI - 0.2;
            b[i] = smooth ? 0.2 + Math.Cos(i * 0.53) : 0.0;
        }
        return (a, b);
    }

    private static double HostLoss(CpuEngine engine, Tensor<double> x, Tensor<double> window, bool center,
        Tensor<double> a, Tensor<double> b)
    {
        engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
        double loss = 0;
        for (int i = 0; i < magnitude.Length; i++) loss += a[i] * magnitude[i] + b[i] * phase[i];
        return loss;
    }

    private static Tensor<double> Loss(IEngine engine, Tensor<double> magnitude, Tensor<double> phase,
        Tensor<double> a, Tensor<double> b) =>
        engine.ReduceSum(engine.TensorAdd(engine.TensorMultiply(magnitude, a), engine.TensorMultiply(phase, b)), null);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TapeGradient_MatchesFiniteDifferences(bool center)
    {
        var engine = new CpuEngine();
        var window = Hann();
        var x = Signal();
        engine.STFT(x, NFft, Hop, window, center, out Tensor<double> m0, out Tensor<double> p0);
        var (a, b) = Weights(m0, p0);

        Tensor<double> grad;
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            var loss = Loss(engine, magnitude, phase, a, b);
            grad = tape.ComputeGradients(loss, new[] { x })[x];
        }

        const double h = 1e-5;
        for (int i = 0; i < x.Length; i++)
        {
            double original = x[i];
            x[i] = original + h;
            double plus = HostLoss(engine, x, window, center, a, b);
            x[i] = original - h;
            double minus = HostLoss(engine, x, window, center, a, b);
            x[i] = original;
            double numeric = (plus - minus) / (2 * h);
            Assert.True(Math.Abs(grad[i] - numeric) <= 1e-6 * Math.Max(1, Math.Abs(numeric)),
                $"center={center}, dL/dx[{i}]: tape {grad[i]:R}, finite difference {numeric:R}");
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TapeWindowGradient_MatchesFiniteDifferences(bool center)
    {
        // A learnable window (as torch.stft trains one) gets dL/dw[i] = Σ_frames dL/df[i] · xPadded[start + i].
        var engine = new CpuEngine();
        var window = Hann();
        var x = Signal();
        engine.STFT(x, NFft, Hop, window, center, out Tensor<double> m0, out Tensor<double> p0);
        var (a, b) = Weights(m0, p0);

        Tensor<double> grad;
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            grad = tape.ComputeGradients(Loss(engine, magnitude, phase, a, b), new[] { window })[window];
        }

        const double h = 1e-5;
        for (int i = 0; i < window.Length; i++)
        {
            double original = window[i];
            window[i] = original + h;
            double plus = HostLoss(engine, x, window, center, a, b);
            window[i] = original - h;
            double minus = HostLoss(engine, x, window, center, a, b);
            window[i] = original;
            double numeric = (plus - minus) / (2 * h);
            Assert.True(Math.Abs(grad[i] - numeric) <= 1e-6 * Math.Max(1, Math.Abs(numeric)),
                $"center={center}, dL/dw[{i}]: tape {grad[i]:R}, finite difference {numeric:R}");
        }
    }

    private static readonly Func<int, double> SignalDirection = i => Math.Cos(i * 0.29) - 0.3;
    private static readonly Func<int, double> WindowDirection = i => 0.4 * Math.Sin(i * 0.83) + 0.1;

    // h(x, w) = ⟨∇ₓL, u⟩ + ⟨∇_wL, v⟩ from first-order gradients only, for the finite differences of the second order.
    private static double DirectionalDerivative(CpuEngine engine, Tensor<double> x, Tensor<double> window, bool center,
        Tensor<double> a, Tensor<double> b)
    {
        using var tape = new GradientTape<double>();
        engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
        var grads = tape.ComputeGradients(Loss(engine, magnitude, phase, a, b), new[] { x, window });
        double h = 0;
        for (int i = 0; i < x.Length; i++) h += grads[x][i] * SignalDirection(i);
        for (int i = 0; i < window.Length; i++) h += grads[window][i] * WindowDirection(i);
        return h;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SecondOrderGradient_InWaveformAndWindow_MatchesFiniteDifferences(bool center)
    {
        var engine = new CpuEngine();
        var window = Hann();
        var x = Signal();
        engine.STFT(x, NFft, Hop, window, center, out Tensor<double> m0, out Tensor<double> p0);
        var (a, b) = Weights(m0, p0);
        var u = new Tensor<double>((int[])x._shape.Clone());
        for (int i = 0; i < u.Length; i++) u[i] = SignalDirection(i);
        var v = new Tensor<double>((int[])window._shape.Clone());
        for (int i = 0; i < v.Length; i++) v[i] = WindowDirection(i);

        Dictionary<Tensor<double>, Tensor<double>> first, plainFirst, second;
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            first = tape.ComputeGradients(Loss(engine, magnitude, phase, a, b), new[] { x, window }, createGraph: true);
            var h = engine.TensorAdd(
                engine.ReduceSum(engine.TensorMultiply(first[x], u), null),
                engine.ReduceSum(engine.TensorMultiply(first[window], v), null));
            second = tape.ComputeGradients(h, new[] { x, window });
        }
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            plainFirst = tape.ComputeGradients(Loss(engine, magnitude, phase, a, b), new[] { x, window });
        }

        // The create-graph first order is the same gradient the first-order kernel computes.
        foreach (var (name, leaf) in new[] { ("x", x), ("w", window) })
            for (int i = 0; i < leaf.Length; i++)
                Assert.True(Math.Abs(first[leaf][i] - plainFirst[leaf][i]) <= 1e-10 * Math.Max(1, Math.Abs(plainFirst[leaf][i])),
                    $"center={center}, dL/d{name}[{i}]: create-graph {first[leaf][i]:R}, first-order {plainFirst[leaf][i]:R}");

        const double step = 1e-5;
        foreach (var (name, leaf) in new[] { ("x", x), ("w", window) })
        {
            Assert.True(second.ContainsKey(leaf), $"center={center}: the second order reached no gradient for {name}");
            for (int i = 0; i < leaf.Length; i++)
            {
                double original = leaf[i];
                leaf[i] = original + step;
                double plus = DirectionalDerivative(engine, x, window, center, a, b);
                leaf[i] = original - step;
                double minus = DirectionalDerivative(engine, x, window, center, a, b);
                leaf[i] = original;
                double numeric = (plus - minus) / (2 * step);
                double analytic = second[leaf][i];
                Assert.True(Math.Abs(analytic - numeric) <= 1e-5 * Math.Max(1, Math.Abs(numeric)),
                    $"center={center}, d/d{name}[{i}] ⟨∇L, (u, v)⟩: create-graph {analytic:R}, finite difference {numeric:R}");
            }
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SilentWaveform_CreateGraphGradient_IsFiniteAndMatchesTheFirstOrder(bool center)
    {
        // Every bin is empty: the magnitude's derivative takes the reported phase's direction and the phase's is
        // floored to zero, in the create-graph path as in the first-order kernel, rather than 0/0.
        var engine = new CpuEngine();
        var window = Hann();
        var x = new Tensor<double>(new[] { Batch, Length });
        engine.STFT(x, NFft, Hop, window, center, out Tensor<double> m0, out Tensor<double> _);
        var a = new Tensor<double>((int[])m0._shape.Clone());
        var b = new Tensor<double>((int[])m0._shape.Clone());
        for (int i = 0; i < a.Length; i++) { a[i] = 0.3 + Math.Sin(i * 0.71); b[i] = 0.2 + Math.Cos(i * 0.53); }

        Dictionary<Tensor<double>, Tensor<double>> plain, graph, second;
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            plain = tape.ComputeGradients(Loss(engine, magnitude, phase, a, b), new[] { x, window });
        }
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            graph = tape.ComputeGradients(Loss(engine, magnitude, phase, a, b), new[] { x, window }, createGraph: true);
            second = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(graph[x], graph[x]), null), new[] { x, window });
        }

        foreach (var (name, leaf) in new[] { ("x", x), ("w", window) })
        {
            // graph[x] reaches both leaves: the window through the analysis, the waveform through the recorded
            // magnitude node's backward, which multiplies by the recorded real and imaginary parts even at empty bins.
            Assert.True(second.TryGetValue(leaf, out var s), $"center={center}: no second-order gradient reached {name}");
            for (int i = 0; i < leaf.Length; i++)
            {
                Assert.True(Math.Abs(graph[leaf][i] - plain[leaf][i]) <= 1e-12 * Math.Max(1, Math.Abs(plain[leaf][i])),
                    $"center={center}, dL/d{name}[{i}]: create-graph {graph[leaf][i]:R}, first-order {plain[leaf][i]:R}");
                Assert.True(!double.IsNaN(s[i]) && !double.IsInfinity(s[i]), $"center={center}, second order d/d{name}[{i}] is {s[i]:R}");
            }
        }
    }

    [Fact]
    public void SilentWaveform_SpectrogramAndStftPhaseCreateGraphGradients_AreFinite()
    {
        var engine = new CpuEngine();
        var window = Hann();
        var x = new Tensor<double>(new[] { Batch, Length });

        foreach (bool phaseOutput in new[] { false, true })
        {
            Tensor<double> plain, graph;
            using (var tape = new GradientTape<double>())
            {
                var y = phaseOutput ? engine.StftPhase(x, NFft, Hop, NFft, window) : engine.Spectrogram(x, NFft, Hop, NFft, window);
                plain = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiplyScalar(y, 0.7), null), new[] { x })[x];
            }
            using (var tape = new GradientTape<double>())
            {
                var y = phaseOutput ? engine.StftPhase(x, NFft, Hop, NFft, window) : engine.Spectrogram(x, NFft, Hop, NFft, window);
                graph = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiplyScalar(y, 0.7), null), new[] { x },
                    createGraph: true)[x];
            }
            for (int i = 0; i < x.Length; i++)
                Assert.True(Math.Abs(graph[i] - plain[i]) <= 1e-12 * Math.Max(1, Math.Abs(plain[i])),
                    $"{(phaseOutput ? "StftPhase" : "Spectrogram")} dL/dx[{i}]: create-graph {graph[i]:R}, first-order {plain[i]:R}");
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void CompiledTrainingGradient_MatchesTheTape(bool center)
    {
        var engine = new CpuEngine();
        var window = Hann();
        var w = Signal();
        engine.STFT(w, NFft, Hop, window, center, out Tensor<double> m0, out Tensor<double> p0);
        var (a, b) = Weights(m0, p0);

        double eagerLoss;
        Tensor<double> eagerGrad;
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(w, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            var loss = Loss(engine, magnitude, phase, a, b);
            eagerLoss = loss[0];
            eagerGrad = tape.ComputeGradients(loss, new[] { w })[w];
        }

        using var cache = new CompiledModelCache<double>();
        var plan = cache.GetOrCompileTraining(w._shape, () =>
        {
            // The scale keeps the STFT operand lazy, which is what used to be read uninitialised.
            engine.STFT(engine.TensorMultiplyScalar(w, 1.0), NFft, Hop, window, center,
                out Tensor<double> magnitude, out Tensor<double> phase);
            return Loss(engine, magnitude, phase, a, b);
        }, new[] { w });

        double compiledLoss = plan.Step()[0];
        var compiledGrad = plan.Gradients[0];

        Assert.Equal(eagerLoss, compiledLoss, 10);
        for (int i = 0; i < w.Length; i++)
            Assert.True(Math.Abs(compiledGrad[i] - eagerGrad[i]) <= 1e-10 * Math.Max(1, Math.Abs(eagerGrad[i])),
                $"center={center}, dL/dw[{i}]: compiled {compiledGrad[i]:R}, tape {eagerGrad[i]:R}");
    }

    [Theory]
    [InlineData(0, false)]   // DC of a constant negative frame: phase π
    [InlineData(4, true)]    // Nyquist of an alternating frame whose Nyquist coefficient is negative: phase π
    public void PhaseGradient_AtTheRealBins_IsExactlyZero(int bin, bool alternating)
    {
        // DC and the even-length Nyquist bin are real for a real frame, so their phase is 0 or π whatever the input and
        // its gradient is zero. The float forward stores π rounded, whose sine is about -8.7e-8; seeding the adjoint from
        // it and dividing by a magnitude of 8e-9 gave a waveform gradient of about 10.9 per sample.
        const int n = 8;
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 1, n });
        for (int i = 0; i < n; i++) x[i] = -1e-9f * (alternating && i % 2 == 1 ? -1f : 1f);
        var window = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) window[i] = 1f;

        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            engine.STFT(x, n, n, window, false, out Tensor<float> _, out Tensor<float> phase);
            Assert.True(Math.Abs(Math.Abs(phase[bin]) - Math.PI) < 1e-6, $"bin {bin} phase {phase[bin]:R}, expected ±π");
            var seed = new Tensor<float>((int[])phase._shape.Clone());
            seed[bin] = 1f;
            grads = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(phase, seed), null), new[] { x, window });
        }

        Assert.True(grads.TryGetValue(x, out var gx), "no waveform gradient was recorded");
        for (int i = 0; i < n; i++) Assert.Equal(0f, gx[i]);
        Assert.True(grads.TryGetValue(window, out var gw), "no window gradient was recorded");
        for (int i = 0; i < n; i++) Assert.Equal(0f, gw[i]);
    }
}
