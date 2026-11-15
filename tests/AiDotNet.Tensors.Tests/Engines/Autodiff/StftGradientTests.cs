// Copyright (c) CrowdTrainer LLC. All rights reserved.
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

        const double step = 1e-6;
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
    [InlineData(true)]
    [InlineData(false)]
    public void BelowFloorNonZeroWaveform_CreateGraphPhaseGradient_IsZeroAndFinite(bool center)
    {
        // Nonzero bins with 0 < |C| <= 1e-12 (PhaseGradientMagnitudeFloor) must contribute zero phase gradient
        // without overflow or NaN under both first-order and create-graph differentiation.
        var engine = new CpuEngine();
        var window = Hann();
        var x = Signal();
        for (int i = 0; i < x.Length; i++) x[i] *= 1e-15;
        engine.STFT(x, NFft, Hop, window, center, out Tensor<double> m0, out Tensor<double> _);
        bool sawNonZeroBelowFloor = false;
        for (int i = 0; i < m0.Length; i++)
        {
            Assert.True(m0[i] <= 1e-12, $"bin {i} magnitude {m0[i]:R} exceeded floor");
            if (m0[i] > 0.0) sawNonZeroBelowFloor = true;
        }
        Assert.True(sawNonZeroBelowFloor, "Expected at least one strictly positive bin below the floor.");

        var zeroMagWeights = new Tensor<double>((int[])m0._shape.Clone());
        var phaseWeights = new Tensor<double>((int[])m0._shape.Clone());
        for (int i = 0; i < phaseWeights.Length; i++) phaseWeights[i] = 0.2 + Math.Cos(i * 0.53);

        Dictionary<Tensor<double>, Tensor<double>> plain, graph, second;
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            plain = tape.ComputeGradients(Loss(engine, magnitude, phase, zeroMagWeights, phaseWeights), new[] { x, window });
        }
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center, out Tensor<double> magnitude, out Tensor<double> phase);
            graph = tape.ComputeGradients(Loss(engine, magnitude, phase, zeroMagWeights, phaseWeights), new[] { x, window }, createGraph: true);
            second = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(graph[x], graph[x]), null), new[] { x, window });
        }

        foreach (var (name, leaf) in new[] { ("x", x), ("w", window) })
        {
            for (int i = 0; i < leaf.Length; i++)
            {
                Assert.Equal(0.0, plain[leaf][i]);
                Assert.Equal(0.0, graph[leaf][i]);
                if (second.TryGetValue(leaf, out var s))
                    Assert.True(!double.IsNaN(s[i]) && !double.IsInfinity(s[i]), $"center={center}, second order d/d{name}[{i}] is {s[i]:R}");
            }
        }
    }

    [Fact]
    public void InPlaceMutationOfOutputs_DoesNotCorruptTapeGradients()
    {
        var engine = new CpuEngine();
        var window = Hann();
        var x = Signal();
        engine.STFT(x, NFft, Hop, window, center: true, out Tensor<double> m0, out Tensor<double> p0);
        var (a, b) = Weights(m0, p0);

        Dictionary<Tensor<double>, Tensor<double>> cleanMagGrad, mutatedMagGrad, cleanPhaseGrad, mutatedPhaseGrad;
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center: true, out Tensor<double> mag, out Tensor<double> _);
            cleanMagGrad = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(mag, a), null), new[] { x, window });
        }
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center: true, out Tensor<double> mag, out Tensor<double> phase);
            var loss = engine.ReduceSum(engine.TensorMultiply(mag, a), null);
            for (int i = 0; i < mag.Length; i++) mag[i] = 999.0;
            for (int i = 0; i < phase.Length; i++) phase[i] = 1.75;
            mutatedMagGrad = tape.ComputeGradients(loss, new[] { x, window });
        }

        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center: true, out Tensor<double> _, out Tensor<double> phase);
            cleanPhaseGrad = tape.ComputeGradients(engine.ReduceSum(engine.TensorMultiply(phase, b), null), new[] { x, window });
        }
        using (var tape = new GradientTape<double>())
        {
            engine.STFT(x, NFft, Hop, window, center: true, out Tensor<double> mag, out Tensor<double> phase);
            var loss = engine.ReduceSum(engine.TensorMultiply(phase, b), null);
            for (int i = 0; i < mag.Length; i++) mag[i] = 999.0;
            for (int i = 0; i < phase.Length; i++) phase[i] = -2.5;
            mutatedPhaseGrad = tape.ComputeGradients(loss, new[] { x, window });
        }

        foreach (var leaf in new[] { x, window })
        {
            for (int i = 0; i < leaf.Length; i++)
            {
                Assert.Equal(cleanMagGrad[leaf][i], mutatedMagGrad[leaf][i], 12);
                Assert.Equal(cleanPhaseGrad[leaf][i], mutatedPhaseGrad[leaf][i], 12);
            }
        }
    }

    [SkippableFact]
    public void DirectGpuStft_FloatAndDoubleTapeGradients_MatchCpu_AndShortCenterSignalThrows()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend.");
            IEngine gpuEngine = gpu;
            var cpu = new CpuEngine();

            // Short centered signal (signalLength <= nFft / 2 while numFrames > 0) must reject cleanly, never launch OOB ReflectPad1d.
            var shortSignal = new Tensor<float>(new[] { 1, 4 });
            var shortWindow = new Tensor<float>(new[] { NFft });
            Assert.Throws<ArgumentException>(() =>
                gpuEngine.STFT(shortSignal, NFft, Hop, shortWindow, center: true, out Tensor<float> _, out Tensor<float> _));

            // Double taped STFT on DirectGpuTensorEngine routes through exact double path and matches CPU to machine precision.
            var wDouble = Hann();
            var xDouble = Signal();
            cpu.STFT(xDouble, NFft, Hop, wDouble, center: true, out Tensor<double> m0, out Tensor<double> p0);
            var (aDouble, bDouble) = Weights(m0, p0);
            Dictionary<Tensor<double>, Tensor<double>> cpuDoubleGrads, gpuDoubleGrads;
            using (var tape = new GradientTape<double>())
            {
                cpu.STFT(xDouble, NFft, Hop, wDouble, center: true, out Tensor<double> mag, out Tensor<double> phase);
                cpuDoubleGrads = tape.ComputeGradients(Loss(cpu, mag, phase, aDouble, bDouble), new[] { xDouble, wDouble });
            }
            using (var tape = new GradientTape<double>())
            {
                gpuEngine.STFT(xDouble, NFft, Hop, wDouble, center: true, out Tensor<double> mag, out Tensor<double> phase);
                // Keep downstream scalar reduction in exact double so we test gpuEngine.STFT's double path directly
                // without float32 rounding from DirectGpuTensorEngine elementwise ops.
                gpuDoubleGrads = tape.ComputeGradients(Loss(cpu, mag, phase, aDouble, bDouble), new[] { xDouble, wDouble });
            }
            for (int i = 0; i < xDouble.Length; i++)
                Assert.True(Math.Abs(gpuDoubleGrads[xDouble][i] - cpuDoubleGrads[xDouble][i]) <= 1e-12 * Math.Max(1.0, Math.Abs(cpuDoubleGrads[xDouble][i])),
                    $"double dL/dx[{i}]: gpu {gpuDoubleGrads[xDouble][i]:R}, cpu {cpuDoubleGrads[xDouble][i]:R}");
            for (int i = 0; i < wDouble.Length; i++)
                Assert.True(Math.Abs(gpuDoubleGrads[wDouble][i] - cpuDoubleGrads[wDouble][i]) <= 1e-12 * Math.Max(1.0, Math.Abs(cpuDoubleGrads[wDouble][i])),
                    $"double dL/dw[{i}]: gpu {gpuDoubleGrads[wDouble][i]:R}, cpu {cpuDoubleGrads[wDouble][i]:R}");

            // Float taped STFT on DirectGpuTensorEngine runs on device and matches CPU float tape gradients.
            var wFloat = new Tensor<float>(new[] { NFft });
            for (int i = 0; i < NFft; i++) wFloat[i] = (float)wDouble[i];
            var xFloat = new Tensor<float>(new[] { Batch, Length });
            for (int i = 0; i < xFloat.Length; i++) xFloat[i] = (float)xDouble[i];
            var aFloat = new Tensor<float>((int[])aDouble._shape.Clone());
            var bFloat = new Tensor<float>((int[])bDouble._shape.Clone());
            for (int i = 0; i < aFloat.Length; i++) { aFloat[i] = (float)aDouble[i]; bFloat[i] = (float)bDouble[i]; }

            Dictionary<Tensor<float>, Tensor<float>> cpuFloatGrads, gpuFloatGrads;
            using (var tape = new GradientTape<float>())
            {
                cpu.STFT(xFloat, NFft, Hop, wFloat, center: true, out Tensor<float> mag, out Tensor<float> phase);
                var loss = cpu.ReduceSum(cpu.TensorAdd(cpu.TensorMultiply(mag, aFloat), cpu.TensorMultiply(phase, bFloat)), null);
                cpuFloatGrads = tape.ComputeGradients(loss, new[] { xFloat, wFloat });
            }
            bool previousThrowOnFallback = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            try
            {
                using (var tape = new GradientTape<float>())
                {
                    gpuEngine.STFT(xFloat, NFft, Hop, wFloat, center: true, out Tensor<float> mag, out Tensor<float> phase);
                    var loss = gpuEngine.ReduceSum(gpuEngine.TensorAdd(gpuEngine.TensorMultiply(mag, aFloat), gpuEngine.TensorMultiply(phase, bFloat)), null);
                    for (int i = 0; i < mag.Length; i++) mag[i] = 999f;
                    for (int i = 0; i < phase.Length; i++) phase[i] = -2.5f;
                    gpuFloatGrads = tape.ComputeGradients(loss, new[] { xFloat, wFloat });
                }
            }
            finally
            {
                DirectGpuTensorEngine.ThrowOnGpuKernelFallback = previousThrowOnFallback;
            }
            for (int i = 0; i < xFloat.Length; i++)
                Assert.True(Math.Abs(gpuFloatGrads[xFloat][i] - cpuFloatGrads[xFloat][i]) <= 1e-4f * Math.Max(1f, Math.Abs(cpuFloatGrads[xFloat][i])),
                    $"float dL/dx[{i}]: gpu {gpuFloatGrads[xFloat][i]}, cpu {cpuFloatGrads[xFloat][i]}");
            for (int i = 0; i < wFloat.Length; i++)
                Assert.True(Math.Abs(gpuFloatGrads[wFloat][i] - cpuFloatGrads[wFloat][i]) <= 1e-4f * Math.Max(1f, Math.Abs(cpuFloatGrads[wFloat][i])),
                    $"float dL/dw[{i}]: gpu {gpuFloatGrads[wFloat][i]}, cpu {cpuFloatGrads[wFloat][i]}");

            // Single-frame STFT with hopLength > padded length (valid 1-frame analysis) must fall back cleanly to the
            // host adjoint instead of throwing ArgumentOutOfRangeException inside RequireStftSignal.
            var xLargeHop = new Tensor<float>(new[] { 1, 8 });
            for (int i = 0; i < xLargeHop.Length; i++) xLargeHop[i] = 0.25f * (i + 1) - 0.7f;
            var wLargeHop = new Tensor<float>(new[] { 4 });
            for (int i = 0; i < wLargeHop.Length; i++) wLargeHop[i] = 0.4f + 0.1f * i;
            Dictionary<Tensor<float>, Tensor<float>> cpuLargeHopGrads, gpuLargeHopGrads;
            using (var tape = new GradientTape<float>())
            {
                cpu.STFT(xLargeHop, nFft: 4, hopLength: 16, window: wLargeHop, center: false, out Tensor<float> mag, out Tensor<float> _);
                cpuLargeHopGrads = tape.ComputeGradients(cpu.ReduceSum(mag, null), new[] { xLargeHop, wLargeHop });
            }
            using (var tape = new GradientTape<float>())
            {
                gpuEngine.STFT(xLargeHop, nFft: 4, hopLength: 16, window: wLargeHop, center: false, out Tensor<float> mag, out Tensor<float> _);
                gpuLargeHopGrads = tape.ComputeGradients(gpuEngine.ReduceSum(mag, null), new[] { xLargeHop, wLargeHop });
            }
            for (int i = 0; i < xLargeHop.Length; i++)
                Assert.True(Math.Abs(gpuLargeHopGrads[xLargeHop][i] - cpuLargeHopGrads[xLargeHop][i]) <= 1e-5f * Math.Max(1f, Math.Abs(cpuLargeHopGrads[xLargeHop][i])),
                    $"large-hop dL/dx[{i}]: gpu {gpuLargeHopGrads[xLargeHop][i]}, cpu {cpuLargeHopGrads[xLargeHop][i]}");
            for (int i = 0; i < wLargeHop.Length; i++)
                Assert.True(Math.Abs(gpuLargeHopGrads[wLargeHop][i] - cpuLargeHopGrads[wLargeHop][i]) <= 1e-5f * Math.Max(1f, Math.Abs(cpuLargeHopGrads[wLargeHop][i])),
                    $"large-hop dL/dw[{i}]: gpu {gpuLargeHopGrads[wLargeHop][i]}, cpu {cpuLargeHopGrads[wLargeHop][i]}");
        }
    }

#if NETCOREAPP
    [SkippableFact]
    public void ReleasedOrPooledGpuStftBuffers_ThrowObjectDisposedException_BeforeLaunch()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend.");
            var backend = gpu.GetBackend() ?? throw new InvalidOperationException("Expected non-null DirectGpu backend.");
            using var live = backend.AllocateBuffer(64);
            var disposed = backend.AllocateBuffer(64);
            disposed.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                backend.StftBackwardFrames(disposed, live, live, live, batch: 1, nFft: 8, numFrames: 2, numFreqs: 5, phaseTarget: 0));
            Assert.Throws<ObjectDisposedException>(() =>
                backend.StftBackwardSignal(live, disposed, live, batch: 1, length: 12, pad: 0, nFft: 8, hop: 4, numFrames: 2));
            Assert.Throws<ObjectDisposedException>(() =>
                backend.StftBackwardWindow(live, live, disposed, batch: 1, length: 12, pad: 0, nFft: 8, hop: 4, numFrames: 2));

            if (backend is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)
            {
                var raw = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.CudaGpuBuffer)backend.AllocateBuffer(64);
                var pooledLease = raw.DetachForPool()
                    ?? throw new InvalidOperationException("Expected DetachForPool to return a pooled lease.");
                try
                {
                    Assert.NotEqual(IntPtr.Zero, pooledLease.Handle);
                    Assert.True(pooledLease.IsReleased);
                    Assert.Throws<ObjectDisposedException>(() =>
                        backend.StftBackwardFrames(pooledLease, live, live, live, batch: 1, nFft: 8, numFrames: 2, numFreqs: 5, phaseTarget: 0));
                    Assert.Throws<ObjectDisposedException>(() =>
                        backend.StftBackwardSignal(pooledLease, live, live, batch: 1, length: 12, pad: 0, nFft: 8, hop: 4, numFrames: 2));
                    Assert.Throws<ObjectDisposedException>(() =>
                        backend.StftBackwardWindow(pooledLease, live, live, batch: 1, length: 12, pad: 0, nFft: 8, hop: 4, numFrames: 2));

                    var pooledView = ((AiDotNet.Tensors.Engines.DirectGpu.IGpuBufferViews)backend).TryCreateView(pooledLease, 0, 64)
                        ?? throw new InvalidOperationException("Expected CUDA view over non-zero pooled handle.");
                    Assert.Throws<ObjectDisposedException>(() =>
                        backend.StftBackwardFrames(pooledView, live, live, live, batch: 1, nFft: 8, numFrames: 2, numFreqs: 5, phaseTarget: 0));
                }
                finally
                {
                    pooledLease.Release();
                }
            }
        }
    }

    [Fact]
    public void PoolReuse_DoesNotReviveDisposedHipWrapper()
    {
#pragma warning disable SYSLIB0050
        var hipOwner = (AiDotNet.Tensors.Engines.DirectGpu.HIP.HipBackend)
            System.Runtime.Serialization.FormatterServices.GetUninitializedObject(
                typeof(AiDotNet.Tensors.Engines.DirectGpu.HIP.HipBackend));
#pragma warning restore SYSLIB0050
        GC.SuppressFinalize(hipOwner);

        using var pool = new AiDotNet.Tensors.Engines.DirectGpu.GpuBufferPool<AiDotNet.Tensors.Engines.DirectGpu.HIP.HipGpuBuffer>(16, 1 << 20);
        var first = new AiDotNet.Tensors.Engines.DirectGpu.HIP.HipGpuBuffer(
            IntPtr.Zero, 64, hipOwner, returned => pool.Return(returned));
        Assert.False(first.IsReleased);

        first.Dispose();
        Assert.True(first.IsReleased);

        Assert.True(pool.TryRent(64, out var reRented));
        var second = reRented ?? throw new InvalidOperationException("Expected rented HipGpuBuffer.");
        Assert.NotSame(first, second);
        Assert.False(second.IsReleased);
        // The stale reference from the previous lease must stay released even after the pool re-rents the buffer.
        Assert.True(first.IsReleased);
        second.Release();
    }
#endif

    [Fact]
    public void CreateGraph_TinyFloatSignalAndStrictShapePolicy_ProducesExactFiniteGradients()
    {
        var engine = new CpuEngine();

        // 1e-30f squared underflows float32 to zero without power-of-two scaling in PolarOfCartesian / RecordedComplexMagnitude.
        var tinyX = new Tensor<float>(new float[] { 1e-30f, 0f, 0f, 0f }, new[] { 1, 4 });
        var unitW = new Tensor<float>(new float[] { 1f, 1f, 1f, 1f }, new[] { 4 });
        Tensor<float> plainGx;
        using (var tape = new GradientTape<float>())
        {
            engine.STFT(tinyX, nFft: 4, hopLength: 4, window: unitW, center: false, out Tensor<float> mag, out Tensor<float> _);
            plainGx = tape.ComputeGradients(engine.ReduceSum(mag, null), new[] { tinyX })[tinyX];
        }
        using (var tape = new GradientTape<float>())
        {
            engine.STFT(tinyX, nFft: 4, hopLength: 4, window: unitW, center: false, out Tensor<float> mag, out Tensor<float> _);
            var grads = tape.ComputeGradients(engine.ReduceSum(mag, null), new[] { tinyX, unitW }, createGraph: true);
            var gx = grads[tinyX];
            for (int i = 0; i < gx.Length; i++)
            {
                Assert.False(float.IsNaN(gx[i]) || float.IsInfinity(gx[i]), $"tiny float createGraph dL/dx[{i}] = {gx[i]}");
                Assert.Equal(plainGx[i], gx[i], 5);
            }
            // Each of the 3 non-negative bins [DC, k=1, Nyquist] has Re = 1e-30, Im = 0; summing cos(2*pi*k*i/4) over k=0,1,2 gives [3, 0, 1, 0].
            Assert.Equal(3f, gx[0], 5);
            Assert.Equal(0f, gx[1], 5);
            Assert.Equal(1f, gx[2], 5);
            Assert.Equal(0f, gx[3], 5);
        }

        // 1D waveform [8], nFft = 4, hopLength = 2 under ShapePolicy.Strict() must broadcast window [1, 1, 4] against frames [1, 3, 4].
        var x1d = new Tensor<float>(new[] { 8 });
        for (int i = 0; i < x1d.Length; i++) x1d[i] = 0.2f * (i + 1);
        using (ShapePolicy.Strict())
        using (var tape = new GradientTape<float>())
        {
            engine.STFT(x1d, nFft: 4, hopLength: 2, window: unitW, center: false, out Tensor<float> mag, out Tensor<float> _);
            var grads = tape.ComputeGradients(engine.ReduceSum(mag, null), new[] { x1d, unitW }, createGraph: true);
            Assert.Equal(8, grads[x1d].Length);
            Assert.Equal(4, grads[unitW].Length);
        }
    }

    [SkippableFact]
    public void DirectGpuPowerOfTwoFftFamily_TapeAndCompiledTrainingGradients_MatchCpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend.");
            IEngine gpuEngine = gpu;
            var cpu = new CpuEngine();

            void CheckTransform(string name, int[] shape, Action<IEngine, Tensor<float>, Tensor<float>, out_pair> invoke)
            {
                var re = new Tensor<float>(shape);
                var im = new Tensor<float>(shape);
                var wRe = new Tensor<float>(shape);
                var wIm = new Tensor<float>(shape);
                for (int i = 0; i < re.Length; i++)
                {
                    re[i] = (float)(0.4 * Math.Sin(0.31 * i + 0.2) + 0.15 * Math.Cos(0.73 * i));
                    im[i] = (float)(0.35 * Math.Cos(0.29 * i - 0.1) - 0.2 * Math.Sin(0.61 * i));
                    wRe[i] = (float)(0.3 + 0.5 * Math.Cos(0.41 * i));
                    wIm[i] = (float)(-0.2 + 0.6 * Math.Sin(0.53 * i));
                }

                Dictionary<Tensor<float>, Tensor<float>> cpuGrads, gpuGrads;
                using (var tape = new GradientTape<float>())
                {
                    var pair = new out_pair();
                    invoke(cpu, re, im, pair);
                    var loss = cpu.ReduceSum(cpu.TensorAdd(cpu.TensorMultiply(pair.Re, wRe), cpu.TensorMultiply(pair.Im, wIm)), null);
                    cpuGrads = tape.ComputeGradients(loss, new[] { re, im });
                }
                using (var tape = new GradientTape<float>())
                {
                    var pair = new out_pair();
                    invoke(gpuEngine, re, im, pair);
                    var loss = gpuEngine.ReduceSum(gpuEngine.TensorAdd(gpuEngine.TensorMultiply(pair.Re, wRe), gpuEngine.TensorMultiply(pair.Im, wIm)), null);
                    gpuGrads = tape.ComputeGradients(loss, new[] { re, im });
                }

                for (int i = 0; i < re.Length; i++)
                {
                    Assert.True(Math.Abs(gpuGrads[re][i] - cpuGrads[re][i]) <= 1e-4f * Math.Max(1f, Math.Abs(cpuGrads[re][i])),
                        $"{name} tape dL/dRe[{i}]: gpu {gpuGrads[re][i]}, cpu {cpuGrads[re][i]}");
                    Assert.True(Math.Abs(gpuGrads[im][i] - cpuGrads[im][i]) <= 1e-4f * Math.Max(1f, Math.Abs(cpuGrads[im][i])),
                        $"{name} tape dL/dIm[{i}]: gpu {gpuGrads[im][i]}, cpu {cpuGrads[im][i]}");
                }

                var previous = AiDotNetEngine.Current;
                try
                {
                    AiDotNetEngine.Current = gpuEngine;
                    ICompiledTrainingPlan<float> plan;
                    using (var scope = GraphMode.EnableTraining(new[] { re, im }))
                    {
                        var pair = new out_pair();
                        invoke(gpuEngine, re, im, pair);
                        gpuEngine.ReduceSum(gpuEngine.TensorAdd(gpuEngine.TensorMultiply(pair.Re, wRe), gpuEngine.TensorMultiply(pair.Im, wIm)), null);
                        plan = scope.CompileTraining(new[] { re, im });
                    }
                    using (plan)
                    {
                        plan.Step();
                        for (int i = 0; i < re.Length; i++)
                        {
                            Assert.True(Math.Abs(plan.Gradients[0][i] - cpuGrads[re][i]) <= 1e-4f * Math.Max(1f, Math.Abs(cpuGrads[re][i])),
                                $"{name} compiled dL/dRe[{i}]: gpu {plan.Gradients[0][i]}, cpu {cpuGrads[re][i]}");
                            Assert.True(Math.Abs(plan.Gradients[1][i] - cpuGrads[im][i]) <= 1e-4f * Math.Max(1f, Math.Abs(cpuGrads[im][i])),
                                $"{name} compiled dL/dIm[{i}]: gpu {plan.Gradients[1][i]}, cpu {cpuGrads[im][i]}");
                        }
                    }
                }
                finally
                {
                    AiDotNetEngine.Current = previous;
                }
            }

            CheckTransform("FFT", new[] { 2, 8 }, (e, r, i, p) => { e.FFT(r, i, out var or, out var oi); p.Re = or; p.Im = oi; });
            CheckTransform("IFFT", new[] { 2, 8 }, (e, r, i, p) => { e.IFFT(r, i, out var or, out var oi); p.Re = or; p.Im = oi; });
            CheckTransform("FFT2D", new[] { 2, 4, 8 }, (e, r, i, p) => { e.FFT2D(r, i, out var or, out var oi); p.Re = or; p.Im = oi; });
            CheckTransform("IFFT2D", new[] { 2, 4, 8 }, (e, r, i, p) => { e.IFFT2D(r, i, out var or, out var oi); p.Re = or; p.Im = oi; });
        }
    }

    private sealed class out_pair
    {
        public Tensor<float> Re = new(new[] { 1 });
        public Tensor<float> Im = new(new[] { 1 });
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

        foreach (bool createGraph in new[] { false, true })
        {
            Dictionary<Tensor<float>, Tensor<float>> grads;
            using (var tape = new GradientTape<float>())
            {
                engine.STFT(x, n, n, window, false, out Tensor<float> _, out Tensor<float> phase);
                Assert.True(Math.Abs(Math.Abs(phase[bin]) - Math.PI) < 1e-6, $"bin {bin} phase {phase[bin]:R}, expected ±π");
                var seed = new Tensor<float>((int[])phase._shape.Clone());
                seed[bin] = 1f;
                grads = tape.ComputeGradients(
                    engine.ReduceSum(engine.TensorMultiply(phase, seed), null),
                    new[] { x, window },
                    createGraph: createGraph);
            }

            Assert.True(grads.TryGetValue(x, out var gx), $"createGraph={createGraph}: no waveform gradient was recorded");
            for (int i = 0; i < n; i++) Assert.Equal(0f, gx[i]);
            Assert.True(grads.TryGetValue(window, out var gw), $"createGraph={createGraph}: no window gradient was recorded");
            for (int i = 0; i < n; i++) Assert.Equal(0f, gw[i]);
        }
    }

    [Fact]
    public void TinyHalfSignal_CreateGraph_ProducesFiniteMagnitudeAndPhaseGradients()
    {
        // Half.MaxValue is 65504 (< 2^16). An input around 2e-5 (< 2^-15) previously chose shift >= 16 in
        // PolarOfCartesian, overflowing scale to Half.PositiveInfinity and poisoning createGraph gradients with NaN.
        const int n = 4;
        var engine = new CpuEngine();
        var window = new Tensor<Half>(new[] { n });
        for (int i = 0; i < n; i++) window[i] = (Half)1f;

        var xMag = new Tensor<Half>(new[] { 1, n });
        xMag[0] = (Half)2e-5f;
        foreach (bool createGraph in new[] { false, true })
        {
            Dictionary<Tensor<Half>, Tensor<Half>> magGrads;
            using (var tape = new GradientTape<Half>())
            {
                engine.STFT(xMag, n, n, window, false, out var mag, out _);
                magGrads = tape.ComputeGradients(engine.ReduceSum(mag, null), new[] { xMag }, createGraph: createGraph);
            }

            Assert.True(magGrads.TryGetValue(xMag, out var gx), $"createGraph={createGraph}: missing Half magnitude gradient");
            float[] expected = { 3f, 0f, 1f, 0f };
            for (int i = 0; i < n; i++)
            {
                float actual = (float)gx[i];
                Assert.False(float.IsNaN(actual) || float.IsInfinity(actual), $"createGraph={createGraph}: gx[{i}]={actual}");
                Assert.True(Math.Abs(actual - expected[i]) <= 5e-2f, $"createGraph={createGraph}: gx[{i}]={actual}, expected {expected[i]}");
            }
        }

        // Phase backward with a subnormal Half amplitude on both seeded and unseeded bins must also stay finite.
        var xPhase = new Tensor<Half>(new[] { 1, n });
        xPhase[1] = (Half)2e-5f;
        xPhase[2] = (Half)1.2e-7f;
        using (var tape = new GradientTape<Half>())
        {
            engine.STFT(xPhase, n, n, window, false, out _, out var phase);
            var seed = new Tensor<Half>((int[])phase._shape.Clone());
            seed[1] = (Half)1e-3f;
            var phaseGrads = tape.ComputeGradients(
                engine.ReduceSum(engine.TensorMultiply(phase, seed), null),
                new[] { xPhase },
                createGraph: true);
            Assert.True(phaseGrads.TryGetValue(xPhase, out var gxp));
            for (int i = 0; i < n; i++)
            {
                float val = (float)gxp[i];
                Assert.False(float.IsNaN(val) || float.IsInfinity(val), $"Half phase grad[{i}]={val}");
            }
            Assert.True((float)gxp[0] > 45f && (float)gxp[0] < 53f, $"Expected Half phase grad[0] ~ 50, got {(float)gxp[0]}");
        }
    }

#if NETCOREAPP
    [Fact]
    public void TryStftBackward_RejectsFramingBeyondPaddedSignalLength()
    {
        var physical = AiDotNet.Tensors.Tests.Engines.DirectGpu.MockDirectGpuBackend.Create(
            new AiDotNet.Tensors.Tests.Engines.DirectGpu.MockBackendState());
        using var direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(physical);
        using var gpu = new DirectGpuTensorEngine(direct);

        var x = new Tensor<float>(new[] { 1, 16 });
        var w = new Tensor<float>(new[] { 8 });
        // Valid numFrames for length=16, nFft=8, hop=4, center=false is 3; 5 frames exceed padded length 16.
        var mismatchedGrad = new Tensor<float>(new[] { 1, 5, 5 });
        bool ok = gpu.TryStftBackward(mismatchedGrad, mismatchedGrad, mismatchedGrad, phaseTarget: false, x, w,
            nFft: 8, hopLength: 4, center: false, out var gw, out var gwin);
        Assert.False(ok);
        Assert.Null(gw);
        Assert.Null(gwin);
    }
    [Fact]
    public void PureRecordingDeferredScope_RefusesStftBackward_ForFloatAndDouble()
    {
        var physical = AiDotNet.Tensors.Tests.Engines.DirectGpu.MockDirectGpuBackend.Create(
            new AiDotNet.Tensors.Tests.Engines.DirectGpu.MockBackendState());
        var recording = new AiDotNet.Tensors.Engines.Gpu.RecordingGpuBackend(physical)
        {
            ExecuteWhileRecording = false,
        };
        using var graph = new AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder();
        using var direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(recording);
        using var gpu = new DirectGpuTensorEngine(direct);

        var xFloat = new Tensor<float>(new[] { 1, 16 });
        var wFloat = new Tensor<float>(new[] { 8 });
        var gFloat = new Tensor<float>(new[] { 1, 5, 3 });
        var xDouble = new Tensor<double>(new[] { 1, 16 });
        var wDouble = new Tensor<double>(new[] { 8 });
        var gDouble = new Tensor<double>(new[] { 1, 5, 3 });

        recording.BeginRecording(graph);
        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                gpu.TryStftBackward(gFloat, gFloat, gFloat, phaseTarget: false, xFloat, wFloat, 8, 4, center: false,
                    out _, out _));
            Assert.Throws<InvalidOperationException>(() =>
                gpu.TryStftBackward(gDouble, gDouble, gDouble, phaseTarget: false, xDouble, wDouble, 8, 4, center: false,
                    out _, out _));
        }
        finally
        {
            recording.EndRecording();
        }
    }
#endif
}
