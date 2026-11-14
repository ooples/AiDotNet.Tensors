using System;
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
}
