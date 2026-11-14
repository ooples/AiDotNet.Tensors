using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Operations that return a host value (TensorMaxValue, TensorSum, ...) or run eagerly on the host (STFT,
/// MelSpectrogram) record no lazy node. Inside a training trace their result was frozen into the plan as a constant,
/// and a lazy operand that had not been computed yet was read as whatever its buffer held: TensorDivideScalar(x,
/// TensorMaxValue(TensorAbs(x))) compiled to a NaN loss, and STFT of a lazy signal to a zero loss and zero gradient.
/// They now fail closed in training traces, so the caller falls back to eager, and read the computed operand in an
/// inference trace.
/// </summary>
[Collection("CompilationGlobalState")]
public class HostScalarTrainingTraceTests
{
    private static Tensor<double> Signal(int length)
    {
        var x = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) x[i] = Math.Sin(i * 0.3) * (1 + i % 5);
        return x;
    }

    [Fact]
    public void TrainingCompile_HostScalarOfALazyOperand_FailsClosed()
    {
        var engine = new CpuEngine();
        var w = Signal(16);
        using var cache = new CompiledModelCache<double>();

        var error = Assert.Throws<GraphCaptureNotSupportedException>(() => cache.GetOrCompileTraining(
            w._shape,
            () => engine.ReduceSum(engine.TensorDivideScalar(w, engine.TensorMaxValue(engine.TensorAbs(w))), null),
            new[] { w }));

        Assert.Equal(GraphCaptureLimitation.HostBoundary, error.Limitation);
        Assert.Equal(nameof(IEngine.TensorMaxValue), error.OperationName);
        Assert.Equal(0, cache.TrainingPlanCount);
    }

    [Fact]
    public void TrainingCompile_StftOfALazySignal_FailsClosed()
    {
        var engine = new CpuEngine();
        var w = Signal(64);
        var window = new Tensor<double>(new[] { 16 });
        for (int i = 0; i < 16; i++) window[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / 16);
        using var cache = new CompiledModelCache<double>();

        var error = Assert.Throws<GraphCaptureNotSupportedException>(() => cache.GetOrCompileTraining(
            w._shape,
            () =>
            {
                engine.STFT(engine.TensorMultiplyScalar(w, 1.0), 16, 4, window, center: true,
                    out Tensor<double> magnitude, out Tensor<double> _);
                return engine.ReduceSum(magnitude, null);
            },
            new[] { w }));

        Assert.Equal(GraphCaptureLimitation.HostBoundary, error.Limitation);
        Assert.Equal(nameof(IEngine.STFT), error.OperationName);
        Assert.Equal(0, cache.TrainingPlanCount);
    }

    [Fact]
    public void InferenceTrace_HostScalarReadsTheComputedLazyOperand()
    {
        var engine = new CpuEngine();
        var x = Signal(16);
        double expectedMax = 0, expectedSum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            expectedMax = Math.Max(expectedMax, Math.Abs(x[i]));
            expectedSum += Math.Abs(x[i]);
        }

        double max, sum;
        using (GraphMode.EnableInference())
        {
            var abs = engine.TensorAbs(x);
            max = engine.TensorMaxValue(abs);
            sum = engine.TensorSum(engine.TensorAbs(x));
        }

        Assert.Equal(expectedMax, max);
        Assert.Equal(expectedSum, sum, 12);
    }
}
