using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Operations that return a host value (TensorMaxValue, TensorSum, ...) record no lazy node. Inside a training trace
/// their result was frozen into the plan as a constant, and a lazy operand that had not been computed yet was read as
/// whatever its buffer held: TensorDivideScalar(x, TensorMaxValue(TensorAbs(x))) compiled to a NaN loss. They now fail
/// closed in training traces, so the caller falls back to eager, and read the computed operand in an inference trace.
/// STFT, which had the same defect, is now recorded instead (StftGradientTests).
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
    public void InferenceCompile_HostScalarOfAPlanInput_RefusesTheTrace()
    {
        // x is a plain input, not a lazy value: TensorVecDot still read it on the host, and the divide would replay
        // the trace-time dot product for every later input.
        var engine = new CpuEngine();
        var x = Signal(16);
        using var cache = new CompiledModelCache<double>();

        var error = Assert.Throws<GraphCaptureNotSupportedException>(() => cache.GetOrCompileInference(
            x._shape, () => engine.TensorDivideScalar(x, engine.TensorVecDot(x, x))));

        Assert.Equal(GraphCaptureLimitation.HostBoundary, error.Limitation);
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
