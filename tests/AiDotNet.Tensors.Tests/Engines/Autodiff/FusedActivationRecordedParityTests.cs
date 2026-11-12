using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The recorded (create-graph) form of every fused activation epilogue must be the SAME function as the fused kernels'
/// handler: equal values, and a first-order gradient equal to the handler's backward. A mismatch here would make the
/// Hessian of a fused layer the Hessian of a different function, which the finite-difference sweeps (run on the fused
/// forward) only catch for the shapes and parameters they happen to use.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class FusedActivationRecordedParityTests : IDisposable
{
    private readonly IEngine _previous;
    private readonly CpuEngine _engine = new();

    public FusedActivationRecordedParityTests()
    {
        _previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _engine;
    }

    public void Dispose() => AiDotNetEngine.Current = _previous;

    public static IEnumerable<object[]> Cases()
    {
        foreach (FusedActivationType type in Enum.GetValues(typeof(FusedActivationType)))
        {
            yield return new object[] { type, false };
            yield return new object[] { type, true };
        }
    }

    // Values clear of every kink (0, ±1, ±3, 6, the 0.5 / 1 thresholds) so both sides are differentiable there.
    private static Tensor<double> Input() => new Tensor<double>(
        new[] { -2.3, -1.4, -0.6, 0.4, 1.3, 2.2, 0.75, -0.35, 1.7, -2.7, 0.15, 2.6 }, new[] { 3, 4 });

    private static FusedActivationParams Custom() => new()
    {
        Alpha = 0.3f, Beta = 0.7f, Theta = 0.55f, PReluSlope = new[] { 0.2f },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void RecordedForm_MatchesTheFusedHandler_InValueAndGradient(FusedActivationType type, bool customParams)
    {
        var parameters = customParams ? Custom() : null;
        var handler = ActivationRegistry.Get(type);
        var x = Input();
        var g = new Tensor<double>(Enumerable.Range(0, 12).Select(i => 0.5 - 0.09 * i).ToArray(), new[] { 3, 4 });

        Tensor<double> recorded;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            recorded = BackwardFunctions<double>.ApplyFusedActivationRecorded(_engine, x, type, parameters);
            grads = tape.ComputeGradients(_engine.ReduceSum(_engine.TensorMultiply(recorded, g), null), new[] { x });
        }

        if (handler is null)
        {
            Assert.Equal(FusedActivationType.None, type);
            for (int i = 0; i < x.Length; i++) Assert.Equal(x[i], recorded[i]);
            return;
        }

        // Gumbel-Softmax samples noise in its forward; its designed gradient (the softmax relaxation) is what the
        // recorded form carries, so only the gradient is compared.
        if (type != FusedActivationType.GumbelSoftmax)
        {
            Tensor<double> expected;
            using (new NoGradScope<double>()) expected = handler.Apply(_engine, x, parameters);
            for (int i = 0; i < x.Length; i++)
                Assert.True(Math.Abs(expected[i] - recorded[i]) <= 1e-12 * (1 + Math.Abs(expected[i])),
                    $"{type}{(customParams ? " (custom params)" : "")}[{i}]: fused {expected[i]:G17} vs recorded {recorded[i]:G17}");
        }

        Tensor<double> expectedGrad;
        using (new NoGradScope<double>()) expectedGrad = handler.ApplyBackward(_engine, g, x, parameters);
        var got = grads.TryGetValue(x, out var gx) ? gx : new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < x.Length; i++)
            Assert.True(Math.Abs(expectedGrad[i] - got[i]) <= 1e-10 * (1 + Math.Abs(expectedGrad[i])),
                $"{type}{(customParams ? " (custom params)" : "")} gradient [{i}]: fused backward {expectedGrad[i]:G17} vs recorded {got[i]:G17}");
    }
}
