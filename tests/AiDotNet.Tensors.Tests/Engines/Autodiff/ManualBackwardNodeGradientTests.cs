using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression tests for manual-backward tape nodes (the mechanism the neural-network layer library
/// uses via <c>LayerBase.RegisterManualBackwardNode</c>). A manual node records a custom
/// <see cref="BackwardFunction{T}"/> that converts the gradient of a freshly-materialized output
/// tensor (e.g. Conv im2col) back into the gradient of a differentiable input (e.g. col2im). The
/// node's input gradient MUST propagate to upstream ops on the tape — the Conv3D im2col training
/// path relied on this and it was silently dropped, freezing every conv layer but the last.
/// </summary>
public class ManualBackwardNodeGradientTests
{
    private readonly CpuEngine _engine = new();

    // Mimics Conv3DLayer.ForwardIm2Col exactly:
    //   input --(manual node: identity backward, like col2im)--> m --MatMul(w)--> y --sum--> loss
    // The manual node is recorded the way RegisterManualBackwardNode records it (InputCount = 0xFF +
    // InputsOverflow). Its input gradient must reach `input`.
    [Fact]
    public void ManualBackwardNode_InputGradient_PropagatesThroughDownstreamMatMul()
    {
        var input = new Tensor<double>(new[] { 2, 3 },
            new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        var w = new Tensor<double>(new[] { 3, 2 },
            new Vector<double>(new double[] { 1, 0, 0, 1, 1, 1 }));

        using var tape = new GradientTape<double>();

        // "im2col": a fresh output tensor filled from input by a non-tape scalar copy.
        var m = new Tensor<double>(new[] { 2, 3 });
        var inSpan = input.Data.Span;
        var mSpan = m.Data.Span;
        for (int i = 0; i < mSpan.Length; i++) mSpan[i] = inSpan[i];

        // Manual backward node: Output = m, differentiable input = input, backward = identity
        // (grad_input = grad_output) — the simplest col2im. Recorded EXACTLY like
        // LayerBase.RegisterManualBackwardNode (InputCount = 0xFF + InputsOverflow + Input0 slots).
        var differentiableInputs = new[] { input };
        var entry = new TapeEntry<double>
        {
            OperationName = "TestManualBackward",
            Output = m,
            InputCount = 0xFF,
            InputsOverflow = differentiableInputs,
            Backward = (gradOutput, inputs, output, saved, eng, grads) =>
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    var g = gradOutput; // identity col2im
                    if (grads.TryGetValue(inputs[i], out var existing))
                        eng.TensorAddInPlace(existing, g);
                    else
                        grads[inputs[i]] = g;
                }
            }
        };
        entry.Input0 = differentiableInputs[0];
        tape.Record(entry);

        var y = _engine.TensorMatMul(m, w);          // [2,2]
        var loss = _engine.ReduceSum(y, new[] { 0, 1 }, keepDims: false);

        var grads = tape.ComputeGradients(loss, new[] { input, w });

        // The weight gradient always worked; the INPUT gradient was silently dropped (the bug).
        Assert.True(grads.ContainsKey(w) && grads[w] is not null, "weight gradient missing");
        Assert.True(grads.ContainsKey(input) && grads[input] is not null,
            "manual-backward node did NOT propagate the input gradient (dropped dX).");

        // Identity backward + dLoss/dy = 1 everywhere => dLoss/dm = w row-sums broadcast, and
        // dLoss/dinput == dLoss/dm. Just assert it is non-zero and finite.
        double sumAbs = 0;
        var gi = grads[input].Data.Span;
        for (int i = 0; i < gi.Length; i++) sumAbs += System.Math.Abs(gi[i]);
        Assert.True(sumAbs > 0, "input gradient is all zeros");
    }
}
