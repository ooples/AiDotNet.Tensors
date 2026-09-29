// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Optimizer updates run under <see cref="NoGradScope{T}"/> so their own arithmetic is not taped (on the GPU engine
/// every op that sees an active tape falls back to the host). Second-order / line-search optimizers still re-evaluate
/// the loss and gradients through the context, and that re-evaluation must record normally -- NoGradScope suppression
/// is thread-wide, so without lifting it the re-evaluated gradients would be empty.
/// </summary>
public sealed class TapeStepContextNoGradTests
{
    private static (TapeStepContext<double> Ctx, Tensor<double> W) Build()
    {
        IEngine e = new CpuEngine();
        var w = new Tensor<double>([3], new Vector<double>(new[] { 0.5, -1.0, 2.0 }));
        var x = new Tensor<double>([2, 3], new Vector<double>(new[] { 1.0, 2.0, 3.0, -1.0, 0.5, 0.25 }));
        var y = new Tensor<double>([2], new Vector<double>(new[] { 1.0, 0.0 }));
        Tensor<double> Forward(Tensor<double> input, Tensor<double> _) =>
            e.ReduceSum(e.TensorMultiply(input, e.Reshape(w, [1, 3])), [1], keepDims: false);
        Tensor<double> Loss(Tensor<double> pred, Tensor<double> target)
        {
            var d = e.TensorSubtract(pred, target);
            return e.ReduceSum(e.TensorMultiply(d, d), null, false);
        }
        var ctx = new TapeStepContext<double>([w], new Dictionary<Tensor<double>, Tensor<double>>(), 0.0, x, y, Forward, Loss);
        return (ctx, w);
    }

    [Fact]
    public void Reevaluate_InsideNoGradScope_StillProducesTheGradient()
    {
        var (reference, refW) = Build();
        reference.Reevaluate();
        var expected = reference.Gradients[refW].ToArray();
        Assert.Contains(expected, v => v != 0.0);

        var (ctx, w) = Build();
        using (new NoGradScope<double>())
        {
            ctx.Reevaluate();
            Assert.True(NoGradScope<double>.IsSuppressed, "the caller's no-grad scope must be restored after re-evaluation");
        }
        Assert.False(NoGradScope<double>.IsSuppressed);
        Assert.True(ctx.Gradients.TryGetValue(w, out var g), "re-evaluation under NoGradScope produced no gradient");
        Assert.Equal(expected, g!.ToArray());
    }
}
