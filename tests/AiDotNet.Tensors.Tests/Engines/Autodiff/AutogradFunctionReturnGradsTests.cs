using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class AutogradFunctionReturnGradsTests
{
    // torch.autograd.Function style: backward returns one gradient per input (null for "no gradient").
    private sealed class ScaledProduct : AutogradFunction<double>
    {
        public override Tensor<double> Forward(AutogradContext ctx, params Tensor<double>[] inputs)
        {
            ctx.SaveForBackward(inputs[0], inputs[1]);
            return AiDotNetEngine.Current.TensorMultiplyScalar(AiDotNetEngine.Current.TensorMultiply(inputs[0], inputs[1]), 3.0);
        }

        public override Tensor<double>?[] Backward(AutogradContext ctx, Tensor<double> gradOutput)
        {
            var e = AiDotNetEngine.Current;
            var a = ctx.GetSaved<double>(0);
            var b = ctx.GetSaved<double>(1);
            return new Tensor<double>?[]
            {
                e.TensorMultiplyScalar(e.TensorMultiply(gradOutput, b), 3.0),
                null, // deliberately no gradient for b
            };
        }
    }

    private sealed class NoBackward : AutogradFunction<double>
    {
        public override Tensor<double> Forward(AutogradContext ctx, params Tensor<double>[] inputs) => inputs[0];
    }

    [Fact]
    public void ReturnedGradients_AccumulateIntoTheirInputs_AndNullMeansNone()
    {
        var a = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });
        var b = new Tensor<double>(new[] { 4.0, 5.0, 6.0 }, new[] { 3 });
        using var tape = new GradientTape<double>();
        var y = new ScaledProduct().Apply(a, b);
        var grads = tape.ComputeGradients(AiDotNetEngine.Current.ReduceSum(y), new[] { a, b });

        Assert.Equal(new[] { 12.0, 15.0, 18.0 }, grads[a].ToArray());
        Assert.False(grads.TryGetValue(b, out var gb) && gb is not null && Array.Exists(gb.ToArray(), v => v != 0));
    }

    [Fact]
    public void ABackwardThatIsNotOverridden_SaysWhichOverloadToOverride()
    {
        var a = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        using var tape = new GradientTape<double>();
        var y = new NoBackward().Apply(a);
        var ex = Assert.Throws<NotImplementedException>(() => tape.ComputeGradients(AiDotNetEngine.Current.ReduceSum(y), new[] { a }));
        Assert.Contains("NoBackward must override", ex.Message);
    }
}
