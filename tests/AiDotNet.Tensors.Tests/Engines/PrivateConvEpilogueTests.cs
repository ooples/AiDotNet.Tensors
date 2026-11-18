using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public sealed class PrivateConvEpilogueTests
{
    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void Private_bias_residual_matches_ordinary_operations_bitwise_and_preserves_offsets(bool bias, bool residual)
    {
        var engine = new CpuEngine();
        int[] shape = { 2, 3, 1, 19 };
        var values = Enumerable.Range(0, 114).Select(i => (float)(Math.Sin(i * .17) * .7)).ToArray();
        var other = Enumerable.Range(0, 114).Select(i => (float)(Math.Cos(i * .21) * .3)).ToArray();
        var storage = Enumerable.Repeat(99.5f, values.Length + 14).ToArray();
        values.CopyTo(storage, 7);
        using var input = new Tensor<float>(values, shape);
        using var destination = Tensor<float>.FromMemory(storage.AsMemory(7, values.Length), shape);
        using var biasTensor = new Tensor<float>(new[] { -.25f, .5f, -.125f }, new[] { 3 });
        using var residualTensor = new Tensor<float>(other, shape);
        using var noGrad = new NoGradScope<float>();
        var expected = bias ? engine.TensorChannelBiasAdd(input, biasTensor) : input;
        if (residual) expected = engine.TensorAdd(expected, residualTensor);
        engine.AddPrivateConvolutionBiasResidual(destination, bias ? biasTensor : null, residual ? residualTensor : null);
        Assert.Equal(expected.ToArray().Select(BitConverter.SingleToInt32Bits), destination.ToArray().Select(BitConverter.SingleToInt32Bits));
        Assert.All(storage.Take(7), v => Assert.Equal(99.5f, v));
        Assert.All(storage.Skip(storage.Length - 7), v => Assert.Equal(99.5f, v));
        Assert.Equal(values, input.ToArray()); Assert.Equal(other, residualTensor.ToArray());
    }

    [Fact]
    public void Private_epilogue_rejects_recording_context_without_changing_destination()
    {
        var engine = new CpuEngine();
        using var output = new Tensor<float>(new[] { 1f, -2f }, new[] { 1, 1, 1, 2 });
        using var bias = new Tensor<float>(new[] { .5f }, new[] { 1 });
        Assert.Throws<InvalidOperationException>(() => engine.AddPrivateConvolutionBiasResidual(output, bias, null));
        Assert.Equal(new[] { 1f, -2f }, output.ToArray());
    }
}
