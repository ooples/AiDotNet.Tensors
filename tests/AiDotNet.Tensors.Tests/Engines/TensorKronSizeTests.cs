// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A Kronecker product has as many elements as its operands' sizes multiplied, which passes what a tensor can hold
/// long before either operand is large. The size is refused by name instead of overflowing.
/// </summary>
public sealed class TensorKronSizeTests
{
    [Theory]
    [InlineData(new[] { 65536 }, new[] { 65536 })]              // one axis alone: 2^32
    [InlineData(new[] { 2, 40000 }, new[] { 2, 40000 })]        // each axis fits, their product does not: 6.4e9
    public void A_result_past_the_element_limit_is_refused_by_name(int[] shapeA, int[] shapeB)
    {
        var engine = new CpuEngine();
        var error = Assert.Throws<InvalidOperationException>(
            () => engine.TensorKron(new Tensor<float>(shapeA), new Tensor<float>(shapeB)));
        Assert.Contains("TensorKron", error.Message);
        Assert.Contains(string.Join(",", shapeA), error.Message);
    }

    [Fact]
    public void A_result_within_the_limit_has_the_product_shape()
    {
        var a = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var b = new Tensor<float>(new[] { 10f, 100f, 1000f }, new[] { 3 });
        var product = new CpuEngine().TensorKron(a, b);
        Assert.Equal(new[] { 6 }, product.Shape.ToArray());
        Assert.Equal(new[] { 10f, 100f, 1000f, 20f, 200f, 2000f }, product.ToArray());
    }
}
