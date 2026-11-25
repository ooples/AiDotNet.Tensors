// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The masked-fill backward with a boolean mask whose shape differs from the gradient's (a flat mask over the same
/// elements): the gradient is zero where the mask filled and passes through elsewhere, as with a same-shape mask.
/// </summary>
public class MaskedFillBackwardMaskShapeTests
{
    [Fact]
    public void A_flat_mask_over_the_same_elements_zeroes_the_filled_gradient()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 2, 3 });
        var grad = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var mask = new Tensor<bool>(new bool[] { true, false, false, true, false, true }, new[] { 6 });
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();

        BackwardFunctions<float>.MaskedFillBackward(grad, new[] { input }, input, new object[] { mask }, engine, grads);

        var g = grads[input];
        Assert.Equal(new[] { 2, 3 }, g.Shape.ToArray());
        Assert.Equal(new float[] { 0, 2, 3, 0, 5, 0 }, g.AsSpan().ToArray());
    }
}
