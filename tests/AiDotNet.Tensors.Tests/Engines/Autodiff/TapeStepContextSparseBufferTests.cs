// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A sparse leaf's ParameterBuffer slot holds only its pattern's non-zeros, and its view is a SparseTensor over that
/// slot rather than a dense view sharing the buffer's storage. TapeStepContext must accept that; it used to count the
/// sparse leaf's dense size and require shared storage, so every sparse model failed its first tape step with
/// "ParameterBuffer total size (...) does not match parameter total (...)".
/// </summary>
public class TapeStepContextSparseBufferTests
{
    [Fact]
    public void ViewsOfAMixedDenseAndSparseBuffer_PassTheAlignmentCheck()
    {
        var pattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 2, 5 }),
            new ParameterLayout(new[] { 3, 3 }, pattern),
            new ParameterLayout(new[] { 4 }),
        });
        var views = buffer.CreateAllViews();
        Assert.IsType<SparseTensor<float>>(views[1]);

        var context = new TapeStepContext<float>(views, new Dictionary<Tensor<float>, Tensor<float>>(), 0f, buffer);

        Assert.Same(views, context.Parameters);
    }

    [Fact]
    public void ATensorThatIsNotTheSparseSlotsView_IsStillRejected()
    {
        var pattern = new SparsityLayout(3, 3, new[] { 0, 1, 2 }, new[] { 0, 1, 2 });
        var buffer = new ParameterBuffer<float>(new[]
        {
            new ParameterLayout(new[] { 2, 5 }),
            new ParameterLayout(new[] { 3, 3 }, pattern),
        });
        var views = buffer.CreateAllViews();
        var impostor = new Tensor<float>[] { views[0], new Tensor<float>(new[] { 3 }) };

        Assert.Throws<ArgumentException>(
            () => new TapeStepContext<float>(impostor, new Dictionary<Tensor<float>, Tensor<float>>(), 0f, buffer));
    }
}
