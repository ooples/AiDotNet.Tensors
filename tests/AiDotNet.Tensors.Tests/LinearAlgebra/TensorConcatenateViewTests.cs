// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Tensor.Concatenate copies one contiguous block per combination of the axes before the concatenation axis. These
/// tests compare it with an independent walk over logical indices, on every axis, for contiguous inputs, permuted
/// (non-contiguous) views and narrowed views that start at a storage offset.
/// </summary>
public class TensorConcatenateViewTests
{
    private static readonly CpuEngine Engine = new();

    private static Tensor<double> Iota(int salt, params int[] shape)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<double>(Enumerable.Range(0, n).Select(i => salt * 1000.0 + i).ToArray(), shape);
    }

    // The values each input holds at its logical indices, laid out in the concatenation's row-major order.
    private static double[] Reference(Tensor<double>[] inputs, int axis)
    {
        int rank = inputs[0].Rank;
        var outShape = inputs[0].Shape.ToArray();
        outShape[axis] = inputs.Sum(t => t.Shape[axis]);
        var result = new double[outShape.Aggregate(1, (a, b) => a * b)];
        var index = new int[rank];
        for (int flat = 0; flat < result.Length; flat++)
        {
            int rest = flat;
            for (int d = rank - 1; d >= 0; d--) { index[d] = rest % outShape[d]; rest /= outShape[d]; }
            int along = index[axis], which = 0;
            while (along >= inputs[which].Shape[axis]) { along -= inputs[which].Shape[axis]; which++; }
            var source = (int[])index.Clone();
            source[axis] = along;
            result[flat] = inputs[which][source];
        }

        return result;
    }

    private static void AssertMatches(Tensor<double>[] inputs, int axis)
    {
        var expected = Reference(inputs, axis);
        var actual = Tensor<double>.Concatenate(inputs, axis);
        Assert.Equal(expected, actual.ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Contiguous_inputs_match_the_index_walk(int axis)
    {
        var a = Iota(1, 2, 3, 4);
        var bShape = new[] { 2, 3, 4 };
        bShape[axis] = 5;
        AssertMatches(new[] { a, Iota(2, bShape) }, axis);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Permuted_views_match_the_index_walk(int axis)
    {
        // [3, 2, 4] swapped to a non-contiguous [2, 3, 4] view.
        var a = Engine.TensorSwapAxes(Iota(1, 3, 2, 4), 0, 1);
        var b = Engine.TensorSwapAxes(Iota(2, 3, 2, 4), 0, 1);
        Assert.False(a.IsContiguous);
        AssertMatches(new[] { a, b }, axis);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Narrowed_views_at_a_storage_offset_match_the_index_walk(int axis)
    {
        // Rows 1..2 of a [4, 3, 4] tensor (contiguous, offset) and columns 1..4 of a [2, 3, 6] one (strided, offset).
        var a = Iota(1, 4, 3, 4).Slice(0, 1, 3);
        var b = Iota(2, 2, 3, 6).Slice(2, 1, 5);
        Assert.False(b.IsContiguous);
        AssertMatches(new[] { a, b, Iota(3, 2, 3, 4) }, axis);
    }

    [Fact]
    public void An_empty_input_contributes_nothing()
    {
        var a = Iota(1, 2, 3);
        var empty = new Tensor<double>(new[] { 2, 0 });
        AssertMatches(new[] { a, empty, Iota(2, 2, 1) }, 1);
    }
}
