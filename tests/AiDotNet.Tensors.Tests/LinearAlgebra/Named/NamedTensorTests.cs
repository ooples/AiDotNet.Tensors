// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Named;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.Named;

/// <summary>
/// Acceptance tests for <see cref="NamedTensor{T}"/> + <see cref="NamedOps"/>.
/// Covers rename / refine / align (permutation), broadcast-by-name, and
/// reduction-by-name.
/// </summary>
public class NamedTensorTests
{
    private static Tensor<float> ZerosTo<T>(int[] shape, Func<int, float> fillByLinear)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = fillByLinear(i);
        return t;
    }

    [Fact]
    public void Names_AreStoredAndQueryable()
    {
        var t = new Tensor<float>(new[] { 2, 3 });
        var n = new NamedTensor<float>(t, "batch", "features");
        Assert.True(n.HasNames);
        Assert.Equal("batch", n.Names[0]);
        Assert.Equal("features", n.Names[1]);
    }

    [Fact]
    public void Rename_ReplacesNamesWithoutCopying()
    {
        var t = new Tensor<float>(new[] { 2, 3 });
        var n = new NamedTensor<float>(t, "batch", "features");
        var renamed = n.Rename("B", "F");
        Assert.Equal("B", renamed.Names[0]);
        Assert.Equal("F", renamed.Names[1]);
        // Underlying tensor identity preserved.
        Assert.Same(t, renamed.Tensor);
    }

    [Fact]
    public void RefineNames_RejectsConflictingLabels()
    {
        var t = new Tensor<float>(new[] { 2, 3 });
        var n = new NamedTensor<float>(t, "batch", null);
        var refined = n.RefineNames("batch", "features");
        Assert.Equal("features", refined.Names[1]);

        Assert.Throws<InvalidOperationException>(() =>
            n.RefineNames("oops", "features"));
    }

    [Fact]
    public void AlignTo_PermutesAxesByName()
    {
        // Original [batch=2, features=3]; AlignTo("features", "batch")
        // should produce [features=3, batch=2] with elements transposed.
        var t = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                t[i, j] = i * 10 + j;

        var n = new NamedTensor<float>(t, "batch", "features");
        var aligned = n.AlignTo("features", "batch");
        Assert.Equal(new[] { 3, 2 }, aligned.Shape);
        Assert.Equal("features", aligned.Names[0]);
        Assert.Equal("batch", aligned.Names[1]);
        // Element [features=2, batch=1] in aligned == [batch=1, features=2] in original = 12.
        Assert.Equal(12f, aligned.Tensor[2, 1]);
    }

    [Fact]
    public void AlignTo_RejectsMissingName()
    {
        var t = new Tensor<float>(new[] { 2, 3 });
        var n = new NamedTensor<float>(t, "batch", "features");
        Assert.Throws<InvalidOperationException>(() => n.AlignTo("batch", "MISSING"));
    }

    [Fact]
    public void Flatten_CollapsesContiguousNamedAxes()
    {
        // [batch=2, height=3, width=4] → flatten height+width to "hw" → [batch=2, hw=12].
        var t = new Tensor<float>(new[] { 2, 3, 4 });
        for (int i = 0; i < t.Length; i++) t.AsWritableSpan()[i] = i;
        var n = new NamedTensor<float>(t, "batch", "height", "width");
        var flat = n.Flatten(new[] { "height", "width" }, "hw");
        Assert.Equal(new[] { 2, 12 }, flat.Shape);
        Assert.Equal("batch", flat.Names[0]);
        Assert.Equal("hw", flat.Names[1]);
    }

    [Fact]
    public void Unflatten_InvertsFlattenWhenSizesAreCompatible()
    {
        var t = new Tensor<float>(new[] { 2, 12 });
        var n = new NamedTensor<float>(t, "batch", "hw");
        var unflat = n.Unflatten("hw", new[] { 3, 4 }, new string?[] { "h", "w" });
        Assert.Equal(new[] { 2, 3, 4 }, unflat.Shape);
        Assert.Equal("h", unflat.Names[1]);
        Assert.Equal("w", unflat.Names[2]);
    }

    [Fact]
    public void Add_BroadcastsByNameNotPosition()
    {
        // a is [batch=2, features=3]. b is [features=3] — but with name "features".
        var a = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                a[i, j] = i * 10 + j;
        var aN = new NamedTensor<float>(a, "batch", "features");

        var b = new Tensor<float>(new[] { 3 });
        b[0] = 100; b[1] = 200; b[2] = 300;
        var bN = new NamedTensor<float>(b, "features");

        var result = NamedOps.Add(aN, bN);
        Assert.Equal(new[] { 2, 3 }, result.Shape);
        Assert.Equal(100f, result.Tensor[0, 0]);
        Assert.Equal(201f, result.Tensor[0, 1]);
        Assert.Equal(312f, result.Tensor[1, 2]);
    }

    [Fact]
    public void Add_BroadcastsSingletonAxisFromLeftOperand()
    {
        var a = new Tensor<float>(new[] { 1, 3 });
        a[0, 0] = 1; a[0, 1] = 2; a[0, 2] = 3;
        var aN = new NamedTensor<float>(a, "batch", "features");

        var b = new Tensor<float>(new[] { 4, 3 });
        for (int batch = 0; batch < 4; batch++)
            for (int feature = 0; feature < 3; feature++)
                b[batch, feature] = batch * 10 + feature;
        var bN = new NamedTensor<float>(b, "batch", "features");

        var result = NamedOps.Add(aN, bN);

        Assert.Equal(new[] { 4, 3 }, result.Shape);
        Assert.Equal(new string?[] { "batch", "features" }, result.Names);
        Assert.Equal(1f, result.Tensor[0, 0]);
        Assert.Equal(13f, result.Tensor[1, 1]);
        Assert.Equal(25f, result.Tensor[2, 2]);
        Assert.Equal(31f, result.Tensor[3, 0]);
    }

    [Fact]
    public void AddAndMultiply_PreserveNamesAndValuesForStridedLeftOperand()
    {
        var storage = new Tensor<float>(new[] { 2, 3 });
        storage[0, 0] = 1; storage[0, 1] = 2; storage[0, 2] = 3;
        storage[1, 0] = 4; storage[1, 1] = 5; storage[1, 2] = 6;
        Tensor<float> strided = storage.Transpose(new[] { 1, 0 });
        Assert.False(strided.IsContiguous);
        var named = new NamedTensor<float>(strided, "features", "batch");

        var perFeature = new Tensor<float>(new[] { 3 });
        perFeature[0] = 10; perFeature[1] = 20; perFeature[2] = 30;
        var namedPerFeature = new NamedTensor<float>(perFeature, "features");

        NamedTensor<float> added = NamedOps.Add(named, namedPerFeature);
        NamedTensor<float> multiplied = NamedOps.Multiply(named, namedPerFeature);

        Assert.Equal(new string?[] { "features", "batch" }, added.Names);
        Assert.Equal(new string?[] { "features", "batch" }, multiplied.Names);
        Assert.Equal(new[] { 3, 2 }, added.Shape);
        Assert.Equal(new[] { 3, 2 }, multiplied.Shape);
        Assert.Equal(new float[] { 11, 14, 22, 25, 33, 36 }, added.Tensor.AsSpan().ToArray());
        Assert.Equal(new float[] { 10, 40, 40, 100, 90, 180 }, multiplied.Tensor.AsSpan().ToArray());
    }

    [Fact]
    public void Sum_ReducesByName()
    {
        var t = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                t[i, j] = i * 10 + j;
        var n = new NamedTensor<float>(t, "batch", "features");
        var summed = NamedOps.Sum(n, "batch");
        Assert.Equal(new[] { 3 }, summed.Shape);
        Assert.Equal(10f, summed.Tensor[0]);  // 0 + 10
        Assert.Equal(12f, summed.Tensor[1]);  // 1 + 11
        Assert.Equal(14f, summed.Tensor[2]);  // 2 + 12
    }
}
