using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>Named dimensions, following torch's named-tensor semantics for the naming operations.</summary>
public class NamedTensorTests
{
    private static Tensor<float> Arange(params int[] shape)
        => new Tensor<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => (float)i).ToArray(), shape);

    [Fact]
    public void NewTensor_IsUnnamed()
    {
        var t = Arange(2, 3);
        Assert.False(t.HasNames());
        Assert.Equal(new string?[] { null, null }, t.Names);
    }

    [Fact]
    public void RenameInPlace_NamesTheTensorItself()
    {
        var t = Arange(2, 3);
        Assert.Same(t, t.RenameInPlace("N", "C"));
        Assert.Equal(new[] { "N", "C" }, t.Names);
        t.RenameInPlace(new Dictionary<string, string> { ["C"] = "features" });
        Assert.Equal(new[] { "N", "features" }, t.Names);
        t.RenameInPlace();
        Assert.False(t.HasNames());
    }

    [Fact]
    public void Rename_IsAViewOfTheSameStorage()
    {
        var t = Arange(2, 3);
        var named = t.Rename("N", "C");
        Assert.False(t.HasNames());
        named[0, 0] = 99f;
        Assert.Equal(99f, t[0, 0]);
    }

    [Fact]
    public void Ellipsis_KeepsTheNamesItCovers()
    {
        var t = Arange(2, 3, 4).Rename("A", "B", "C");
        Assert.Equal(new[] { "X", "B", "C" }, t.Rename("X", "...").Names);
        Assert.Equal(new[] { "A", "B", "Z" }, t.Rename("...", "Z").Names);
    }

    [Fact]
    public void RefineNames_FillsUnnamedDims_AndRejectsChangingANamedOne()
    {
        var t = Arange(2, 3).Rename("N", null);
        Assert.Equal(new[] { "N", "C" }, t.RefineNames("N", "C").Names);
        Assert.Throws<InvalidOperationException>(() => t.RefineNames("M", "C"));
    }

    [Fact]
    public void DuplicateNames_AreRejected()
        => Assert.Throws<ArgumentException>(() => Arange(2, 3).RenameInPlace("N", "N"));

    [Fact]
    public void AlignTo_PermutesAndInsertsSizeOneDims()
    {
        var t = Arange(2, 3).Rename("N", "C");          // values n * 3 + c
        var a = t.AlignTo("C", "H", "N");
        Assert.Equal(new[] { 3, 1, 2 }, a.Shape.ToArray());
        Assert.Equal(new[] { "C", "H", "N" }, a.Names);
        for (int n = 0; n < 2; n++) for (int c = 0; c < 3; c++) Assert.Equal(n * 3 + c, a[c, 0, n]);
        Assert.Equal(new[] { "C", "N" }, t.AlignTo("C", "...").Names);
        Assert.Throws<InvalidOperationException>(() => t.AlignTo("N"));   // would drop C
    }

    [Fact]
    public void AlignAs_And_AlignTensors_UseTheOtherNames()
    {
        var a = Arange(3).Rename("C");
        var b = Arange(2, 3).Rename("N", "C");
        Assert.Equal(new[] { 1, 3 }, a.AlignAs(b).Shape.ToArray());
        var aligned = Tensor<float>.AlignTensors(a, b);
        Assert.Equal(new[] { "N", "C" }, aligned[0].Names);
        Assert.Equal(new[] { 1, 3 }, aligned[0].Shape.ToArray());
        Assert.Equal(new[] { 2, 3 }, aligned[1].Shape.ToArray());
    }
}
