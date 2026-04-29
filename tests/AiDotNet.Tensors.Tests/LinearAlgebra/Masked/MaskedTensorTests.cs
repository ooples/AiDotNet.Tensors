// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Masked;
using AiDotNet.Tensors.LinearAlgebra.Nested;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.Masked;

/// <summary>
/// Acceptance tests for <see cref="MaskedTensor{T}"/> + <see cref="MaskedOps"/>
/// + the nested ↔ masked round-trip in <see cref="StructuredBridges"/>.
/// </summary>
public class MaskedTensorTests
{
    [Fact]
    public void Mean_IgnoresMaskedLanes()
    {
        // Values: [1, 2, 999 (masked), 4, 999 (masked)] → mean = (1+2+4)/3 = 7/3.
        var values = new Tensor<float>(new[] { 5 });
        values.AsWritableSpan()[0] = 1; values.AsWritableSpan()[1] = 2;
        values.AsWritableSpan()[2] = 999; values.AsWritableSpan()[3] = 4;
        values.AsWritableSpan()[4] = 999;
        var mask = new[] { true, true, false, true, false };
        var masked = new MaskedTensor<float>(values, mask);

        var mean = MaskedOps.Mean(masked);
        Assert.Equal(7f / 3f, mean.AsSpan()[0], 4);
    }

    [Fact]
    public void Sum_MatchesManualReductionOverValidLanes()
    {
        var values = new Tensor<float>(new[] { 4 });
        values.AsWritableSpan()[0] = 10; values.AsWritableSpan()[1] = 20;
        values.AsWritableSpan()[2] = float.NaN; values.AsWritableSpan()[3] = 30;
        var mask = new[] { true, true, false, true };
        var masked = new MaskedTensor<float>(values, mask);
        Assert.Equal(60f, MaskedOps.Sum(masked).AsSpan()[0]);
        // The NaN at masked-out lane 2 must NOT pollute the sum — that
        // was the original PyTorch motivator for MaskedTensor.
    }

    [Fact]
    public void Var_AndStd_OnlyConsiderValidLanes()
    {
        var values = new Tensor<float>(new[] { 5 });
        // Valid: [1, 2, 3, 4]; masked: [-1000].
        var arr = values.AsWritableSpan();
        arr[0] = 1; arr[1] = 2; arr[2] = 3; arr[3] = 4; arr[4] = -1000;
        var mask = new[] { true, true, true, true, false };
        var masked = new MaskedTensor<float>(values, mask);

        // mean = 2.5, var = ((1.5²+0.5²+0.5²+1.5²)/4) = 1.25.
        Assert.Equal(1.25f, MaskedOps.Var(masked).AsSpan()[0], 4);
        Assert.Equal((float)Math.Sqrt(1.25), MaskedOps.Std(masked).AsSpan()[0], 4);
    }

    [Fact]
    public void Min_Max_ArgMin_ArgMax_RespectMask()
    {
        var values = new Tensor<float>(new[] { 5 });
        var arr = values.AsWritableSpan();
        // Masked-out lane has the lowest value (-100); valid lanes are [3, 7, 1, 5].
        arr[0] = 3; arr[1] = 7; arr[2] = 1; arr[3] = 5; arr[4] = -100;
        var mask = new[] { true, true, true, true, false };
        var masked = new MaskedTensor<float>(values, mask);
        Assert.Equal(1f, MaskedOps.Min(masked).AsSpan()[0]);
        Assert.Equal(7f, MaskedOps.Max(masked).AsSpan()[0]);
        Assert.Equal(2, MaskedOps.ArgMin(masked));
        Assert.Equal(1, MaskedOps.ArgMax(masked));
    }

    [Fact]
    public void Add_OutputMaskIsAndOfInputs()
    {
        var a = new Tensor<float>(new[] { 4 });
        var b = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) { a.AsWritableSpan()[i] = i + 1; b.AsWritableSpan()[i] = (i + 1) * 10; }
        var maskA = new[] { true, true, false, true };
        var maskB = new[] { true, false, true, true };
        var ma = new MaskedTensor<float>(a, maskA);
        var mb = new MaskedTensor<float>(b, maskB);
        var sum = MaskedOps.Add(ma, mb);
        // AND of masks = [true, false, false, true]
        Assert.True(sum.IsValid(0));
        Assert.False(sum.IsValid(1));
        Assert.False(sum.IsValid(2));
        Assert.True(sum.IsValid(3));
        Assert.Equal(11f, sum.Values.AsSpan()[0]);
        Assert.Equal(44f, sum.Values.AsSpan()[3]);
        Assert.Equal(2, sum.ValidCount);
    }

    [Fact]
    public void ToDense_ReplacesMaskedLanesWithFill()
    {
        var values = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) values.AsWritableSpan()[i] = i + 1;
        var mask = new[] { true, false, true, false };
        var masked = new MaskedTensor<float>(values, mask);
        var dense = masked.ToDense(fill: -1f);
        Assert.Equal(1f, dense.AsSpan()[0]);
        Assert.Equal(-1f, dense.AsSpan()[1]);
        Assert.Equal(3f, dense.AsSpan()[2]);
        Assert.Equal(-1f, dense.AsSpan()[3]);
    }

    [Fact]
    public void PackedMask_StorageIsBitPacked()
    {
        // 17-element mask should fit in 3 bytes (17 / 8 = 2.125 → 3).
        var values = new Tensor<float>(new[] { 17 });
        var mask = new bool[17];
        var masked = new MaskedTensor<float>(values, mask);
        // 1 bit per element ⇒ 17 / 8 rounded up = 3 bytes.
        Assert.Equal(3, masked.PackedMask.Length);
    }

    [Fact]
    public void NestedToMasked_RoundTripPreservesValues()
    {
        // 3 rows of (varying length, 2 features).
        var rows = new[]
        {
            MakeRow(2, 2, 1f),
            MakeRow(3, 2, 100f),
            MakeRow(1, 2, 1000f),
        };
        var nested = NestedTensor<float>.FromList(rows);
        var (masked, lengths) = StructuredBridges.NestedToMaskedFromPadding(nested, padding: -999f);

        Assert.Equal(new[] { 2, 3, 1 }, lengths);
        Assert.Equal(new[] { 3, 3, 2 }, masked.Shape);
        // Valid count = 2*2 + 3*2 + 1*2 = 12.
        Assert.Equal(12, masked.ValidCount);

        // Round-trip: masked → nested.
        var roundtrip = StructuredBridges.MaskedToNested(masked);
        Assert.Equal(nested.BatchSize, roundtrip.BatchSize);
        Assert.Equal(nested.FeatureSize, roundtrip.FeatureSize);
        Assert.Equal(nested.Offsets, roundtrip.Offsets);
        for (int i = 0; i < nested.Values.Length; i++)
            Assert.Equal(nested.Values.AsSpan()[i], roundtrip.Values.AsSpan()[i]);
    }

    [Fact]
    public void NestedToPadded_PaddedToNested_IsIdentityWhenLengthsMatch()
    {
        var rows = new[]
        {
            MakeRow(1, 3, 5f),
            MakeRow(2, 3, 50f),
            MakeRow(3, 3, 500f),
        };
        var nested = NestedTensor<float>.FromList(rows);
        var padded = StructuredBridges.NestedToPadded(nested, padding: 0f);
        var lengths = new[] { 1, 2, 3 };
        var roundtrip = StructuredBridges.NestedFromPadded(padded, lengths);

        for (int i = 0; i < nested.Values.Length; i++)
            Assert.Equal(nested.Values.AsSpan()[i], roundtrip.Values.AsSpan()[i]);
    }

    private static Tensor<float> MakeRow(int seqLen, int features, float baseVal)
    {
        var t = new Tensor<float>(new[] { seqLen, features });
        for (int s = 0; s < seqLen; s++)
            for (int f = 0; f < features; f++)
                t[s, f] = baseVal + s * 10 + f;
        return t;
    }
}
