// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #624 Stage 1 — copy-on-write <see cref="TensorBase{T}.CloneShared"/> isolation guarantee.
/// CloneShared() shares the backing storage O(1)-until-write; the first in-place write on either
/// side must privatize so neither observes the other's mutation. These tests drive EVERY write path
/// through the clone (and the source) and assert the peer is unchanged — the completeness guard
/// against a missed gate. Reads must NOT privatize.
/// </summary>
public class TensorCowCloneTests
{
    private static readonly float[] Original = { 1, 2, 3, 4, 5, 6 };

    private static Tensor<float> MakeTensor() => new Tensor<float>((float[])Original.Clone(), new[] { 2, 3 });

    [Fact]
    public void CloneShared_SharesStorage_BothFlaggedCow()
    {
        var src = MakeTensor();
        var clone = (Tensor<float>)src.CloneShared();

        Assert.True(src.IsCowShared, "source should be flagged COW after CloneShared");
        Assert.True(clone.IsCowShared, "clone should be flagged COW after CloneShared");
        Assert.Equal(Original, clone.ToArray());
        Assert.Equal(Original, src.ToArray());
    }

    [Fact]
    public void CloneShared_ReadsDoNotPrivatize()
    {
        var src = MakeTensor();
        var clone = (Tensor<float>)src.CloneShared();

        // Pure reads on both sides — none of these is a write path.
        _ = clone[new[] { 0, 0 }];
        _ = clone.GetFlat(2);
        _ = clone.ToArray();
        _ = clone.AsSpan().Length;
        _ = src[new[] { 1, 2 }];

        Assert.True(src.IsCowShared, "reads must not privatize the source");
        Assert.True(clone.IsCowShared, "reads must not privatize the clone");
    }

    // ---------- every write path must isolate the clone from the source ----------

    private static void AssertCloneWriteIsolates(Action<Tensor<float>> mutateClone, int changedFlat, float expected)
    {
        var src = MakeTensor();
        var clone = (Tensor<float>)src.CloneShared();

        mutateClone(clone);

        Assert.Equal(Original, src.ToArray());                                  // source untouched
        Assert.False(clone.IsCowShared, "clone must privatize on first write");  // privatized
        Assert.Equal(expected, clone.ToArray()[changedFlat]);                   // write landed
    }

    [Fact] public void Write_Indexer_Isolates() => AssertCloneWriteIsolates(t => t[new[] { 0, 0 }] = 99f, 0, 99f);
    [Fact] public void Write_FlatIndexer_Isolates() => AssertCloneWriteIsolates(t => t[2] = 99f, 2, 99f);
    [Fact] public void Write_SetFlat_Isolates() => AssertCloneWriteIsolates(t => t.SetFlat(3, 99f), 3, 99f);
    [Fact] public void Write_SetFlatIndex_Isolates() => AssertCloneWriteIsolates(t => t.SetFlatIndex(4, 99f), 4, 99f);
    [Fact] public void Write_SetFlatIndexValue_Isolates() => AssertCloneWriteIsolates(t => t.SetFlatIndexValue(5, 99f), 5, 99f);
    [Fact] public void Write_CopyFromArray_Isolates() => AssertCloneWriteIsolates(t => t.CopyFromArray(new float[] { 9, 9, 9, 9, 9, 9 }), 0, 9f);
    [Fact] public void Write_Fill_Isolates() => AssertCloneWriteIsolates(t => t.Fill(7f), 0, 7f);
    [Fact] public void Write_AsWritableSpan_Isolates() => AssertCloneWriteIsolates(t => t.AsWritableSpan()[1] = 99f, 1, 99f);
    [Fact] public void Write_GetDataArray_Isolates() => AssertCloneWriteIsolates(t => t.GetDataArray()[1] = 99f, 1, 99f);
    [Fact] public void Write_RawWritableStorageSpan_Isolates() => AssertCloneWriteIsolates(t => t.RawWritableStorageSpan[1] = 99f, 1, 99f);
    [Fact] public void Write_Memory_Isolates() => AssertCloneWriteIsolates(t => t.Memory.Span[1] = 99f, 1, 99f);

    [Fact]
    public void Write_AsVector_Isolates()
    {
        var src = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var clone = (Tensor<float>)src.CloneShared();

        var vec = clone.AsVector();
        vec[0] = 99f;

        Assert.Equal(new float[] { 1, 2, 3, 4 }, src.ToArray());
        Assert.Equal(99f, clone.ToArray()[0]);
    }

    // ---------- the reverse: writing the SOURCE must not disturb the clone ----------

    [Fact]
    public void Write_Source_DoesNotDisturbClone()
    {
        var src = MakeTensor();
        var clone = (Tensor<float>)src.CloneShared();

        src[new[] { 0, 0 }] = 42f;

        Assert.Equal(Original, clone.ToArray());     // clone still original
        Assert.Equal(42f, src.ToArray()[0]);
        Assert.False(src.IsCowShared, "source must privatize on its own first write");
    }

    [Fact]
    public void DoubleWrite_OnSamePeer_IsStable()
    {
        var src = MakeTensor();
        var clone = (Tensor<float>)src.CloneShared();

        clone[0] = 10f;   // privatizes
        clone[1] = 20f;   // already owned — must still write correctly
        Assert.Equal(Original, src.ToArray());
        Assert.Equal(10f, clone.ToArray()[0]);
        Assert.Equal(20f, clone.ToArray()[1]);
    }

    // ---------- non-shareable layouts fall back to a correct deep copy ----------

    [Fact]
    public void CloneShared_OnView_FallsBackToDeepCopy()
    {
        var src = MakeTensor();
        var view = src.Transpose(new[] { 1, 0 }); // non-contiguous view → not shareable

        var clone = (Tensor<float>)((TensorBase<float>)view).CloneShared();

        Assert.False(clone.IsCowShared, "a view must deep-copy, not COW-share");
        // Deep copy is value-correct and independent of the source.
        var expected = view.ToArray();
        clone[0] = 123f;
        Assert.Equal(expected, src.Transpose(new[] { 1, 0 }).ToArray());
    }

    // ---------- Stage 3 (#624): the public Clone() itself routes through COW by default ----------

    [Fact]
    public void Clone_RoutesThroughCow_SharesUntilWrite()
    {
        Assert.True(TensorBase<float>.UseCopyOnWriteClone, "COW clone is the default");
        var src = MakeTensor();
        var clone = (Tensor<float>)src.Clone();

        // O(1) proof: CloneShared only flags _cowShared when it actually shared storage (the eager
        // CloneDeepCopy fallback never does), so IsCowShared==true means no full-buffer copy happened.
        Assert.True(clone.IsCowShared, "Clone() must share storage (O(1)) by default");
        Assert.True(src.IsCowShared, "the source is flagged COW too");
        Assert.Equal(Original, clone.ToArray());
    }

    [Fact]
    public void Clone_ThenWrite_IsolatesLikeDeepCopy()
    {
        var src = MakeTensor();
        var clone = (Tensor<float>)src.Clone();

        clone[2] = 77f;                                  // first write privatizes

        Assert.Equal(Original, src.ToArray());           // source untouched — deep-copy semantics preserved
        Assert.False(clone.IsCowShared);
        Assert.Equal(77f, clone.ToArray()[2]);
    }

    [Fact]
    public void Clone_LargeTensor_IsCheapShare()
    {
        // A 4M-element tensor: an eager Clone would copy 16 MB; COW shares at O(1). We assert the
        // share happened (IsCowShared) rather than timing, then prove isolation still holds.
        var big = new Tensor<float>(new float[4_000_000], new[] { 4_000_000 });
        var clone = (Tensor<float>)big.Clone();

        Assert.True(clone.IsCowShared, "large Clone() must be an O(1) share, not a 16 MB copy");
        clone[0] = 5f;
        Assert.Equal(0f, big.ToArray()[0]);              // privatized write didn't touch the source
    }

    [Fact]
    public void CloneDeepCopy_AlwaysEager_NeverShares()
    {
        var src = MakeTensor();
        var eager = (Tensor<float>)src.CloneDeepCopy();

        Assert.False(eager.IsCowShared, "CloneDeepCopy is always an independent eager copy");
        Assert.False(src.IsCowShared, "an eager copy must not flag the source COW");
        eager[0] = 55f;
        Assert.Equal(Original, src.ToArray());
    }
}
