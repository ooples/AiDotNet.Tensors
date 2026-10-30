// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// #653: Clone() shares storage copy-on-write, which privatizes only when a write goes through the shared storage.
/// After Reset an arena hands the same array to the next step's op output, and the op writes it without privatizing
/// anything, so a copy-on-write clone of an arena result changed under its owner (a diffusion predictor copying its
/// output out of its inference arena: the copy changed on the next prediction). A clone of an arena tensor must own
/// its data.
/// </summary>
[Collection(nameof(TensorArenaPinnedTests))]
public class ArenaTensorCloneTests
{
    private static readonly CpuEngine Eng = new CpuEngine();

    private static Tensor<float> Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { 256, n / 256 });
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Fact]
    public void CloneOfArenaResult_OwnsItsData_AndKeepsItsValues_AfterTheNextStepReusesTheArray()
    {
        const int n = 256 * 1152;   // a DiT-sized activation
        var a = Rand(n, 1); var b = Rand(n, 2); var c = Rand(n, 3); var d = Rand(n, 4);
        using var arena = TensorArena.CreateDetached();

        Tensor<float> first;
        using (arena.Activate()) first = Eng.TensorAdd(a, b);
        var expected = first.AsSpan().ToArray();
        var clone = first.Clone();
        // The contract: the clone owns its data. (The overwrite itself depends on which path writes the reused array
        // next; the diffusion predictor that hit it is covered in AiDotNet's InferenceArenaTests.)
        Assert.NotSame(first.GetLiveBackingArrayOrNull(), clone.GetLiveBackingArrayOrNull());
        arena.Reset();

        Tensor<float> second;
        using (arena.Activate()) second = Eng.TensorAdd(c, d);

        // Only meaningful when the arena really handed the first result's array to the second.
        Assert.Same(first.GetLiveBackingArrayOrNull(), second.GetLiveBackingArrayOrNull());
        var values = clone.AsSpan();
        for (int i = 0; i < n; i++)
            Assert.True(values[i] == expected[i], $"element {i} of the clone is {values[i]}, expected {expected[i]}: the next step's output overwrote it");
    }
}