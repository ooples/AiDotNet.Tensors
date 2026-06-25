// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Validates the fused grouped/depthwise deformable conv (DCNv3) CPU reference kernel (#1691) against
/// the per-group composition over the existing groups=1 DeformableConv2D — the correctness oracle the
/// future GPU backends must also match.
/// </summary>
public class GroupedDeformableConv2DTests
{
    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var d = new float[n];
        var rng = new Random(seed);
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return new Tensor<float>(shape, new Vector<float>(d));
    }

    [Theory]
    [InlineData(2, 2)]   // groups=2, deformGroups=2
    [InlineData(4, 4)]   // depthwise-ish
    [InlineData(2, 1)]   // shared deformable group across output groups
    public void FusedGrouped_MatchesPerGroupComposition(int groups, int deformGroups)
    {
        var engine = new CpuEngine();
        const int B = 1, inC = 8, outC = 8, H = 8, W = 8, k = 3;
        int inCpg = inC / groups, outCpg = outC / groups, kk = k * k;

        var input = Rand(1, B, inC, H, W);
        var kernel = Rand(2, outC, inCpg, k, k);                 // [outC, inC/groups, k, k]
        var offset = Rand(3, B, 2 * kk * deformGroups, H, W);    // stride 1, pad 1 -> out = H,W
        int[] stride = [1, 1], pad = [1, 1], dil = [1, 1];

        var fused = engine.DeformableConv2DGrouped(input, kernel, offset, null, stride, pad, dil, groups, deformGroups);

        // Reference: per-group groups=1 deformable conv over channel slices, concatenated.
        var groupOuts = new Tensor<float>[groups];
        for (int g = 0; g < groups; g++)
        {
            int dg = deformGroups == 1 ? 0 : g * deformGroups / groups;
            var inSlice = input.Slice(1, g * inCpg, (g + 1) * inCpg);
            var wSlice = kernel.Slice(0, g * outCpg, (g + 1) * outCpg);
            var offSlice = offset.Slice(1, dg * 2 * kk, (dg + 1) * 2 * kk);
            groupOuts[g] = engine.DeformableConv2D(inSlice, wSlice, offSlice, null, stride, pad, dil);
        }
        var reference = Tensor<float>.Concatenate(groupOuts, axis: 1);

        Assert.Equal(reference.Shape, fused.Shape);
        for (int i = 0; i < reference.Length; i++)
            Assert.True(MathF.Abs(reference[i] - fused[i]) < 1e-4f,
                $"groups={groups} dg={deformGroups}: mismatch at {i}: ref={reference[i]} fused={fused[i]}");
    }

    [Fact]
    public void Groups1_ReducesToBaseDeformableConv()
    {
        var engine = new CpuEngine();
        const int B = 1, C = 4, H = 6, W = 6, k = 3, kk = 9;
        var input = Rand(5, B, C, H, W);
        var kernel = Rand(6, C, C, k, k);
        var offset = Rand(7, B, 2 * kk, H, W);
        int[] stride = [1, 1], pad = [1, 1], dil = [1, 1];
        var baseOut = engine.DeformableConv2D(input, kernel, offset, null, stride, pad, dil);
        var grouped = engine.DeformableConv2DGrouped(input, kernel, offset, null, stride, pad, dil, 1, 1);
        for (int i = 0; i < baseOut.Length; i++) Assert.Equal(baseOut[i], grouped[i]);
    }
}
