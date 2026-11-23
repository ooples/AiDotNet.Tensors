// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// <c>AIDOTNET_CPU_CORES</c>: confining a process's worker pool to a set of physical cores, so two processes on one
/// machine stop binding their workers to the same cores.
/// </summary>
public class CpuCoreSetTests
{
    [Theory]
    [InlineData("3", new[] { 3 })]
    [InlineData("0-3", new[] { 0, 1, 2, 3 })]
    [InlineData(" 32-34 , 8 ,33", new[] { 32, 33, 34, 8 })]   // order kept, duplicates dropped
    public void ParseCoreList_ReadsIndexesAndRanges(string text, int[] expected)
        => Assert.Equal(expected, CpuTopology.ParseCoreList(text));

    [Theory]
    [InlineData("")]
    [InlineData("a")]
    [InlineData("5-2")]
    [InlineData("-1")]
    [InlineData("1,,2")]
    [InlineData("0x4")]
    public void ParseCoreList_RejectsMalformedLists(string text)
    {
        var ex = Assert.Throws<InvalidOperationException>(() => CpuTopology.ParseCoreList(text));
        Assert.Contains(CpuTopology.CoresEnvironmentVariable, ex.Message);
    }

    // Two groups of two cores, each core an SMT pair: group 0 = processors 0-3, group 1 = processors 0-3 of group 1.
    private static readonly CpuTopology.Domain[] Machine =
    {
        new(0b0011, 0, 1), new(0b1100, 0, 1), new(0b0011, 1, 1), new(0b1100, 1, 1),
    };

    [Fact]
    public void MapRequestedCores_OneGroup_GivesTheFirstSiblingAndTheGroupMask()
    {
        var (cores, group, mask) = CpuTopology.MapRequestedCores(new[] { 1, 0 }, Machine);
        Assert.Equal(new (ushort, byte)[] { (0, 2), (0, 0) }, cores);
        Assert.Equal((ushort)0, group);
        Assert.Equal(0b1111UL, mask);   // SMT siblings included
    }

    [Fact]
    public void MapRequestedCores_AcrossGroups_HasNoSingleGroupMask()
    {
        var (cores, group, mask) = CpuTopology.MapRequestedCores(new[] { 1, 2 }, Machine);
        Assert.Equal(new (ushort, byte)[] { (0, 2), (1, 0) }, cores);
        Assert.Null(group);
        Assert.Equal(0UL, mask);
    }

    [Fact]
    public void MapRequestedCores_RejectsACoreTheMachineLacks()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => CpuTopology.MapRequestedCores(new[] { 0, 4 }, Machine));
        Assert.Contains("core 4", ex.Message);
    }
}
#endif
