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
    [InlineData("   ")]
    [InlineData("99999999999")]
    [InlineData("0-99999999999")]
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

    // A sysfs CPU directory: each entry is a logical processor and the SMT siblings of its core.
    private static string Sysfs(params (int Processor, string Siblings)[] processors)
    {
        string root = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet-cpu-" + Guid.NewGuid().ToString("N"));
        foreach (var (processor, siblings) in processors)
        {
            string topology = System.IO.Path.Combine(root, "cpu" + processor, "topology");
            System.IO.Directory.CreateDirectory(topology);
            System.IO.File.WriteAllText(System.IO.Path.Combine(topology, "thread_siblings_list"), siblings + "\n");
        }

        System.IO.Directory.CreateDirectory(System.IO.Path.Combine(root, "cpufreq"));   // not a processor
        System.IO.Directory.CreateDirectory(System.IO.Path.Combine(root, "cpu9"));      // offline: no topology
        return root;
    }

    [Fact]
    public void LinuxCores_AreReadFromSysfs_InOrderOfTheirFirstProcessor_WithTheirSiblings()
    {
        // Four cores of two threads each, numbered the way a 4-core / 8-thread machine numbers them (0,4 1,5 2,6 3,7),
        // and listed out of order.
        string root = Sysfs((5, "1,5"), (0, "0,4"), (4, "0,4"), (1, "1,5"), (2, "2,6"), (6, "2,6"), (3, "3,7"), (7, "3,7"));
        try
        {
            int[][] cores = CpuTopology.ReadLinuxPhysicalCores(root);
            Assert.Equal(new[] { new[] { 0, 4 }, new[] { 1, 5 }, new[] { 2, 6 }, new[] { 3, 7 } }, cores);
            Assert.Equal(0b0010_0010UL, CpuTopology.MapLinuxCores(new[] { 1 }, cores));
            Assert.Equal(0b1100_1100UL, CpuTopology.MapLinuxCores(new[] { 2, 3 }, cores));

            var missing = Assert.Throws<InvalidOperationException>(() => CpuTopology.MapLinuxCores(new[] { 4 }, cores));
            Assert.Contains(CpuTopology.CoresEnvironmentVariable, missing.Message);
        }
        finally
        {
            System.IO.Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void LinuxCores_RangesAreSiblingsToo_AndAMachineWithoutTopologyIsAnError()
    {
        string root = Sysfs((0, "0-1"), (1, "0-1"), (2, "2-3"), (3, "2-3"));
        try
        {
            int[][] cores = CpuTopology.ReadLinuxPhysicalCores(root);
            Assert.Equal(new[] { new[] { 0, 1 }, new[] { 2, 3 } }, cores);
            Assert.Equal(0b1100UL, CpuTopology.MapLinuxCores(new[] { 1 }, cores));
        }
        finally
        {
            System.IO.Directory.Delete(root, recursive: true);
        }

        Assert.Empty(CpuTopology.ReadLinuxPhysicalCores(System.IO.Path.Combine(root, "absent")));
        var none = Assert.Throws<InvalidOperationException>(() => CpuTopology.MapLinuxCores(new[] { 0 }, Array.Empty<int[]>()));
        Assert.Contains(CpuTopology.CoresEnvironmentVariable, none.Message);

        // A logical processor the affinity mask cannot name is refused, not truncated into another core.
        var beyond = Assert.Throws<InvalidOperationException>(
            () => CpuTopology.MapLinuxCores(new[] { 0 }, new[] { new[] { 3, IntPtr.Size * 8 } }));
        Assert.Contains("taskset", beyond.Message);
    }
}
#endif
