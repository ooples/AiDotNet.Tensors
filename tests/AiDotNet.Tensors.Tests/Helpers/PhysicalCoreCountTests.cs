using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The physical-core count that now sets the default parallel width, parsed from /proc/cpuinfo text: one (physical id,
/// core id) pair per processor record, and an unknown topology (a record missing either id) falls back to the logical
/// processor count rather than merging cores.
/// </summary>
public sealed class PhysicalCoreCountTests
{
    private static string[] Record(int processor, int? physical, int? core)
    {
        var lines = new System.Collections.Generic.List<string> { $"processor\t: {processor}", "vendor_id\t: AuthenticAMD" };
        if (physical is int p) lines.Add($"physical id\t: {p}");
        if (core is int c) lines.Add($"core id\t\t: {c}");
        lines.Add("");
        return lines.ToArray();
    }

    private static string[] Join(params string[][] records)
    {
        var all = new System.Collections.Generic.List<string>();
        foreach (var r in records) all.AddRange(r);
        return all.ToArray();
    }

    [Fact]
    public void SmtSiblings_CountOnce()
        => Assert.Equal(2, CpuParallelSettings.CountPhysicalCores(Join(Record(0, 0, 0), Record(1, 0, 1), Record(2, 0, 0), Record(3, 0, 1))));

    [Fact]
    public void SameCoreIdOnTwoSockets_CountsTwice()
        => Assert.Equal(2, CpuParallelSettings.CountPhysicalCores(Join(Record(0, 0, 0), Record(1, 1, 0))));

    [Fact]
    public void MissingPhysicalId_FallsBackToLogicalCount()
        // Without the per-record check, core id 0 and 1 of a socket-less record would be added under the previous
        // record's physical id and the count would read 2 for distinct cores.
        => Assert.Equal(Environment.ProcessorCount,
            CpuParallelSettings.CountPhysicalCores(Join(Record(0, 0, 0), Record(1, null, 0), Record(2, null, 1))));

    [Fact]
    public void MissingCoreId_FallsBackToLogicalCount()
        => Assert.Equal(Environment.ProcessorCount, CpuParallelSettings.CountPhysicalCores(Join(Record(0, 0, null))));

    [Fact]
    public void NoTrailingBlankLine_StillCountsTheLastRecord()
        => Assert.Equal(2, CpuParallelSettings.CountPhysicalCores(new[] { "physical id : 0", "core id : 0", "", "physical id : 0", "core id : 1" }));

    [Fact]
    public void Empty_FallsBackToLogicalCount()
        => Assert.Equal(Environment.ProcessorCount, CpuParallelSettings.CountPhysicalCores(Array.Empty<string>()));
}