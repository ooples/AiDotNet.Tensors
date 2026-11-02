#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #653: on a host with more than one Windows processor group the worker pool's core list must cover the cores of
/// every group, not just the process's own group, so a thread budget above one group's core count gets a distinct
/// physical core per worker instead of doubling up on SMT siblings.
/// </summary>
public class CpuTopologyProcessorGroupTests
{
    [Fact]
    public void UsableCoresStartsWithTheOwnGroupAndCoversEveryPhysicalCoreOnce()
    {
        var own = CpuTopology.UsableCoresInCurrentGroup();
        var all = CpuTopology.UsableCores();
        if (own.Length == 0)
        {
            Assert.Empty(all); // non-Windows or no topology: nothing to bind
            return;
        }

        // The dispatcher's core (index 0) and the rest of the own group come first, in order.
        Assert.Equal(own, all.Take(own.Length).ToArray());
        // One entry per physical core: no core listed twice.
        Assert.Equal(all.Length, all.Distinct().Count());
        // Each entry is the first logical processor of a distinct physical core.
        var physical = CpuTopology.DetectPhysicalCores();
        foreach (var (group, number) in all)
            Assert.Contains(physical, c => c.Group == group && (c.Mask & (1UL << number)) != 0);
        // Unrestricted affinity (the test host's default) reaches every group's cores.
        int groups = physical.Select(c => c.Group).Distinct().Count();
        if (groups > 1 && own.Length == physical.Count(c => c.Group == own[0].Group))
            Assert.Equal(physical.Length, all.Length);
    }

    [Fact]
    public void SpanGroupsKillSwitchKeepsTheOwnGroup()
    {
        string? saved = Environment.GetEnvironmentVariable("AIDOTNET_CPU_SPAN_GROUPS");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CPU_SPAN_GROUPS", "0");
            Assert.Equal(CpuTopology.UsableCoresInCurrentGroup(), CpuTopology.UsableCores());
        }
        finally { Environment.SetEnvironmentVariable("AIDOTNET_CPU_SPAN_GROUPS", saved); }
    }
}
#endif
