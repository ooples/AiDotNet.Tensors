// Copyright (c) AiDotNet. All rights reserved.
// PTX-vs-CUDA-vs-CPU parity scaffold. Coverage audit: every direct-PTX kernel
// must carry an explicit parity decision, so coverage is 100% auditable and a
// new kernel cannot be added without one.
#if !NETFRAMEWORK

using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.PtxParity;

public sealed class PtxKernelCoverageTests
{
    private readonly ITestOutputHelper _out;

    public PtxKernelCoverageTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Every kernel discovered in the assembly must have an explicit registry
    /// decision (a three-way parity spec or a documented deferral). This is what
    /// makes coverage 100% auditable: a newly added Ptx*Kernel with no decision
    /// fails this test until someone consciously classifies it.
    /// </summary>
    [Fact]
    public void EveryPtxKernelHasAnExplicitCoverageDecision()
    {
        Assert.NotEmpty(PtxKernelInventory.KernelTypeNames);
        string[] undecided = PtxKernelInventory.KernelTypeNames
            .Where(name => !PtxParityRegistry.TryGet(name, out _))
            .ToArray();
        Assert.True(undecided.Length == 0,
            "These direct-PTX kernels have no parity coverage decision in PtxParityRegistry. " +
            "Add a ThreeWayParity spec or a documented Deferred entry: " +
            string.Join(", ", undecided));
    }

    /// <summary>The registry must not name kernels that no longer exist (stale entries).</summary>
    [Fact]
    public void RegistryHasNoStaleEntries()
    {
        var inventory = PtxKernelInventory.KernelTypeNames.ToHashSet(StringComparer.Ordinal);
        string[] stale = PtxParityRegistry.Specs
            .Select(s => s.KernelTypeName)
            .Where(name => !inventory.Contains(name))
            .ToArray();
        Assert.True(stale.Length == 0,
            "PtxParityRegistry references kernels that are not in the assembly (remove or rename): " +
            string.Join(", ", stale));
    }

    /// <summary>
    /// Deferred entries must state a reason, so a deferral is a real decision and
    /// not an empty placeholder.
    /// </summary>
    [Fact]
    public void DeferredEntriesStateAReason()
    {
        var reasonless = PtxParityRegistry.Specs
            .Where(s => s.Status == PtxParityStatus.Deferred &&
                        string.IsNullOrWhiteSpace(s.Note))
            .Select(s => s.KernelTypeName)
            .ToArray();
        Assert.True(reasonless.Length == 0,
            "Deferred parity entries must document why: " + string.Join(", ", reasonless));
    }

    /// <summary>Prints an auditable coverage summary to the test output.</summary>
    [Fact]
    public void CoverageSummary_IsAuditable()
    {
        int total = PtxKernelInventory.KernelTypeNames.Count;
        int threeWay = PtxParityRegistry.Specs.Count(s => s.Status == PtxParityStatus.ThreeWayParity);
        int deferred = PtxParityRegistry.Specs.Count(s => s.Status == PtxParityStatus.Deferred);
        _out.WriteLine($"Direct-PTX kernels: {total}");
        _out.WriteLine($"  three-way parity: {threeWay}");
        _out.WriteLine($"  deferred (documented): {deferred}");
        foreach (var s in PtxParityRegistry.Specs.OrderBy(s => s.Status).ThenBy(s => s.KernelTypeName, StringComparer.Ordinal))
            _out.WriteLine($"  [{s.Status,-14}] {s.KernelTypeName} — {s.BackingPublicOp}");
        Assert.Equal(total, threeWay + deferred);
    }
}
#endif
