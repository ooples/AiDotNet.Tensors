// Copyright (c) AiDotNet. All rights reserved.
// PTX transfer tracking. Reports transfer progress and signals — via a single
// gate that flips from skipped to passing — when every CUDA kernel has a
// promoted PTX replacement and the CUDA kernels can be deleted.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.PtxParity;

public sealed class CudaToPtxTransferTests
{
    private readonly ITestOutputHelper _out;

    public CudaToPtxTransferTests(ITestOutputHelper output) => _out = output;

    /// <summary>The ledger may only reference CUDA kernels that actually exist (no stale entries).</summary>
    [Fact]
    public void LedgerReferencesRealCudaKernels()
    {
        Assert.NotEmpty(CudaKernelCensus.KernelNames);
        var census = CudaKernelCensus.KernelNames.ToHashSet(StringComparer.Ordinal);
        string[] unknown = CudaToPtxTransferLedger.Entries
            .Select(e => e.CudaKernel)
            .Where(k => !census.Contains(k))
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        Assert.True(unknown.Length == 0,
            "Transfer ledger references CUDA kernels not present in the census (rename or remove): " +
            string.Join(", ", unknown));
    }

    /// <summary>No CUDA kernel may have two conflicting ledger records.</summary>
    [Fact]
    public void LedgerHasNoDuplicateCudaKernels()
    {
        var dupes = CudaToPtxTransferLedger.Entries
            .GroupBy(e => e.CudaKernel, StringComparer.Ordinal)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToArray();
        Assert.True(dupes.Length == 0,
            "Transfer ledger has duplicate CUDA-kernel entries: " + string.Join(", ", dupes));
    }

    /// <summary>Prints the transfer scoreboard: how many CUDA kernels are replaced, in progress, remaining.</summary>
    [Fact]
    public void TransferProgress_IsReported()
    {
        int total = CudaKernelCensus.KernelNames.Count;
        int replaced = CudaToPtxTransferLedger.Replaced.Count();
        int inProgress = CudaToPtxTransferLedger.InProgress.Count();
        int notPlanned = CudaToPtxTransferLedger.NotPlanned.Count();
        int remaining = total - replaced - notPlanned;

        _out.WriteLine($"CUDA kernels total:            {total}");
        _out.WriteLine($"  replaced by promoted PTX:    {replaced}");
        _out.WriteLine($"  PTX in progress (not promoted): {inProgress}");
        _out.WriteLine($"  not planned (infra/gap/reassigned): {notPlanned}");
        _out.WriteLine($"  remaining before full delete:  {remaining}");
        _out.WriteLine("");
        _out.WriteLine("In-progress ports:");
        foreach (var e in CudaToPtxTransferLedger.InProgress.OrderBy(e => e.CudaKernel, StringComparer.Ordinal))
            _out.WriteLine($"  {e.CudaKernel} -> {e.PtxKernel}: {e.Note}");

        // Always informational — never fails; it is the human-readable scoreboard.
        Assert.True(replaced <= total);
    }

    /// <summary>
    /// The transfer-complete signal. This test is SKIPPED while any CUDA kernel
    /// lacks a promoted PTX replacement, and PASSES only once every census kernel
    /// is ledger-marked PtxPromotedReplaced. When this test stops skipping, the
    /// CUDA kernels are safe to delete — that transition is the deliberate,
    /// automated "we can do the full transfer now" gate.
    /// </summary>
    [SkippableFact]
    public void FullTransferComplete_AllCudaKernelsHavePromotedPtxReplacement()
    {
        var replaced = CudaToPtxTransferLedger.Replaced
            .Select(e => e.CudaKernel)
            .ToHashSet(StringComparer.Ordinal);
        // Kernels deliberately excluded from the transfer (infra utilities,
        // genuine gaps, or category-subsumed) do not block completion.
        var excluded = CudaToPtxTransferLedger.NotPlanned
            .Select(e => e.CudaKernel)
            .ToHashSet(StringComparer.Ordinal);
        string[] remaining = CudaKernelCensus.KernelNames
            .Where(k => !replaced.Contains(k) && !excluded.Contains(k))
            .OrderBy(k => k, StringComparer.Ordinal)
            .ToArray();

        Skip.If(remaining.Length > 0,
            $"Full PTX transfer not complete: {remaining.Length} CUDA kernel(s) still lack a promoted " +
            $"PTX replacement. First few: {string.Join(", ", remaining.Take(8))}" +
            (remaining.Length > 8 ? ", ..." : "."));

        Assert.Empty(remaining); // reached only when transfer is 100% complete.
    }
}
#endif
