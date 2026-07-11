// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). Full-surface coverage audit.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Enumerates the ENTIRE tensor-returning IEngine op surface and audits how much of it the parity
/// registry covers, writing a coverage report (covered / pending lists + %). This is the "scaffold
/// for all operations" spine: it makes full-surface coverage measurable and forces every registry
/// entry to name a real IEngine op, so the registry can grow toward 100% with the gaps always visible.
/// </summary>
[Collection("OpParity")]
public sealed class OpParityCoverageTests
{
    /// <summary>Every registry OpMethod must be a real tensor-returning IEngine op — guards typos and
    /// stale specs, and keeps the coverage denominator honest.</summary>
    [Fact]
    public void EveryRegistryCase_NamesARealIEngineOp()
    {
        var inventory = IEngineOpInventory.TensorReturningOps().ToHashSet(StringComparer.Ordinal);
        foreach (var op in OpParityRegistry.ViTPath())
        {
            Assert.True(inventory.Contains(op.OpMethod),
                $"OpCase '{op.Name}' claims to cover IEngine op '{op.OpMethod}', which is not a " +
                $"tensor-returning IEngine method (typo, or it returns something other than Tensor<T>).");
        }
    }

    /// <summary>Writes the full-surface coverage report and asserts we have positive coverage and
    /// no regression below a tracked floor. The report lists exactly which ops still need a spec —
    /// the worklist for driving the scaffold to 100%.</summary>
    [Fact]
    public void FullSurface_CoverageReport()
    {
        var inventory = IEngineOpInventory.TensorReturningOps();
        var covered = OpParityRegistry.ViTPath()
            .Select(o => o.OpMethod)
            .Distinct(StringComparer.Ordinal)
            .ToHashSet(StringComparer.Ordinal);
        var pending = inventory.Where(n => !covered.Contains(n)).OrderBy(n => n, StringComparer.Ordinal).ToList();

        WriteReport(inventory, covered, pending);

        Assert.NotEmpty(covered);
        // Coverage floor: keep this at/above the current count so coverage can only grow. Raise it
        // as the registry expands; a drop means an op silently lost its spec.
        const int coverageFloor = 10;
        Assert.True(covered.Count >= coverageFloor,
            $"Parity op coverage dropped to {covered.Count} distinct IEngine ops (floor {coverageFloor}). " +
            $"An op likely lost its registry spec.");
    }

    private static void WriteReport(IReadOnlyList<string> inventory, ISet<string> covered, IReadOnlyList<string> pending)
    {
        try
        {
            var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                      ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
            Directory.CreateDirectory(dir);
            var sb = new StringBuilder();
            double pct = inventory.Count == 0 ? 0 : 100.0 * covered.Count / inventory.Count;
            sb.AppendLine($"# CPU-vs-GPU op-parity coverage (#775)");
            sb.AppendLine($"tensor-returning IEngine ops : {inventory.Count}");
            sb.AppendLine($"covered by parity registry   : {covered.Count} ({pct:F1}%)");
            sb.AppendLine($"total public IEngine overloads: {IEngineOpInventory.TotalPublicMethodOverloads()}");
            sb.AppendLine();
            sb.AppendLine("## covered");
            foreach (var c in covered.OrderBy(x => x, StringComparer.Ordinal)) sb.AppendLine($"  [x] {c}");
            sb.AppendLine();
            sb.AppendLine("## pending (needs a parity spec)");
            foreach (var p in pending) sb.AppendLine($"  [ ] {p}");
            File.WriteAllText(Path.Combine(dir, "op-parity-coverage.md"), sb.ToString());
        }
        catch { /* best-effort */ }
    }
}
#endif
