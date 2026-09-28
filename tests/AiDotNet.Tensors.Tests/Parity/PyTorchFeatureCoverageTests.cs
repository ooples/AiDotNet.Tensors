using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using static AiDotNet.Tensors.Tests.Parity.PyTorchParityInventory;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// Feature parity against PyTorch, in two lanes (issue #1056).
/// </summary>
/// <remarks>
/// <para>
/// <b>Ratchet</b> (<see cref="Coverage_DoesNotRegress"/>, every PR): every item PyTorch exposes has a
/// coverage entry, every mapped Tensors symbol really exists, and the number of gaps has not grown. Closing a
/// gap must lower <c>gapBaseline</c> in the same change, so progress is recorded rather than absorbed.
/// </para>
/// <para>
/// <b>Parity</b> (<see cref="FullParity_NoGapRemains"/>, nightly, <c>Category=PyTorchParity</c>): fails while
/// any gap remains, and prints the full list grouped by category. It is the absolute distance to parity, and
/// it is expected to be red until that distance is zero.
/// </para>
/// </remarks>
public class PyTorchFeatureCoverageTests
{
    private readonly ITestOutputHelper _output;

    public PyTorchFeatureCoverageTests(ITestOutputHelper output) => _output = output;

    private static string RequireRoot()
        => FindRepositoryRoot()
           ?? throw new InvalidOperationException(
               $"Could not find {SurfaceFile} above {AppContext.BaseDirectory}. The parity inventory is checked in; " +
               "a missing file is a broken checkout, not a deferred check.");

    [Fact]
    public void Coverage_DoesNotRegress()
    {
        string root = RequireRoot();
        var surface = LoadSurface(root);
        var coverage = LoadCoverage(root);
        var problems = new StringBuilder();

        Assert.True(surface.TorchVersion == coverage.TorchVersion,
            $"{CoverageFile} was reviewed against torch {coverage.TorchVersion} but {SurfaceFile} was extracted from " +
            $"torch {surface.TorchVersion}. Re-run the seeder for new items and review the result.");

        var surfaceKeys = new HashSet<string>(surface.Items.Select(i => i.Key), StringComparer.Ordinal);
        var unrecorded = surface.Items.Where(i => !coverage.Entries.ContainsKey(i.Key)).Select(i => i.Key).ToList();
        var orphaned = coverage.Entries.Keys.Where(k => !surfaceKeys.Contains(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        var unresolved = coverage.Entries
            .Where(e => e.Value.Status == CoverageStatus.Mapped)
            .Select(e => (e.Key, Error: e.Value.Symbol is null ? "mapped with no symbol" : ResolveSymbol(e.Value.Symbol)))
            .Where(r => r.Error is not null)
            .ToList();
        var unexplained = coverage.Entries
            .Where(e => e.Value.Status == CoverageStatus.NotApplicable && string.IsNullOrWhiteSpace(e.Value.Reason))
            .Select(e => e.Key)
            .ToList();
        int gaps = coverage.Entries.Count(e => e.Value.Status == CoverageStatus.Gap && surfaceKeys.Contains(e.Key));

        if (unrecorded.Count > 0)
            problems.AppendLine($"{unrecorded.Count} PyTorch item(s) have no coverage entry:").AppendLine(Indent(unrecorded));
        if (orphaned.Count > 0)
            problems.AppendLine($"{orphaned.Count} coverage entr(ies) name items PyTorch no longer exposes:").AppendLine(Indent(orphaned));
        if (unresolved.Count > 0)
            problems.AppendLine($"{unresolved.Count} mapped symbol(s) do not resolve:")
                .AppendLine(Indent(unresolved.Select(u => $"{u.Key} -> {u.Error}")));
        if (unexplained.Count > 0)
            problems.AppendLine($"{unexplained.Count} not-applicable entr(ies) give no reason:").AppendLine(Indent(unexplained));
        if (gaps > coverage.GapBaseline)
            problems.AppendLine($"Gaps rose from {coverage.GapBaseline} to {gaps}: a mapping was removed or downgraded.");
        if (gaps < coverage.GapBaseline)
            problems.AppendLine($"Gaps fell from {coverage.GapBaseline} to {gaps}. Lower gapBaseline in {CoverageFile} to {gaps} " +
                                "in this change, so the progress is recorded rather than silently absorbed.");

        WriteSummary(surface, coverage);
        Assert.True(problems.Length == 0, problems.ToString());
    }

    [Fact]
    [Trait("Category", "PyTorchParity")]
    public void FullParity_NoGapRemains()
    {
        string root = RequireRoot();
        var surface = LoadSurface(root);
        var coverage = LoadCoverage(root);

        var gaps = surface.Items
            .Where(i => coverage.Entries.TryGetValue(i.Key, out var e) && e.Status == CoverageStatus.Gap)
            .GroupBy(i => i.Category)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .ToList();

        WriteSummary(surface, coverage);
        if (gaps.Count == 0) return;

        var message = new StringBuilder($"Tensors is missing {gaps.Sum(g => g.Count())} PyTorch {surface.TorchVersion} feature(s):");
        message.AppendLine();
        foreach (var group in gaps)
            message.AppendLine($"  {group.Key} ({group.Count()}): {string.Join(", ", group.Select(i => i.Name))}");
        Assert.Fail(message.ToString());
    }

    /// <summary>
    /// Proposes <c>parity/coverage.json</c> from names. Runs only when <c>PARITY_SEED=1</c>, and refuses to
    /// overwrite a reviewed manifest unless <c>PARITY_SEED_OVERWRITE=1</c>: its output is a starting point for
    /// review, never a claim.
    /// </summary>
    [SkippableFact]
    public void Seed_CoverageManifest()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("PARITY_SEED") == "1", "Set PARITY_SEED=1 to regenerate the proposal.");
        string root = RequireRoot();
        string path = Path.Combine(root, CoverageFile);
        Skip.If(File.Exists(path) && Environment.GetEnvironmentVariable("PARITY_SEED_OVERWRITE") != "1",
            $"{CoverageFile} exists; set PARITY_SEED_OVERWRITE=1 to replace it.");

        var seeded = Seed(LoadSurface(root));
        File.WriteAllText(path, seeded.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
        _output.WriteLine($"Wrote {path}");
    }

    private void WriteSummary(Surface surface, Coverage coverage)
    {
        _output.WriteLine($"PyTorch {surface.TorchVersion}: {surface.Items.Count} feature(s)");
        foreach (var group in surface.Items.GroupBy(i => i.Category).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            int mapped = 0, gap = 0, na = 0;
            foreach (var item in group)
            {
                if (!coverage.Entries.TryGetValue(item.Key, out var e)) continue;
                if (e.Status == CoverageStatus.Mapped) mapped++;
                else if (e.Status == CoverageStatus.Gap) gap++;
                else na++;
            }

            _output.WriteLine($"  {group.Key,-20} {group.Count(),5}  mapped {mapped,5}  gap {gap,5}  n/a {na,5}");
        }
    }

    private static string Indent(System.Collections.Generic.IEnumerable<string> lines)
        => string.Join(Environment.NewLine, lines.Select(l => "  " + l));
}
