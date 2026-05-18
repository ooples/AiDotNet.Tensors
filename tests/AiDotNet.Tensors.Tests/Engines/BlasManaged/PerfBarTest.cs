using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue A (#369) task A.7: gate test that asserts BlasManaged meets the
/// codified perf bar in <see cref="PerfBar"/>.
///
/// <para>
/// <see cref="Catalog_Size_Matches_PerfBar"/> always runs and verifies the catalog
/// size matches the bar's <see cref="PerfBar.CatalogShapeCount"/>. If a new shape
/// is added without updating <see cref="PerfBar"/>, this fails — the constant
/// is intentionally load-bearing so the bar gets re-evaluated when the catalog grows.
/// </para>
///
/// <para>
/// <see cref="WinRate_And_MaxLoss_Meet_PerfBar"/> only runs on the authoritative
/// self-hosted runner (env var <c>AIDOTNET_PERF_RUNNER=1</c>) and only when the
/// host's <see cref="HardwareFingerprint"/> matches <see cref="PerfBar.TargetHardwareFingerprint"/>.
/// Skipped elsewhere because perf numbers aren't comparable across hardware.
/// </para>
///
/// <para>
/// <b>Expected state during the sprint:</b> immediately after A.7 lands and before
/// B/C/D/E close their bars, <see cref="WinRate_And_MaxLoss_Meet_PerfBar"/> will
/// fail on the authoritative runner because BlasManaged hasn't caught up yet.
/// That failure is the gate Sub-issue G (#375 — native removal) must satisfy.
/// </para>
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class PerfBarTest
{
    [Fact]
    public void Catalog_Size_Matches_PerfBar()
    {
        Assert.Equal(PerfBar.CatalogShapeCount, ShapeCatalog.All.Count);
    }

    /// <summary>
    /// One-shot baseline generator: sweeps the full catalog through
    /// <see cref="PerfHarness"/> and writes the JSON to <c>artifacts/perf/baseline.json</c>
    /// (or to <c>AIDOTNET_BASELINE_OUT</c> if set). Gated by env var
    /// <c>AIDOTNET_GENERATE_BASELINE=1</c> so it doesn't run on normal test invocations
    /// (it's slow — full catalog sweep can take minutes).
    /// </summary>
    [Fact]
    public void Generate_Baseline_From_Full_Catalog()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_GENERATE_BASELINE") != "1") return;

        var outPath = Environment.GetEnvironmentVariable("AIDOTNET_BASELINE_OUT");
        if (string.IsNullOrWhiteSpace(outPath))
        {
            // Default: walk up to find repo root, write to artifacts/perf/baseline.json.
            var dir = AppContext.BaseDirectory;
            for (int i = 0; i < 10 && dir != null; i++)
            {
                if (Directory.Exists(Path.Combine(dir, ".git")) ||
                    Directory.Exists(Path.Combine(dir, "artifacts")))
                {
                    outPath = Path.Combine(dir, "artifacts", "perf", "baseline.json");
                    break;
                }
                dir = Path.GetDirectoryName(dir);
            }
        }
        Assert.False(string.IsNullOrWhiteSpace(outPath), "Could not resolve baseline output path");

        PerfHarness.RunAll(ShapeCatalog.All, outPath!);
        Assert.True(File.Exists(outPath), $"Expected baseline at {outPath}");
    }

    [Fact]
    public void WinRate_And_MaxLoss_Meet_PerfBar()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") != "1")
        {
            // Not the authoritative runner — skip cleanly. Perf numbers aren't
            // comparable across hardware.
            return;
        }

        if (HardwareFingerprint.Current.ToString() != PerfBar.TargetHardwareFingerprint)
        {
            // Wrong hardware — skip. The bar is calibrated for
            // PerfBar.TargetHardwareFingerprint only.
            return;
        }

        // Run the full catalog through PerfHarness; assert against PerfBar.
        var tmpPath = Path.Combine(Path.GetTempPath(), $"perfbar-{Guid.NewGuid():N}.json");
        try
        {
            PerfHarness.RunAll(ShapeCatalog.All, tmpPath);
            var output = JsonSerializer.Deserialize<HarnessOutput>(
                File.ReadAllText(tmpPath),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

            int wins = output.Shapes
                .Count(r => r.NativeAvailable && r.RatioBmOverNative > 0 && r.RatioBmOverNative < 1.0);
            int total = output.Shapes.Count(r => r.NativeAvailable);
            int winPct = total > 0 ? (wins * 100) / total : 0;

            double worstLoss = output.Shapes
                .Where(r => r.NativeAvailable && r.RatioBmOverNative > 0)
                .Select(r => r.RatioBmOverNative)
                .DefaultIfEmpty(0)
                .Max();

            Assert.True(
                winPct >= PerfBar.MinWinRatePercent,
                $"Win rate {winPct}% < bar {PerfBar.MinWinRatePercent}% ({wins}/{total} wins)");

            Assert.True(
                worstLoss <= PerfBar.MaxLossMultiple,
                $"Worst loss {worstLoss:F2}x > bar {PerfBar.MaxLossMultiple:F2}x");
        }
        finally
        {
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
        }
    }
}
