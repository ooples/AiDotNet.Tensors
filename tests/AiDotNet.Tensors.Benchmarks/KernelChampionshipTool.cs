// Copyright (c) AiDotNet. All rights reserved.
// One command for the complete measured schedule-selection loop.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using static AiDotNet.Tensors.Benchmarks.KernelToolArgs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Tunes every candidate, proves numerical equivalence, measures the strongest stable
/// competitor, and diagnoses every result that still fails the win threshold.
/// </summary>
/// <remarks>
/// Keeping these stages behind one command closes two easy evidence holes: benchmarking
/// the modelled fallback because an ignored cache was absent, and calling a partial table
/// a clean catalog result. The individual tools remain available for focused development.
/// </remarks>
internal static class KernelChampionshipTool
{
    private const double WinThreshold = 1.10;

    private sealed record CompetitorRow(string Kernel, double Ratio);

    internal static void Run(string[] args)
    {
        string selector = Selector(args);
        int expected = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All.Count
            : CodegenKernelCatalog.Find(selector) is null ? 0 : 1;
        RequireNonEmptySelection(selector, expected, "kernel-championship");

        string evidenceDirectory = Path.GetFullPath(
            ValueOf(args, "--evidence-dir") ?? Path.Combine("artifacts"));
        Directory.CreateDirectory(evidenceDirectory);
        string autotunePath = Path.Combine(evidenceDirectory, "autotune.tsv");
        string competitorPath = Path.Combine(evidenceDirectory, "competitor-ratios.tsv");
        string limiterPath = Path.Combine(evidenceDirectory, "limiter.tsv");
        string diagnosisPath = Path.Combine(evidenceDirectory, "kernel-diagnosis.tsv");

        // The competitor lane launches a fresh copy of this assembly. Keep the standard
        // repository-relative cache location unless the caller explicitly supplies an
        // evidence directory, in which case publish its path for every child process.
        string? priorCacheEnvironment =
            Environment.GetEnvironmentVariable("AIDOTNET_CODEGEN_AUTOTUNE_CACHE");
        string priorCachePath = CodegenAutotuneCache.CachePath;
        Environment.SetEnvironmentVariable("AIDOTNET_CODEGEN_AUTOTUNE_CACHE", autotunePath);
        CodegenAutotuneCache.CachePath = autotunePath;
        try
        {
            Console.WriteLine();
            Console.WriteLine("KERNEL CHAMPIONSHIP - tune, prove, compete, diagnose");
            Console.WriteLine("selector " + selector + ", win threshold " +
                              WinThreshold.ToString("F2", CultureInfo.InvariantCulture) + "x");

            KernelAutotuneTool.Run(new[] { selector, "--out", autotunePath });
            CodegenAutotuneCache.Invalidate();
            KernelConveyorTool.Run("verify", new[] { selector });

            var competitorArgs = new List<string>
            {
                selector, "--out", competitorPath,
            };
            ForwardValue(args, competitorArgs, "--runner-python");
            ForwardValue(args, competitorArgs, "--competitor-python");
            ForwardValue(args, competitorArgs, "--max-spread-pct");
            KernelCompetitorTool.Run(competitorArgs.ToArray());

            IReadOnlyList<CompetitorRow> rows = ReadCompetitorRows(competitorPath, expected);
            CompetitorRow[] nonWins = rows
                .Where(row => row.Ratio < WinThreshold)
                .OrderBy(row => row.Ratio)
                .ToArray();
            if (nonWins.Length == 0)
            {
                Console.WriteLine();
                Console.WriteLine("CHAMPIONSHIP GATE: " + rows.Count.ToString(
                    CultureInfo.InvariantCulture) + "/" + expected.ToString(
                    CultureInfo.InvariantCulture) + " stable competitor wins; PASS");
                return;
            }

            Console.WriteLine();
            Console.WriteLine(nonWins.Length.ToString(CultureInfo.InvariantCulture) +
                              " non-win(s) remain; collecting counters before failing:");
            foreach (CompetitorRow row in nonWins)
                Console.WriteLine("  " + row.Kernel + " " +
                                  row.Ratio.ToString("F3", CultureInfo.InvariantCulture) + "x");

            var limiterArgs = new List<string> { selector, "--out", limiterPath };
            ForwardValue(args, limiterArgs, "--ncu");
            KernelLimiterTool.Run(limiterArgs.ToArray());
            KernelCatalogOracleTool.Run(new[]
            {
                selector,
                "--competitor", competitorPath,
                "--limiter", limiterPath,
                "--out", diagnosisPath,
            });

            throw new InvalidOperationException(
                "Championship gate failed: " + nonWins.Length.ToString(
                    CultureInfo.InvariantCulture) + " of " + expected.ToString(
                    CultureInfo.InvariantCulture) + " selected kernels remain below " +
                WinThreshold.ToString("F2", CultureInfo.InvariantCulture) +
                "x. See " + diagnosisPath + ".");
        }
        finally
        {
            CodegenAutotuneCache.CachePath = priorCachePath;
            CodegenAutotuneCache.Invalidate();
            Environment.SetEnvironmentVariable(
                "AIDOTNET_CODEGEN_AUTOTUNE_CACHE", priorCacheEnvironment);
        }
    }

    private static void ForwardValue(
        string[] source, ICollection<string> destination, string flag)
    {
        string? value = ValueOf(source, flag);
        if (value is null) return;
        destination.Add(flag);
        destination.Add(value);
    }

    private static IReadOnlyList<CompetitorRow> ReadCompetitorRows(
        string path, int expected)
    {
        string[] lines = File.ReadAllLines(path);
        int kernelColumn = -1;
        int ratioColumn = -1;
        var rows = new List<CompetitorRow>(expected);
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (string line in lines)
        {
            if (line.Length == 0 || line[0] == '#') continue;
            string[] cells = line.Split('\t');
            if (kernelColumn < 0)
            {
                kernelColumn = Array.IndexOf(cells, "kernel");
                ratioColumn = Array.IndexOf(cells, "ratio");
                if (kernelColumn < 0 || ratioColumn < 0)
                    throw new InvalidOperationException(
                        "Competitor evidence is missing kernel or ratio columns: " + path);
                continue;
            }

            int required = Math.Max(kernelColumn, ratioColumn);
            if (cells.Length <= required ||
                !double.TryParse(cells[ratioColumn], NumberStyles.Float,
                    CultureInfo.InvariantCulture, out double ratio) ||
                !double.IsFinite(ratio) || ratio <= 0)
            {
                throw new InvalidOperationException(
                    "Competitor evidence contains an invalid ratio row: " + line);
            }
            if (!names.Add(cells[kernelColumn]))
                throw new InvalidOperationException(
                    "Competitor evidence contains duplicate kernel '" +
                    cells[kernelColumn] + "'.");
            rows.Add(new CompetitorRow(cells[kernelColumn], ratio));
        }

        if (rows.Count != expected)
            throw new InvalidOperationException(
                "Competitor evidence contains " + rows.Count.ToString(
                    CultureInfo.InvariantCulture) + " stable rows; expected " +
                expected.ToString(CultureInfo.InvariantCulture) + ".");
        return rows;
    }
}
