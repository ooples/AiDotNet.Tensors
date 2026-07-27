// Copyright (c) AiDotNet. All rights reserved.
// Blueprint #4: record WHICH hardware unit each kernel saturates.
//
// Release used to gate on static machine-code metrics -- SASS instructions, LDG,
// registers, spills. Those called vectorised loads a 24.7% improvement while wall clock
// moved 3.7%, because they measure what is easy to measure rather than what decides
// whether a kernel is competitive.
//
// "Is it fast?" is unanswerable without a competitor. "What is stopping it?" is
// answerable from hardware counters, and it is the question that says what to do next.
// A kernel is finished when it is at a NAMED roofline, or when it records which one it
// is short of and by how much.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelLimiterTool
{
    /// <summary>A kernel at or above this fraction of a roofline is done with that lever.</summary>
    private const double SaturatedAt = 70.0;

    private static readonly string[] Metrics =
    {
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__inst_executed_op_global_ld.sum",
        "gpu__time_duration.sum",

        // WHY A WARP IS NOT ISSUING. The throughput percentages above say which unit is
        // busiest; they do not say what the kernel is waiting on, and reading them as if
        // they did killed two levers on this branch.
        //
        // Per-dimension staging was aimed at "L1 59%" read as "too many global loads".
        // It raised L1 to 77.45% instead, because shared memory IS L1TEX and ld.shared is
        // counted by the same metric. Register reuse was aimed at the same number and
        // moved it not at all: 64.27 / 64.11 / 64.20 at coarsening 2 / 4 / 8.
        //
        // The stall breakdown said what neither could: mio_throttle 3.03%, so the load
        // pipe was never the bottleneck and no amount of load reduction could have helped.
        // These counters exist so that is visible BEFORE the work, not after it.
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    };

    /// <summary>
    /// The lever a stall profile actually points at.
    /// </summary>
    /// <remarks>
    /// Ordered by how specific the signal is. A high mio_throttle means the load/store
    /// pipe's queue is full, so fewer memory INSTRUCTIONS is the fix -- vectorising or
    /// staging. A high long_scoreboard means warps wait on global latency that occupancy
    /// is not hiding, which is what cp.async and prefetching address. A high
    /// short_scoreboard is the same for shared memory. A high `wait` is arithmetic
    /// dependency: more independent accumulators. And a profile where nothing dominates is
    /// a balanced kernel, where a code generator has no lever at all and only a different
    /// algorithm helps -- which is exactly what dense 3x3 turned out to be.
    /// </remarks>
    private static string LeverFor(double mio, double longSb, double shortSb, double wait, double noInst)
    {
        if (mio >= 15.0) return "fewer LSU instructions (vectorise/stage)";
        if (longSb >= 20.0) return "hide global latency (cp.async/prefetch)";
        if (shortSb >= 15.0) return "reduce shared-memory dependency";
        if (wait >= 25.0) return "more independent accumulators (ILP)";
        if (noInst >= 10.0) return "instruction fetch: shorten the body";
        return "balanced -- no codegen lever; needs a different algorithm";
    }

    internal static void Run(string[] args)
    {
        string? ncu = ValueOf(args, "--ncu") ?? FindNcu();
        if (ncu is null)
        {
            Console.WriteLine("Nsight Compute not found. Counters need it, and on Windows they");
            Console.WriteLine("also need an elevated session (otherwise ERR_NVGPUCTRPERM).");
            Console.WriteLine("Pass --ncu <path> to point at it.");
            return;
        }

        string selector = KernelToolArgs.Selector(args);
        var entries = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All
            : new[] { CodegenKernelCatalog.Find(selector)! }.Where(e => e != null).ToList();
        KernelToolArgs.RequireNonEmptySelection(selector, entries.Count, "kernel-limiter");

        string outputPath = ValueOf(args, "--out") ??
            Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "limiter.tsv");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        Console.WriteLine();
        Console.WriteLine("LIMITER GATE - which unit is saturated, measured");
        Console.WriteLine("protocol " + CodegenMeasurementProtocol.Tag);
        Console.WriteLine();
        Console.WriteLine("kernel                            L1%  DRAM%    SM%    wait% longSb%  mio%  status    next lever");

        var rows = new List<string>();
        int satisfied = 0, unresolved = 0;

        foreach (var entry in entries)
        {
            // The launched kernel carries the SPEC name, not the catalog name.
            var counters = Profile(ncu, entry.Name, entry.Bench.Name);
            if (counters is null)
            {
                Console.WriteLine(entry.Name.PadRight(32) + "   profiling failed");
                unresolved++;
                continue;
            }

            double l1 = counters.GetValueOrDefault(Metrics[0]);
            double dram = counters.GetValueOrDefault(Metrics[1]);
            double l2 = counters.GetValueOrDefault(Metrics[2]);
            double sm = counters.GetValueOrDefault(Metrics[3]);

            (string unit, double value) = new[]
            {
                ("L1", l1), ("DRAM", dram), ("L2", l2), ("SM", sm)
            }.OrderByDescending(x => x.Item2).First();

            bool saturated = value >= SaturatedAt;
            if (saturated) satisfied++; else unresolved++;

            double longSb = counters.GetValueOrDefault(Metrics[6]);
            double shortSb = counters.GetValueOrDefault(Metrics[7]);
            double mio = counters.GetValueOrDefault(Metrics[8]);
            double wait = counters.GetValueOrDefault(Metrics[9]);
            double noInst = counters.GetValueOrDefault(Metrics[10]);
            string lever = LeverFor(mio, longSb, shortSb, wait, noInst);

            Console.WriteLine(entry.Name.PadRight(32) +
                l1.ToString("F1", CultureInfo.InvariantCulture).PadLeft(6) +
                dram.ToString("F1", CultureInfo.InvariantCulture).PadLeft(7) +
                sm.ToString("F1", CultureInfo.InvariantCulture).PadLeft(7) +
                wait.ToString("F1", CultureInfo.InvariantCulture).PadLeft(8) +
                longSb.ToString("F1", CultureInfo.InvariantCulture).PadLeft(8) +
                mio.ToString("F1", CultureInfo.InvariantCulture).PadLeft(6) +
                "  " + (saturated ? "roofline" : "headroom").PadRight(10) + lever);

            rows.Add(string.Join("\t", entry.Name, unit,
                value.ToString("F2", CultureInfo.InvariantCulture),
                l1.ToString("F2", CultureInfo.InvariantCulture),
                dram.ToString("F2", CultureInfo.InvariantCulture),
                l2.ToString("F2", CultureInfo.InvariantCulture),
                sm.ToString("F2", CultureInfo.InvariantCulture),
                saturated ? "at-roofline" : "headroom",
                wait.ToString("F2", CultureInfo.InvariantCulture),
                longSb.ToString("F2", CultureInfo.InvariantCulture),
                shortSb.ToString("F2", CultureInfo.InvariantCulture),
                mio.ToString("F2", CultureInfo.InvariantCulture),
                noInst.ToString("F2", CultureInfo.InvariantCulture),
                lever,
                CodegenMeasurementProtocol.Tag));
        }

        var text = new StringBuilder();
        text.AppendLine("# protocol " + CodegenMeasurementProtocol.Tag + ": " + CodegenMeasurementProtocol.Description);
        text.AppendLine("kernel\tlimiter\tpct_of_peak\tl1\tdram\tl2\tsm\tstatus" +
                        "\tstall_wait\tstall_long_sb\tstall_short_sb\tstall_mio\tstall_no_inst" +
                        "\tlever\tprotocol");
        foreach (string row in rows) text.AppendLine(row);
        File.WriteAllText(outputPath, text.ToString());

        Console.WriteLine();
        Console.WriteLine(satisfied + " at a named roofline, " + unresolved + " with headroom");
        Console.WriteLine("recorded to " + outputPath);
        Console.WriteLine();
        Console.WriteLine("A kernel with headroom is not a failure -- it is a kernel whose next");
        Console.WriteLine("lever is known. One at a roofline cannot be improved by any change to");
        Console.WriteLine("the code generator, only by changing what the kernel has to move.");
    }

    private static Dictionary<string, double>? Profile(string ncu, string catalogName, string specName)
    {
        string dll = Path.Combine(AppContext.BaseDirectory, "AiDotNet.Tensors.Benchmarks.dll");
        var start = new ProcessStartInfo
        {
            FileName = ncu,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("--target-processes");
        start.ArgumentList.Add("all");
        start.ArgumentList.Add("--csv");
        start.ArgumentList.Add("--metrics");
        start.ArgumentList.Add(string.Join(",", Metrics));
        start.ArgumentList.Add("-k");
        start.ArgumentList.Add("regex:" + specName);
        start.ArgumentList.Add("dotnet");
        start.ArgumentList.Add(dll);
        start.ArgumentList.Add("--kernel-once");
        start.ArgumentList.Add(catalogName);

        using var process = Process.Start(start);
        if (process is null) return null;
        string stdout = process.StandardOutput.ReadToEnd();
        process.StandardError.ReadToEnd();
        if (!process.WaitForExit((int)TimeSpan.FromMinutes(5).TotalMilliseconds))
        {
            process.Kill(entireProcessTree: true);
            return null;
        }

        var values = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (string line in stdout.Split('\n'))
        {
            var cells = SplitCsv(line);
            if (cells.Count < 15) continue;
            string metric = cells[12];
            if (!Metrics.Contains(metric)) continue;
            if (double.TryParse(cells[14].Replace(",", ""), NumberStyles.Any,
                                CultureInfo.InvariantCulture, out double v))
                values[metric] = Math.Max(values.GetValueOrDefault(metric), v);
        }
        return values.Count > 0 ? values : null;
    }

    private static List<string> SplitCsv(string line)
    {
        var cells = new List<string>();
        bool quoted = false;
        var current = new StringBuilder();
        foreach (char c in line)
        {
            if (c == '"') { quoted = !quoted; continue; }
            if (c == ',' && !quoted) { cells.Add(current.ToString()); current.Clear(); continue; }
            current.Append(c);
        }
        cells.Add(current.ToString());
        return cells;
    }

    private static string? FindNcu()
    {
        foreach (string root in new[] { @"C:\Program Files\NVIDIA Corporation" })
        {
            if (!Directory.Exists(root)) continue;
            foreach (string dir in Directory.GetDirectories(root, "Nsight Compute*").OrderByDescending(d => d))
            {
                string candidate = Path.Combine(dir, "target", "windows-desktop-win7-x64", "ncu.exe");
                if (File.Exists(candidate)) return candidate;
            }
        }
        return null;
    }

    private static string? ValueOf(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], flag, StringComparison.Ordinal))
                return args[i + 1];
        return null;
    }
}
