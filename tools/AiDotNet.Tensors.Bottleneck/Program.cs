// Copyright (c) AiDotNet. All rights reserved.

using System.Diagnostics;
using System.Reflection;
using AiDotNet.Tensors.Engines.Profiling;
using AiDotNet.Tensors.Engines.Profiling.Memory;
using AiDotNet.Tensors.Engines.Profiling.Trace;

namespace AiDotNet.Tensors.Bottleneck;

/// <summary>
/// One-shot diagnostic that wraps a workload assembly's entry point in a
/// profiler + memory snapshot + dispatch-tier counter pass and emits a
/// readable summary. Modelled on <c>torch.utils.bottleneck</c>.
///
/// <para>Usage:
/// <code>
/// dotnet aidotnet-tensors-bottleneck --assembly &lt;path-to-dll&gt; [--method NS.Type.Method]
///                                    [--out &lt;dir&gt;] [--repeat 3]
/// </code></para>
///
/// <para>The tool loads the target assembly, locates either the entry point
/// or the named static method, runs it under a fully-instrumented profiler,
/// and writes:</para>
/// <list type="bullet">
///   <item><c>trace.json</c> — chrome-trace JSON for visual inspection in
///     <c>chrome://tracing</c> / Perfetto.</item>
///   <item><c>memory.txt</c> — peak / current / total bytes plus the top
///     live allocations.</item>
///   <item><c>summary.txt</c> — top per-op timings (descending), wall clock,
///     event count, NVTX/ITT availability, dispatch hints.</item>
/// </list>
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        try { return Run(args); }
        catch (Exception ex) { Console.Error.WriteLine($"bottleneck: {ex.Message}"); return 2; }
    }

    private static int Run(string[] args)
    {
        var opts = ParseArgs(args);
        if (opts is null) return 1;

        Directory.CreateDirectory(opts.OutputDir);

        // Load the workload assembly + locate the entry method.
        var asm = Assembly.LoadFrom(opts.AssemblyPath);
        var entry = ResolveEntry(asm, opts.MethodName)
            ?? throw new InvalidOperationException(
                "Could not resolve an entry method. Provide --method NS.Type.Method " +
                "or ensure the assembly has a public static void Main()/Run() method.");

        // Engage the profiler. We capture every step (no schedule) so the
        // chrome-trace shows the full workload timeline rather than a
        // wait/warmup/active band — bottleneck is a one-shot tool, not a
        // long-running training run.
        var memOriginal = MemoryProfiler.Mode;
        MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.State);
        MemoryProfiler.Reset();

        using var prof = Profiler.Profile(new ProfilerOptions
        {
            Activities = ProfilerActivities.All,
        });

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < opts.Repeat; i++)
        {
            using (Profiler.Range($"iteration_{i}"))
            {
                InvokeEntry(entry);
            }
        }
        sw.Stop();

        // Reports: read the live event log + memory state, write the artifacts.
        string tracePath  = Path.Combine(opts.OutputDir, "trace.json");
        string memPath    = Path.Combine(opts.OutputDir, "memory.txt");
        string summaryPath = Path.Combine(opts.OutputDir, "summary.txt");

        prof.ExportChromeTrace(tracePath);
        MemoryProfiler.DumpSnapshot(memPath);
        WriteSummary(summaryPath, prof, sw.Elapsed, opts);

        MemoryProfiler.RecordHistory(memOriginal);

        Console.WriteLine($"bottleneck: wrote {tracePath}");
        Console.WriteLine($"bottleneck: wrote {memPath}");
        Console.WriteLine($"bottleneck: wrote {summaryPath}");
        return 0;
    }

    private static MethodInfo? ResolveEntry(Assembly asm, string? methodName)
    {
        if (methodName is not null)
        {
            int dot = methodName.LastIndexOf('.');
            if (dot < 0) return null;
            string typeName = methodName.Substring(0, dot);
            string method = methodName.Substring(dot + 1);
            var t = asm.GetType(typeName, throwOnError: false);
            return t?.GetMethod(method,
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance);
        }

        // Fall back to the assembly's entry point, or a public static Run.
        if (asm.EntryPoint is not null) return asm.EntryPoint;
        foreach (var t in asm.GetTypes())
        {
            var m = t.GetMethod("Run", BindingFlags.Public | BindingFlags.Static);
            if (m is not null) return m;
        }
        return null;
    }

    private static void InvokeEntry(MethodInfo entry)
    {
        // Match either parameterless or string[] entrypoints — the two
        // shapes a real workload-style harness uses. If the entrypoint
        // returns a Task, block on completion so async work is included
        // in the profile (otherwise the harness exits with the workload
        // still running and the trace would miss most of the events).
        var ps = entry.GetParameters();
        object? instance = entry.IsStatic ? null : Activator.CreateInstance(entry.DeclaringType!);
        object? result;
        if (ps.Length == 0)
        {
            result = entry.Invoke(instance, null);
        }
        else if (ps.Length == 1 && ps[0].ParameterType == typeof(string[]))
        {
            result = entry.Invoke(instance, new object?[] { Array.Empty<string>() });
        }
        else
        {
            throw new InvalidOperationException(
                $"Entry method {entry.DeclaringType?.FullName}.{entry.Name} has unsupported signature; " +
                "must be either parameterless or take a single string[].");
        }
        if (result is System.Threading.Tasks.Task task)
        {
            // Block until the async entry completes; GetResult unwraps any exception.
            task.GetAwaiter().GetResult();
        }
    }

    private static void WriteSummary(string path, ProfilerSession session, TimeSpan wall, Options opts)
    {
        // Aggregate Complete events by name, descending by total time.
        var byName = new Dictionary<string, (long total, long count)>(StringComparer.Ordinal);
        long totalEventCount = 0;
        foreach (var ev in session.Events)
        {
            if (ev.Phase != 'X') continue;
            totalEventCount++;
            if (!byName.TryGetValue(ev.Name, out var entry)) entry = (0, 0);
            byName[ev.Name] = (entry.total + ev.DurationMicros, entry.count + 1);
        }

        var ordered = byName.OrderByDescending(kv => kv.Value.total).Take(40).ToList();

        using var sw = new StreamWriter(path);
        sw.WriteLine("# AiDotNet.Tensors Bottleneck Summary");
        sw.WriteLine($"Assembly:    {opts.AssemblyPath}");
        sw.WriteLine($"Method:      {opts.MethodName ?? "(entry point)"}");
        sw.WriteLine($"Repeat:      {opts.Repeat}");
        sw.WriteLine($"WallClock:   {wall.TotalMilliseconds:N1} ms");
        sw.WriteLine($"Events:      {totalEventCount:N0}");
        sw.WriteLine($"NvtxAvail:   {AiDotNet.Tensors.Engines.Profiling.Native.NvtxBridge.IsAvailable}");
        sw.WriteLine($"IttAvail:    {AiDotNet.Tensors.Engines.Profiling.Native.IttBridge.IsAvailable}");
        sw.WriteLine();
        sw.WriteLine("## Top ops by total time");
        sw.WriteLine($"{"Op",-40} {"calls",10} {"total_us",14} {"avg_us",12}");
        foreach (var kv in ordered)
        {
            double avg = kv.Value.count == 0 ? 0 : (double)kv.Value.total / kv.Value.count;
            sw.WriteLine($"{kv.Key,-40} {kv.Value.count,10:N0} {kv.Value.total,14:N0} {avg,12:F1}");
        }
        sw.WriteLine();
        sw.WriteLine("## Memory");
        sw.WriteLine($"PeakBytes:    {MemoryProfiler.PeakBytes:N0}");
        sw.WriteLine($"TotalAlloc:   {MemoryProfiler.TotalAllocatedBytes:N0}");
        sw.WriteLine($"CurrentBytes: {MemoryProfiler.CurrentBytes:N0}");
    }

    private sealed class Options
    {
        public required string AssemblyPath { get; init; }
        public string? MethodName { get; init; }
        public string OutputDir { get; init; } = "./bottleneck-out";
        public int Repeat { get; init; } = 1;
    }

    private static Options? ParseArgs(string[] args)
    {
        string? assembly = null;
        string? method = null;
        string outDir = "./bottleneck-out";
        int repeat = 1;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--assembly" when i + 1 < args.Length: assembly = args[++i]; break;
                case "--method"   when i + 1 < args.Length: method   = args[++i]; break;
                case "--out"      when i + 1 < args.Length: outDir   = args[++i]; break;
                case "--repeat"   when i + 1 < args.Length:
                    if (!int.TryParse(args[++i], out repeat) || repeat < 1)
                    {
                        Console.Error.WriteLine("bottleneck: --repeat must be a positive integer.");
                        return null;
                    }
                    break;
                case "-h": case "--help":
                    PrintHelp(); return null;
                default:
                    Console.Error.WriteLine($"bottleneck: unknown arg {args[i]}");
                    PrintHelp();
                    return null;
            }
        }

        if (string.IsNullOrEmpty(assembly))
        {
            Console.Error.WriteLine("bottleneck: --assembly <path> is required.");
            PrintHelp();
            return null;
        }
        if (!File.Exists(assembly))
        {
            Console.Error.WriteLine($"bottleneck: assembly not found: {assembly}");
            return null;
        }

        return new Options
        {
            AssemblyPath = Path.GetFullPath(assembly),
            MethodName = method,
            OutputDir = outDir,
            Repeat = repeat,
        };
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
            aidotnet-tensors-bottleneck — one-shot performance diagnostic.

            Usage:
              aidotnet-tensors-bottleneck --assembly <path> [--method NS.Type.Method]
                                          [--out <dir>] [--repeat N]

            Options:
              --assembly  Path to the workload assembly (.dll). Required.
              --method    Fully-qualified method to invoke. Defaults to the
                          assembly's entry point or any public static Run().
              --out       Output directory. Default: ./bottleneck-out
              --repeat    Number of iterations. Default: 1.

            Outputs:
              <out>/trace.json   Chrome Trace Format (open in chrome://tracing).
              <out>/memory.txt   Peak/current/total + top live allocations.
              <out>/summary.txt  Per-op timings and dispatch hints.
            """);
    }
}
