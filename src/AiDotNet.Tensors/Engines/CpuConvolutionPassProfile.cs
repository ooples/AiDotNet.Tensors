using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

// Instrument-only branch. Process CPU time includes every runtime/worker thread;
// it distinguishes a serial pass from a busy process, not FMA from busy waiting.
internal static class CpuConvolutionPassProfile
{
    private static readonly string? DirectoryPath = Environment.GetEnvironmentVariable("AIDOTNET_CPU_CONV_PASS_PROFILE_DIR");
    internal static readonly bool Enabled = !string.IsNullOrEmpty(DirectoryPath) && RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
    private static readonly Dictionary<string, Row> Rows = new();
    private static readonly object Gate = new();
    internal sealed class Row
    {
        internal long Calls, WallTicks, Cpu100ns;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct FileTime { internal uint Low, High; internal long Value => (long)((ulong)High << 32 | Low); }
    [DllImport("kernel32.dll")] private static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll")] private static extern bool GetProcessTimes(IntPtr process, out FileTime created, out FileTime exited, out FileTime kernel, out FileTime user);
    static CpuConvolutionPassProfile()
    {
        if (Enabled) AppDomain.CurrentDomain.ProcessExit += (_, _) => Write();
    }
    private static long CpuTime() => GetProcessTimes(GetCurrentProcess(), out _, out _, out var kernel, out var user) ? kernel.Value + user.Value : 0;
    internal static Scope Start(string pass, int[] shape, int[] kernel)
    {
        if (!Enabled) return default;
        string key = pass + "\t" + string.Join(",", shape) + "\t" + string.Join(",", kernel)
            + "\t" + (CpuParallelSettings.IsInParallelRegion ? "1" : "0");
        Row row;
        lock (Gate)
        {
            if (!Rows.TryGetValue(key, out row!)) { row = new Row(); Rows.Add(key, row); }
        }
        return new Scope(row, CpuTime(), Stopwatch.GetTimestamp());
    }
    internal readonly struct Scope : IDisposable
    {
        private readonly Row? _row;
        private readonly long _cpu, _wall;
        internal Scope(Row row, long cpu, long wall) { _row = row; _cpu = cpu; _wall = wall; }
        public void Dispose()
        {
            if (_row is null) return;
            long wall = Stopwatch.GetTimestamp() - _wall, cpu = CpuTime() - _cpu;
            Interlocked.Increment(ref _row.Calls);
            Interlocked.Add(ref _row.WallTicks, wall);
            Interlocked.Add(ref _row.Cpu100ns, cpu);
        }
    }
    private static void Write()
    {
        var text = new StringBuilder("# diagnostic_only=true formal_performance_win=false process_cpu_includes_workers_gc_jit_and_spinning=true\n");
        text.Append("pass\tinput_shape\tkernel_shape\tnested\tcalls\twall_us\tprocess_cpu_us\tcpu_per_wall\n");
        lock (Gate) foreach (var pair in Rows)
        {
            double wall = pair.Value.WallTicks * 1e6 / Stopwatch.Frequency;
            double cpu = pair.Value.Cpu100ns / 10d;
            text.Append(pair.Key).Append('\t').Append(pair.Value.Calls).Append('\t')
                .Append(wall.ToString("R", CultureInfo.InvariantCulture)).Append('\t')
                .Append(cpu.ToString("R", CultureInfo.InvariantCulture)).Append('\t')
                .Append((wall == 0 ? 0 : cpu / wall).ToString("R", CultureInfo.InvariantCulture)).Append('\n');
        }
        Directory.CreateDirectory(DirectoryPath!);
        using var process = Process.GetCurrentProcess();
        File.WriteAllText(Path.Combine(DirectoryPath!, "pid" + process.Id + ".tsv"), text.ToString());
    }
}
