using System.Diagnostics;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Benchmarks;

internal static class GpuBenchmarkEnvironment
{
    private const int MixedComputeConflictThresholdPercent = 5;

    internal static void RequireIdleGpu(string label)
    {
        RequireNoForeignCompute(label);

        string status = RunNvidiaSmi(
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits");
        string[] cells = status.Split(',', StringSplitOptions.TrimEntries);
        if (cells.Length >= 3 && int.TryParse(cells[0], out int utilization)
            && int.TryParse(cells[1], out int usedMegabytes)
            && int.TryParse(cells[2], out int temperatureCelsius)
            && (utilization > 20 || usedMegabytes > 2048 || temperatureCelsius > 75))
        {
            throw new InvalidOperationException(
                $"[{label}] GPU is not benchmark-ready (utilization={utilization}%, " +
                $"memory.used={usedMegabytes} MiB, temperature={temperatureCelsius} C).");
        }
    }

    internal static void RequireNoForeignCompute(
        string label,
        bool ignoreMixedWddmProcesses = false)
    {
        string processMonitor = RunNvidiaSmi("pmon", "-c", "1", "-s", "u");
        string[] conflicts = FindComputeWorkloadConflicts(
            processMonitor, Environment.ProcessId, ignoreMixedWddmProcesses);
        if (conflicts.Length != 0)
            throw new InvalidOperationException(
                $"[{label}] Foreign GPU workload detected; clean benchmark refused: {string.Join("; ", conflicts)}");

        string temperature = RunNvidiaSmi(
            "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits");
        if (int.TryParse(temperature, out int temperatureCelsius) && temperatureCelsius > 75)
            throw new InvalidOperationException(
                $"[{label}] GPU temperature {temperatureCelsius} C exceeds the 75 C evidence ceiling.");
    }

    internal static string[] FindComputeWorkloadConflicts(
        string processMonitor,
        int currentProcessId,
        bool ignoreMixedWddmProcesses = false)
    {
        var conflicts = new List<string>();
        foreach (string line in processMonitor.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string trimmed = line.Trim();
            if (trimmed.StartsWith('#'))
                continue;

            string[] cells = trimmed.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (cells.Length < 9 || !int.TryParse(cells[1], out int processId)
                || processId == currentProcessId)
                continue;

            string processType = cells[2];
            string smUtilization = cells[3];
            bool isComputeOnly = string.Equals(processType, "C", StringComparison.OrdinalIgnoreCase);
            // Under WDDM, ordinary desktop applications can be reported as C+G
            // with a 0-1% sample. Treat a mixed process as competing compute only
            // when its measured SM use is material; the separate whole-device
            // guard still rejects >20% utilization at every suite boundary.
            bool isActiveMixedCompute = !ignoreMixedWddmProcesses && processType.Contains('C') &&
                int.TryParse(smUtilization, out int smPercent) &&
                smPercent > MixedComputeConflictThresholdPercent;
            if (isComputeOnly || isActiveMixedCompute)
                conflicts.Add($"pid={processId} {cells[^1]} type={processType} sm={smUtilization}%");
        }
        return conflicts.ToArray();
    }

    internal static void PrintSnapshot(string label)
    {
        Console.WriteLine($"[{label}] OS={RuntimeInformation.OSDescription}; .NET={Environment.Version}; " +
            $"process={Environment.ProcessId}; UTC={DateTime.UtcNow:O}");
        try
        {
            string output = RunNvidiaSmi(
                "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits");
            if (output.Length != 0)
                Console.WriteLine($"[{label}] GPU name, uuid, driver, pstate, SM MHz, memory MHz, C, W, limit W: {output}");
        }
        catch
        {
            Console.WriteLine($"[{label}] nvidia-smi metadata unavailable");
        }
    }

    private static string RunNvidiaSmi(params string[] arguments)
    {
        var start = new ProcessStartInfo
        {
            FileName = "nvidia-smi",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = false,
            CreateNoWindow = true
        };
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start nvidia-smi.");
        Task<string> output = process.StandardOutput.ReadToEndAsync();
        if (!process.WaitForExit(5000))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("nvidia-smi did not respond within five seconds.");
        }
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"nvidia-smi exited with code {process.ExitCode}.");
        return output.GetAwaiter().GetResult().Trim();
    }
}
