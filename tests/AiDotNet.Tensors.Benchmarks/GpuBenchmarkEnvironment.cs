using System.Diagnostics;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Benchmarks;

internal static class GpuBenchmarkEnvironment
{
    private const int MixedComputeConflictThresholdPercent = 5;
    private const int DeviceUtilizationCeilingPercent = 20;
    private const int PostSuiteUtilizationAttempts = 6;
    private const int PostSuiteUtilizationDelayMilliseconds = 250;
    private const int DeviceMemoryCeilingMegabytes = 2048;
    private const int DeviceTemperatureCeilingCelsius = 75;
    private const int HostUtilizationCeilingPercent = 20;
    private const int HostUtilizationSampleMilliseconds = 750;

    internal static void RequireIdleGpu(string label)
    {
        RequireNoForeignCompute(label);
        RequireHostQuiescence(label);

        string status = RunNvidiaSmi(
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits");
        string[] cells = status.Split(',', StringSplitOptions.TrimEntries);
        if (cells.Length >= 3 && int.TryParse(cells[0], out int utilization)
            && int.TryParse(cells[1], out int usedMegabytes)
            && int.TryParse(cells[2], out int temperatureCelsius)
            && (utilization > DeviceUtilizationCeilingPercent ||
                usedMegabytes > DeviceMemoryCeilingMegabytes ||
                temperatureCelsius > DeviceTemperatureCeilingCelsius))
        {
            throw new InvalidOperationException(
                $"[{label}] GPU is not benchmark-ready (utilization={utilization}%, " +
                $"memory.used={usedMegabytes} MiB, temperature={temperatureCelsius} C).");
        }
    }

    internal static void RequireNoForeignCompute(
        string label,
        bool afterSuite = false,
        bool ignoreMixedWddmProcesses = false)
    {
        RequireNoForeignPython(label);
        string processMonitor = RunNvidiaSmi("pmon", "-c", "1", "-s", "u");
        int? trustedOrchestrator = int.TryParse(
            Environment.GetEnvironmentVariable("AIDOTNET_BENCHMARK_ORCHESTRATOR_PID"),
            out int orchestratorId) && orchestratorId > 0
                ? orchestratorId
                : null;
        string[] conflicts = FindComputeWorkloadConflicts(
            processMonitor, Environment.ProcessId, trustedOrchestrator,
            afterSuite || ignoreMixedWddmProcesses);
        if (conflicts.Length != 0)
            throw new InvalidOperationException(
                $"[{label}] Foreign GPU workload detected; clean benchmark refused: {string.Join("; ", conflicts)}");

        string temperature = RunNvidiaSmi(
            "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits");
        if (int.TryParse(temperature, out int temperatureCelsius) &&
            temperatureCelsius > DeviceTemperatureCeilingCelsius)
            throw new InvalidOperationException(
                $"[{label}] GPU temperature {temperatureCelsius} C exceeds the " +
                $"{DeviceTemperatureCeilingCelsius} C evidence ceiling.");

        if (afterSuite)
        {
            RequirePostSuiteDeviceQuiescence(label);
            RequireHostQuiescence(label);
        }
    }

    private static void RequireNoForeignPython(string label)
    {
        var conflicts = new List<string>();
        foreach (Process process in Process.GetProcesses())
        {
            using (process)
            {
                try
                {
                    if (process.Id == Environment.ProcessId)
                        continue;
                    string name = process.ProcessName;
                    if (string.Equals(name, "python", StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(name, "python3", StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(name, "pythonw", StringComparison.OrdinalIgnoreCase))
                        conflicts.Add($"pid={process.Id} {name}");
                }
                catch (InvalidOperationException)
                {
                    // The process exited between enumeration and inspection.
                }
                catch (System.ComponentModel.Win32Exception)
                {
                    // An inaccessible system process cannot be a normal Python
                    // benchmark process; the NVIDIA process gate remains active.
                }
            }
        }
        if (conflicts.Count != 0)
            throw new InvalidOperationException(
                $"[{label}] OS-level Python workload detected before CUDA registration; " +
                $"clean benchmark refused: {string.Join("; ", conflicts)}");
    }

    private static void RequireHostQuiescence(string label)
    {
        Dictionary<int, TimeSpan> before = ReadForeignProcessCpuTimes();
        var interval = Stopwatch.StartNew();
        System.Threading.Thread.Sleep(HostUtilizationSampleMilliseconds);
        Dictionary<int, TimeSpan> after = ReadForeignProcessCpuTimes();
        interval.Stop();

        double busyMilliseconds = 0;
        foreach (KeyValuePair<int, TimeSpan> sample in after)
        {
            if (before.TryGetValue(sample.Key, out TimeSpan start) && sample.Value > start)
                busyMilliseconds += (sample.Value - start).TotalMilliseconds;
        }

        double capacityMilliseconds = interval.Elapsed.TotalMilliseconds *
            Math.Max(1, Environment.ProcessorCount);
        int utilizationPercent = (int)Math.Round(
            busyMilliseconds / capacityMilliseconds * 100.0,
            MidpointRounding.AwayFromZero);
        if (utilizationPercent > HostUtilizationCeilingPercent)
            throw new InvalidOperationException(
                $"[{label}] Host is not benchmark-ready (foreign CPU utilization=" +
                $"{utilizationPercent}%, ceiling={HostUtilizationCeilingPercent}%).");
    }

    private static Dictionary<int, TimeSpan> ReadForeignProcessCpuTimes()
    {
        var times = new Dictionary<int, TimeSpan>();
        int currentProcessId = Environment.ProcessId;
        foreach (Process process in Process.GetProcesses())
        {
            using (process)
            {
                try
                {
                    if (process.Id != 0 && process.Id != currentProcessId)
                        times[process.Id] = process.TotalProcessorTime;
                }
                catch (Exception)
                {
                    // A process can exit or become inaccessible between enumeration
                    // and sampling. Other surviving processes still form the gate.
                }
            }
        }
        return times;
    }

    private static void RequirePostSuiteDeviceQuiescence(string label)
    {
        int utilizationPercent = 0;
        for (int attempt = 0; attempt < PostSuiteUtilizationAttempts; attempt++)
        {
            string utilization = RunNvidiaSmi(
                "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits");
            if (!int.TryParse(utilization, out utilizationPercent) ||
                utilizationPercent <= DeviceUtilizationCeilingPercent)
                return;

            // NVIDIA reports utilization over a trailing sample window. Immediately
            // after our own synchronized launches, the first snapshot can therefore
            // describe work that has already finished. Give that sample a bounded
            // opportunity to age out; sustained foreign work remains above the ceiling
            // and still fails closed after the final sample.
            if (attempt + 1 < PostSuiteUtilizationAttempts)
                System.Threading.Thread.Sleep(PostSuiteUtilizationDelayMilliseconds);
        }

        throw new InvalidOperationException(
            $"[{label}] GPU utilization remains {utilizationPercent}% after " +
            $"{PostSuiteUtilizationAttempts} post-suite quiescence samples, above the " +
            $"{DeviceUtilizationCeilingPercent}% evidence ceiling.");
    }

    internal static string[] FindComputeWorkloadConflicts(
        string processMonitor, int currentProcessId, int? trustedOrchestratorId = null,
        bool afterSuite = false)
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
            // --kernel-competitor is a parent .NET process orchestrating two child lanes.
            // Merely loading CUDA-capable benchmark dependencies can make that parent appear
            // in pmon as type C at 0% SM. Trust only the explicitly supplied parent PID and
            // only while it reports no SM sample ('-') or remains below the same
            // material-compute threshold used for C+G; if it starts doing work, it becomes
            // a conflict like anything else.
            if (trustedOrchestratorId is > 0 && processId == trustedOrchestratorId &&
                (smUtilization == "-" ||
                 (int.TryParse(smUtilization, out int orchestratorSm) &&
                  orchestratorSm <= MixedComputeConflictThresholdPercent)))
            {
                continue;
            }
            bool isComputeOnly = string.Equals(processType, "C", StringComparison.OrdinalIgnoreCase);
            // Under WDDM, ordinary desktop applications can be reported as C+G
            // with a 0-1% sample. Treat a mixed process as competing compute only
            // when its measured SM use is material; the separate whole-device
            // guard still rejects >20% utilization at every suite boundary.
            // A single WDDM pmon sample can retain a C+G percentage after the timed
            // work has ended, including values inconsistent with a quiet whole-device
            // snapshot. Enforce mixed-process admission before every suite; afterward,
            // the row-level spread/clock gates have already judged the timed region and
            // every compute-only process remains an unconditional conflict.
            bool isActiveMixedCompute = !afterSuite && processType.Contains('C') &&
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

    /// <summary>
    /// Current SM clock in MHz, or 0 if it cannot be read.
    /// </summary>
    /// <remarks>
    /// RequireIdleGpu and RequireNoForeignCompute only check the START and END of a
    /// run, which is not enough. A depthwise row was published three times with a
    /// 7.2-7.5% run spread and a P95/median near 2.0; re-measuring the SAME lowering on
    /// the SAME code path later gave 0.1-0.9% and 1.25. Nothing about the kernel had
    /// changed, so the earlier numbers were contaminated by concurrent activity that
    /// the boundary checks did not catch. Sampling the clock around each measurement
    /// lets a contaminated row be flagged instead of quietly reported as evidence.
    /// </remarks>
    internal static int SampleSmClockMhz()
    {
        try
        {
            string output = RunNvidiaSmi("--query-gpu=clocks.sm", "--format=csv,noheader,nounits");
            return int.TryParse(output.Trim().Split('\n')[0].Trim(), out int mhz) ? mhz : 0;
        }
        catch (Exception)
        {
            return 0;
        }
    }

    /// <summary>
    /// Reports SM-clock movement across a measurement. A drift beyond a couple of
    /// percent means the two halves of the measurement did not run on the same machine
    /// state, so any ratio taken across them is suspect.
    /// </summary>
    internal static string DescribeClockDrift(int startMhz, int endMhz)
    {
        if (startMhz <= 0 || endMhz <= 0) return "clock unknown";
        double drift = (endMhz - startMhz) / (double)startMhz * 100.0;
        string text = startMhz.ToString(System.Globalization.CultureInfo.InvariantCulture) + "->" +
                      endMhz.ToString(System.Globalization.CultureInfo.InvariantCulture) + " MHz (" +
                      drift.ToString("+0.0;-0.0", System.Globalization.CultureInfo.InvariantCulture) + "%)";
        return Math.Abs(drift) > 2.0 ? text + " SUSPECT" : text;
    }
}
