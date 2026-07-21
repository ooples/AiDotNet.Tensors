using System.Diagnostics;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Benchmarks;

internal static class GpuBenchmarkEnvironment
{
    internal static void PrintSnapshot(string label)
    {
        Console.WriteLine($"[{label}] OS={RuntimeInformation.OSDescription}; .NET={Environment.Version}; " +
            $"process={Environment.ProcessId}; UTC={DateTime.UtcNow:O}");
        try
        {
            var start = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            start.ArgumentList.Add("--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit");
            start.ArgumentList.Add("--format=csv,noheader,nounits");
            using Process process = Process.Start(start)!;
            string output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit(5000);
            if (process.ExitCode == 0 && output.Length != 0)
                Console.WriteLine($"[{label}] GPU name, uuid, driver, pstate, SM MHz, memory MHz, C, W, limit W: {output}");
        }
        catch
        {
            Console.WriteLine($"[{label}] nvidia-smi metadata unavailable");
        }
    }
}
