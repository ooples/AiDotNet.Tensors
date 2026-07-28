using System.Diagnostics;

namespace AiDotNet.Tensors.Benchmarks;

internal static class DirectPtxExternalBaselines
{
    internal static void Run()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("external-gpu-baselines-start");
        GpuBenchmarkEnvironment.PrintSnapshot("external-gpu-baselines-start");
        string script = Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py", "run_direct_ptx_gpu_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory, "run_direct_ptx_gpu_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("The direct-PTX Python competitor harness was not copied to output.", script);

        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            Arguments = "\"" + script.Replace("\"", "\\\"") + "\"",
            UseShellExecute = false,
            RedirectStandardOutput = false,
            RedirectStandardError = false
        };
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the Python GPU baseline process.");
        if (!process.WaitForExit((int)TimeSpan.FromMinutes(60).TotalMilliseconds))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("Python GPU baselines did not complete within 60 minutes.");
        }
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"Python GPU baselines failed with exit code {process.ExitCode}.");
        GpuBenchmarkEnvironment.RequireNoForeignCompute("external-gpu-baselines-end");
        GpuBenchmarkEnvironment.PrintSnapshot("external-gpu-baselines-end");
    }
}
