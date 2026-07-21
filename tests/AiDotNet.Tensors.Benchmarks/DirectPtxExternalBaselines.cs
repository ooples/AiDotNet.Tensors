using System.Diagnostics;

namespace AiDotNet.Tensors.Benchmarks;

internal static class DirectPtxExternalBaselines
{
    internal static void Run()
    {
        string script = Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py", "run_direct_ptx_gpu_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory, "run_direct_ptx_gpu_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("The direct-PTX Python competitor harness was not copied to output.", script);

        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            UseShellExecute = false,
            RedirectStandardOutput = false,
            RedirectStandardError = false
        };
        start.ArgumentList.Add(script);
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the Python GPU baseline process.");
        process.WaitForExit();
        if (process.ExitCode != 0)
            Console.WriteLine($"Python GPU baselines unavailable or failed (exit {process.ExitCode}).");
    }
}
