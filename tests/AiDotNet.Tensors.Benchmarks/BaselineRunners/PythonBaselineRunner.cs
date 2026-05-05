using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Benchmarks.BaselineRunners;

/// <summary>
/// Runs a Python subprocess against one of the Parity-210 baseline scripts
/// and parses the timed-median-ms output. Used by the BenchmarkDotNet runs
/// in <see cref="Parity210"/> to compare against <c>torch.eager</c>,
/// <c>torch.compile</c>, <c>opt_einsum</c>, and <c>jnp.einsum</c> under jit.
/// </summary>
/// <remarks>
/// <para>
/// The runner writes operand shapes + equation to stdin of the Python
/// process and reads a single CSV-line response:
///   <c>&lt;median_ms&gt;,&lt;p90_ms&gt;,&lt;iters&gt;</c>
/// </para>
/// <para>
/// When Python is not installed or the requested baseline library is
/// missing, <see cref="TryRun"/> returns a null result — the benchmark
/// records "skipped" rather than failing. This is the "eager-only fallback"
/// mode documented in the #210 plan.
/// </para>
/// </remarks>
public sealed class PythonBaselineRunner : IDisposable
{
    private readonly string _pythonExe;
    private readonly string _scriptDir;
    private Process? _proc;

    /// <summary>Result of a single baseline run.</summary>
    public sealed class Result
    {
        /// <summary>Median wall time in milliseconds.</summary>
        public double MedianMs { get; }
        /// <summary>P90 wall time in milliseconds.</summary>
        public double P90Ms { get; }
        /// <summary>Number of iterations contributing to the median.</summary>
        public int Iterations { get; }
        /// <summary>Constructs a result.</summary>
        public Result(double medianMs, double p90Ms, int iterations)
        {
            MedianMs = medianMs;
            P90Ms = p90Ms;
            Iterations = iterations;
        }
    }

    /// <summary>Baseline library to invoke inside Python.</summary>
    public enum Baseline
    {
        /// <summary>torch in eager mode.</summary>
        TorchEager,
        /// <summary>torch.compile fullgraph mode.</summary>
        TorchCompile,
        /// <summary>opt_einsum path optimizer.</summary>
        OptEinsum,
        /// <summary>jax.numpy under jit.</summary>
        JaxJit,
        /// <summary>Issue #294 PyTorch CPU baseline — matmul, conv2d,
        /// FlashAttention (scaled_dot_product_attention), LayerNorm,
        /// BCE forward+backward.</summary>
        Torch294,
    }

    /// <summary>Creates a runner; auto-detects Python location via PATH.</summary>
    public PythonBaselineRunner(string? pythonExe = null, string? scriptDir = null)
    {
        _pythonExe = pythonExe ?? "python";
        _scriptDir = scriptDir ?? Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py");
    }

    /// <summary>
    /// Quick check: whether the baseline runner can be used at all. Returns
    /// false if Python isn't on PATH or the script directory is missing.
    /// </summary>
    public bool IsAvailable
    {
        get
        {
            try
            {
                using var p = new Process();
                p.StartInfo = new ProcessStartInfo(_pythonExe, "--version")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                if (!p.Start()) return false;
                p.WaitForExit(2000);
                return p.ExitCode == 0 && Directory.Exists(_scriptDir);
            }
            catch
            {
                return false;
            }
        }
    }

    /// <summary>
    /// Run one op against one baseline. Returns null if the subprocess
    /// fails or the library is missing — the caller should skip that data
    /// point in the benchmark report.
    /// </summary>
    /// <param name="baseline">Which Python library to invoke.</param>
    /// <param name="opName">Op identifier that the script recognises
    /// (e.g. "einsum", "scatter_reduce", "cumsum").</param>
    /// <param name="args">Op-specific arguments as a semicolon-separated
    /// string. For einsum this is <c>equation;shape1,shape2,…</c>.</param>
    /// <param name="warmup">Warmup iterations before timing.</param>
    /// <param name="iters">Number of timed iterations.</param>
    public Result? TryRun(Baseline baseline, string opName, string args,
        int warmup = 20, int iters = 100)
    {
        if (!IsAvailable) return null;

        string scriptName = baseline switch
        {
            Baseline.TorchEager => "run_torch_eager.py",
            Baseline.TorchCompile => "run_torch_compile.py",
            Baseline.OptEinsum => "run_opt_einsum.py",
            Baseline.JaxJit => "run_jax_jit.py",
            Baseline.Torch294 => "run_torch_294.py",
            _ => throw new ArgumentOutOfRangeException(nameof(baseline))
        };
        string scriptPath = Path.Combine(_scriptDir, scriptName);
        if (!File.Exists(scriptPath)) return null;

        using var p = new Process();
        p.StartInfo = new ProcessStartInfo(_pythonExe, scriptPath)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        try
        {
            if (!p.Start()) return null;
            p.StandardInput.WriteLine($"{opName}|{args}|{warmup}|{iters}");
            p.StandardInput.Close();
            string stdout = p.StandardOutput.ReadToEnd();
            if (!p.WaitForExit(60000)) { p.Kill(); return null; }
            if (p.ExitCode != 0) return null;

            // Expected CSV: median_ms,p90_ms,iters
            var parts = stdout.Trim().Split(',');
            if (parts.Length < 3) return null;
            if (!double.TryParse(parts[0], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out double median)) return null;
            if (!double.TryParse(parts[1], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out double p90)) return null;
            if (!int.TryParse(parts[2], out int it)) return null;
            return new Result(median, p90, it);
        }
        catch
        {
            return null;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _proc?.Dispose();
    }
}
