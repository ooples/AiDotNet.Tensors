// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using static AiDotNet.Tensors.Benchmarks.KernelToolArgs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Runs the versioned PyTorch/cuDNN competitor stage and validates its evidence.</summary>
internal static class KernelCompetitorTool
{
    internal static void Run(string[] args)
    {
        string script = Path.GetFullPath(Path.Combine(
            "tools", "bakeoff", "run_bakeoff.py"));
        if (!File.Exists(script))
            throw new FileNotFoundException(
                "Run --kernel-competitor from the repository root; script not found.", script);

        string runnerPython = ValueOf(args, "--runner-python") ?? "python";
        string output = Path.GetFullPath(ValueOf(args, "--out") ??
            Path.Combine("artifacts", "competitor-ratios.tsv"));
        string selector = KernelToolArgs.Selector(args);
        int selected = string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? CodegenKernelCatalog.All.Count
            : CodegenKernelCatalog.Find(selector) is null ? 0 : 1;
        KernelToolArgs.RequireNonEmptySelection(selector, selected, "kernel-competitor");
        string maxSpread = ValueOf(args, "--max-spread-pct") ?? "5.0";
        if (!double.TryParse(maxSpread, NumberStyles.Float, CultureInfo.InvariantCulture,
                out double spread) || spread <= 0)
        {
            throw new ArgumentException("--max-spread-pct must be a positive number.");
        }
        string dispatch = KernelEvidenceIdentity.CurrentDispatch();
        // A failed refresh must not leave an older current-tag artifact available to the
        // release reader, and the child may create a partial replacement before failing.
        if (File.Exists(output)) File.Delete(output);

        var start = new ProcessStartInfo
        {
            FileName = runnerPython,
            WorkingDirectory = Directory.GetCurrentDirectory(),
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add(script);
        start.ArgumentList.Add("--protocol");
        start.ArgumentList.Add(CodegenMeasurementProtocol.Tag);
        start.ArgumentList.Add("--dispatch");
        start.ArgumentList.Add(dispatch);
        start.ArgumentList.Add("--output");
        start.ArgumentList.Add(output);
        start.ArgumentList.Add("--max-spread-pct");
        start.ArgumentList.Add(spread.ToString("R", CultureInfo.InvariantCulture));
        start.ArgumentList.Add("--selector");
        start.ArgumentList.Add(selector);

        // The outer Python script launches both lanes. Pin our lane to this exact assembly,
        // rather than relying on a default path that may name a stale build.
        start.Environment["BAKEOFF_DOTNET"] = Assembly.GetExecutingAssembly().Location;
        start.Environment["AIDOTNET_BENCHMARK_ORCHESTRATOR_PID"] =
            Environment.ProcessId.ToString(CultureInfo.InvariantCulture);
        start.Environment["BAKEOFF_CONV_TRANSPOSE_CONTRACT"] = ConvTransposeContract();
        string? competitorPython = ValueOf(args, "--competitor-python");
        if (!string.IsNullOrEmpty(competitorPython))
            start.Environment["BAKEOFF_PYTHON"] = Path.GetFullPath(competitorPython);

        Console.WriteLine("COMPETITOR LANE - " + CodegenMeasurementProtocol.Stamp("current CUDA device"));
        using Process? process = Process.Start(start);
        if (process is null) throw new InvalidOperationException("Could not start competitor lane.");
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        bool timedOut = !process.WaitForExit(
            (int)TimeSpan.FromMinutes(30).TotalMilliseconds);
        string terminationDiagnostic = string.Empty;
        if (timedOut)
        {
            try
            {
                process.Kill(entireProcessTree: true);
                _ = process.WaitForExit(5000);
            }
            catch (Exception ex)
            {
                terminationDiagnostic = " Termination also failed: " + ex.Message;
            }
        }

        string outputText = timedOut && !stdout.IsCompletedSuccessfully
            ? string.Empty
            : stdout.GetAwaiter().GetResult();
        string errorText = timedOut && !stderr.IsCompletedSuccessfully
            ? string.Empty
            : stderr.GetAwaiter().GetResult();
        if (outputText.Length != 0) Console.Write(outputText);
        if (errorText.Length != 0) Console.Error.Write(errorText);
        if (timedOut)
        {
            if (File.Exists(output)) File.Delete(output);
            throw new TimeoutException(
                "Competitor lane did not finish within thirty minutes; evidence discarded." +
                terminationDiagnostic + " " + FirstDiagnostic(errorText, outputText));
        }
        if (process.ExitCode != 0)
        {
            if (File.Exists(output)) File.Delete(output);
            throw new InvalidOperationException(
                "Competitor lane failed with exit code " +
                process.ExitCode.ToString(CultureInfo.InvariantCulture) +
                "; no release evidence accepted. " +
                FirstDiagnostic(errorText, outputText));
        }

        string dispatchAfter = KernelEvidenceIdentity.CurrentDispatch();
        if (!string.Equals(dispatch, dispatchAfter, StringComparison.Ordinal))
        {
            if (File.Exists(output)) File.Delete(output);
            throw new InvalidOperationException(
                "Generated dispatch changed during the competitor run; evidence discarded.");
        }
        ValidateEvidence(output, dispatch);
        Console.WriteLine("validated competitor evidence: " + output);
    }

    /// <summary>
    /// Supplies the transposed-convolution shape from the catalog authority. Keeping this
    /// in Python as an independent literal once compared our corrected 28 -> 55 program
    /// with cuDNN output_padding=1 (28 -> 56), i.e. two different operators.
    /// </summary>
    private static string ConvTransposeContract()
    {
        CodegenCatalogEntry entry = CodegenKernelCatalog.Find(
            "conv_transpose2d_3x3_stride2") ??
            throw new InvalidOperationException("Transposed-convolution catalog entry is missing.");
        CodegenKernelSpec spec = entry.Bench;
        if (spec.Inputs.Count < 2 || spec.Inputs[0].Shape.Count != 4 ||
            spec.Output.Shape.Count != 4)
            throw new InvalidOperationException("Unexpected transposed-convolution catalog shape.");

        int n = spec.Inputs[0].Shape[0];
        int c = spec.Inputs[0].Shape[1];
        int ih = spec.Inputs[0].Shape[2];
        int iw = spec.Inputs[0].Shape[3];
        int oh = spec.Output.Shape[2];
        int ow = spec.Output.Shape[3];
        const int stride = 2, padding = 1, kernel = 3;
        int outputPaddingH = oh - ((ih - 1) * stride - 2 * padding + kernel);
        int outputPaddingW = ow - ((iw - 1) * stride - 2 * padding + kernel);
        if (outputPaddingH != outputPaddingW || outputPaddingH < 0 ||
            outputPaddingH >= stride)
            throw new InvalidOperationException(
                "Catalog transposed extent cannot be represented by cuDNN output_padding.");

        return string.Join(",",
            n.ToString(CultureInfo.InvariantCulture),
            c.ToString(CultureInfo.InvariantCulture),
            ih.ToString(CultureInfo.InvariantCulture),
            iw.ToString(CultureInfo.InvariantCulture),
            oh.ToString(CultureInfo.InvariantCulture),
            ow.ToString(CultureInfo.InvariantCulture),
            outputPaddingH.ToString(CultureInfo.InvariantCulture));
    }

    private static string FirstDiagnostic(string stderr, string stdout)
    {
        string diagnostic = stderr.Length != 0 ? stderr : stdout;
        foreach (string line in diagnostic.Split(new[] { '\r', '\n' },
                     StringSplitOptions.RemoveEmptyEntries))
            return line.Trim();
        return "No subprocess diagnostic was produced.";
    }

    private static void ValidateEvidence(string path, string expectedDispatch)
    {
        if (!File.Exists(path))
            throw new InvalidOperationException("Competitor lane completed without evidence: " + path);

        int accepted = 0;
        foreach (string line in File.ReadAllLines(path))
        {
            if (line.Length == 0 || line[0] == '#') continue;
            string[] cells = line.Split('\t');
            if (cells.Length < 9 || cells[0] == "kernel") continue;
            if (!string.Equals(cells[cells.Length - 1], CodegenMeasurementProtocol.Tag,
                    StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "Competitor evidence contains a stale protocol row for " + cells[0] + ".");
            }
            if (!string.Equals(cells[cells.Length - 2], expectedDispatch,
                    StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "Competitor evidence contains a stale dispatch row for " + cells[0] + ".");
            }
            accepted++;
        }

        if (accepted == 0)
            throw new InvalidOperationException("Competitor evidence contains no stable rows.");
    }

}
