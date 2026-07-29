// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

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
        string maxSpread = ValueOf(args, "--max-spread-pct") ?? "5.0";
        if (!double.TryParse(maxSpread, NumberStyles.Float, CultureInfo.InvariantCulture,
                out double spread) || spread <= 0)
        {
            throw new ArgumentException("--max-spread-pct must be a positive number.");
        }

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
        start.ArgumentList.Add("--output");
        start.ArgumentList.Add(output);
        start.ArgumentList.Add("--max-spread-pct");
        start.ArgumentList.Add(spread.ToString("R", CultureInfo.InvariantCulture));

        // The outer Python script launches both lanes. Pin our lane to this exact assembly,
        // rather than relying on a default path that may name a stale build.
        start.Environment["BAKEOFF_DOTNET"] = Assembly.GetExecutingAssembly().Location;
        start.Environment["AIDOTNET_BENCHMARK_ORCHESTRATOR_PID"] =
            Environment.ProcessId.ToString(CultureInfo.InvariantCulture);
        string? competitorPython = ValueOf(args, "--competitor-python");
        if (!string.IsNullOrEmpty(competitorPython))
            start.Environment["BAKEOFF_PYTHON"] = Path.GetFullPath(competitorPython);

        Console.WriteLine("COMPETITOR LANE - " + CodegenMeasurementProtocol.Stamp("current CUDA device"));
        using Process? process = Process.Start(start);
        if (process is null) throw new InvalidOperationException("Could not start competitor lane.");
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        process.WaitForExit();
        string outputText = stdout.GetAwaiter().GetResult();
        string errorText = stderr.GetAwaiter().GetResult();
        if (outputText.Length != 0) Console.Write(outputText);
        if (errorText.Length != 0) Console.Error.Write(errorText);
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                "Competitor lane failed with exit code " +
                process.ExitCode.ToString(CultureInfo.InvariantCulture) +
                "; no release evidence accepted. " +
                FirstDiagnostic(errorText, outputText));

        ValidateEvidence(output);
        Console.WriteLine("validated competitor evidence: " + output);
    }

    private static string FirstDiagnostic(string stderr, string stdout)
    {
        string diagnostic = stderr.Length != 0 ? stderr : stdout;
        foreach (string line in diagnostic.Split(new[] { '\r', '\n' },
                     StringSplitOptions.RemoveEmptyEntries))
            return line.Trim();
        return "No subprocess diagnostic was produced.";
    }

    private static void ValidateEvidence(string path)
    {
        if (!File.Exists(path))
            throw new InvalidOperationException("Competitor lane completed without evidence: " + path);

        int accepted = 0;
        foreach (string line in File.ReadAllLines(path))
        {
            if (line.Length == 0 || line[0] == '#') continue;
            string[] cells = line.Split('\t');
            if (cells.Length < 8 || cells[0] == "kernel") continue;
            if (!string.Equals(cells[cells.Length - 1], CodegenMeasurementProtocol.Tag,
                    StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "Competitor evidence contains a stale protocol row for " + cells[0] + ".");
            }
            accepted++;
        }

        if (accepted == 0)
            throw new InvalidOperationException("Competitor evidence contains no stable rows.");
    }

    private static string? ValueOf(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], flag, StringComparison.Ordinal)) return args[i + 1];
        return null;
    }
}
