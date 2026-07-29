// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Device-free emission and ptxas validation across supported NVIDIA targets.</summary>
internal static class KernelArchitectureTool
{
    private static readonly (int Major, int Minor)[] Targets =
    {
        (8, 0), (8, 6), (8, 9), (9, 0),
    };

    internal static void Run(string[] args)
    {
        string? ptxas = ValueOf(args, "--ptxas") ?? FindPtxas();
        bool requirePtxas = args.Contains("--require-ptxas", StringComparer.Ordinal);
        if (requirePtxas && ptxas is null)
            throw new InvalidOperationException("ptxas is required but was not found.");

        string output = Path.GetFullPath(ValueOf(args, "--out") ??
            Path.Combine("artifacts", "codegen-architecture-validation.tsv"));
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        if (File.Exists(output)) File.Delete(output);

        string temporary = Path.Combine(
            Path.GetTempPath(), "aidotnet-codegen-arch-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temporary);
        var rows = new List<string>();
        int assembled = 0;

        Console.WriteLine();
        Console.WriteLine("CODEGEN ARCHITECTURE VALIDATION");
        Console.WriteLine("targets: " + string.Join(", ", Targets.Select(TargetName)));
        Console.WriteLine("ptxas: " + (ptxas ?? "not found; emission-only"));

        try
        {
            foreach (var entry in CodegenKernelCatalog.All)
            {
                foreach (var target in Targets)
                {
                    var emitter = new PtxAffineEmitter();
                    string ptx = emitter.Emit(entry.Bench, target.Major, target.Minor);
                    string hash = Sha256(ptx);
                    string status = "emitted";

                    if (ptxas is not null)
                    {
                        string stem = Safe(entry.Name) + "-" + TargetName(target);
                        string ptxPath = Path.Combine(temporary, stem + ".ptx");
                        string cubinPath = Path.Combine(temporary, stem + ".cubin");
                        File.WriteAllText(ptxPath, ptx, new UTF8Encoding(false));
                        Assemble(ptxas, target, ptxPath, cubinPath);
                        status = "ptxas-pass";
                        assembled++;
                    }

                    rows.Add(string.Join("\t",
                        entry.Name,
                        TargetName(target),
                        hash,
                        status,
                        CodegenMeasurementProtocol.Tag));
                }
            }
        }
        finally
        {
            if (Directory.Exists(temporary)) Directory.Delete(temporary, recursive: true);
        }

        var text = new StringBuilder();
        text.AppendLine("kernel\ttarget\tptx_sha256\tstatus\tprotocol");
        foreach (string row in rows) text.AppendLine(row);
        File.WriteAllText(output, text.ToString(), new UTF8Encoding(false));

        Console.WriteLine("validated {0} kernel-target pairs; ptxas assembled {1}",
            rows.Count.ToString(CultureInfo.InvariantCulture),
            assembled.ToString(CultureInfo.InvariantCulture));
        Console.WriteLine("manifest: " + output);
        Console.WriteLine("This proves emission/assembly portability, not runtime performance;");
        Console.WriteLine("release still requires the physical second-architecture lane.");
    }

    private static void Assemble(
        string ptxas, (int Major, int Minor) target, string ptxPath, string cubinPath)
    {
        var start = new ProcessStartInfo
        {
            FileName = ptxas,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        start.ArgumentList.Add("-arch=" + TargetName(target));
        start.ArgumentList.Add("-o");
        start.ArgumentList.Add(cubinPath);
        start.ArgumentList.Add(ptxPath);

        using Process? process = Process.Start(start);
        if (process is null) throw new InvalidOperationException("Could not start ptxas.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                TargetName(target) + " ptxas failed for " + Path.GetFileName(ptxPath) +
                ": " + (stderr.Length != 0 ? stderr : stdout).Replace('\n', ' ').Trim());
        }
    }

    private static string TargetName((int Major, int Minor) target) =>
        "sm_" + target.Major.ToString(CultureInfo.InvariantCulture) +
        target.Minor.ToString(CultureInfo.InvariantCulture);

    private static string Safe(string value)
    {
        var text = new StringBuilder(value.Length);
        foreach (char c in value)
            text.Append(char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_');
        return text.ToString();
    }

    private static string Sha256(string value)
    {
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(value));
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    private static string? FindPtxas()
    {
        string? configured = Environment.GetEnvironmentVariable("AIDOTNET_PTXAS_PATH");
        if (!string.IsNullOrEmpty(configured) && File.Exists(configured))
            return Path.GetFullPath(configured);

        string? path = Environment.GetEnvironmentVariable("PATH");
        if (!string.IsNullOrEmpty(path))
        {
            foreach (string directory in path.Split(Path.PathSeparator))
            {
                if (directory.Length == 0) continue;
                string candidate = Path.Combine(directory,
                    OperatingSystem.IsWindows() ? "ptxas.exe" : "ptxas");
                if (File.Exists(candidate)) return candidate;
            }
        }

        if (OperatingSystem.IsWindows())
        {
            string root = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
            if (Directory.Exists(root))
            {
                return Directory.GetDirectories(root)
                    .OrderByDescending(Directory.GetLastWriteTimeUtc)
                    .Select(v => Path.Combine(v, "bin", "ptxas.exe"))
                    .FirstOrDefault(File.Exists);
            }
        }
        return null;
    }

    private static string? ValueOf(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], flag, StringComparison.Ordinal)) return args[i + 1];
        return null;
    }
}
