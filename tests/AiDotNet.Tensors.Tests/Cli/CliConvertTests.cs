// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER

using System;
using System.IO;
using AiDotNet.Tensors.Cli;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Serialization.Safetensors;
using Xunit;

namespace AiDotNet.Tensors.Tests.Cli;

/// <summary>
/// End-to-end smoke tests for the <c>ai-tensors</c> CLI. Each test
/// drives <see cref="AiDotNet.Tensors.Cli.Program.Main"/> directly so
/// the harness is self-contained — no <c>dotnet</c> subprocess, no
/// path discovery, no PATH dependency. The CLI's correctness gates
/// are: arg parsing → format dispatch → reader/writer pairing →
/// licence guard. The reader/writer formats themselves are covered
/// in their own round-trip suites; here we verify the verbs glue
/// them together correctly.
/// </summary>
[Collection("PersistenceGuard")]
public class CliConvertTests
{
    private static IDisposable IsolatedTrial()
    {
        var path = Path.Combine(Path.GetTempPath(), "aidotnet-cli-trial-" + Guid.NewGuid().ToString("N") + ".json");
        return PersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    private static string FreshTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-cli-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void WriteSampleSafetensors(string path)
    {
        var t1 = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var t2 = new Tensor<int>(new[] { 10, 20, 30, 40 }, new[] { 4 });
        using var w = SafetensorsWriter.Create(path);
        w.Add("weights.0", t1);
        w.Add("indices", t2);
        w.Save();
    }

    [Fact]
    public void Help_ReturnsZeroAndPrintsUsage()
    {
        using var _ = IsolatedTrial();
        // Capture stdout so the test doesn't pollute xUnit output.
        var prior = Console.Out;
        using var sw = new StringWriter();
        Console.SetOut(sw);
        try
        {
            int rc = Program.Main(new[] { "--help" });
            Assert.Equal(0, rc);
            string written = sw.ToString();
            Assert.Contains("ai-tensors", written, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("convert", written, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("inspect", written, StringComparison.OrdinalIgnoreCase);
        }
        finally { Console.SetOut(prior); }
    }

    [Fact]
    public void NoArgs_ReturnsOneAndPrintsUsage()
    {
        using var _ = IsolatedTrial();
        var prior = Console.Out;
        using var sw = new StringWriter();
        Console.SetOut(sw);
        try
        {
            int rc = Program.Main(Array.Empty<string>());
            Assert.Equal(1, rc);
            // Verify the usage banner actually printed — without this
            // assertion a regression in PrintUsage() would silently
            // pass since we only check the exit code.
            Assert.Contains("USAGE", sw.ToString(), StringComparison.OrdinalIgnoreCase);
        }
        finally { Console.SetOut(prior); }
    }

    [Fact]
    public void UnknownVerb_PrintsErrorAndReturnsOne()
    {
        using var _ = IsolatedTrial();
        var priorOut = Console.Out;
        var priorErr = Console.Error;
        using var sw = new StringWriter();
        using var sErr = new StringWriter();
        Console.SetOut(sw);
        Console.SetError(sErr);
        try
        {
            int rc = Program.Main(new[] { "frobnicate" });
            Assert.Equal(1, rc);
            Assert.Contains("unknown command", sErr.ToString(), StringComparison.OrdinalIgnoreCase);
        }
        finally { Console.SetOut(priorOut); Console.SetError(priorErr); }
    }

    [Fact]
    public void Convert_SafetensorsToSharded_AndBack_PreservesAllTensorsBitExact()
    {
        using var _ = IsolatedTrial();
        string dir = FreshTempDir();
        try
        {
            string srcPath = Path.Combine(dir, "src.safetensors");
            string shardedDir = Path.Combine(dir, "sharded");
            string roundTripPath = Path.Combine(dir, "rt.safetensors");
            WriteSampleSafetensors(srcPath);

            // Force a small shard size so we actually exercise the
            // multi-shard path on a tiny payload.
            int rc1 = Program.Main(new[]
            {
                "convert",
                "--from", "safetensors",
                "--to", "safetensors-sharded",
                "--input", srcPath,
                "--output", shardedDir,
                "--shard-size", "32B",
            });
            Assert.Equal(0, rc1);
            Assert.True(File.Exists(Path.Combine(shardedDir, "model.safetensors.index.json")),
                "sharded writer did not produce model.safetensors.index.json");

            int rc2 = Program.Main(new[]
            {
                "convert",
                "--from", "safetensors-sharded",
                "--to", "safetensors",
                "--input", Path.Combine(shardedDir, "model.safetensors.index.json"),
                "--output", roundTripPath,
            });
            Assert.Equal(0, rc2);

            // Bit-exact recovery of every tensor.
            using var rA = SafetensorsReader.Open(srcPath);
            using var rB = SafetensorsReader.Open(roundTripPath);
            Assert.Equal(rA.Entries.Count, rB.Entries.Count);
            foreach (var name in rA.Names)
            {
                var aBytes = rA.ReadRawBytes(name);
                var bBytes = rB.ReadRawBytes(name);
                Assert.Equal(aBytes, bBytes);
            }
        }
        finally { try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ } }
    }

    [Fact]
    public void Inspect_Safetensors_ListsEntriesAndReturnsZero()
    {
        using var _ = IsolatedTrial();
        string dir = FreshTempDir();
        try
        {
            string p = Path.Combine(dir, "x.safetensors");
            WriteSampleSafetensors(p);

            var prior = Console.Out;
            using var sw = new StringWriter();
            Console.SetOut(sw);
            try
            {
                int rc = Program.Main(new[] { "inspect", p });
                Assert.Equal(0, rc);
                string text = sw.ToString();
                Assert.Contains("weights.0", text, StringComparison.Ordinal);
                Assert.Contains("indices", text, StringComparison.Ordinal);
            }
            finally { Console.SetOut(prior); }
        }
        finally { try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ } }
    }

    [Fact]
    public void Convert_RejectsUnsupportedFormatPair()
    {
        using var _ = IsolatedTrial();
        string dir = FreshTempDir();
        try
        {
            string p = Path.Combine(dir, "x.safetensors");
            WriteSampleSafetensors(p);

            var priorErr = Console.Error;
            using var sErr = new StringWriter();
            Console.SetError(sErr);
            try
            {
                int rc = Program.Main(new[]
                {
                    "convert",
                    "--from", "gguf",
                    "--to", "pt",
                    "--input", p,
                    "--output", Path.Combine(dir, "out.pt"),
                });
                Assert.Equal(1, rc);
                Assert.Contains("Unsupported conversion", sErr.ToString(), StringComparison.OrdinalIgnoreCase);
            }
            finally { Console.SetError(priorErr); }
        }
        finally { try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ } }
    }
}

#endif
