#if NET5_0_OR_GREATER
// Net-core only: it drives an external Python process (ProcessStartInfo.ArgumentList), as HeadToHeadNetworkHarness does.
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Tests.Parity;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Our device memory release and allocator against PyTorch's CUDA caching allocator, on the same device and sizes
/// (tools/parity/torch_cuda_allocator_reference.py runs the PyTorch side):
/// <list type="bullet">
/// <item>release: results nobody references, then <see cref="AiDotNetEngine.ResetToCpu"/> against
/// <c>torch.cuda.empty_cache()</c>: the device memory each returns, and how long the call takes;</item>
/// <item>churn: one allocate + free pair per size, against <c>torch.empty</c> from Python.</item>
/// </list>
/// The release must return at least what PyTorch's does. The timings are reported (test output and the job summary)
/// until a baseline from the CUDA runner sets their gates. Torch missing skips, unless
/// <c>AIDOTNET_REQUIRE_TORCH_CUDA=1</c> (the CUDA lane), where it fails: that lane installs a CUDA torch for this.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class PyTorchCachingAllocatorComparisonTests
{
    private const int Elements = 8 << 20; // 32 MiB per result, as in GpuEngineReleaseTests
    private const int Count = 8;
    private const int ChurnIterations = 5000, ChurnWarmup = 200;
    private static readonly int[] ChurnSizes = { 1 << 10, 1 << 18, 1 << 24 }; // 4 KiB, 1 MiB, 64 MiB

    private readonly ITestOutputHelper _output;

    public PyTorchCachingAllocatorComparisonTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Release_and_churn_against_the_pytorch_caching_allocator()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            Skip.IfNot(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false), "No GPU was adopted");
            var gpu = Assert.IsType<DirectGpuTensorEngine>(AiDotNetEngine.Current);
            var backend = gpu.QueryBackend;
            var memory = backend as IDeviceMemoryPressure;
            if (backend is null || memory is null || !memory.TryGetDeviceMemory(out _, out _))
            {
                Skip.If(true, "Not applicable: this backend cannot report free device memory, so the release cannot be measured.");
                return;
            }

            var torch = Torch(out string? unavailable);
            if (torch is null)
            {
                string reason = unavailable ?? "torch did not run.";
                Assert.False(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_TORCH_CUDA") == "1",
                    "AIDOTNET_REQUIRE_TORCH_CUDA=1 but the PyTorch side cannot run: " + reason);
                Skip.If(true, reason);
                return;
            }

            // Both sides must be measuring one device, or every number below compares two different GPUs.
            Assert.True(string.Equals(backend.DeviceName.Trim(), torch.Device.Trim(), StringComparison.Ordinal),
                $"our engine runs on '{backend.DeviceName}' but PyTorch measured '{torch.Device}'");

            // Ours, release: the same drop as GpuEngineReleaseTests, then the reset.
            GpuEngineReleaseTests.DropDeviceResults(gpu, Elements, Count);
            Assert.True(memory.TryGetDeviceMemory(out ulong before, out _), "free device memory before the reset was not readable");
            var watch = Stopwatch.StartNew();
            AiDotNetEngine.ResetToCpu();
            double releaseSeconds = watch.Elapsed.TotalSeconds;
            Assert.True(memory.TryGetDeviceMemory(out ulong after, out _), "free device memory after the reset was not readable");
            long freed = (long)after - (long)before;

            // Ours, churn: the backend's allocate + free, as torch.empty + del exercises PyTorch's.
            var churn = ChurnSizes.Select(size => (Elements: size, Seconds: Churn(backend, size))).ToArray();

            var rows = new List<(string Measure, double Ours, double Torch, string Unit)>
            {
                ("release: device memory returned", freed / 1048576.0, torch.FreedBytes / 1048576.0, "MiB"),
                ("release: call duration", releaseSeconds * 1e3, torch.ReleaseSeconds * 1e3, "ms"),
            };
            foreach (var (size, seconds) in churn)
                rows.Add(($"churn: allocate + free, {size * 4L / 1024} KiB", seconds * 1e6,
                    torch.Churn[size] * 1e6, "us"));
            Report(torch.Version, torch.Device, rows);

            Assert.True(freed >= torch.FreedBytes,
                $"the reset returned {freed >> 20} MiB; torch.cuda.empty_cache returned {torch.FreedBytes >> 20} MiB");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    private static double Churn(IDirectGpuBackend backend, int elements)
    {
        for (int i = 0; i < ChurnWarmup; i++) backend.AllocateBuffer(elements).Dispose();
        backend.Synchronize();
        var watch = Stopwatch.StartNew();
        for (int i = 0; i < ChurnIterations; i++) backend.AllocateBuffer(elements).Dispose();
        backend.Synchronize();
        return watch.Elapsed.TotalSeconds / ChurnIterations;
    }

    private void Report(string version, string device, List<(string Measure, double Ours, double Torch, string Unit)> rows)
    {
        var table = new StringBuilder();
        table.AppendLine($"### Device memory release and allocator vs PyTorch {version} ({device})");
        table.AppendLine();
        table.AppendLine("| Measure | Ours | PyTorch | Ours / PyTorch |");
        table.AppendLine("|---|---:|---:|---:|");
        foreach (var (measure, ours, torch, unit) in rows)
        {
            string ratio = torch == 0 ? "n/a" : (ours / torch).ToString("0.00", CultureInfo.InvariantCulture) + "x";
            table.AppendLine(string.Format(CultureInfo.InvariantCulture, "| {0} | {1:0.###} {3} | {2:0.###} {3} | {4} |",
                measure, ours, torch, unit, ratio));
        }
        string text = table.ToString();
        _output.WriteLine(text);
        if (Environment.GetEnvironmentVariable("GITHUB_STEP_SUMMARY") is { Length: > 0 } summary)
            File.AppendAllText(summary, text + Environment.NewLine);
    }

    private sealed class TorchResult
    {
        public string Version = string.Empty, Device = string.Empty;
        public long FreedBytes;
        public double ReleaseSeconds;
        public Dictionary<int, double> Churn = new();
    }

    private static TorchResult? Torch(out string? unavailable)
    {
        unavailable = null;
        string? root = PyTorchParityInventory.FindRepositoryRoot();
        if (root is null) { unavailable = "tools/parity is missing from this checkout."; return null; }
        string script = Path.Combine(root, "tools", "parity", "torch_cuda_allocator_reference.py");
        string dir = Path.Combine(Path.GetTempPath(), "aidotnet-torch-allocator-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string casePath = Path.Combine(dir, "case.json"), resultPath = Path.Combine(dir, "result.json");
            File.WriteAllText(casePath, JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["elements"] = Elements, ["count"] = Count, ["churn_sizes"] = ChurnSizes,
                ["churn_iterations"] = ChurnIterations, ["churn_warmup"] = ChurnWarmup,
            }));
            var psi = new ProcessStartInfo(HeadToHeadNetworkHarness.PythonExecutable)
            {
                // stdout is not read (the result is a file), so it is not redirected: an unread pipe that fills blocks the child.
                RedirectStandardError = true, RedirectStandardOutput = false, UseShellExecute = false,
            };
            foreach (var a in new[] { script, casePath, resultPath }) psi.ArgumentList.Add(a);
            Process? p;
            try { p = Process.Start(psi); }
            catch (System.ComponentModel.Win32Exception) { unavailable = $"{psi.FileName} is not on PATH (set PARITY_PYTHON)."; return null; }
            if (p is null) { unavailable = "python did not start."; return null; }
            using (p)
            {
                // Read stderr while waiting: reading it to the end first would never return from a child that hangs.
                var errRead = p.StandardError.ReadToEndAsync();
                if (!p.WaitForExit(300_000))
                {
                    try { p.Kill(); } catch (InvalidOperationException) { }   // it exited between the wait and the kill
                    throw new TimeoutException($"{script} did not finish within 300 s.");
                }
                string err = errRead.GetAwaiter().GetResult();
                if (p.ExitCode == 3) { unavailable = "torch has no CUDA device here (" + err.Trim() + ")."; return null; }
                if (p.ExitCode != 0)
                {
                    if (err.Contains("No module named 'torch'", StringComparison.Ordinal))
                    {
                        unavailable = "torch is not installed for " + psi.FileName + ".";
                        return null;
                    }
                    throw new InvalidOperationException($"torch_cuda_allocator_reference.py failed ({p.ExitCode}): {err}");
                }
            }
            using var doc = JsonDocument.Parse(File.ReadAllText(resultPath));
            var json = doc.RootElement;
            var release = json.GetProperty("release");
            var result = new TorchResult
            {
                Version = json.GetProperty("torch").GetString() ?? string.Empty,
                Device = json.GetProperty("device").GetString() ?? string.Empty,
                FreedBytes = release.GetProperty("freed_bytes").GetInt64(),
                ReleaseSeconds = release.GetProperty("seconds").GetDouble(),
            };
            foreach (var c in json.GetProperty("churn").EnumerateArray())
                result.Churn[c.GetProperty("elements").GetInt32()] = c.GetProperty("seconds_per_pair").GetDouble();
            return result;
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
#endif
