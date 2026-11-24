using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CUDA_VISIBLE_DEVICES must hide a GPU from every backend, not only CUDA's: Vulkan enumerates every GPU in the
/// machine, and its backend opened a context on a GPU the process was told not to use (measured 2026-11-24: with only
/// the RTX 3090 visible, a test host appeared on the RTX 3080 when the Vulkan backend initialized, and it picked the
/// 3080 as its device; CUDA and NVIDIA's OpenCL stayed on the 3090).
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaVisibleDevicesTests
{
    private readonly ITestOutputHelper _output;
    public CudaVisibleDevicesTests(ITestOutputHelper output) => _output = output;

    private const string Uuid = "GPU-3cc6845a-128e-b862-3de9-7b3ea1e8495d";

    [Fact]
    public void An_unset_variable_filters_nothing()
        => Assert.Null(CudaVisibleDevices.Resolve(null, () => throw new InvalidOperationException("not consulted")));

    [Fact]
    public void Without_the_cuda_driver_full_uuids_are_read_from_the_value()
    {
        var visible = CudaVisibleDevices.Resolve(" " + Uuid + " ,", () => null);
        Assert.Equal(new[] { "3cc6845a128eb8623de97b3ea1e8495d" }, visible!.ToArray());
        // Indices (or UUID prefixes) need the driver's ordering: unmapped, nothing is filtered.
        Assert.Null(CudaVisibleDevices.Resolve("0,1", () => null));
        Assert.Null(CudaVisibleDevices.Resolve("GPU-3cc6845a", () => null));
        // An empty value hides every device, as it does for CUDA.
        Assert.Empty(CudaVisibleDevices.Resolve("", () => null)!);
    }

    [Fact]
    public void The_cuda_drivers_view_wins_when_present()
    {
        var visible = CudaVisibleDevices.Resolve("1", () => new List<string> { "00112233445566778899aabbccddeeff" });
        Assert.Equal(new[] { "00112233445566778899aabbccddeeff" }, visible!.ToArray());
        Assert.True(CudaVisibleDevices.Hex(new byte[] { 0x0a, 0xf0 }) == "0af0");
    }

    /// <summary>
    /// On a machine with two or more NVIDIA GPUs and CUDA_VISIBLE_DEVICES naming fewer, the Vulkan backend opens its
    /// device on a visible GPU, and nvidia-smi lists this process on visible GPUs only.
    /// </summary>
    [SkippableFact]
    public void The_vulkan_backend_opens_no_context_on_a_hidden_gpu()
    {
        string? setting = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
        Skip.If(setting is null, "CUDA_VISIBLE_DEVICES is not set.");
        var all = NvidiaSmi("--query-gpu=uuid --format=csv,noheader");
        Skip.If(all is null || all.Length < 2, "Needs nvidia-smi and two or more NVIDIA GPUs.");
        var visible = CudaVisibleDevices.VisibleNvidiaUuids;
        Skip.If(visible is null || visible.Count >= all!.Length, "CUDA_VISIBLE_DEVICES does not hide a GPU here.");
        var hidden = all!.Select(u => u.Trim()).Where(u => !visible!.Contains(u.Substring(4).Replace("-", "").ToLowerInvariant())).ToArray();
        Assert.NotEmpty(hidden);

        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize(), "No Vulkan device.");
        _output.WriteLine($"Vulkan device: {backend.DeviceName}; hidden: {string.Join(" ", hidden)}");
        System.Threading.Thread.Sleep(1000);
        int me = Environment.ProcessId;
        var mine = (NvidiaSmi("--query-compute-apps=pid,gpu_uuid --format=csv,noheader") ?? Array.Empty<string>())
            .Select(l => l.Split(',')).Where(f => f.Length == 2 && int.TryParse(f[0].Trim(), out int pid) && pid == me)
            .Select(f => f[1].Trim()).Distinct().ToArray();
        _output.WriteLine($"this process is on: {string.Join(" ", mine)}");
        Assert.NotEmpty(mine);   // the Vulkan device itself is listed, on a visible GPU
        Assert.Empty(mine.Intersect(hidden));
    }

    private static string[]? NvidiaSmi(string arguments)
    {
        try
        {
            var start = new ProcessStartInfo("nvidia-smi", arguments) { RedirectStandardOutput = true, UseShellExecute = false };
            using var process = Process.Start(start);
            if (process is null) return null;
            string text = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            return process.ExitCode == 0 ? text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => l.Trim()).ToArray() : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }
}
