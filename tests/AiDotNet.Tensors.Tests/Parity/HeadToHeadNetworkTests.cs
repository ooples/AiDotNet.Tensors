#if NET5_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// Head-to-head networks against PyTorch (issue #1057): the same network, built from one spec with identical
/// weights and data, trained on both sides and compared.
/// </summary>
/// <remarks>
/// <para>
/// Every run checks that the two loss curves agree, because a timing is meaningless if the two sides are not
/// doing the same arithmetic. Then:
/// </para>
/// <list type="bullet">
/// <item><b>Ratchet</b> (every PR): the ours/theirs step-time ratio may not rise above the ratio recorded for
/// this machine class in <c>parity/head-to-head-baseline.json</c>, beyond the measured noise.</item>
/// <item><b>Parity</b> (nightly, <c>Category=PyTorchParity</c>): fails whenever Tensors is slower than PyTorch
/// beyond the noise.</item>
/// </list>
/// <para>
/// Without Python and torch the case is skipped with the reason, never passed.
/// </para>
/// </remarks>
public abstract class HeadToHeadTestBase
{
    private const string BaselineFile = "parity/head-to-head-baseline.json";
    private readonly ITestOutputHelper _output;

    protected HeadToHeadTestBase(ITestOutputHelper output) => _output = output;

    private protected HeadToHeadNetworkHarness.CaseResult RunCase(string network, string device)
    {
        string root = PyTorchParityInventory.FindRepositoryRoot()
            ?? throw new InvalidOperationException("parity/ is missing from this checkout.");
        var (result, deferred) = HeadToHeadNetworkHarness.Run(root, network, device);
        Skip.If(deferred is not null, $"DEFERRED: {deferred?.Reason}");
        if (result is null) throw new InvalidOperationException("The harness returned neither a result nor a deferral.");

        Report(result);
        AssertLossesAgree(result);
        return result;
    }

    private void Report(HeadToHeadNetworkHarness.CaseResult r)
    {
        _output.WriteLine($"{r.Network} on {r.Device} [{r.MachineKey}]  torch {r.Torch.Version} vs {r.Tensors.Framework} {r.Tensors.Version}");
        _output.WriteLine($"  {"phase",-10} {"torch ms",10} {"ours ms",10} {"ratio",7}");
        foreach (var phase in new[] { "forward", "backward", "optimizer", "step" })
        {
            double theirs = r.Torch.Phases[phase].MedianMs, ours = r.Tensors.Phases[phase].MedianMs;
            _output.WriteLine($"  {phase,-10} {theirs,10:F3} {ours,10:F3} {ours / theirs,7:F2}");
        }

        _output.WriteLine($"  step ratio (min/min, gated) {r.StepRatio:F3}; median/median {r.MedianStepRatio:F3}; run noise {r.NoiseBand:P1}");
    }

    private static void AssertLossesAgree(HeadToHeadNetworkHarness.CaseResult r)
    {
        int n = Math.Min(r.Torch.Losses.Count, r.Tensors.Losses.Count);
        Assert.True(n > 0, "Neither side recorded a loss curve.");
        for (int i = 0; i < n; i++)
        {
            double theirs = r.Torch.Losses[i], ours = r.Tensors.Losses[i];
            // The first loss sees identical weights and data, so only the reduction order differs; later steps
            // compound that through SGD, so their tolerance is looser.
            double tolerance = i == 0 ? 1e-4 : 1e-3;
            Assert.True(Math.Abs(ours - theirs) <= tolerance * Math.Max(1.0, Math.Abs(theirs)),
                $"{r.Network}: loss at step {i} disagrees (torch {theirs:G9}, ours {ours:G9}). The two sides are not " +
                "training the same network, so their timings are not comparable.");
        }
    }

    private protected void AssertRatchet(HeadToHeadNetworkHarness.CaseResult r)
    {
        string root = PyTorchParityInventory.FindRepositoryRoot()
            ?? throw new InvalidOperationException("parity/ is missing from this checkout.");
        string key = $"{r.Network}:{r.Device}";
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, BaselineFile)));
        double band = doc.RootElement.GetProperty("noiseFloor").GetDouble();

        bool recorded = doc.RootElement.GetProperty("cases").TryGetProperty(key, out var perMachine)
                        && perMachine.TryGetProperty(r.MachineKey, out var baselineElement);
        Skip.IfNot(recorded,
            $"DEFERRED: no baseline for {key} on {r.MachineKey}. Measured ratio {r.StepRatio:F3}; add " +
            $"\"{r.MachineKey}\": {r.StepRatio:F3} under \"{key}\" in {BaselineFile} to start the ratchet.");

        double baseline = perMachine.GetProperty(r.MachineKey).GetDouble();
        Assert.True(r.StepRatio <= baseline * (1 + band),
            $"{key} regressed on {r.MachineKey}: Tensors takes {r.StepRatio:F3}x PyTorch's step time, above the " +
            $"recorded {baseline:F3}x plus {band:P0} noise.");
        if (r.StepRatio < baseline * (1 - band))
            _output.WriteLine($"  IMPROVED: {r.StepRatio:F3}x against a recorded {baseline:F3}x. Lower the baseline in " +
                              $"{BaselineFile} so the gain is locked in.");
    }

    private protected static void AssertParity(HeadToHeadNetworkHarness.CaseResult r)
    {
        const double band = 0.05;
        Assert.True(r.StepRatio <= 1 + band,
            $"{r.Network} on {r.Device}: Tensors takes {r.StepRatio:F3}x PyTorch's step time " +
            $"(fastest step {r.Tensors.Phases["step"].MinMs:F3} ms against {r.Torch.Phases["step"].MinMs:F3} ms). " +
            "Slowest phase: " + r.Tensors.Phases.Where(p => p.Key != "step")
                .OrderByDescending(p => p.Value.MinMs / r.Torch.Phases[p.Key].MinMs)
                .Select(p => $"{p.Key} {p.Value.MinMs / r.Torch.Phases[p.Key].MinMs:F2}x").First());
    }
}

/// <summary>CPU cases: the PR ratchet and the nightly parity check (see <see cref="HeadToHeadTestBase"/>).</summary>
[Collection("PyTorchHeadToHead")]
public sealed class HeadToHeadNetworkTests : HeadToHeadTestBase
{
    public HeadToHeadNetworkTests(ITestOutputHelper output) : base(output) { }

    [SkippableFact]
    public void Mlp_Cpu_DoesNotRegress() => AssertRatchet(RunCase("mlp", "cpu"));

    [SkippableFact]
    [Trait("Category", "PyTorchParity")]
    public void Mlp_Cpu_IsAsFastAsPyTorch() => AssertParity(RunCase("mlp", "cpu"));

    [SkippableFact]
    public void Cnn_Cpu_DoesNotRegress() => AssertRatchet(RunCase("cnn", "cpu"));

    [SkippableFact]
    [Trait("Category", "PyTorchParity")]
    public void Cnn_Cpu_IsAsFastAsPyTorch() => AssertParity(RunCase("cnn", "cpu"));
}

/// <summary>
/// CUDA cases, run by tools/parity/run-gpu.ps1 (<c>Category=PyTorchParityGpu</c>); hosted CI has no GPU. They share
/// the serial GPU collection so no other GPU test runs on the device while a step is being timed.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HeadToHeadGpuNetworkTests : HeadToHeadTestBase
{
    public HeadToHeadGpuNetworkTests(ITestOutputHelper output) : base(output) { }

    [SkippableFact]
    [Trait("Category", "PyTorchParityGpu")]
    public void Mlp_Cuda_DoesNotRegress() => AssertRatchet(RunCase("mlp", "cuda"));

    [SkippableFact]
    [Trait("Category", "PyTorchParityGpu")]
    public void Mlp_Cuda_IsAsFastAsPyTorch() => AssertParity(RunCase("mlp", "cuda"));

    [SkippableFact]
    [Trait("Category", "PyTorchParityGpu")]
    public void Cnn_Cuda_DoesNotRegress() => AssertRatchet(RunCase("cnn", "cuda"));

    [SkippableFact]
    [Trait("Category", "PyTorchParityGpu")]
    public void Cnn_Cuda_IsAsFastAsPyTorch() => AssertParity(RunCase("cnn", "cuda"));
}
#endif
