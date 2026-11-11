#if NET5_0_OR_GREATER
using System;
using System.IO;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// #1096 P0.7: deterministic per-step counts of every reference network's compiled training step, ratcheted against
/// <c>parity/step-count-baseline.json</c>. A count may fall (lower the baseline then) but never rise.
/// <list type="bullet">
/// <item>CPU: managed bytes allocated per steady-state step on one thread (exact, so the slack only absorbs runtime
/// noise such as a lazily grown pool).</item>
/// <item>CUDA: memcpy and memset nodes of the captured step graph, which must not rise at all; kernel nodes, which may
/// rise by 5% (at least 1) because the GEMM route depends on which vendor libraries load (the tuned-kernel routes are
/// pinned to their references); plus the allocated bytes.</item>
/// </list>
/// Runs from random weights: no PyTorch needed, so the CPU cases run in hosted CI.
/// </summary>
public abstract class StepCountRatchetBase
{
    internal const string BaselineFile = "parity/step-count-baseline.json";
    private readonly ITestOutputHelper _output;

    private protected StepCountRatchetBase(ITestOutputHelper output) => _output = output;

    private protected void Ratchet(string network, DirectGpuTensorEngine? gpu)
    {
        string root = PyTorchParityInventory.FindRepositoryRoot()
            ?? throw new InvalidOperationException("parity/ is missing from this checkout.");
        var counts = HeadToHeadNetworkHarness.MeasureCompiledStepCounts(root, network, gpu);
        // CPU allocations can differ between the AVX2 and AVX-512 code paths, and CUDA graphs between devices (routes are
        // tuned per device), so each count is keyed by what it depends on. An unseeded combination defers with its value.
        string device = gpu is null
            ? "cpu:" + (System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated ? "avx512"
                : System.Runtime.Intrinsics.X86.Avx2.IsSupported ? "avx2" : "scalar")
            : "cuda:" + (gpu.GetBackend() is { } backend ? backend.DeviceName : "unknown");
        string key = $"{network}:{device}";
        _output.WriteLine($"{key}: {counts.AllocatedBytesPerStep} bytes/step" +
            (counts.KernelNodes is { } k ? $", graph kernels {k}, memcpy {counts.MemcpyNodes}, memset {counts.MemsetNodes}" : "") +
            (counts.GraphCaptureFailure is { } why ? $", no graph: {why}" : ""));

        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, BaselineFile)));
        bool recorded = doc.RootElement.GetProperty("cases").TryGetProperty(key, out var baseline);
        string measuredJson = $"\"{key}\": {{ \"allocatedBytesPerStep\": {counts.AllocatedBytesPerStep}" +
            (counts.KernelNodes is { } kn ? $", \"kernelNodes\": {kn}, \"memcpyNodes\": {counts.MemcpyNodes}, \"memsetNodes\": {counts.MemsetNodes}" : "") +
            " }";
        Skip.IfNot(recorded, $"DEFERRED: no baseline for {key}. Measured {measuredJson}; add it under \"cases\" in {BaselineFile}.");

        long allowedBytes = baseline.GetProperty("allocatedBytesPerStep").GetInt64();
        // 5% of the baseline, with a floor for runtime noise. The CPU floor (4 KB) absorbs a lazily grown pool. The CUDA baselines
        // are 0.9-8 KB, so a 4 KB floor there would let an 896-byte step allocate five times its baseline unnoticed; the
        // CUDA floor is 256 bytes.
        long slack = Math.Max(gpu is null ? 4096 : 256, allowedBytes / 20);
        Assert.True(counts.AllocatedBytesPerStep <= allowedBytes + slack,
            $"{key}: a compiled step allocates {counts.AllocatedBytesPerStep} bytes, above the recorded {allowedBytes} " +
            $"(+{slack} slack). Find the new per-step allocation (GC.GetAllocatedBytesForCurrentThread around Step()).");
        if (counts.AllocatedBytesPerStep + slack < allowedBytes)
            _output.WriteLine($"  IMPROVED: {counts.AllocatedBytesPerStep} bytes against {allowedBytes}; lower the baseline.");

        foreach (var (name, value) in new[]
                 {
                     ("kernelNodes", counts.KernelNodes), ("memcpyNodes", counts.MemcpyNodes), ("memsetNodes", counts.MemsetNodes),
                 })
        {
            if (!baseline.TryGetProperty(name, out var recordedCount)) continue;
            Assert.True(value is not null,
                $"{key}: the baseline records {name} but no step graph was captured ({counts.GraphCaptureFailure}).");
            // The CUDA ratchet pins the tuned-kernel routes, but the GEMM route still depends on the vendor libraries
            // that load (the CNN step is one kernel longer with cuBLASLt than without it), so kernel nodes keep a small
            // slack; memcpy and memset nodes are exact.
            int limit = name == "kernelNodes" ? recordedCount.GetInt32() + Math.Max(1, recordedCount.GetInt32() / 20) : recordedCount.GetInt32();
            Assert.True(value <= limit,
                $"{key}: the captured step graph has {value} {name}, above the recorded {recordedCount.GetInt32()} (limit {limit}).");
            if (value < recordedCount.GetInt32())
                _output.WriteLine($"  IMPROVED: {name} {value} against {recordedCount.GetInt32()}; lower the baseline.");
        }
    }
}

/// <summary>
/// The head-to-heads and the step-count ratchets set process-wide state (CpuParallelSettings.MaxDegreeOfParallelism,
/// AIDOTNET_AUTOCAST, AiDotNetEngine.Current) for a whole run, so they run with no other collection in parallel: a test
/// elsewhere that set the thread count itself could otherwise run serially and still pass, and overlapping
/// save-and-restore pairs could leave the setting changed after both finished.
/// </summary>
[CollectionDefinition("PyTorchHeadToHead", DisableParallelization = true)]
public sealed class PyTorchHeadToHeadCollection
{
}

/// <summary>The CPU cases (hosted CI).</summary>
[Collection("PyTorchHeadToHead")]
public sealed class StepCountRatchetTests : StepCountRatchetBase
{
    public StepCountRatchetTests(ITestOutputHelper output) : base(output) { }

    [SkippableTheory]
    [InlineData("mlp")]
    [InlineData("cnn")]
    [InlineData("lstm")]
    [InlineData("transformer")]
    public void CompiledStep_Cpu_CountsDoNotRise(string network) => Ratchet(network, gpu: null);
}

/// <summary>The CUDA cases, run by the GPU lane (tools/parity/run-gpu.ps1, <c>Category=PyTorchParityGpu</c>).</summary>
[Collection("DirectGpuSerial")]
public sealed class StepCountRatchetGpuTests : StepCountRatchetBase
{
    public StepCountRatchetGpuTests(ITestOutputHelper output) : base(output) { }

    [SkippableTheory]
    [Trait("Category", "PyTorchParityGpu")]
    [InlineData("mlp")]
    [InlineData("cnn")]
    [InlineData("lstm")]
    [InlineData("transformer")]
    [InlineData("resnet18")]
    public void CompiledStep_Cuda_CountsDoNotRise(string network)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) { Skip.If(true, $"no GPU backend ({ex.GetType().Name}: {ex.Message})."); return; }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "no GPU available to AiDotNet.Tensors.");
            // The keys and the graph-node counts are CUDA's; another backend (HIP, OpenCL) would defer under a "cuda:" key.
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "the CUDA ratchet needs the CUDA backend.");
            // Kernel nodes follow the tuned-kernel route of every conv shape (a Winograd conv is several kernels where
            // the reference is one), and a slot decides by timing in this process, by the tuning budget left, and by
            // decisions persisted under ~/.aidotnet/autotune by any earlier process on the machine. Unpinned, one
            // commit measured ResNet-18 at 147, 148, 149, 157 and 166 kernel nodes on one RTX 1000 Ada. The ratchet
            // tracks the step's structure, not the machine's tuning history, so every slot serves its reference route
            // here and no persisted decision is read or written. The engine's backend is new, so no slot holds an
            // earlier decision.
            TunedKernelMode? previousMode = TunedKernelPolicy.ModeOverride;
            bool previousIgnoreProfiles = TunedKernelPolicy.IgnoreProfiles;
            bool previousPersist = TunedKernelPolicy.PersistDecisions;
            TunedKernelPolicy.ModeOverride = TunedKernelMode.Off;
            TunedKernelPolicy.IgnoreProfiles = true;
            TunedKernelPolicy.PersistDecisions = false;
            try
            {
                Ratchet(network, gpu);
            }
            finally
            {
                TunedKernelPolicy.ModeOverride = previousMode;
                TunedKernelPolicy.IgnoreProfiles = previousIgnoreProfiles;
                TunedKernelPolicy.PersistDecisions = previousPersist;
            }
        }
    }
}
#endif
