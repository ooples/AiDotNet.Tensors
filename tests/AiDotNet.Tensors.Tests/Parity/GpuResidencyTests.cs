#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// GPU device residency (issue #1058): every host/device crossing is counted, so a GPU step that silently
/// leaves the device is visible and gated.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuResidencyTests
{
    private readonly ITestOutputHelper _output;

    public GpuResidencyTests(ITestOutputHelper output) => _output = output;

    private static DirectGpuTensorEngine RequireGpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) { Skip.If(true, $"DEFERRED: no GPU backend ({ex.GetType().Name})."); throw; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "DEFERRED: no GPU available."); }
        return gpu;
    }

    /// <summary>
    /// A device-side write made after the host has read the tensor must reach the next host read. The pending
    /// download was re-registered under the vector while the read path checked only the backing array, so a
    /// GPU optimizer step on an already-read weight left the host (and every CPU-fallback op) on the old values:
    /// the head-to-head CNN trained on stale conv weights and its loss drifted from PyTorch's.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeviceWriteAfterHostRead_ReachesTheHost(bool readBeforeWrite)
    {
        using var gpu = RequireGpu();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var shape = new[] { 4, 1, 3, 3 };
            var p = gpu.UploadToGpu(new Tensor<float>(Enumerable.Repeat(1f, 36).ToArray(), shape), GpuTensorRole.General);
            var g = gpu.UploadToGpu(new Tensor<float>(Enumerable.Repeat(1f, 36).ToArray(), shape), GpuTensorRole.General);
            if (readBeforeWrite) Assert.Equal(1f, p.AsSpan()[0]);

            Assert.True(GpuOptimizer.TrySgdStep(p, g, 0.25f), "The device-side SGD step was refused.");

            Assert.All(p.AsSpan().ToArray(), v => Assert.Equal(0.75f, v));
            Assert.Equal(0.75f, p.GetFlat(35));
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
    /// <summary>The scope sees a real upload and a real download, with their byte counts and backend.</summary>
    [SkippableFact]
    public void Scope_CountsUploadsAndDownloadsWithBytes()
    {
        using var gpu = RequireGpu();
        var host = new Tensor<float>(Enumerable.Range(0, 1024).Select(i => (float)i).ToArray(), new[] { 1024 });

        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
        {
            var resident = gpu.UploadToGpu(host, GpuTensorRole.General);
            var back = resident.Contiguous().AsSpan().ToArray();
            Assert.Equal(1023f, back[1023]);
        }

        foreach (var e in scope.Events) _output.WriteLine($"  {e.Kind,-13} {e.Bytes,8} B  {e.Backend,-7} {e.Operation}");
        Assert.True(scope.Uploads >= 1, "The upload of a 1024-element tensor was not counted.");
        Assert.True(scope.BytesUploaded >= 1024 * sizeof(float), $"Upload bytes {scope.BytesUploaded} are below the 4096 the tensor holds.");
        Assert.True(scope.Downloads >= 1, "Reading the resident tensor back on the host was not counted as a download.");
        Assert.All(scope.Events, e => Assert.False(string.IsNullOrEmpty(e.Backend)));
    }

    /// <summary>Transfers made outside a scope, or on another thread, are not attributed to it.</summary>
    [SkippableFact]
    public void Scope_IgnoresTransfersOutsideIt()
    {
        using var gpu = RequireGpu();
        var host = new Tensor<float>(new float[256], new[] { 256 });
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin()) { }
        gpu.UploadToGpu(host, GpuTensorRole.General);
        Assert.Empty(scope.Events);
    }

    private const string BaselineFile = "parity/residency-baseline.json";

    /// <summary>
    /// Ratchet (GPU lane): in a warmed-up MLP training step, no operation may cross the host/device boundary more
    /// often than recorded for its backend, and no new operation may start. Per operation rather than a total,
    /// because one new leak can hide behind one that disappears: an added host read of the output was counted,
    /// but it also spared a later cache invalidation its download, so the total stayed at 35.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "PyTorchParityGpu")]
    public void MlpTrainingStep_TransfersDoNotRegress()
    {
        var (scope, report) = RunMlpTrainingStep();
        string backend = scope.Events.FirstOrDefault().Backend ?? "unknown";
        var measured = CrossingsByOperation(scope);
        string root = PyTorchParityInventory.FindRepositoryRoot()
            ?? throw new InvalidOperationException("parity/ is missing from this checkout.");
        using var doc = System.Text.Json.JsonDocument.Parse(System.IO.File.ReadAllText(System.IO.Path.Combine(root, BaselineFile)));
        bool recorded = doc.RootElement.GetProperty("mlp").TryGetProperty(backend, out var baselineElement);
        string snapshot = "{ " + string.Join(", ", measured.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"\"{kv.Key}\": {kv.Value}")) + " }";
        // An unmeasured backend is not a passing ratchet: fail with the snapshot to record. A step that produced no
        // events at all reports backend "unknown", which has no baseline either, so a dead probe fails here too.
        Assert.True(recorded, $"No residency baseline for mlp on {backend}. Measure it on that backend and add " +
                              $"\"{backend}\": {snapshot} under \"mlp\" in {BaselineFile}.{Environment.NewLine}{report}");

        var baseline = baselineElement.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal);
        var rises = measured
            .Where(kv => kv.Value > (baseline.TryGetValue(kv.Key, out int allowed) ? allowed : 0))
            .Select(kv => $"  {kv.Key}: {kv.Value} (recorded {(baseline.TryGetValue(kv.Key, out int b) ? b : 0)})")
            .ToList();
        Assert.True(rises.Count == 0,
            $"mlp on {backend}: these operations now cross the host/device boundary more often than recorded:" +
            Environment.NewLine + string.Join(Environment.NewLine, rises) + Environment.NewLine + report);

        var falls = baseline.Where(kv => (measured.TryGetValue(kv.Key, out int m) ? m : 0) < kv.Value).Select(kv => kv.Key).ToList();
        if (falls.Count > 0)
            _output.WriteLine($"IMPROVED: {string.Join(", ", falls)}. Record {snapshot} in {BaselineFile} to lock it in.");
    }

    private static Dictionary<string, int> CrossingsByOperation(GpuResidencyScope scope)
        => scope.Events
            .Where(e => e.Kind != GpuTransferKind.Synchronize)
            .GroupBy(e => $"{e.Kind} {e.Operation ?? "<outside the engine>"}")
            .ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);

    /// <summary>
    /// Parity (GPU lane, nightly): a training step that stays resident moves no data between host and device
    /// beyond the batch it is given, so every reported operation is residency work still to do.
    /// </summary>
    [SkippableFact]
    // The absolute target, reported by run-gpu.ps1 but not gating it: the per-operation ratchet above is the gate
    // until a step reaches zero crossings.
    [Trait("Category", "PyTorchParityGpuTarget")]
    public void MlpTrainingStep_StaysResident()
    {
        var (scope, report) = RunMlpTrainingStep();
        int crossings = scope.Uploads + scope.Downloads;
        Assert.True(crossings == 0,
            $"A warmed-up MLP training step on the GPU crossed the host/device boundary {crossings} time(s) " +
            $"({scope.BytesUploaded + scope.BytesDownloaded:N0} bytes). Each operation below leaves the device:" +
            Environment.NewLine + report);
    }

    /// <summary>
    /// Runs warmed-up MLP training steps on the GPU engine and records one step's crossings, grouped by the engine
    /// operation that caused each.
    /// </summary>
    private (GpuResidencyScope Scope, string Report) RunMlpTrainingStep()
    {
        using var gpu = RequireGpu();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var rng = new Random(1058);
            Tensor<float> Rand(params int[] shape)
            {
                int n = shape.Aggregate(1, (a, b) => a * b);
                var data = new float[n];
                for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
                return new Tensor<float>(data, shape);
            }

            var weights = new[] { Rand(784, 512), Rand(512, 256), Rand(256, 10) };
            var biases = new[] { Rand(512), Rand(256), Rand(10) };
            var x = gpu.UploadToGpu(Rand(128, 784), GpuTensorRole.General);
            var y = gpu.UploadToGpu(Rand(128, 10), GpuTensorRole.General);
            for (int i = 0; i < 3; i++)
            {
                weights[i] = gpu.UploadToGpu(weights[i], GpuTensorRole.General);
                biases[i] = gpu.UploadToGpu(biases[i], GpuTensorRole.General);
            }

            var sources = weights.Concat(biases).ToArray();
            IEngine engine = gpu;
            void Step()
            {
                using var tape = new GradientTape<float>();
                var h = x;
                for (int i = 0; i < 3; i++)
                    h = engine.FusedLinear(h, weights[i], biases[i], i < 2 ? FusedActivationType.ReLU : FusedActivationType.None);
                var loss = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(h, y)), new[] { 0, 1 }, keepDims: false);
                var grads = tape.ComputeGradients(loss, sources);
                Assert.Equal(sources.Length, grads.Count);
                // A training step includes the update: the device-side SGD kernel, so the measured crossings cover
                // forward, backward and optimizer exactly as the head-to-head CUDA step runs them.
                foreach (var parameter in sources)
                    Assert.True(GpuOptimizer.TrySgdStep(parameter, grads[parameter], 0.01f),
                        $"The device-side SGD step was refused for a [{string.Join(", ", parameter.Shape.ToArray())}] parameter " +
                        $"(parameter on device: {parameter.TryGetGpuBuffer() is not null}, gradient on device: " +
                        $"{grads[parameter].TryGetGpuBuffer() is not null}).");
            }

            for (int i = 0; i < 5; i++) Step();

            GpuResidencyScope scope;
            using (scope = GpuResidencyScope.Begin(captureOperations: true)) Step();

            var lines = scope.Events
                .GroupBy(e => (e.Kind, Op: e.Operation ?? "<outside the engine>"))
                .OrderByDescending(g => g.Sum(e => e.Bytes))
                .Select(g => $"  {g.Key.Kind,-13} x{g.Count(),-3} {g.Sum(e => e.Bytes),12:N0} B  {g.Key.Op}")
                .ToList();
            string report = $"MLP training step on {scope.Events.FirstOrDefault().Backend ?? "GPU"}: " +
                            $"{scope.Uploads} upload(s) {scope.BytesUploaded:N0} B, {scope.Downloads} download(s) {scope.BytesDownloaded:N0} B, " +
                            $"{scope.Synchronizations} sync(s)" + Environment.NewLine + string.Join(Environment.NewLine, lines);
            _output.WriteLine(report);
            return (scope, report);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
#endif
