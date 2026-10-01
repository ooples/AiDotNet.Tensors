#if NET5_0_OR_GREATER
using System.Linq;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// A warmed-up training step of every network the PyTorch head-to-head trains stays on the device: no engine operation
/// moves data between host and device. The only transfers allowed are explicit uploads of host data the caller hands
/// in (the LSTM harness builds its zero initial states on the host each step). Measured on OpenCL before these fixes:
/// LSTM 114 crossings per step (slices under the tape took the host path), Transformer 22 (permute tables uploaded per
/// call, attention's batched matmuls of views ran on the host), ResNet-18 2 (adaptive-pool matrices per backward).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class NetworkTrainingStepResidencyTests
{
    private const int WarmupSteps = 5;
    private const string ExplicitUpload = "HostToDevice DirectGpuTensorEngine.UploadToGpu";

    [SkippableTheory]
    [InlineData("mlp")]
    [InlineData("cnn")]
    [InlineData("lstm")]
    [InlineData("transformer")]
    [InlineData("resnet18")]
    public void WarmTrainingStep_StaysOnTheDevice(string network)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        string root = PyTorchParityInventory.FindRepositoryRoot()
            ?? throw new System.InvalidOperationException("parity/ is missing from this checkout.");

        var crossings = HeadToHeadNetworkHarness.MeasureStepCrossings(root, network, gpu, WarmupSteps);

        var leaks = crossings.Where(kv => kv.Key != ExplicitUpload).OrderByDescending(kv => kv.Value).ToList();
        Assert.True(leaks.Count == 0,
            $"a warm {network} training step crossed the host/device boundary: " +
            string.Join("; ", leaks.Select(kv => $"{kv.Value}x {kv.Key}")));
    }
}
#endif
