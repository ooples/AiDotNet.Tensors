// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("VulkanGlobalState")]
public sealed class ClampNaNAndResidentCopyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ClampNaNAndResidentCopyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    // 45 elements: the CPU clamp's 32-wide unrolled body, its 8-wide loop and its scalar tail all see special values.
    private static Tensor<float> SpecialValues()
    {
        var t = new Tensor<float>(new[] { 45 });
        for (int i = 0; i < t.Length; i++)
            t[i] = (i % 5) switch { 0 => float.NaN, 1 => float.PositiveInfinity, 2 => float.NegativeInfinity, 3 => 5f, _ => 0.25f };
        return t;
    }

    private static void AssertTorchClamp(float[] got, string engine)
    {
        for (int i = 0; i < got.Length; i++)
        {
            float expected = (i % 5) switch { 0 => float.NaN, 1 => 1f, 2 => -1f, 3 => 1f, _ => 0.25f };
            bool ok = float.IsNaN(expected) ? float.IsNaN(got[i]) : got[i] == expected;
            Assert.True(ok, $"{engine} clamp[{i}] = {got[i]}, expected {expected} (NaN must propagate, +/-Inf clamp)");
        }
    }

    /// <summary>
    /// torch.clamp propagates NaN. The CPU clamp turned NaN into the lower bound inside its AVX body (MAXPS returns
    /// its second operand on NaN) but kept it NaN in the scalar tail; the CUDA/HIP kernels turned it into the upper
    /// bound. Value-based gradient clipping then hid a NaN gradient from the optimizer's non-finite guard.
    /// </summary>
    [Fact]
    public void Clamp_PropagatesNaN_AndClampsInfinities_OnCpu()
        => AssertTorchClamp(new CpuEngine().TensorClamp(SpecialValues(), -1f, 1f).ToArray(), "CPU");

    [SkippableFact]
    public void Clamp_PropagatesNaN_AndClampsInfinities_OnGpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        AssertTorchClamp(gpu.TensorClamp(SpecialValues(), -1f, 1f).ToArray(), "GPU");
    }

    /// <summary>
    /// TensorCopy had no GPU implementation: into a device-resident destination it wrote a host array while the
    /// destination's bound device buffer -- what the fused optimizer kernels read through TryGetGpuBuffer -- kept
    /// the old values. The copy must land in that device buffer, and a host read must see it too.
    /// </summary>
    [SkippableFact]
    public void TensorCopy_IntoDeviceResidentTensor_UpdatesTheDeviceBuffer()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var engine = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = engine;
            var destination = GpuOptimizer.CreateStateTensor(new[] { 64 });   // device-resident zeros
            var deviceBuffer = destination.TryGetGpuBuffer();
            Assert.NotNull(deviceBuffer);
            var source = new Tensor<float>(new[] { 64 });
            for (int i = 0; i < source.Length; i++) source[i] = i * 0.5f;

            ((IEngine)engine).TensorCopy(source, destination);

            // Read the DEVICE buffer the optimizer kernels use, not the host copy.
            Assert.Same(deviceBuffer, destination.TryGetGpuBuffer());
            var onDevice = engine.GetBackend()!.DownloadBuffer(deviceBuffer!);
            for (int i = 0; i < source.Length; i++)
                Assert.True(onDevice[i] == i * 0.5f, $"device[{i}] = {onDevice[i]}, expected {i * 0.5f}");
            var host = destination.ToArray();
            for (int i = 0; i < source.Length; i++)
                Assert.True(host[i] == i * 0.5f, $"host[{i}] = {host[i]}, expected {i * 0.5f}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
