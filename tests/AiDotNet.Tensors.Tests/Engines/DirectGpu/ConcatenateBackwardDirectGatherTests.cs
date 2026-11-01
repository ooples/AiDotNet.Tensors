using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// In a resident step ConcatenateBackward gathers each input's piece straight into that input's gradient buffer
/// (TryRectSliceInto) instead of a fresh TensorSlice device tensor, which only a finalizer freed: DenseNet-121's
/// channel concatenations held 1.9 GB of them.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ConcatenateBackwardDirectGatherTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ConcatenateBackwardDirectGatherTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(0, 2)]
    [InlineData(2, 4)]
    [InlineData(5, 1)]
    public void RectSliceInto_GathersTheChannelPiece(int channelStart, int channels)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");

        int[] shape = { 2, 6, 3, 4 };
        var data = new float[2 * 6 * 3 * 4];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.5f - 17f;
        var backend = gpu.PlacementBackend;
        Skip.If(backend is null, "No placement backend.");
        // ConcatenateBackward's gradOutput is device-resident; the gather reads it without an upload.
        var source = Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(data), shape);
        // The gradient accumulator is a resident buffer holding last step's values, which the gather overwrites.
        var stale = new float[2 * channels * 3 * 4];
        for (int i = 0; i < stale.Length; i++) stale[i] = 999f;
        var into = Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(stale), new[] { 2, channels, 3, 4 });
        var start = new[] { 0, channelStart, 0, 0 };
        var length = new[] { 2, channels, 3, 4 };

        // Outside a resident step the engine declines, so the caller keeps the TensorSlice path.
        Assert.False(gpu.TryRectSliceInto(source, start, length, into));

        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
            {
                Assert.True(gpu.ResidentStepActive);
                Assert.True(gpu.TryRectSliceInto(source, start, length, into));
            }
        }
        finally
        {
            gpu.ResumeActivationEviction();
        }

        var expected = new float[into.Length];
        int k = 0;
        for (int b = 0; b < 2; b++)
            for (int c = channelStart; c < channelStart + channels; c++)
                for (int s = 0; s < 12; s++)
                    expected[k++] = data[(b * 6 + c) * 12 + s];
        Assert.Equal(expected, into.ToArray());
    }

    [SkippableFact]
    public void RectSliceInto_RejectsAPieceOutsideTheSource()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");

        var source = new Tensor<float>(new[] { 2, 6, 3, 4 });
        var into = new Tensor<float>(new[] { 2, 3, 3, 4 });
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
                Assert.False(gpu.TryRectSliceInto(source, new[] { 0, 4, 0, 0 }, new[] { 2, 3, 3, 4 }, into));
        }
        finally
        {
            gpu.ResumeActivationEviction();
        }
    }
}
