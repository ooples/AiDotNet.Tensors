using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class ResidentMirrorViewLifetimeTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public ResidentMirrorViewLifetimeTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (Tensor<float> View, WeakReference<Tensor<float>> Producer) MakeView(
        DirectGpuTensorEngine gpu, bool enrolViewControl)
    {
        var input = new Tensor<float>(new[] { 1f, -2f, 3f, -4f, 5f, -6f }, new[] { 6 });
        var producer = gpu.TensorMultiplyScalar(input, 2f);
        var contribution = gpu.TensorMultiplyScalar(input, .5f);
        ((IEngine)gpu).TensorAddInPlace(producer, contribution);
        Assert.NotNull(producer._gpuMaterializerCallback);
        Assert.Contains("ResidentHostMirror", producer._gpuMaterializerCallback!.Target!.GetType().FullName!);
        var view = gpu.Reshape(producer, new[] { 2, 3 });
        Assert.Same(producer.DataVector, view.DataVector);
        if (enrolViewControl) Assert.NotNull(ResidentHostMirror.Attach(view));
        return (view, new WeakReference<Tensor<float>>(producer));
    }

    [SkippableTheory]
    [InlineData(false)] [InlineData(true)]
    public void Retained_view_downloads_native_value_after_original_mirror_member_is_collected(bool enrolViewControl)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException();
        var result = MakeView(gpu, enrolViewControl);
        using var view = result.View;
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        Assert.False(result.Producer.TryGetTarget(out _), "The original producer must be collected for this test.");
        var buffer = view._gpuBuffer ?? throw new InvalidOperationException("Retained view lost native buffer.");
        var backend = view._gpuBackend ?? throw new InvalidOperationException("Retained view lost native backend.");
        float[] expected = { 2.5f, -5f, 7.5f, -10f, 12.5f, -15f };
        // This independent native read does not invoke or repair the host materializer.
        var native = backend.DownloadBuffer(buffer);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], native[i]);
        var actual = view.ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(actual[i] == expected[i],
                $"enrolViewControl={enrolViewControl}, index={i}: live native={native[i]}, host materialized={actual[i]}");
        GC.KeepAlive(view);
    }

    [SkippableFact]
    public void Missing_mirror_reader_does_not_publish_an_unwritten_host_payload()
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException();
        var result = MakeView(gpu, enrolViewControl: false);
        using var view = result.View;
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        Assert.False(result.Producer.TryGetTarget(out _));
        ResidentHostMirror.Detach(view);
        Assert.Null(view.GetBackingArrayForCacheLookupUnsafe());
        Assert.True(HostSync.IsPending(view.DataVector));

        var error = Assert.Throws<InvalidOperationException>(() => view.ToArray());

        Assert.Contains("no live reader", error.Message);
        Assert.True(HostSync.IsPending(view.DataVector));
        Assert.Throws<InvalidOperationException>(() => view.ToArray());
        // A failed materialization must retain the pending device value so a valid
        // reader can be re-armed without treating an allocated host shadow as current.
        Assert.True(ResidentHostMirror.ArmDownload(view));
        Assert.Equal(new[] { 2.5f, -5f, 7.5f, -10f, 12.5f, -15f }, view.ToArray());
    }
}
