using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class ResidentFloatEmbeddingIdsTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public ResidentFloatEmbeddingIdsTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ResidentIds_UseRoundedLogicalValues_ForForwardAndBackward(bool partialView)
    {
        Skip.If(!_fixture.IsAvailable, "No direct GPU backend available.");
        var gpu = _fixture.Engine!;
        var backend = gpu.GetBackend()!;
        Skip.If(backend is not IResidentIndexBackend, "Backend cannot convert resident indices.");
        var storage = partialView
            ? new Tensor<float>(new[] { 9f, 9f, 2f, 5f }, new[] { 2, 2 })
            : new Tensor<float>(new[] { 1.75f, 2.5f, 3.5f, -1f, 10f }, new[] { 5 });
        var ids = partialView ? storage.Slice(1) : storage;
        int[] expectedIds = partialView ? new[] { 2, 5 } : new[] { 2, 2, 4, -1, 10 };
        var table = new Tensor<float>(Enumerable.Range(0, 10).Select(i => (float)(i * 10)).ToArray(), new[] { 10, 1 });
        using var inputBuffer = backend.AllocateBuffer(storage.ToArray());
        using var tableBuffer = backend.AllocateBuffer(table.ToArray());
        var output = new Tensor<float>(new[] { ids.Length, 1 });
        gpu.SuspendActivationEviction();
        using var capturePath = gpu.EnterCompiledCapturePath();
        try
        {
            gpu.BindResidentBuffer(storage, inputBuffer, backend);
            gpu.BindResidentBuffer(table, tableBuffer, backend);
            Assert.True(gpu.TryEmbeddingFromFloatIndicesResidentInto(output, table, ids, ids.Length, 1));
            Assert.Equal(expectedIds.Select(i => i >= 0 && i < 10 ? (float)(i * 10) : 0f), output.ToArray());
            var upstream = new Tensor<float>(Enumerable.Repeat(1f, ids.Length).ToArray(), new[] { ids.Length, 1 });
            using var gradBuffer = backend.AllocateBuffer(upstream.ToArray());
            gpu.BindResidentBuffer(upstream, gradBuffer, backend);
            var gradient = gpu.TryEmbeddingBackwardResident(upstream, ids, ids.Length, 10, 1, new[] { 10, 1 });
            Assert.NotNull(gradient);
            var expectedGradient = new float[10];
            foreach (int id in expectedIds.Where(i => i >= 0 && i < 10)) expectedGradient[id]++;
            Assert.Equal(expectedGradient, gradient!.ToArray());
        }
        finally
        {
            gpu.ResumeActivationEviction();
        }
    }
}
