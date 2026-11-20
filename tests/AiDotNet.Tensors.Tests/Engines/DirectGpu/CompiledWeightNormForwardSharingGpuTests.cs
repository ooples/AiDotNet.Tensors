using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledWeightNormForwardSharingGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public CompiledWeightNormForwardSharingGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Cuda_shared_native_storage_survives_replays_and_unused_pool_release(bool secondNoGrad)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        var engine = _fixture.Engine ?? throw new InvalidOperationException();
        Skip.If(engine.GetBackend() is not CudaBackend, "CUDA runtime proof.");
        var baseline = WeightNormSharingTestGraph.Run(engine, false, secondNoGrad);
        var shared = WeightNormSharingTestGraph.Run(engine, true, secondNoGrad);
        WeightNormSharingTestGraph.AssertEquivalent(baseline, shared, expectSharing: true);
    }

    [SkippableFact]
    public void Whole_vector_native_alias_keeps_ownership_after_both_producer_wrappers_are_disposed()
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        var engine = _fixture.Engine ?? throw new InvalidOperationException();
        Skip.If(engine.GetBackend() is not CudaBackend, "CUDA runtime proof.");
        IEngine previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            float[] original = { .5f, -.75f, .125f, .375f, -.625f };
            using var direction = new Tensor<float>((float[])original.Clone(), new[] { 1, 5 });
            using var gain = new Tensor<float>(new[] { .75f }, new[] { 1 });
            engine.EnsureResidentInput(direction);
            engine.EnsureResidentInput(gain);
            using var first = new Tensor<float>(new[] { 1, 5 });
            using var second = new Tensor<float>(new[] { 1, 5 });
            var steps = new List<CompiledStep<float>> { WeightNormSharingFixture.Step(first, direction, gain),
                WeightNormSharingFixture.Step(second, direction, gain) };
            Assert.Equal(1, WeightNormSharingFixture.Apply(steps, direction, gain, new Tensor<float>(new[] { 1 }), new(), engine));
            foreach (var step in steps) step.Execute(engine, step.OutputBuffer);
            Assert.NotNull(first._gpuBuffer);
            Assert.Same(first._gpuBuffer, second._gpuBuffer);
            IntPtr handle = first._gpuBuffer!.Handle;
            Assert.NotEqual(IntPtr.Zero, handle);
            using var view = second.Reshape(5);
            first.Dispose();
            second.Dispose();
            using var unrelated = engine.TensorMultiplyScalar(direction, 2f);
            _ = unrelated.ToArray();
            engine.ReleaseUnusedDeviceMemory();
            Assert.Equal(handle, view._gpuBuffer!.Handle);
            float[] actual = view.ToArray();
            double squares = .03125;
            foreach (float value in original) squares += (double)value * value;
            for (int index = 0; index < actual.Length; ++index)
                Assert.InRange(Math.Abs(actual[index] - .75 * original[index] / Math.Sqrt(squares)), 0, 2e-6);
            Assert.Equal(original, direction.ToArray());
            Assert.Equal(.75f, gain[0]);
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
