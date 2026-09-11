using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

[Collection("EngineCurrentGlobalState")]
public sealed class TensorGpuCachePhysicalTests
{
    public enum InPlaceOperation
    {
        MultiplyScalar,
        AddCachedOperand,
        AddUncachedOperand,
    }

    public enum ExternalWrapperKind
    {
        Array,
        Vector,
        OffsetVector,
    }

    [Theory]
    [InlineData(ExternalWrapperKind.Array, false)]
    [InlineData(ExternalWrapperKind.Array, true)]
    [InlineData(ExternalWrapperKind.Vector, false)]
    [InlineData(ExternalWrapperKind.Vector, true)]
    [InlineData(ExternalWrapperKind.OffsetVector, false)]
    [InlineData(ExternalWrapperKind.OffsetVector, true)]
    public void IndependentExternalWrapperMutation_InvalidatesSharedArrayGpuSnapshot(
        ExternalWrapperKind kind, bool inference)
    {
        WithPhysicalGpu((gpu, _) =>
        {
            float[] values = { 1, 2, 3, 4 };
            using var source = new Tensor<float>(values, new[] { 4 });
            using Tensor<float> wrapper = kind switch
            {
                ExternalWrapperKind.Array => new Tensor<float>(values, new[] { 4 }),
                ExternalWrapperKind.Vector => new Tensor<float>(new[] { 4 }, Vector<float>.WrapMemory(values)),
                ExternalWrapperKind.OffsetVector => new Tensor<float>(new[] { 2 }, Vector<float>.WrapMemory(values.AsMemory(1, 2))),
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
            Assert.NotSame(source._storage, wrapper._storage);
            gpu.RegisterResidentParamBuffer(source);
            Assert.NotNull(source.TryGetGpuBuffer());
            int sourceVersion = source.Version;
            int wrapperVersion = wrapper.Version;
            int epoch = source.GpuCacheVersion;

            WithInference(inference, () => wrapper.CopyFromArray(kind == ExternalWrapperKind.OffsetVector
                ? new float[] { 20, 30 }
                : new float[] { 10, 20, 30, 40 }));

            Assert.Equal(sourceVersion, source.Version);
            if (inference) Assert.Equal(wrapperVersion, wrapper.Version);
            Assert.Equal(epoch + 1, source.GpuCacheVersion);
            Assert.Equal(source.GpuCacheVersion, wrapper.GpuCacheVersion);
            Assert.Equal(kind == ExternalWrapperKind.OffsetVector
                ? new float[] { 2, 40, 60, 8 }
                : new float[] { 20, 40, 60, 80 }, gpu.TensorMultiplyScalar(source, 2f).ToArray());
        });
    }

    [Fact]
    public void ExternalWrapperCreatedAfterGpuRefresh_InheritsCurrentArrayEpoch()
    {
        WithPhysicalGpu((gpu, _) =>
        {
            float[] values = { 1, 2, 3, 4 };
            using var source = new Tensor<float>(values, new[] { 4 });
            gpu.RegisterResidentParamBuffer(source);
            source.CopyFromArray(new float[] { 10, 20, 30, 40 });
            gpu.RegisterResidentParamBuffer(source);
            Assert.True(source.GpuCacheVersion > 0);

            using var wrapper = new Tensor<float>(values, new[] { 4 });
            Assert.Equal(source.GpuCacheVersion, wrapper.GpuCacheVersion);
            gpu.RegisterResidentParamBuffer(wrapper);
            Assert.Same(source.TryGetGpuBuffer(), wrapper.TryGetGpuBuffer());

            wrapper.CopyFromArray(new float[] { 100, 200, 300, 400 });
            Assert.Equal(new float[] { 200, 400, 600, 800 }, gpu.TensorMultiplyScalar(source, 2f).ToArray());
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CopyOnWriteDetach_CopiesButDoesNotShareExternalArrayEpoch(bool inference)
    {
        WithPhysicalGpu((gpu, _) =>
        {
            float[] values = { 1, 2, 3, 4 };
            using var source = new Tensor<float>(values, new[] { 4 });
            using var external = new Tensor<float>(values, new[] { 4 });
            gpu.RegisterResidentParamBuffer(source);
            int sourceEpoch = source.GpuCacheVersion;
            using var clone = (Tensor<float>)source.CloneShared();
            Assert.Same(source._storage, clone._storage);

            WithInference(inference, () => clone.CopyFromArray(new float[] { 10, 20, 30, 40 }));
            Assert.NotSame(source._storage, clone._storage);
            Assert.Equal(sourceEpoch, source.GpuCacheVersion);
            gpu.RegisterResidentParamBuffer(clone);
            int cloneEpoch = clone.GpuCacheVersion;

            WithInference(inference, () => external.CopyFromArray(new float[] { 5, 6, 7, 8 }));
            Assert.Equal(sourceEpoch + 1, source.GpuCacheVersion);
            Assert.Equal(cloneEpoch, clone.GpuCacheVersion);
            Assert.Equal(new float[] { 10, 12, 14, 16 }, gpu.TensorMultiplyScalar(source, 2f).ToArray());
            Assert.Equal(new float[] { 20, 40, 60, 80 }, gpu.TensorMultiplyScalar(clone, 2f).ToArray());
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ViewCreatedAfterHostMutation_PreservesStaleParentSnapshot(bool inference)
    {
        WithPhysicalGpu((gpu, backend) =>
        {
            using var source = Tensor<float>.FromGpuBuffer(
                backend, backend.AllocateBuffer(new float[] { 1, 2, 3, 4 }), new[] { 4 });
            Assert.True(source.IsGpuResident);
            Assert.Equal(new float[] { 1, 2, 3, 4 }, source.ToArray());
            int version = source.Version;

            WithInference(inference, () => source.CopyFromArray(new float[] { 10, 20, 30, 40 }));
            if (inference) Assert.Equal(version, source.Version);

            using Tensor<float> view = source.Reshape(new[] { 2, 2 });
            Assert.Equal(source._gpuBufferVersion, view._gpuBufferVersion);
            Assert.NotEqual(view.GpuCacheVersion, view._gpuBufferVersion);
            Assert.Equal(new float[] { 20, 40, 60, 80 }, gpu.TensorMultiplyScalar(view, 2f).ToArray());
        });
    }

    [Theory]
    [InlineData(InPlaceOperation.MultiplyScalar, false)]
    [InlineData(InPlaceOperation.MultiplyScalar, true)]
    [InlineData(InPlaceOperation.AddCachedOperand, false)]
    [InlineData(InPlaceOperation.AddCachedOperand, true)]
    [InlineData(InPlaceOperation.AddUncachedOperand, false)]
    [InlineData(InPlaceOperation.AddUncachedOperand, true)]
    public void InPlaceOperation_AfterHostMutationUsesCurrentValues(InPlaceOperation operation, bool inference)
    {
        WithPhysicalGpu((gpu, _) =>
        {
            using var target = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
            using var other = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 4 });
            gpu.RegisterResidentParamBuffer(target);
            Assert.NotNull(target.TryGetGpuBuffer());
            if (operation == InPlaceOperation.AddCachedOperand)
            {
                gpu.RegisterResidentParamBuffer(other);
                Assert.NotNull(other.TryGetGpuBuffer());
            }

            int version = target.Version;
            WithInference(inference, () =>
            {
                target.CopyFromArray(new float[] { 10, 20, 30, 40 });
                other.CopyFromArray(new float[] { 50, 60, 70, 80 });
            });
            if (inference) Assert.Equal(version, target.Version);

            if (operation == InPlaceOperation.MultiplyScalar)
            {
                ((IEngine)gpu).TensorMultiplyScalarInPlace(target, 2f);
                Assert.Equal(new float[] { 20, 40, 60, 80 }, target.ToArray());
            }
            else
            {
                gpu.TensorAddInPlace(target, other);
                Assert.Equal(new float[] { 60, 80, 100, 120 }, target.ToArray());
            }
        });
    }

    [Theory]
    [InlineData(InPlaceOperation.MultiplyScalar)]
    [InlineData(InPlaceOperation.AddCachedOperand)]
    [InlineData(InPlaceOperation.AddUncachedOperand)]
    public void InPlaceOperation_PreservesCurrentPersistentAllocation(InPlaceOperation operation)
    {
        WithPhysicalGpu((gpu, _) =>
        {
            using var target = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
            using var other = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 4 });
            gpu.RegisterResidentParamBuffer(target);
            var originalBuffer = target.TryGetGpuBuffer()
                ?? throw new InvalidOperationException("The physical GPU parameter cache was not populated.");
            if (operation == InPlaceOperation.AddCachedOperand)
                gpu.RegisterResidentParamBuffer(other);

            for (int iteration = 0; iteration < 2; iteration++)
            {
                if (operation == InPlaceOperation.MultiplyScalar)
                    ((IEngine)gpu).TensorMultiplyScalarInPlace(target, 2f);
                else
                    gpu.TensorAddInPlace(target, other);

                gpu.RegisterResidentParamBuffer(target);
                Assert.Same(originalBuffer, target.TryGetGpuBuffer());
                Assert.NotEqual(IntPtr.Zero, originalBuffer.Handle);
            }

            Assert.Equal(operation == InPlaceOperation.MultiplyScalar
                ? new float[] { 4, 8, 12, 16 }
                : new float[] { 11, 14, 17, 20 }, target.ToArray());
        });
    }

    private static void WithInference(bool enabled, Action action)
    {
        if (!enabled) { action(); return; }
        using (new InferenceModeScope<float>()) action();
    }

    private static void WithPhysicalGpu(Action<DirectGpuTensorEngine, IDirectGpuBackend> action)
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "A physical GPU was required, but no GPU backend initialized.");
        if (!gpu.IsGpuAvailable) return;
        var backend = gpu.TestBackend
            ?? throw new InvalidOperationException("GPU availability did not produce an initialized backend.");
        bool previousStrict = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try { action(gpu, backend); }
        finally { DirectGpuTensorEngine.ThrowOnGpuKernelFallback = previousStrict; }
    }
}
