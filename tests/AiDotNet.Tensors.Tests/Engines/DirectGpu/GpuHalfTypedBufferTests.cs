// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The FP16 kernels over typed <see cref="GpuElementType.Float16"/> buffers: a buffer allocated and transferred as
/// halves is what the half element-wise ops and the half matrix multiply compute on, as the capability table says.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuHalfTypedBufferTests
{
    private static IDirectGpuBackend? Device()
    {
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu || gpu.GetBackend() is not { } backend) return null;
        if (!backend.SupportsElementType(GpuElementType.Float16)) return null;
        return backend is IGpuFp16ElementwiseBackend { SupportsFp16NativeOps: true } ? backend : null;
    }

    private static Half[] Values(int length, int seed)
    {
        var values = new Half[length];
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < length; i++) values[i] = (Half)(float)((random.NextDouble() - 0.5) * 64.0);
        return values;
    }

    [SkippableFact]
    public void Half_add_and_relu_compute_on_typed_buffers_and_equal_the_host()
    {
        var backend = Device();
        Skip.If(backend is null, "No device with half storage and FP16 kernels on this host.");
        var fp16 = (IGpuFp16ElementwiseBackend)backend;
        Assert.True(backend.SupportsOperation(GpuOperationFamily.Elementwise, GpuElementType.Float16));
        foreach (int size in new[] { 1, 255, 257, 4097 })
        {
            var left = Values(size, 1);
            var right = Values(size, 2);
            using var a = backend.AllocateBuffer(size, GpuElementType.Float16);
            using var b = backend.AllocateBuffer(size, GpuElementType.Float16);
            using var output = backend.AllocateBuffer(size, GpuElementType.Float16);
            Assert.Equal(size, a.Size);
            backend.UploadBuffer<Half>(a, left);
            backend.UploadBuffer<Half>(b, right);

            fp16.Fp16Add(a, b, output, size);
            var actual = new Half[size];
            backend.DownloadBuffer<Half>(output, actual);
            // The sum of two halves is exact in float; the store rounds it to half, as the host conversion does.
            for (int i = 0; i < size; i++)
                Assert.True((Half)((float)left[i] + (float)right[i]) == actual[i], $"add at {i} of {size}: {left[i]} + {right[i]} gave {actual[i]}.");

            fp16.Fp16Relu(a, output, size);
            backend.DownloadBuffer<Half>(output, actual);
            for (int i = 0; i < size; i++)
                Assert.True(((float)left[i] > 0f ? left[i] : (Half)0f) == actual[i], $"relu at {i} of {size}: {left[i]} gave {actual[i]}.");

            // A float read of a half buffer is refused rather than returning half the bytes as floats.
            Assert.Throws<InvalidOperationException>(() => backend.DownloadBuffer(output));
        }
    }

    [SkippableFact]
    public void Half_matrix_multiply_computes_on_typed_buffers()
    {
        var backend = Device();
        Skip.If(backend is null, "No device with half storage and FP16 kernels on this host.");
        Skip.If(backend is not IGpuHalfPrecisionBackend { SupportsHgemm: true }, "No half matrix multiply on this device.");
        var half = (IGpuHalfPrecisionBackend)backend;
        Assert.True(backend.SupportsOperation(GpuOperationFamily.MatrixMultiply, GpuElementType.Float16));
        const int M = 5, K = 7, N = 3;
        // Small integers: every product and partial sum is exact in half, whatever order the device accumulates in.
        var left = new Half[M * K];
        var right = new Half[K * N];
        for (int i = 0; i < left.Length; i++) left[i] = (Half)(float)((i * 5 % 7) - 3);
        for (int i = 0; i < right.Length; i++) right[i] = (Half)(float)((i * 3 % 5) - 2);
        using var a = backend.AllocateBuffer(left.Length, GpuElementType.Float16);
        using var b = backend.AllocateBuffer(right.Length, GpuElementType.Float16);
        using var c = backend.AllocateBuffer(M * N, GpuElementType.Float16);
        backend.UploadBuffer<Half>(a, left);
        backend.UploadBuffer<Half>(b, right);
        half.Hgemm(a, b, c, M, N, K);
        var actual = new Half[M * N];
        backend.DownloadBuffer<Half>(c, actual);
        for (int row = 0; row < M; row++)
        {
            for (int column = 0; column < N; column++)
            {
                float expected = 0f;
                for (int j = 0; j < K; j++) expected += (float)left[row * K + j] * (float)right[j * N + column];
                Assert.Equal(expected, (float)actual[row * N + column]);
            }
        }
    }
}

#endif
