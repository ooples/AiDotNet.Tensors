using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The declared deterministic scatter-add and segment reductions on the device, against a CPU loop in increasing source
/// order. Their destination axis is read only through the indirect shift, so a lowering that tiles it must give each
/// lane its own range test (#1441: the emitter once keyed every lane to lane 0's load).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredScatterAddRouteTests
{
    [SkippableTheory]
    [InlineData(4096, 256, 64)]
    [InlineData(1000, 37, 16)]
    [InlineData(300, 7, 33)]
    public void DriverOnlyDeclaredScatterAdd_MatchesTheOrderedSum(int count, int destinations, int features)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.DeclaredKernelsEnabled,
            "The declared route is off for this process or this architecture has no released codegen kernels.");

        var random = RandomHelper.CreateSeededRandom(1441);
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = i % 13 == 12 ? -1 : random.Next(destinations);
        var values = new float[count * features];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 4 - 2);
        var expected = new float[destinations * features];
        for (int i = 0; i < count; i++)
            if (ids[i] >= 0)
                for (int f = 0; f < features; f++) expected[ids[i] * features + f] += values[i * features + f];

        using var source = backend.AllocateBuffer(values);
        using var indices = backend.AllocateByteBuffer(count * sizeof(int));
        var idBytes = new byte[count * sizeof(int)];
        Buffer.BlockCopy(ids, 0, idBytes, 0, idBytes.Length);
        backend.UploadByteBuffer(indices, idBytes);
        using var output = backend.AllocateBuffer(destinations * features);

        long before = backend.DeclaredProgramLaunches;
        Assert.True(backend.TryDeclaredScatterAddDeterministic(source, indices, output, count, destinations, features),
            "The declared scatter-add declined on a device that supports it.");
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        var actual = backend.DownloadBuffer(output);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"index {i}: ordered sum {expected[i]:G9}, declared {actual[i]:G9}");
    }
}