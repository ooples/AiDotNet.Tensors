using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The declared row-wise scatter mean and softmax (#1441) through CudaBackend's public entries, against the hand-written
/// resident_scatter_mean_rows_counts / resident_scatter_softmax_rows they replace. The counts must match exactly; the
/// means and the softmax to 1e-5 relative, since the declared reduction need not add in the hand-written kernel's
/// sequential order and the softmax multiplies by a reciprocal where that divides. Skips only where the declared
/// route is unavailable by design; this is the verification <see cref="CudaBackend.ScatterRowsDeclaredPromoted"/>
/// waits for.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredScatterRowsRouteTests
{
    [SkippableTheory]
    [InlineData(1000, 16, 37)]
    [InlineData(4096, 8, 256)]
    [InlineData(300, 33, 7)]
    public void DriverOnlyDeclaredScatterRows_MatchTheHandWrittenKernels(int sourceRows, int innerSize, int groups)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = CudaBackend.CreateOrThrow();
        backend.DeclaredScatterRowsSwitch = true;
        Skip.IfNot(backend.DeclaredKernelsEnabled,
            "The declared route is off for this process or this architecture has no released codegen kernels.");

        var random = RandomHelper.CreateSeededRandom(1441);
        var ids = new int[sourceRows];
        for (int i = 0; i < sourceRows; i++) ids[i] = i % 11 == 10 ? -1 : random.Next(groups - 1);   // the last group stays empty
        var values = new float[sourceRows * innerSize];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 6 - 3);
        // A row outside the groups writes 0 for the softmax whatever it holds, +Inf and an overflowing value included.
        for (int i = 0; i < sourceRows; i++)
            if (ids[i] < 0) { values[i * innerSize] = float.PositiveInfinity; values[i * innerSize + innerSize - 1] = 500f; }

        using var source = backend.AllocateBuffer(values);
        using var indices = backend.AllocateByteBuffer(ids.Length * sizeof(int));
        var idBytes = new byte[ids.Length * sizeof(int)];
        Buffer.BlockCopy(ids, 0, idBytes, 0, idBytes.Length);
        backend.UploadByteBuffer(indices, idBytes);
        using var meanHand = backend.AllocateBuffer(groups * innerSize);
        using var meanDeclared = backend.AllocateBuffer(groups * innerSize);
        using var countsHand = backend.AllocateBuffer(groups);
        using var countsDeclared = backend.AllocateBuffer(groups);
        using var softmaxHand = backend.AllocateBuffer(sourceRows * innerSize);
        using var softmaxDeclared = backend.AllocateBuffer(sourceRows * innerSize);

        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterMeanRowsWithCounts(source, indices, meanHand, countsHand, sourceRows, innerSize, groups);
        backend.ScatterSoftmaxRows(source, indices, softmaxHand, sourceRows, innerSize, groups);

        backend.DeclaredScatterRowsSwitch = true;
        long before = backend.DeclaredProgramLaunches;
        backend.ScatterMeanRowsWithCounts(source, indices, meanDeclared, countsDeclared, sourceRows, innerSize, groups);
        backend.ScatterSoftmaxRows(source, indices, softmaxDeclared, sourceRows, innerSize, groups);
        Assert.True(backend.DeclaredProgramLaunches == before + 2,
            "The declared scatter-rows route declined on a device that supports it: " +
            (backend.DeclaredProgramLaunches - before) + " of 2 calls ran it.");

        Assert.Equal(backend.DownloadBuffer(countsHand), backend.DownloadBuffer(countsDeclared));
        AssertClose(backend.DownloadBuffer(meanHand), backend.DownloadBuffer(meanDeclared));
        var softmax = backend.DownloadBuffer(softmaxDeclared);
        AssertClose(backend.DownloadBuffer(softmaxHand), softmax);
        for (int i = 0; i < sourceRows; i++)
            if (ids[i] < 0)
                for (int j = 0; j < innerSize; j++) Assert.Equal(0f, softmax[i * innerSize + j]);
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
    }
}
