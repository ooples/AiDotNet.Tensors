using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The declared strided gather and scatter (#1441) through CudaBackend's public entries, against the hand-written
/// strided_gather / strided_scatter they replace. Both only move values, so they must agree bit for bit, and the scatter
/// must leave every cell between the strided ones as the caller left it. Skips without a CUDA device; this is the
/// verification <see cref="CudaBackend.StridedDeclaredPromoted"/> waits for.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredStridedRouteTests
{
    [SkippableTheory]
    [InlineData(20000, 3, 0)]
    [InlineData(65536, 7, 5)]
    [InlineData(100000, 1, 11)]
    public void DriverOnlyDeclaredStrided_MatchesTheHandWrittenKernelsBitForBit(int count, int stride, int offset)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = CudaBackend.CreateOrThrow();
        int length = offset + (count - 1) * stride + 1 + 3;   // a tail past the last strided element
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1441);
        var wide = new float[length];
        var narrow = new float[count];
        for (int i = 0; i < wide.Length; i++) wide[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < narrow.Length; i++) narrow[i] = (float)(random.NextDouble() * 2 - 1);

        using var wideBuffer = backend.AllocateBuffer(wide);
        using var narrowBuffer = backend.AllocateBuffer(narrow);
        using var gatheredHand = backend.AllocateBuffer(count);
        using var gatheredDeclared = backend.AllocateBuffer(count);
        using var scatteredHand = backend.AllocateBuffer(wide);
        using var scatteredDeclared = backend.AllocateBuffer(wide);

        backend.DeclaredStridedSwitch = false;
        backend.StridedGather(wideBuffer, gatheredHand, offset, stride, count);
        backend.StridedScatter(narrowBuffer, scatteredHand, offset, stride, count);

        backend.DeclaredStridedSwitch = true;
        long before = backend.DeclaredProgramLaunches;
        backend.StridedGather(wideBuffer, gatheredDeclared, offset, stride, count);
        backend.StridedScatter(narrowBuffer, scatteredDeclared, offset, stride, count);
        Skip.If(backend.DeclaredProgramLaunches == before,
            "The declared route did not run on this device (switched off or the architecture has no released kernels).");
        Assert.Equal(before + 2, backend.DeclaredProgramLaunches);

        Assert.Equal(backend.DownloadBuffer(gatheredHand), backend.DownloadBuffer(gatheredDeclared));
        Assert.Equal(backend.DownloadBuffer(scatteredHand), backend.DownloadBuffer(scatteredDeclared));
    }
}
