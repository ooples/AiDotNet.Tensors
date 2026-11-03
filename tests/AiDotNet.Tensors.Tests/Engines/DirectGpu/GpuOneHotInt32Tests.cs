using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorOneHot on the GPU engine reads the int32 indices where they live, in one launch (#1664). It used to download
/// the indices and issue one single-element copy launch per index, so a MoE router's one-hot of a device TopK result
/// cost a host readback plus rows launches per call. Indices outside [0, depth) give a zero row, as on the CPU.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuOneHotInt32Tests
{
    public enum Placement { Host, ResidentRawInt32, ResidentNumeric }

    private static DirectGpuTensorEngine? TryGpu()
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) return gpu;
            gpu.Dispose();
            return null;
        }
        catch
        {
            return null;
        }
    }

    private static int[] Indices(int rows, int depth)
    {
        var random = new Random(1664 + rows);
        var ids = Enumerable.Range(0, rows).Select(_ => random.Next(depth)).ToArray();
        // Out-of-range rows: negative, exactly depth, far past depth.
        if (rows > 3) { ids[1] = -1; ids[2] = depth; ids[rows - 1] = int.MaxValue; }
        return ids;
    }

    private static Tensor<int> Place(DirectGpuTensorEngine gpu, int[] ids, int[] shape, Placement placement)
    {
        var backend = gpu.GetBackend() ?? throw new InvalidOperationException("GPU engine without a backend.");
        switch (placement)
        {
            case Placement.ResidentRawInt32:
                return Tensor<int>.FromGpuBuffer(backend, backend.AllocateIntBuffer(ids), shape,
                    ownsBuffer: true, bufferContainsRawInt32: true);
            case Placement.ResidentNumeric:
                return Tensor<int>.FromGpuBuffer(backend, backend.AllocateBuffer(ids.Select(i => (float)i).ToArray()), shape,
                    ownsBuffer: true);
            default:
                return new Tensor<int>(ids, shape);
        }
    }

    [SkippableTheory]
    [InlineData(Placement.Host, 4, 5)]
    [InlineData(Placement.Host, 1, 1)]
    [InlineData(Placement.ResidentRawInt32, 1024, 64)]
    [InlineData(Placement.ResidentRawInt32, 7, 3)]
    [InlineData(Placement.ResidentNumeric, 300, 17)]
    public void OneHot_MatchesTheCpu_ForEveryIndexPlacement(Placement placement, int rows, int depth)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU engine unavailable.");
        using var gpuScope = gpu;
        var ids = Indices(rows, depth);
        int[] shape = rows % 2 == 0 && rows > 2 ? new[] { 2, rows / 2 } : new[] { rows };

        var expected = new CpuEngine().TensorOneHot<float>(new Tensor<int>(ids, shape), depth);
        var actual = ((IEngine)gpu!).TensorOneHot<float>(Place(gpu!, ids, shape, placement), depth);

        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var want = expected.ToArray();
        var got = actual.ToArray();
        for (int i = 0; i < want.Length; i++)
            Assert.True(want[i] == got[i], $"element {i} (row {i / depth}, index {ids[i / depth]}): GPU {got[i]}, CPU {want[i]}");
    }

    [SkippableFact]
    public void OneHot_OfResidentIndices_ReadsNothingBack_AndIsOneLaunch()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU engine unavailable.");
        using var gpuScope = gpu;
        const int rows = 8 * 128 * 8, depth = 64;   // a MoE router's top-8 of 64 experts over B=8, S=128
        IEngine engine = gpu!;
        using (engine.TensorOneHot<float>(Place(gpu!, Indices(rows, depth), new[] { rows }, Placement.ResidentRawInt32), depth)) { }   // warm: kernel compile
        // Fresh indices: a host copy left by an earlier read would hide the download being measured.
        var indices = Place(gpu!, Indices(rows, depth), new[] { rows }, Placement.ResidentRawInt32);

        GpuLaunchProbe.Reset();
        GpuLaunchProbe.CaptureReadbackSites = true;
        try
        {
            using var oneHot = engine.TensorOneHot<float>(indices, depth);
            Assert.True(GpuLaunchProbe.Readbacks == 0,
                $"{GpuLaunchProbe.Readbacks} readbacks ({GpuLaunchProbe.ReadbackBytes:N0} bytes): " + string.Join("; ", GpuLaunchProbe.ReadbackSites));
            Assert.True(GpuLaunchProbe.Count <= 1, $"{GpuLaunchProbe.Count} launches for one one-hot of {rows} rows");
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = false;
        }
    }
}
