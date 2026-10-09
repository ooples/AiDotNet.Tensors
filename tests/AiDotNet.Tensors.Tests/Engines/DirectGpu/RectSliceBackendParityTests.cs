#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every backend's <see cref="IRectSliceKernels"/> port must gather and scatter exactly like a CPU reference,
/// across ranks and windows, and leave the rest of the full tensor untouched on scatter.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class RectSliceBackendParityTests
{
    private static readonly (int[] Shape, int[] Start, int[] Length)[] Cases =
    {
        (new[] { 10 }, new[] { 3 }, new[] { 4 }),
        (new[] { 3, 5, 7, 4 }, new[] { 1, 0, 2, 1 }, new[] { 2, 5, 3, 2 }),
        (new[] { 2, 3, 2, 3, 4 }, new[] { 0, 0, 0, 0, 0 }, new[] { 2, 3, 2, 3, 4 }),
    };

    private static void Check(IDirectGpuBackend backend)
    {
        var slicer = (IRectSliceKernels)backend;
        foreach (var (shape, start, length) in Cases)
        {
            int full = 1, total = 1;
            foreach (int s in shape) full *= s;
            foreach (int l in length) total *= l;
            var data = new float[full];
            for (int i = 0; i < full; i++) data[i] = i * 0.5f + 1f;

            // CPU reference: index of each slice element in the full tensor, row-major.
            var map = new int[total];
            for (int idx = 0; idx < total; idx++)
            {
                int remaining = idx, offset = 0, stride = 1;
                for (int d = shape.Length - 1; d >= 0; d--)
                {
                    int c = remaining % length[d];
                    remaining /= length[d];
                    offset += (start[d] + c) * stride;
                    stride *= shape[d];
                }
                map[idx] = offset;
            }

            using var fullBuf = backend.AllocateBuffer(data);
            using var sliceBuf = backend.AllocateBuffer(new float[total]);
            slicer.RectSlice(fullBuf, sliceBuf, shape, start, length, scatter: false);
            var gathered = backend.DownloadBuffer(sliceBuf);
            for (int i = 0; i < total; i++)
                Assert.Equal(data[map[i]], gathered[i]);

            var sliceValues = new float[total];
            for (int i = 0; i < total; i++) sliceValues[i] = -100f - i;
            var background = new float[full];
            for (int i = 0; i < full; i++) background[i] = -7f;
            using var target = backend.AllocateBuffer(background);
            using var src = backend.AllocateBuffer(sliceValues);
            slicer.RectSlice(target, src, shape, start, length, scatter: true);
            var scattered = backend.DownloadBuffer(target);
            var expected = (float[])background.Clone();
            for (int i = 0; i < total; i++) expected[map[i]] = sliceValues[i];
            for (int i = 0; i < full; i++)
                Assert.Equal(expected[i], scattered[i]);
        }
    }

    [SkippableFact]
    public void OpenCl_RectSlice_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_RectSlice_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable,
            "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_RectSlice_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
