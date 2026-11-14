#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
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
        // Rows of contiguous runs (the CUDA strided row copy): an NCHW channel slice with float4-aligned runs and
        // offsets, the same slice with runs and offsets off the float4 grid, and unit axes around a merged run.
        (new[] { 3, 6, 4, 4 }, new[] { 0, 2, 0, 0 }, new[] { 3, 3, 4, 4 }),
        (new[] { 3, 6, 3, 3 }, new[] { 0, 1, 0, 0 }, new[] { 3, 4, 3, 3 }),
        (new[] { 4, 1, 5, 6 }, new[] { 2, 0, 1, 0 }, new[] { 1, 1, 3, 6 }),
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
    public void Cuda_RectSlice_MatchesCpu()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsAvailable, "No CUDA driver.");
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend unavailable.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_RectSlice_RejectsStartOverflowAndReleasedBuffers_WithoutCorruptingContext()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsAvailable, "No CUDA driver.");
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend unavailable.");
        var slicer = (IRectSliceKernels)backend;

        using var fullBuf = backend.AllocateBuffer(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
        using var sliceBuf = backend.AllocateBuffer(new float[4]);

        // start + length wraps negative in 32-bit signed arithmetic; must reject before touching the driver.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            slicer.RectSlice(fullBuf, sliceBuf, new[] { 8 }, new[] { int.MaxValue }, new[] { 2 }, scatter: false));

        // Released full/slice buffers with non-zero baseOffset (TryCollapseToRows path) and general N-D path must
        // throw ObjectDisposedException instead of dereferencing IntPtr.Zero + baseOffset * sizeof(float).
        var releasedFull = backend.AllocateBuffer(new float[8]);
        releasedFull.Dispose();
        Assert.Throws<ObjectDisposedException>(() =>
            slicer.RectSlice(releasedFull, sliceBuf, new[] { 8 }, new[] { 3 }, new[] { 4 }, scatter: false));
        Assert.Throws<ObjectDisposedException>(() =>
            slicer.RectSlice(releasedFull, sliceBuf, new[] { 2, 2, 2 }, new[] { 0, 1, 0 }, new[] { 2, 1, 2 }, scatter: false));

        var releasedSlice = backend.AllocateBuffer(new float[4]);
        releasedSlice.Dispose();
        Assert.Throws<ObjectDisposedException>(() =>
            slicer.RectSlice(fullBuf, releasedSlice, new[] { 8 }, new[] { 2 }, new[] { 4 }, scatter: false));

        // Verify the shared primary CUDA context remains healthy after all rejected calls.
        slicer.RectSlice(fullBuf, sliceBuf, new[] { 8 }, new[] { 2 }, new[] { 4 }, scatter: false);
        Assert.Equal(new float[] { 3f, 4f, 5f, 6f }, backend.DownloadBuffer(sliceBuf));
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
    [SkippableFact]
    public void Metal_RectSlice_MatchesCpu()
    {
        // The constructor throws PlatformNotSupportedException off Apple platforms.
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal needs macOS or iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void WebGpu_RectSlice_MatchesCpu()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        Check(backend);
    }
}
#endif
