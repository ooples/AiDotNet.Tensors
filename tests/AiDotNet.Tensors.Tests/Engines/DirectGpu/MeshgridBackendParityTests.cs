#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IMeshgridBackend.Meshgrid2DPair"/> on every backend against torch.meshgrid semantics (#922, #1142), for
/// both indexing modes and the issue's (256, 256) and (1024, 256) targets plus an odd small shape.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MeshgridBackendParityTests
{
    private static readonly (int N0, int N1)[] Shapes = { (3, 5), (256, 256), (1024, 256) };

    private static void Check(IDirectGpuBackend backend)
    {
        var mesh = Assert.IsAssignableFrom<IMeshgridBackend>(backend);
        foreach (var (n0, n1) in Shapes)
        {
            var s0 = new float[n0];
            var s1 = new float[n1];
            for (int i = 0; i < n0; i++) s0[i] = i * 0.5f - 7f;
            for (int j = 0; j < n1; j++) s1[j] = 1000f + j;
            foreach (bool xy in new[] { false, true })
            {
                using var b0 = backend.AllocateBuffer(s0);
                using var b1 = backend.AllocateBuffer(s1);
                using var o0 = backend.AllocateBuffer(n0 * n1);
                using var o1 = backend.AllocateBuffer(n0 * n1);
                mesh.Meshgrid2DPair(b0, b1, o0, o1, n0, n1, xy);
                var g0 = backend.DownloadBuffer(o0);
                var g1 = backend.DownloadBuffer(o1);
                int rows = xy ? n1 : n0, cols = xy ? n0 : n1;
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                    {
                        int k = r * cols + c;
                        float e0 = xy ? s0[c] : s0[r];
                        float e1 = xy ? s1[r] : s1[c];
                        Assert.True(g0[k] == e0 && g1[k] == e1,
                            $"[{n0}x{n1}] {(xy ? "xy" : "ij")} ({r},{c}): got ({g0[k]}, {g1[k]}), expected ({e0}, {e1})");
                    }
            }
        }
    }

    [SkippableFact]
    public void OpenCl_Meshgrid2DPair_MatchesTorch()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_Meshgrid2DPair_MatchesTorch()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_Meshgrid2DPair_MatchesTorch()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_Meshgrid2DPair_MatchesTorch()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
