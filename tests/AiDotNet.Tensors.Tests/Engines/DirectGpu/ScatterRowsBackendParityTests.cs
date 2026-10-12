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
/// <see cref="IScatterRowsKernels.ScatterMeanRows"/> and <see cref="IScatterRowsKernels.ScatterMaxRows"/> on every
/// backend against a CPU reference (#1142): source [srcRows, inner] scattered by per-row group index into
/// [outRows, inner]. Group 2 receives no rows, so the empty-group contract (mean 0, max -inf) is checked too.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ScatterRowsBackendParityTests
{
    private static readonly (int SrcRows, int Inner, int OutRows)[] Shapes =
    {
        (1, 1, 3),
        (7, 5, 4),
        (300, 3, 5),   // more source rows than a 256-wide work-group
        (9, 257, 3),   // inner wider than a work-group, not a power of two
    };

    private static int[] Groups(int srcRows, int outRows)
    {
        var g = new int[srcRows];
        for (int i = 0; i < srcRows; i++)
        {
            int group = (i * 5 + 1) % outRows;
            g[i] = group == 2 && outRows > 2 ? (group + 1) % outRows : group;
        }
        return g;
    }

    private static float[] Values(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((i * 7919) % 53 - 26) * 0.25f;
        return v;
    }

    private static void Check(IDirectGpuBackend backend)
    {
        var rows = Assert.IsAssignableFrom<IScatterRowsKernels>(backend);
        foreach (var (srcRows, inner, outRows) in Shapes)
        {
            var source = Values(srcRows * inner);
            var groups = Groups(srcRows, outRows);
            var expectedMean = new double[outRows * inner];
            var expectedMax = new double[outRows * inner];
            var counts = new int[outRows];
            for (int i = 0; i < expectedMax.Length; i++) expectedMax[i] = double.NegativeInfinity;
            for (int r = 0; r < srcRows; r++)
            {
                counts[groups[r]]++;
                for (int c = 0; c < inner; c++)
                {
                    int o = groups[r] * inner + c;
                    expectedMean[o] += source[r * inner + c];
                    expectedMax[o] = Math.Max(expectedMax[o], source[r * inner + c]);
                }
            }
            for (int o = 0; o < expectedMean.Length; o++)
                if (counts[o / inner] > 0) expectedMean[o] /= counts[o / inner];

            using var src = backend.AllocateBuffer(source);
            using var idx = backend.AllocateIntBuffer(groups);
            using var mean = backend.AllocateBuffer(new float[outRows * inner]);
            using var max = backend.AllocateBuffer(new float[outRows * inner]);
            rows.ScatterMeanRows(src, idx, mean, srcRows, inner, outRows);
            rows.ScatterMaxRows(src, idx, max, srcRows, inner, outRows);
            var actualMean = backend.DownloadBuffer(mean);
            var actualMax = backend.DownloadBuffer(max);

            string shape = $"[{srcRows}x{inner}]->{outRows}";
            for (int o = 0; o < expectedMean.Length; o++)
            {
                Assert.True(Math.Abs(expectedMean[o] - actualMean[o]) <= 1e-5 * (1 + Math.Abs(expectedMean[o])),
                    $"{shape} mean[{o}]: expected {expectedMean[o]}, got {actualMean[o]}");
                Assert.True(double.IsNegativeInfinity(expectedMax[o])
                        ? float.IsNegativeInfinity(actualMax[o])
                        : expectedMax[o] == actualMax[o],
                    $"{shape} max[{o}]: expected {expectedMax[o]}, got {actualMax[o]}");
            }
        }
    }

    [SkippableFact]
    public void OpenCl_ScatterRows_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_ScatterRows_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable,
            "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_ScatterRows_MatchesCpu()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_ScatterRows_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
