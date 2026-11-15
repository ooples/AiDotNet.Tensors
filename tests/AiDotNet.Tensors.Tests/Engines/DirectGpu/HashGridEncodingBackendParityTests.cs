#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IInstantNgpBackend"/> of every backend against the CPU multiresolution hash encoding, forward and
/// backward, over levels that are indexed densely and levels that are hashed. A first check pins three points to
/// grid vertices of a hashed level whose table holds each entry's own index, so the output is the tiny-cuda-nn
/// coherent prime hash itself (hand-computed), not merely agreement with the CPU.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HashGridEncodingBackendParityTests
{
    // 256 points so the CUDA direct-PTX path's shape gate can take it; 4 levels from N_min = 4, b = 2: scales
    // 3, 7, 15, 31 and 4, 8, 16, 32 vertices per axis, so the first level (64 <= 256 entries) is dense, the rest hashed.
    private const int Points = 256, Levels = 4, Features = 2, TableSize = 256, BaseResolution = 4;
    private const double PerLevelScale = 2.0;

    private static float[] Random(int seed, int length, double low, double high)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(low + rng.NextDouble() * (high - low));
        return values;
    }

    private static IInstantNgpBackend AsInstantNgp(IDirectGpuBackend backend)
    {
        Skip.IfNot(backend is IInstantNgpBackend, $"{backend.GetType().Name} has no Instant-NGP kernels.");
        return (IInstantNgpBackend)backend;
    }

    private static void AssertClose(float[] expected, float[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{what}[{i}]: cpu {expected[i]}, gpu {actual[i]} (tol {tolerance})");
    }

    // Vertices (1,2,3), (7,11,13), (2,1,1) of a level with scale 64 (N_min = 65; 65^3 > 4096, hashed):
    // (x ^ y*2654435761 ^ z*805459861) mod 4096 = 1500, 2061, 3622.
    private static void CheckKnownHashIndices(IDirectGpuBackend backend, IInstantNgpBackend ngp)
    {
        const float scale = 64f;
        int[][] vertices = { new[] { 1, 2, 3 }, new[] { 7, 11, 13 }, new[] { 2, 1, 1 } };
        float[] expected = { 1500f, 2061f, 3622f };
        var positions = new float[Points * 3];
        for (int n = 0; n < Points; n++)
            for (int d = 0; d < 3; d++)
                positions[n * 3 + d] = (vertices[n % 3][d] - 0.5f) / scale;   // pos = 64x + 0.5 = the vertex exactly
        var table = new float[4096];
        for (int i = 0; i < table.Length; i++) table[i] = i;
        var level = InstantNGPOperations.ComputeHashGridLevel(0, 65, 2.0, HashGridAlignment.TinyCudaNn);
        Assert.Equal(scale, level.Scale);

        using var positionBuffer = backend.AllocateBuffer(positions);
        using var tableBuffer = backend.AllocateBuffer(table);
        using var output = backend.AllocateBuffer(Points);
        ngp.HashGridEncodeLevel(positionBuffer, tableBuffer, output, Points,
            level.Scale, level.PositionOffset, level.GridResolution, 4096, 1, 0, 1);
        backend.Synchronize();
        var actual = backend.DownloadBuffer(output);
        for (int n = 0; n < Points; n++)
            Assert.True(Math.Abs(expected[n % 3] - actual[n]) <= 1e-3,
                $"point {n}: expected hash index {expected[n % 3]}, gpu {actual[n]}");
    }

    private static void Check(IDirectGpuBackend backend)
    {
        var ngp = AsInstantNgp(backend);
        CheckKnownHashIndices(backend, ngp);

        float[] positions = Random(2490, Points * 3, 0.0, 1.0);
        // The far faces and out-of-domain coordinates: x = 1 exactly (tiny-cuda-nn does not clamp it), below 0, above 1
        // and NaN, which every backend maps to 0 even when its compiler assumes finite math.
        float[] edges = { 1f, 1f, 1f, float.NaN, 0.3f, 0.7f, -0.25f, 1.5f, 0.5f, 1f, 0.999f, 0f };
        Array.Copy(edges, positions, edges.Length);
        var tables = new float[Levels][];
        var tableTensors = new Tensor<float>[Levels];
        for (int l = 0; l < Levels; l++)
        {
            tables[l] = Random(2491 + l, TableSize * Features, -1.0, 1.0);
            tableTensors[l] = new Tensor<float>(tables[l], new[] { TableSize, Features });
        }
        float[] upstream = Random(2499, Points * Levels * Features, -1.0, 1.0);
        var positionTensor = new Tensor<float>(positions, new[] { Points, 3 });

        var expected = InstantNGPOperations.MultiresolutionHashEncoding(
            positionTensor, tableTensors, BaseResolution, PerLevelScale, Features).ToArray();
        var expectedGradients = InstantNGPOperations.MultiresolutionHashEncodingBackward(
            positionTensor, tableTensors, BaseResolution, PerLevelScale, Features,
            new Tensor<float>(upstream, new[] { Points, Levels * Features }));
        var levels = InstantNGPOperations.ComputeHashGridLevels(Levels, BaseResolution, PerLevelScale, HashGridAlignment.TinyCudaNn);
        Assert.True(InstantNGPOperations.UsesDenseHashGridIndex(levels[0].GridResolution, TableSize));
        Assert.False(InstantNGPOperations.UsesDenseHashGridIndex(levels[1].GridResolution, TableSize));

        using var positionBuffer = backend.AllocateBuffer(positions);
        using var upstreamBuffer = backend.AllocateBuffer(upstream);
        using var output = backend.AllocateBuffer(Points * Levels * Features);
        for (int l = 0; l < Levels; l++)
        {
            using var tableBuffer = backend.AllocateBuffer(tables[l]);
            ngp.HashGridEncodeLevel(positionBuffer, tableBuffer, output, Points,
                levels[l].Scale, levels[l].PositionOffset, levels[l].GridResolution, TableSize, Features,
                l * Features, Levels * Features);
            backend.Synchronize();
        }
        AssertClose(expected, backend.DownloadBuffer(output), 1e-4, "encoding");

        for (int l = 0; l < Levels; l++)
        {
            using var tableGradient = backend.AllocateBuffer(TableSize * Features);
            ngp.HashGridEncodeLevelBackward(positionBuffer, upstreamBuffer, tableGradient, Points,
                levels[l].Scale, levels[l].PositionOffset, levels[l].GridResolution, TableSize, Features,
                l * Features, Levels * Features);
            backend.Synchronize();
            // Up to 256 points add into one entry in a different order than the CPU's double sum.
            AssertClose(expectedGradients[l].ToArray(), backend.DownloadBuffer(tableGradient), 1e-3, $"level {l} table gradient");
        }
    }

    [Fact]
    public void EveryKernel_ClampsToTheClosedUnitCube_AndTestsNaNOnTheBits()
    {
        // HIP (-ffast-math), OpenCL (-cl-finite-math-only), Metal (fast math by default) and WGSL may fold a float
        // comparison with NaN, so the NaN-to-0 clamp reads the bits; the upper clamp is 1, tiny-cuda-nn's domain edge.
        var sources = new (string Name, string Source, string NaNTest)[]
        {
            ("CUDA/HIP", AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaInstantNgpKernels.GetSource(),
                "(__float_as_uint(value) & 0x7FFFFFFFu) > 0x7F800000u"),
            ("OpenCL", OpenClInstantNgpKernels.GetSource(), "(as_uint(value) & 0x7FFFFFFFu) > 0x7F800000u"),
            ("Metal", MetalInstantNgpKernels.Source, "(as_type<uint>(value) & 0x7FFFFFFFu) > 0x7F800000u"),
            ("Vulkan", VulkanInstantNgpKernels.Forward, "(floatBitsToUint(value) & 0x7FFFFFFFu) > 0x7F800000u"),
#if NET7_0_OR_GREATER
            ("WebGPU", AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuInstantNgpKernels.Forward,
                "(bitcast<u32>(value) & 0x7FFFFFFFu) > 0x7F800000u"),
#endif
        };
        foreach (var (name, source, nanTest) in sources)
        {
            int start = source.IndexOf("instant_ngp_clamp_position(", StringComparison.Ordinal);
            Assert.True(start >= 0, $"{name}: no clamp function");
            string clamp = source.Substring(start, source.IndexOf('}', start) - start);
            Assert.Contains(nanTest, clamp, StringComparison.Ordinal);
            Assert.DoesNotContain("0.999999", clamp, StringComparison.Ordinal);
        }
    }

    [SkippableFact]
    public void Cuda_HashGridEncoding_MatchesCpu()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_HashGridEncoding_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void OpenCl_HashGridEncoding_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_HashGridEncoding_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_HashGridEncoding_MatchesCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_HashGridEncoding_MatchesCpu()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        Check(backend);
    }
#endif
}
#endif
