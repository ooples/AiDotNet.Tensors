using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Off CUDA, the float-id embedding and the class gather/scatter are composed from backend primitives, and each
/// primitive's edge semantics must match the CPU:
/// - Round rounds halves to even, as the CPU (RoundToNearestInteger) and torch.round do. OpenCL and Metal rounded them
///   away from zero.
/// - Clamp propagates NaN, as torch.clamp does.
/// - NaN classification, masked fill and take_along_dim together read a class the way CpuEngine.ClassIndexOf does:
///   rounded, with NaN and out-of-range classes gathering 0.
/// Each check runs on every backend this machine has.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class PortableIdPrimitiveTests
{
    private static readonly float[] Halves = { 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 3.2f, 3.7f };
    private const float NoId = -1f;

    [SkippableFact]
    public void OpenCl_RoundsHalvesToEven_PropagatesNaNThroughClamp_AndReadsClassesAsTheCpu()
    {
        using var engine = new DirectGpuTensorEngine();
        var backend = engine.IsGpuAvailable ? engine.GetBackend() : null;
        // The engine takes the first available backend (cuda, opencl, hip); only an OpenCL selection exercises the
        // OpenCL kernels this test is named for.
        Skip.IfNot(backend is AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend,
            $"needs the OpenCL backend; the engine selected {backend?.GetType().Name ?? "none"}.");
        if (backend is AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend openCl)
            CheckAll(openCl);
    }

    [SkippableFact]
    public void Vulkan_RoundsHalvesToEven_PropagatesNaNThroughClamp_AndReadsClassesAsTheCpu()
    {
        VulkanBackend? backend = null;
        bool available;
        try
        {
            backend = VulkanBackend.Instance;
            available = backend.Initialize();
        }
        catch (Exception)
        {
            available = false;
        }
        Skip.If(!available || backend is null, "Vulkan not available on this system.");
        // Round and clamp are built in; the classification, masked fill and take_along_dim kernels are GLSL compiled
        // at run time.
        Skip.If(!backend!.IsGlslCompilerAvailable, "Vulkan's runtime GLSL compiler (libshaderc) is unavailable.");
        CheckAll(backend);
    }

    private static void CheckAll(IDirectGpuBackend backend)
    {
        CheckRound(backend);
        CheckClamp(backend);
        CheckClassGather(backend);
    }

    private static void CheckRound(IDirectGpuBackend backend)
    {
        using var input = backend.AllocateBuffer(Halves);
        using var output = backend.AllocateBuffer(Halves.Length);
        backend.Round(input, output, Halves.Length);
        var got = backend.DownloadBuffer(output);
        for (int i = 0; i < Halves.Length; i++)
            Assert.True(got[i] == MathF.Round(Halves[i]), $"round({Halves[i]}) = {got[i]}, expected {MathF.Round(Halves[i])}");
    }

    private static void CheckClamp(IDirectGpuBackend backend)
    {
        float[] values = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 5f, 0.25f };
        float[] expected = { float.NaN, 1f, -1f, 1f, 0.25f };
        using var input = backend.AllocateBuffer(values);
        using var output = backend.AllocateBuffer(values.Length);
        backend.Clamp(input, output, -1f, 1f, values.Length);
        var got = backend.DownloadBuffer(output);
        for (int i = 0; i < values.Length; i++)
            Assert.True(float.IsNaN(expected[i]) ? float.IsNaN(got[i]) : got[i] == expected[i],
                $"clamp({values[i]}) = {got[i]}, expected {expected[i]}");
    }

    private static void CheckClassGather(IDirectGpuBackend backend)
    {
        const int Rows = 6, Classes = 4;
        float[] classes = { 0f, 3f, -1f, 4f, 1.5f, float.NaN };    // in range, last, below, == C, a half, NaN
        int[] expectedClass = { 0, 3, -1, -1, 2, -1 };
        var values = new float[Rows * Classes];
        for (int i = 0; i < values.Length; i++) values[i] = i + 1;

        using var classBuffer = backend.AllocateBuffer(classes);
        using var valueBuffer = backend.AllocateBuffer(values);
        using var normalized = backend.AllocateBuffer(Rows);
        using var nanMask = backend.AllocateBuffer(Rows);
        using var picked = backend.AllocateBuffer(Rows);
        backend.Round(classBuffer, normalized, Rows);
        backend.ClassifyFloat(classBuffer, nanMask, (int)DirectGpuTensorEngine.FloatClassification.IsNaN, Rows);
        backend.MaskedFillKernel(normalized, nanMask, normalized, NoId, Rows);
        backend.TakeAlongDim(valueBuffer, normalized, picked, Rows, 1, 1, Classes);

        var got = backend.DownloadBuffer(picked);
        for (int r = 0; r < Rows; r++)
        {
            float expected = expectedClass[r] < 0 ? 0f : values[r * Classes + expectedClass[r]];
            Assert.True(got[r] == expected, $"row {r} (class {classes[r]}): {got[r]}, expected {expected}");
        }
    }
}
