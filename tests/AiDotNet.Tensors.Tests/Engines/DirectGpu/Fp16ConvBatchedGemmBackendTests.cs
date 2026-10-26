// System.Half (the FP16 oracle) only exists on .NET 5+; the hosted WebGPU backend needs .NET 7+.
#if NET7_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IFp16ConvBatchedGemmBackend.GemmFp16ConvBatched"/> on every non-CUDA backend (CUDA's is covered in
/// <c>Fp16ConvBatchLayoutTests</c>): the FP16 im2col conv's per-image GEMM must write each image's
/// <c>W[M,K] @ col[K, b*P + p]</c> in NCHW, against a CPU reference from the same FP16-rounded operands, with FP32 and
/// FP16 output. A backend that is not present on this machine skips (or throws under AIDOTNET_REQUIRE_GPU_TESTS=1).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class Fp16ConvBatchedGemmBackendTests
{
    // One method per backend, so a backend's CI lane selects its case by name (the Metal lane runs ~Metal).
    [SkippableTheory, InlineData(false), InlineData(true)]
    public void OpenCl_GemmFp16ConvBatched_WritesEveryImageInNchw(bool halfOutput) => Check("OpenCL", halfOutput);

    [SkippableTheory, InlineData(false), InlineData(true)]
    public void Vulkan_GemmFp16ConvBatched_WritesEveryImageInNchw(bool halfOutput) => Check("Vulkan", halfOutput);

    [SkippableTheory, InlineData(false), InlineData(true)]
    public void WebGpu_GemmFp16ConvBatched_WritesEveryImageInNchw(bool halfOutput) => Check("WebGpu", halfOutput);

    [SkippableTheory, InlineData(false), InlineData(true)]
    public void Metal_GemmFp16ConvBatched_WritesEveryImageInNchw(bool halfOutput) => Check("Metal", halfOutput);

    [SkippableTheory, InlineData(false), InlineData(true)]
    public void Hip_GemmFp16ConvBatched_WritesEveryImageInNchw(bool halfOutput) => Check("Hip", halfOutput);

    private static void Check(string backendName, bool halfOutput)
    {
        IDisposable? owner = null;
        IDirectGpuBackend? backend = null;
        bool byteSizedHalves = false;
        try
        {
            switch (backendName)
            {
                case "OpenCL":
                    var openCl = new OpenClBackendTestFixture();
                    owner = openCl;
                    backend = openCl.IsAvailable ? openCl.Backend : null;
                    byteSizedHalves = true;
                    break;
                case "Hip":
                    var hip = new HipBackendTestFixture();
                    owner = hip;
                    backend = hip.IsAvailable ? hip.Backend : null;
                    break;
                case "Vulkan":
                    var vulkan = VulkanBackend.Instance;   // process-wide singleton: not disposed here
                    backend = vulkan.IsAvailable ? vulkan : null;
                    break;
                case "WebGpu":
                    backend = DawnHostedWebGpu.BackendOrNull();   // shared by the test process
                    break;
                case "Metal":
                    var metal = new MetalBackend();
                    owner = metal;
                    backend = metal.IsAvailable ? metal : null;
                    break;
            }
        }
        catch (Exception ex) when (!RequireGpu())
        {
            owner?.Dispose();
            Skip.If(true, $"{backendName} backend unavailable: {ex.GetType().Name}: {ex.Message}");
        }

        try
        {
            bool ready = backend is IGpuHalfPrecisionBackend { SupportsHgemm: true };
            if (!ready && RequireGpu())
                throw new InvalidOperationException($"GPU tests were required but the {backendName} FP16 GEMM was unavailable.");
            Skip.IfNot(ready, $"{backendName} FP16 GEMM not available on this system " +
                $"(backend {(backend is null ? "absent" : "present")}" +
                (backendName == "Vulkan" ? $", GLSL compiler {VulkanBackend.Instance.IsGlslCompilerAvailable})." : ")."));
            Run(backend!, byteSizedHalves, halfOutput);
        }
        finally
        {
            owner?.Dispose();
        }
    }

    private static bool RequireGpu() =>
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal);

    private static void Run(IDirectGpuBackend backend, bool byteSizedHalves, bool halfOutput)
    {
        const int m = 24, pixels = 40, k = 72, batch = 3;
        int n = batch * pixels;
        float[] w = Random(m * k, 31), col = Random(k * n, 32);

        var expected = new float[m * n];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < m; i++)
                for (int p = 0; p < pixels; p++)
                {
                    float acc = 0f;
                    for (int q = 0; q < k; q++)
                        acc += (float)(Half)w[i * k + q] * (float)(Half)col[q * n + b * pixels + p];
                    expected[(b * m + i) * pixels + p] = acc;
                }

        IGpuBuffer Half(float[] values)
        {
            using var wide = backend.AllocateBuffer(values);
            var half = byteSizedHalves ? backend.AllocateByteBuffer(values.Length * 2) : backend.AllocateBuffer(values.Length);
            backend.ConvertToFp16(wide, half, values.Length);
            return half;
        }
        using var dw = Half(w);
        using var dcol = Half(col);
        using var result = backend.AllocateBuffer(m * n);
        using var halfResult = halfOutput
            ? (byteSizedHalves ? backend.AllocateByteBuffer(m * n * 2) : backend.AllocateBuffer(m * n))
            : null;
        ((IFp16ConvBatchedGemmBackend)backend).GemmFp16ConvBatched(dw, dcol, halfResult ?? result, m, pixels, k, batch, halfOutput);
        if (halfResult is not null) backend.ConvertToFp32(halfResult, result, m * n);
        float[] actual = backend.DownloadBuffer(result);

        // FP32 accumulation order only, plus the final FP16 rounding for a half output.
        double absTol = halfOutput ? 5e-2 : 1e-2, relTol = halfOutput ? 3e-2 : 1e-2;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= absTol + relTol * Math.Abs(expected[i]),
                $"out[{i / (m * pixels)}][{i / pixels % m}][{i % pixels}] = {actual[i]}, expected {expected[i]}");
    }

    private static float[] Random(int count, int seed)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }
}

#endif