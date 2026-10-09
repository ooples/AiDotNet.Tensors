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
/// Every backend's <see cref="ITransposedAGemm"/> port must compute row-major <c>C = alpha · Aᵀ · B + beta · C</c>
/// like a CPU reference, including shapes where M, N and K all differ (so a swapped stride or dimension shows).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class TransposedAGemmBackendParityTests
{
    private static readonly (int M, int N, int K)[] Shapes =
    {
        (1, 1, 1),
        (5, 7, 3),
        (33, 17, 64),
    };

    private static void Check(IDirectGpuBackend backend)
    {
        var gemm = (ITransposedAGemm)backend;
        foreach (var (m, n, k) in Shapes)
        {
            var a = new float[k * m];
            var b = new float[k * n];
            var c0 = new float[m * n];
            for (int i = 0; i < a.Length; i++) a[i] = ((i * 37) % 23 - 11) * 0.125f;
            for (int i = 0; i < b.Length; i++) b[i] = ((i * 53) % 19 - 9) * 0.25f;
            for (int i = 0; i < c0.Length; i++) c0[i] = (i % 7) - 3f;

            foreach (var (alpha, beta) in new[] { (1f, 0f), (0.5f, 2f) })
            {
                var expected = new float[m * n];
                for (int row = 0; row < m; row++)
                    for (int col = 0; col < n; col++)
                    {
                        double acc = 0;
                        for (int kk = 0; kk < k; kk++) acc += (double)a[kk * m + row] * b[kk * n + col];
                        expected[row * n + col] = (float)(alpha * acc + beta * c0[row * n + col]);
                    }

                using var bufA = backend.AllocateBuffer(a);
                using var bufB = backend.AllocateBuffer(b);
                using var bufC = backend.AllocateBuffer(c0);
                gemm.MatMulTransposedA(bufA, bufB, bufC, m, n, k, alpha, beta);
                var actual = backend.DownloadBuffer(bufC);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                        $"M={m} N={n} K={k} alpha={alpha} beta={beta} at {i}: expected {expected[i]}, got {actual[i]}");
            }
        }
    }

    [SkippableFact]
    public void OpenCl_MatMulTransposedA_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_MatMulTransposedA_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable,
            "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_MatMulTransposedA_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
    [SkippableFact]
    public void Metal_MatMulTransposedA_MatchesCpu()
    {
        // The constructor throws PlatformNotSupportedException off Apple platforms.
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal needs macOS or iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void WebGpu_MatMulTransposedA_MatchesCpu()
    {
        // WebGPU runs through the browser's JavaScript interop; elsewhere the backend throws PlatformNotSupportedException.
        Skip.IfNot(OperatingSystem.IsBrowser(), "WebGPU needs a browser host.");
        using var backend = new WebGpuBackend();
        Skip.IfNot(backend.InitializeAsync().GetAwaiter().GetResult(), "WebGPU is not available.");
        Check(backend);
    }
}
#endif