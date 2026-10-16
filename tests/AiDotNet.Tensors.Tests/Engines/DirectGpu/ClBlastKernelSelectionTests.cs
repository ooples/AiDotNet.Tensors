using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The OpenCL backend picks between CLBlast's direct and indirect GEMM kernels the way CLBlast does: by the problem's
/// volume M·N·K against MinIndirectSize³ (#377). It used to go indirect as soon as M or N alone reached MinIndirectSize,
/// which ran skinny problems at 0.34x to 0.40x the native library's speed.
/// </summary>
public sealed class ClBlastKernelSelectionTests
{
    [Theory]
    [InlineData(32, 1024, 1024, false)]   // 33.5M < 448^3 = 89.9M: direct, as in CLBlast (indirect before)
    [InlineData(1024, 64, 1024, false)]   // 67.1M: direct (indirect before)
    [InlineData(447, 448, 448, false)]
    [InlineData(448, 448, 448, true)]
    [InlineData(512, 512, 512, true)]
    [InlineData(64, 64, 4096, false)]
    [InlineData(4096, 4096, 8, true)]     // 134M: indirect
    public void Selection_FollowsTheProblemVolume(int m, int n, int k, bool indirect)
        => Assert.Equal(indirect, OpenClBackend.UsesClBlastIndirectKernel(m, n, k, 448));

    [Fact]
    public void Selection_WithoutAThreshold_IsDirect()
    {
        Assert.False(OpenClBackend.UsesClBlastIndirectKernel(4096, 4096, 4096, 0));
    }

    [Fact]
    public void Selection_DoesNotOverflow_OnLargeProblems()
    {
        Assert.True(OpenClBackend.UsesClBlastIndirectKernel(65536, 65536, 65536, 2048));
    }

    [SkippableTheory]
    [InlineData(32, 1024, 1024)]
    [InlineData(1024, 64, 1024)]
    [InlineData(33, 1000, 1000)]
    [InlineData(1000, 70, 999)]
    public void SkinnyGemm_OnTheDirectKernel_MatchesTheCpu(int m, int n, int k)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.GetBackend() is OpenClBackend, "needs the OpenCL backend");
        var backend = (OpenClBackend)engine.GetBackend()!;

        var a = new float[m * k];
        var b = new float[k * n];
        var rng = new Random(11);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);

        var previous = Environment.GetEnvironmentVariable("AIDOTNET_GPU_GEMM_IMPL");
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_GEMM_IMPL", "builtin");
        float[] gpu;
        try
        {
            using var ga = backend.AllocateBuffer(a);
            using var gb = backend.AllocateBuffer(b);
            using var gc = backend.AllocateBuffer(m * n);
            backend.Gemm(ga, gb, gc, m, n, k);
            gpu = backend.DownloadBuffer(gc);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_GPU_GEMM_IMPL", previous);
        }

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                double expected = 0;
                for (int p = 0; p < k; p++) expected += (double)a[row * k + p] * b[p * n + col];
                float actual = gpu[row * n + col];
                Assert.True(Math.Abs(actual - expected) <= 1e-3 * (1 + Math.Abs(expected)),
                    $"C[{row},{col}] = {actual}, expected {expected} ({m}x{n}x{k})");
            }
        }
    }
}
