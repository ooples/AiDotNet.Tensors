using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// OpenCL's <see cref="OpenClBackend.BatchedGemm"/> (gemm_batched_tiled) against a CPU product accumulated in double:
/// sizes that are not multiples of the 16-wide tile in every dimension, several batches, and both the beta = 0 and the
/// beta != 0 paths (the latter reads C back).
/// </summary>
[Collection("DirectGpuSerial")]
public class OpenClBatchedGemmTests
{
    public static readonly TheoryData<int, int, int, int, float, float> Cases = new()
    {
        { 1, 1, 1, 1, 1f, 0f },
        { 3, 17, 5, 33, 1f, 0f },
        { 16, 64, 512, 64, 1f, 0f },
        { 4, 31, 47, 19, 0.5f, 2f },
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Batched_gemm_matches_the_cpu_product(int batch, int m, int n, int k, float alpha, float beta)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        var rng = new Random(7);
        float[] Fill(int count) { var a = new float[count]; for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var a = Fill(batch * m * k);
        var b = Fill(batch * k * n);
        var c = Fill(batch * m * n);
        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bc = backend.AllocateBuffer(c);
        backend.BatchedGemm(ba, bb, bc, m, n, k, batch, alpha, beta);
        var actual = backend.DownloadBuffer(bc);

        for (int p = 0; p < batch; p++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int q = 0; q < k; q++) sum += (double)a[(p * m + i) * k + q] * b[(p * k + q) * n + j];
                    int idx = (p * m + i) * n + j;
                    double expected = alpha * sum + beta * c[idx];
                    // fp32 accumulation of k products of magnitude <= 1.
                    Assert.True(Math.Abs(expected - actual[idx]) <= 1e-5 * (k + 1) * (1 + Math.Abs(beta)),
                        $"batch {p} [{i},{j}]: expected {expected}, got {actual[idx]}");
                }
    }
}