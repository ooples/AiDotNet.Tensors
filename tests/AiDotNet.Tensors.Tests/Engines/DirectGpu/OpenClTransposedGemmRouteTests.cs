using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The transposed GEMMs of a linear layer's backward pass (input gradient dY · Wᵀ, weight gradient Xᵀ · dY) went to
/// CLBlast's transposed GEMM at every size. Past its packed-kernel threshold the library copies and pads the operands
/// into buffers it allocates itself, outside the pool: VGG-11's 4096 x 4096 classifier held a 64 MB copy of the weight
/// and another of the gradient at every training step, and those set its GPU memory peak (1,098 MB, PyTorch 1,055).
/// Those problems now take transposed copies from the pool and an on-device GEMM; the library keeps the problems its
/// direct kernel multiplies in place.
/// </summary>
[Collection("DirectGpuSerial")]
public class OpenClTransposedGemmRouteTests
{
    private readonly ITestOutputHelper _output;

    public OpenClTransposedGemmRouteTests(ITestOutputHelper output) => _output = output;

    private static OpenClBackend RequireBackend()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        var backend = new OpenClBackend(deviceIndex: 0);
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "OpenCL backend did not initialize.");
        }
        return backend;
    }

    private static float[] Values(int count, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    // C[M,N] = alpha · op(A) · op(B) + beta · C in double, op(A) and op(B) read element by element.
    private static float[] Reference(Func<int, int, float> a, Func<int, int, float> b, float[] c, int m, int n, int k,
        float alpha, float beta)
    {
        var result = new float[m * n];
        for (int r = 0; r < m; r++)
            for (int col = 0; col < n; col++)
            {
                double sum = 0;
                for (int i = 0; i < k; i++) sum += (double)a(r, i) * b(i, col);
                result[r * n + col] = (float)(alpha * sum + beta * c[r * n + col]);
            }
        return result;
    }

    private void AssertClose(float[] expected, float[] actual, string what)
    {
        double maxError = 0, maxMagnitude = 1;
        for (int i = 0; i < expected.Length; i++)
        {
            maxError = Math.Max(maxError, Math.Abs(expected[i] - actual[i]));
            maxMagnitude = Math.Max(maxMagnitude, Math.Abs(expected[i]));
        }
        _output.WriteLine($"{what}: max error {maxError:G3} of {maxMagnitude:G3}");
        Assert.True(maxError <= 1e-4 * maxMagnitude, $"{what}: max error {maxError:G3} against magnitude {maxMagnitude:G3}");
    }

    // MatMulTransposed: C[M,N] = A[M,K] · Bᵀ with B stored [N, K]. The shapes cover the library's direct kernel, a
    // weight larger than the batch's activations (alpha 1, beta 0 and a general alpha/beta), and a batch larger than
    // the weight.
    [SkippableTheory]
    [InlineData(8, 16, 24, 1f, 0f)]
    [InlineData(32, 1024, 4096, 1f, 0f)]
    [InlineData(32, 1024, 4096, 0.5f, -1f)]
    [InlineData(1024, 256, 512, 1f, 0f)]
    public void MatMulTransposed_matches_the_reference_on_every_route(int m, int n, int k, float alpha, float beta)
    {
        using var backend = RequireBackend();
        float[] hostA = Values(m * k, 1), hostB = Values(n * k, 2), hostC = Values(m * n, 3);
        using var a = backend.AllocateBuffer(hostA);
        using var b = backend.AllocateBuffer(hostB);
        using var c = backend.AllocateBuffer(hostC);
        backend.MatMulTransposed(a, b, c, m, n, k, alpha, beta);
        float[] expected = Reference((r, i) => hostA[r * k + i], (i, col) => hostB[col * k + i], hostC, m, n, k, alpha, beta);
        AssertClose(expected, backend.DownloadBuffer(c), $"MatMulTransposed {m}x{n}x{k} alpha {alpha} beta {beta}");
    }

    // MatMulTransposedA: C[M,N] = Aᵀ · B with A stored [K, M]: the library's direct kernel, and a weight gradient over
    // a short batch.
    [SkippableTheory]
    [InlineData(8, 16, 24, 1f, 0f)]
    [InlineData(1024, 1024, 128, 1f, 0f)]
    [InlineData(1024, 1024, 128, 1f, 1f)]
    public void MatMulTransposedA_matches_the_reference_on_every_route(int m, int n, int k, float alpha, float beta)
    {
        using var backend = RequireBackend();
        float[] hostA = Values(k * m, 4), hostB = Values(k * n, 5), hostC = Values(m * n, 6);
        using var a = backend.AllocateBuffer(hostA);
        using var b = backend.AllocateBuffer(hostB);
        using var c = backend.AllocateBuffer(hostC);
        backend.MatMulTransposedA(a, b, c, m, n, k, alpha, beta);
        float[] expected = Reference((r, i) => hostA[i * m + r], (i, col) => hostB[i * n + col], hostC, m, n, k, alpha, beta);
        AssertClose(expected, backend.DownloadBuffer(c), $"MatMulTransposedA {m}x{n}x{k} alpha {alpha} beta {beta}");
    }

    [SkippableFact]
    public void Only_problems_below_the_packed_kernel_threshold_go_to_the_clblast_library()
    {
        using var backend = RequireBackend();
        Skip.IfNot(OpenClBackend.IsClBlastAvailable, "The CLBlast library is not available.");
        // VGG-11's classifier at batch 64: the input gradient (64 x 4096 x 4096) and the weight gradient
        // (4096 x 4096 x 64), past the library's packed-kernel threshold on a device with a database entry.
        const int batch = 64, features = 4096;
        Skip.If(backend.UsesClBlastDirectKernel(batch, features, features), "No packed-kernel threshold for this device.");
        Assert.False(backend.UsesClBlastDirectKernel(features, features, batch));
        Assert.True(backend.UsesClBlastDirectKernel(8, 16, 24));

        using var small = backend.AllocateBuffer(Values(16 * 24, 7));
        using var smallC = backend.AllocateBuffer(8 * 16);
        using var dY = backend.AllocateBuffer(Values(batch * features, 8));
        using var x = backend.AllocateBuffer(Values(batch * features, 9));
        using var weight = backend.AllocateBuffer(Values(features * features, 10));
        using var dX = backend.AllocateBuffer(batch * features);
        using var dW = backend.AllocateBuffer(features * features);

        backend.MatMulTransposed(small, small, smallC, 8, 16, 24);
        Assert.Equal(OpenClGemmDispatchPath.ClBlastLibrary, backend.LastGemmDispatchPath);
        backend.MatMulTransposed(dY, weight, dX, batch, features, features);
        Assert.NotEqual(OpenClGemmDispatchPath.ClBlastLibrary, backend.LastGemmDispatchPath);

        backend.MatMulTransposedA(small, small, smallC, 8, 16, 24);
        Assert.Equal(OpenClGemmDispatchPath.ClBlastLibrary, backend.LastGemmDispatchPath);
        backend.MatMulTransposedA(x, dY, dW, features, features, batch);
        Assert.NotEqual(OpenClGemmDispatchPath.ClBlastLibrary, backend.LastGemmDispatchPath);
    }
}
