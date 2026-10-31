#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="ISumAroundKeptBackend"/>: output[c] = scale * sum over l and t of input[l, c, t], the compiled GPU step's
/// convolution bias gradient, against a double-precision sum. On OpenCL the step permuted the whole gradient instead.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class SumAroundKeptBackendTests
{
    private static readonly (int L, int C, int T)[] Shapes =
    {
        (1, 1, 1), (2, 3, 5), (1, 7, 300), (64, 32, 196), (3, 300, 4), (500, 2, 1), (17, 5, 33),
    };

    private static void Check(IDirectGpuBackend backend)
    {
        var around = (ISumAroundKeptBackend)backend;
        foreach (var (l, c, t) in Shapes)
        {
            var x = new float[l * c * t];
            for (int i = 0; i < x.Length; i++) x[i] = ((i * 7919 % 97) - 48) * 0.01f;
            const float scale = 0.5f;
            var expected = new double[c];
            for (int li = 0; li < l; li++)
                for (int ci = 0; ci < c; ci++)
                    for (int ti = 0; ti < t; ti++)
                        expected[ci] += x[(li * c + ci) * t + ti];
            using var input = backend.AllocateBuffer(x);
            using var scratch = backend.AllocateBuffer(Math.Max(1, around.SumAroundKeptScratchFloats(l, c, t)));
            using var output = backend.AllocateBuffer(new float[c]);
            Assert.True(around.TrySumAroundKept(input, scratch, output, l, c, t, scale), $"L={l} C={c} T={t} declined");
            var actual = backend.DownloadBuffer(output);
            for (int ci = 0; ci < c; ci++)
            {
                double want = scale * expected[ci];
                Assert.True(Math.Abs(want - actual[ci]) <= 1e-4 * (1 + Math.Abs(want)),
                    $"L={l} C={c} T={t} channel {ci}: expected {want}, got {actual[ci]}");
            }
        }
    }

    [SkippableFact]
    public void OpenCl_SumAroundKept_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_SumAroundKept_MatchesCpu()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }
}
#endif