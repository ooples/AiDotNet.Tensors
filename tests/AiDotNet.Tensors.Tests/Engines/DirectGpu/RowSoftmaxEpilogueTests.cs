#if NET6_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A softmax activation on a fused linear normalizes each row of the <c>[rows, features]</c> result on its own: every
/// row sums to 1 and matches the CPU <c>softmax(x W + b)</c>. Covers the stream-ordered fused GEMM epilogue on each
/// backend and the engine's <c>FusedLinearGpu</c>, at one row and at several (a whole-buffer softmax would pass at
/// one row only).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class RowSoftmaxEpilogueTests
{
    private const int K = 6, N = 5;

    private static float[] Values(int n, int salt)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((i * 37 + salt * 101) % 23 - 11) * 0.09f;
        return v;
    }

    /// <summary>softmax(x W + b) per row, in double.</summary>
    private static double[] Expected(float[] x, float[] w, float[] b, int rows)
    {
        var y = new double[rows * N];
        for (int r = 0; r < rows; r++)
        {
            var z = new double[N];
            for (int j = 0; j < N; j++)
            {
                double s = b[j];
                for (int k = 0; k < K; k++) s += x[r * K + k] * w[k * N + j];
                z[j] = s;
            }
            double max = z.Max(), sum = z.Sum(v => Math.Exp(v - max));
            for (int j = 0; j < N; j++) y[r * N + j] = Math.Exp(z[j] - max) / sum;
        }
        return y;
    }

    private static void AssertRows(double[] expected, float[] actual, int rows, string path)
    {
        for (int r = 0; r < rows; r++)
        {
            double rowSum = 0;
            for (int j = 0; j < N; j++)
            {
                int i = r * N + j;
                rowSum += actual[i];
                Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-5, $"{path} row {r} col {j}: {actual[i]} vs {expected[i]}");
            }
            Assert.True(Math.Abs(rowSum - 1) <= 1e-5, $"{path} row {r} sums to {rowSum}");
        }
    }

    private static void CheckAsync(IDirectGpuBackend backend, int rows)
    {
        var async = Assert.IsAssignableFrom<IAsyncGpuBackend>(backend);
        float[] x = Values(rows * K, 1), w = Values(K * N, 2), b = Values(N, 3);
        var stream = async.CreateStream(GpuStreamType.Compute);
        using var bufX = async.AllocateBufferAsync(x, stream);
        using var bufW = async.AllocateBufferAsync(w, stream);
        using var bufB = async.AllocateBufferAsync(b, stream);
        using var output = backend.AllocateBuffer(rows * N);
        async.FusedGemmBiasActivationAsync(bufX, bufW, bufB, output, rows, N, K, FusedActivationType.Softmax, stream);
        var host = new float[rows * N];
        async.DownloadBufferAsync(output, host, stream);
        async.SynchronizeStream(stream);
        AssertRows(Expected(x, w, b, rows), host, rows, $"{backend.GetType().Name} async rows={rows}");
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(4)]
    public void Cuda_FusedGemmBiasSoftmax_NormalizesEachRow(int rows)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        CheckAsync(backend, rows);
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(4)]
    public void OpenCl_FusedGemmBiasSoftmax_NormalizesEachRow(int rows)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        CheckAsync(backend, rows);
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(4)]
    public void Hip_FusedGemmBiasSoftmax_NormalizesEachRow(int rows)
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        CheckAsync(backend, rows);
    }

    [SkippableTheory]
    [InlineData(1, true)]
    [InlineData(3, true)]
    [InlineData(3, false)]
    public void FusedLinearGpu_Softmax_NormalizesEachRow(int rows, bool withBias)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "No GPU backend is available.");
        float[] x = Values(rows * K, 4), w = Values(K * N, 5);
        float[] b = withBias ? Values(N, 6) : new float[N];
        var bias = withBias ? new Tensor<float>(new[] { N }, new Vector<float>(b)) : null;
        using var result = engine.FusedLinearGpu(
            new Tensor<float>(new[] { rows, K }, new Vector<float>(x)),
            new Tensor<float>(new[] { K, N }, new Vector<float>(w)),
            bias, FusedActivationType.Softmax);
        AssertRows(Expected(x, w, b, rows), result.ToArray(), rows, $"FusedLinearGpu rows={rows} bias={withBias}");
    }
}
#endif
