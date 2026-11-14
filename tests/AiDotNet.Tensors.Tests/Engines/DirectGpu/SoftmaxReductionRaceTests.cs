using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The block softmax kernels reduce the row max into shared memory, read it, then reuse the same shared memory for the
/// sum. Without a barrier between the read and the reuse, a warp still reading the max can see a sum instead and write a
/// garbage row. Each row here is spread over several warps (65+ columns) with logits wide enough that a wrong max is far
/// outside tolerance; enough rows make a missing barrier show up on every run, not one in three (MultiHeadAttention's
/// 65-key composed attention).
/// </summary>
[Collection("DirectGpuSerial")]
public class SoftmaxReductionRaceTests
{
    private static float[] Logits(int n, int seed)
    {
        var r = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(r.NextDouble() * 60 - 30);
        return v;
    }

    private static void AssertRowsMatch(float[] x, float[] got, int rows, int cols, string what)
    {
        for (int row = 0; row < rows; row++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < cols; c++) max = Math.Max(max, x[row * cols + c]);
            for (int c = 0; c < cols; c++) sum += Math.Exp(x[row * cols + c] - max);
            for (int c = 0; c < cols; c++)
            {
                double expected = Math.Exp(x[row * cols + c] - max) / sum;
                double actual = got[row * cols + c];
                Assert.True(Math.Abs(expected - actual) <= 1e-5 + 1e-4 * expected,
                    $"{what}: row {row} column {c} expected {expected}, got {actual}");
            }
        }
    }

    [SkippableTheory]
    [InlineData(65)]
    [InlineData(97)]
    [InlineData(255)]
    public void SoftmaxRows_and_Softmax_agree_with_a_double_reference_on_multi_warp_rows(int cols)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "The CUDA backend did not initialize.");
        const int rows = 8192;
        float[] x = Logits(rows * cols, cols);
        using var input = backend.AllocateBuffer(x);
        using var rowsOut = backend.AllocateBuffer(rows * cols);
        using var softmaxOut = backend.AllocateBuffer(rows * cols);
        for (int repeat = 0; repeat < 4; repeat++)
        {
            backend.SoftmaxRows(input, rowsOut, rows, cols);
            backend.Softmax(input, softmaxOut, rows, cols);
            backend.Synchronize();
            AssertRowsMatch(x, backend.DownloadBuffer(rowsOut), rows, cols, $"SoftmaxRows (repeat {repeat})");
            AssertRowsMatch(x, backend.DownloadBuffer(softmaxOut), rows, cols, $"Softmax (repeat {repeat})");
        }
    }
}
