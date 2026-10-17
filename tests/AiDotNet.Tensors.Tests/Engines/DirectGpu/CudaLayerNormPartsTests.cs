using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CudaBackend's column sums (IColumnSumBackend: column_sum_small for short matrices, the chunked partial/finish pair
/// for tall ones) and its one-pass LayerNorm backward (ILayerNormBackwardPartsBackend.TryLayerNormRowBackward), each
/// against a double-precision reference, written and added, and run twice to check the result is bitwise repeatable.
/// </summary>
public class CudaLayerNormPartsTests
{
    [SkippableTheory]
    [InlineData(256, 64)]      // a d_model-64 Transformer's bias gradients: column_sum_small
    [InlineData(256, 192)]
    [InlineData(1024, 10)]     // the most rows column_sum_small takes, a partial column tile
    [InlineData(1, 3)]
    [InlineData(1025, 64)]     // one row past it: the chunked pair
    [InlineData(4096, 100)]
    public void ColumnSums_EqualTheReference(int rows, int cols)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var sums = (IColumnSumBackend)backend;
        float[] x = Fill(rows * cols, 1), prior = Fill(cols, 2);
        using var xBuf = backend.AllocateBuffer(x);
        using var scratch = backend.AllocateBuffer(Math.Max(1, sums.ColumnSumScratchFloats(rows, cols)));
        foreach (bool add in new[] { false, true })
        {
            float[]? first = null;
            for (int run = 0; run < 2; run++)
            {
                using var outBuf = backend.AllocateBuffer(add ? prior : Enumerable.Repeat(float.NaN, cols).ToArray());
                Assert.True(sums.TryColumnSums(xBuf, scratch, outBuf, rows, cols, add), $"declined {rows}x{cols}");
                var got = backend.DownloadBuffer(outBuf);
                for (int c = 0; c < cols; c++)
                {
                    double s = add ? prior[c] : 0, bound = add ? Math.Abs(prior[c]) : 0;
                    for (int r = 0; r < rows; r++) { s += x[r * cols + c]; bound += Math.Abs(x[r * cols + c]); }
                    AssertClose(got[c], s, bound, $"{rows}x{cols} add={add} column {c}");
                }
                if (first is null) first = got;
                else Assert.Equal(first, got);
            }
        }
    }

    [SkippableTheory]
    [InlineData(256, 64)]      // the parity Transformer's LayerNorms: layernorm_backward_rows2, one row per warp
    [InlineData(300, 100)]     // ragged rows and columns: rows4
    [InlineData(2048, 256)]    // more rows than 128 one-row-per-warp chunks: rows8
    [InlineData(17, 512)]      // the widest shape taken: rows16
    [InlineData(5, 1)]
    public void RowBackward_EqualsTheReference(int rows, int cols)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var parts = (ILayerNormBackwardPartsBackend)backend;
        const float eps = 1e-5f;
        float[] dy = Fill(rows * cols, 3), x = Fill(rows * cols, 4), gamma = Fill(cols, 5);
        float[] mean = new float[rows], variance = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            double m = 0, v = 0;
            for (int c = 0; c < cols; c++) m += x[r * cols + c];
            m /= cols;
            for (int c = 0; c < cols; c++) v += (x[r * cols + c] - m) * (x[r * cols + c] - m);
            mean[r] = (float)m; variance[r] = (float)(v / cols);
        }
        float[] priorDx = Fill(rows * cols, 6), priorG = Fill(cols, 7), priorB = Fill(cols, 8);
        using var dyBuf = backend.AllocateBuffer(dy);
        using var xBuf = backend.AllocateBuffer(x);
        using var gammaBuf = backend.AllocateBuffer(gamma);
        using var meanBuf = backend.AllocateBuffer(mean);
        using var varBuf = backend.AllocateBuffer(variance);
        int scratchFloats = parts.LayerNormRowBackwardScratchFloats(rows, cols);
        Assert.True(scratchFloats > 0, $"{rows}x{cols} not served");
        using var scratch = backend.AllocateBuffer(scratchFloats);

        foreach (bool add in new[] { false, true })
        {
            float[]? firstDx = null, firstG = null, firstB = null;
            for (int run = 0; run < 2; run++)
            {
                float[] nan(int n) => Enumerable.Repeat(float.NaN, n).ToArray();
                using var dxBuf = backend.AllocateBuffer(add ? priorDx : nan(rows * cols));
                using var gBuf = backend.AllocateBuffer(add ? priorG : nan(cols));
                using var bBuf = backend.AllocateBuffer(add ? priorB : nan(cols));
                Assert.True(parts.TryLayerNormRowBackward(dyBuf, xBuf, gammaBuf, meanBuf, varBuf, scratch, dxBuf, gBuf, bBuf,
                    rows, cols, eps, add, add, add), $"declined {rows}x{cols}");
                float[] dx = backend.DownloadBuffer(dxBuf), dg = backend.DownloadBuffer(gBuf), db = backend.DownloadBuffer(bBuf);

                for (int r = 0; r < rows; r++)
                {
                    double inv = 1.0 / Math.Sqrt(variance[r] + (double)eps);
                    double s1 = 0, s2 = 0, b1 = 0;
                    for (int c = 0; c < cols; c++)
                    {
                        double gdy = gamma[c] * (double)dy[r * cols + c], xh = (x[r * cols + c] - mean[r]) * inv;
                        s1 += gdy; s2 += gdy * xh; b1 += Math.Abs(gdy);
                    }
                    s1 /= cols; s2 /= cols;
                    for (int c = 0; c < cols; c++)
                    {
                        int i = r * cols + c;
                        double gdy = gamma[c] * (double)dy[i], xh = (x[i] - mean[r]) * inv;
                        double expected = inv * (gdy - s1 - xh * s2) + (add ? priorDx[i] : 0);
                        double bound = inv * (Math.Abs(gdy) + b1 * (1 + Math.Abs(xh) * 4) / cols) + (add ? Math.Abs(priorDx[i]) : 0);
                        AssertClose(dx[i], expected, bound, $"{rows}x{cols} add={add} dx[{i}]");
                    }
                }
                for (int c = 0; c < cols; c++)
                {
                    double sg = add ? priorG[c] : 0, sb = add ? priorB[c] : 0, bg = Math.Abs(sg), bb = Math.Abs(sb);
                    for (int r = 0; r < rows; r++)
                    {
                        int i = r * cols + c;
                        double inv = 1.0 / Math.Sqrt(variance[r] + (double)eps), xh = (x[i] - mean[r]) * inv;
                        sg += dy[i] * xh; bg += Math.Abs(dy[i] * xh);
                        sb += dy[i]; bb += Math.Abs(dy[i]);
                    }
                    AssertClose(dg[c], sg, bg, $"{rows}x{cols} add={add} dGamma[{c}]");
                    AssertClose(db[c], sb, bb, $"{rows}x{cols} add={add} dBeta[{c}]");
                }
                if (firstDx is null) { firstDx = dx; firstG = dg; firstB = db; }
                else { Assert.Equal(firstDx, dx); Assert.Equal(firstG, dg); Assert.Equal(firstB, db); }
            }
        }
    }

    [SkippableFact]
    public void RowBackward_DeclinesRowsWiderThanItHolds()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var parts = (ILayerNormBackwardPartsBackend)backend;
        Assert.Equal(0, parts.LayerNormRowBackwardScratchFloats(16, 513));
        using var buf = backend.AllocateBuffer(16 * 513 * 2);
        using var small = backend.AllocateBuffer(513);
        using var other = backend.AllocateBuffer(513);
        Assert.False(parts.TryLayerNormRowBackward(buf, buf, small, small, small, buf, buf, small, other, 16, 513, 1e-5f,
            false, false, false));
    }

    private static void AssertClose(float got, double expected, double bound, string what)
    {
        if (!(Math.Abs(got - expected) <= 1e-5 * bound + 1e-6))
            Assert.Fail($"{what} = {got}, expected {expected} (term magnitude {bound})");
    }

    private static float[] Fill(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
    }
}
