// Copyright (c) AiDotNet. All rights reserved.
#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The HIP entry points whose kernels were never compiled (see HipKernelLookupIntegrityTests) now run on the device
/// for the first time, so each is compared with a CPU reference written from its documented contract: MaxAxis,
/// Copy2DStrided, CsrSegmented{Max,Min,StdDev}, BatchedDotProduct, TileBatch and BroadcastMultiplyLastAxis.
/// The octonion kernels are covered by HipHyperbolicOctonionGpuCorrectnessTests.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipRestoredKernelParityTests : IClassFixture<HipTensorEngineTestFixture>
{
    private const float Sentinel = -12345f;
    private readonly HipTensorEngineTestFixture _fixture;

    public HipRestoredKernelParityTests(HipTensorEngineTestFixture fixture) => _fixture = fixture;

    private IDirectGpuBackend Backend()
    {
        Skip.IfNot(_fixture.IsAvailable, "HIP GPU backend not available");
        var engine = _fixture.Engine ?? throw new InvalidOperationException("HIP fixture reports available without an engine.");
        return engine.GetBackend();
    }

    private static float[] Values(int count, int seed)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 4 - 2);
        return a;
    }

    private static void AssertClose(float[] expected, float[] actual, double tolerance, string what)
    {
        Assert.True(actual.Length >= expected.Length, $"{what}: {actual.Length} outputs for {expected.Length} expected");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1.0, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static IGpuBuffer Filled(IDirectGpuBackend backend, int n, float value)
    {
        var buffer = backend.AllocateBuffer(n);
        backend.Fill(buffer, value, n);
        return buffer;
    }

    [SkippableTheory]
    [InlineData(1, 1)]
    [InlineData(7, 300)]
    [InlineData(1000, 3)]
    public void MaxAxis_MatchesCpu(int outer, int reduce)
    {
        var backend = Backend();
        float[] x = Values(outer * reduce, 1 + outer);
        var expected = new float[outer];
        for (int o = 0; o < outer; o++)
        {
            float m = x[o * reduce];
            for (int i = 1; i < reduce; i++) m = Math.Max(m, x[o * reduce + i]);
            expected[o] = m;
        }
        using var dx = backend.AllocateBuffer(x);
        using var o1 = Filled(backend, outer, Sentinel);
        backend.MaxAxis(dx, o1, outer, reduce);
        AssertClose(expected, backend.DownloadBuffer(o1), 0, "MaxAxis");
    }

    [SkippableTheory]
    [InlineData(1, 1, 3, 2)]
    [InlineData(5, 300, 700, 250)]
    [InlineData(70000, 2, 5, 3)]
    public void Copy2DStrided_WritesOnlyTheTargetColumns(int rows, int srcCols, int destTotalCols, int destColOffset)
    {
        var backend = Backend();
        float[] src = Values(rows * srcCols, 2 + rows);
        var expected = new float[rows * destTotalCols];
        for (int i = 0; i < expected.Length; i++) expected[i] = Sentinel;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < srcCols; c++)
                expected[r * destTotalCols + destColOffset + c] = src[r * srcCols + c];
        using var ds = backend.AllocateBuffer(src);
        using var dd = Filled(backend, expected.Length, Sentinel);
        backend.Copy2DStrided(ds, dd, rows, srcCols, destTotalCols, destColOffset);
        AssertClose(expected, backend.DownloadBuffer(dd), 0, $"Copy2DStrided rows={rows}");
    }

    private enum CsrReduction { Max, Min, StdDev }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void CsrSegmented_MatchesCpu(int which)
    {
        var backend = Backend();
        var op = (CsrReduction)which;
        // Row 2 is empty, which every reduction writes as 0.
        const int K = 9, N = 300;
        int[] rowPointers = { 0, 3, 7, 7, 9, 12 };
        int[] cols = { 0, 4, 8, 1, 2, 3, 5, 6, 7, 8, 0, 2 };
        int M = rowPointers.Length - 1;
        const float epsilon = 1e-5f;
        float[] input = Values(K * N, 3 + which);
        var expected = new float[M * N];
        for (int row = 0; row < M; row++)
            for (int j = 0; j < N; j++)
            {
                int start = rowPointers[row], end = rowPointers[row + 1];
                if (start == end) { expected[row * N + j] = 0f; continue; }
                double acc = op == CsrReduction.Max ? double.MinValue : op == CsrReduction.Min ? double.MaxValue : 0;
                for (int i = start; i < end; i++)
                {
                    double v = input[cols[i] * N + j];
                    acc = op == CsrReduction.Max ? Math.Max(acc, v) : op == CsrReduction.Min ? Math.Min(acc, v) : acc + v;
                }
                if (op == CsrReduction.StdDev)
                {
                    double mean = acc / (end - start), sq = 0;
                    for (int i = start; i < end; i++) { double d = input[cols[i] * N + j] - mean; sq += d * d; }
                    acc = Math.Sqrt(sq / (end - start) + epsilon);
                }
                expected[row * N + j] = (float)acc;
            }
        using var dCols = backend.AllocateBuffer(Array.ConvertAll(cols, c => (float)c));
        using var dRows = backend.AllocateBuffer(Array.ConvertAll(rowPointers, r => (float)r));
        using var dIn = backend.AllocateBuffer(input);
        using var dOut = Filled(backend, M * N, Sentinel);
        switch (op)
        {
            case CsrReduction.Max: backend.CsrSegmentedMax(dCols, dRows, dIn, dOut, M, K, N); break;
            case CsrReduction.Min: backend.CsrSegmentedMin(dCols, dRows, dIn, dOut, M, K, N); break;
            default: backend.CsrSegmentedStdDev(dCols, dRows, dIn, dOut, M, K, N, epsilon); break;
        }
        AssertClose(expected, backend.DownloadBuffer(dOut), 1e-5, $"CsrSegmented{op}");
    }

    [SkippableTheory]
    [InlineData(1, 1)]
    [InlineData(3, 255)]
    [InlineData(300, 1000)]
    public void BatchedDotProduct_MatchesCpu(int batch, int dim)
    {
        var backend = Backend();
        float[] a = Values(batch * dim, 4 + dim), b = Values(batch * dim, 5 + dim);
        var expected = new float[batch];
        for (int r = 0; r < batch; r++)
        {
            double s = 0;
            for (int i = 0; i < dim; i++) s += (double)a[r * dim + i] * b[r * dim + i];
            expected[r] = (float)s;
        }
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(b);
        using var dOut = Filled(backend, batch, Sentinel);
        backend.BatchedDotProduct(da, db, dOut, batch, dim);
        // An absolute bound scaled by sqrt(dim): float32 accumulation of dim products of magnitude <= 4.
        float[] actual = backend.DownloadBuffer(dOut);
        for (int r = 0; r < batch; r++)
            Assert.True(Math.Abs(expected[r] - actual[r]) <= 1e-4 * Math.Sqrt(dim) * 4,
                $"BatchedDotProduct[{r}]: expected {expected[r]:R}, got {actual[r]:R}");
    }

    [SkippableTheory]
    [InlineData(1, 1)]
    [InlineData(4, 300)]
    public void TileBatch_RepeatsEachElement(int repeats, int innerSize)
    {
        var backend = Backend();
        float[] x = Values(innerSize, 6 + innerSize);
        var expected = new float[innerSize * repeats];
        for (int i = 0; i < innerSize; i++)
            for (int r = 0; r < repeats; r++)
                expected[i * repeats + r] = x[i];
        using var dx = backend.AllocateBuffer(x);
        using var dOut = Filled(backend, expected.Length, Sentinel);
        backend.TileBatch(dx, dOut, repeats, innerSize);
        AssertClose(expected, backend.DownloadBuffer(dOut), 0, "TileBatch");
    }

    [SkippableTheory]
    [InlineData(1, 1)]
    [InlineData(37, 300)]
    public void BroadcastMultiplyLastAxis_MatchesCpu(int outer, int inner)
    {
        var backend = Backend();
        float[] a = Values(outer * inner, 7 + outer), b = Values(inner, 8 + inner);
        var expected = new float[outer * inner];
        for (int i = 0; i < expected.Length; i++) expected[i] = a[i] * b[i % inner];
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(b);
        using var dOut = Filled(backend, expected.Length, Sentinel);
        backend.BroadcastMultiplyLastAxis(da, db, dOut, outer, inner);
        AssertClose(expected, backend.DownloadBuffer(dOut), 1e-6, "BroadcastMultiplyLastAxis");
    }
}
#endif