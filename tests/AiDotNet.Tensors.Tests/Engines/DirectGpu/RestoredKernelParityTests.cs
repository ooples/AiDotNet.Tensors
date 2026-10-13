// Copyright (c) AiDotNet. All rights reserved.
#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Entry points whose kernels no compiled program provided (see KernelLookupIntegrityTests) now run on the device for
/// the first time, so each is compared with a CPU reference written from its documented contract: MaxAxis,
/// Copy2DStrided, CsrSegmented{Max,Min,StdDev}, BatchedDotProduct, TileBatch, BroadcastMultiplyLastAxis, Fma and the
/// SNN RbfForward / UpdateTraces / StdpUpdate. The octonion kernels are covered by
/// HyperbolicOctonionGpuCorrectnessTests.
/// </summary>
public abstract class RestoredKernelParityTestsBase
{
    private const float Sentinel = -12345f;
    private readonly IDirectGpuBackend? _backend;
    private readonly string _skipReason;

    protected RestoredKernelParityTestsBase(IDirectGpuBackend? backend, string skipReason)
    {
        _backend = backend;
        _skipReason = skipReason;
    }

    private IDirectGpuBackend Backend()
    {
        Skip.If(_backend is null, _skipReason);
        return _backend ?? throw new InvalidOperationException(_skipReason);
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
    // The outputs start as a sentinel, so a result accumulated without zeroing first (the atomicAdd kernels) fails.
    [SkippableTheory]
    [InlineData(false, 1)]
    [InlineData(false, 1000)]
    [InlineData(false, 100_003)]
    [InlineData(true, 1000)]
    [InlineData(true, 100_003)]
    public void DotProduct_MatchesCpu(bool deterministic, int size)
    {
        var backend = Backend();
        float[] a = Values(size, 20 + size), b = Values(size, 21 + size);
        double expected = 0;
        for (int i = 0; i < size; i++) expected += (double)a[i] * b[i];
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(b);
        using var dOut = Filled(backend, 1, Sentinel);
        bool previous = AiDotNet.Tensors.Engines.AiDotNetEngine.DeterministicMode;
        AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(deterministic);
        try { backend.DotProduct(da, db, dOut, size); }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(previous); }
        float actual = backend.DownloadBuffer(dOut)[0];
        Assert.True(Math.Abs(expected - actual) <= 1e-4 * Math.Sqrt(size) * 4, $"DotProduct({size}, deterministic={deterministic}): expected {expected:R}, got {actual:R}");
    }

    // result = sum(a[i] * b[bOffset + i * bStride]); a stride of 0 reads one element of b throughout.
    [SkippableTheory]
    [InlineData(false, 300, 3, 2)]
    [InlineData(false, 5000, 7, 0)]
    [InlineData(true, 300, 3, 2)]
    [InlineData(true, 5000, 7, 0)]
    public void StridedDotProduct_MatchesCpu(bool deterministic, int aSize, int bOffset, int bStride)
    {
        var backend = Backend();
        int bSize = bOffset + Math.Max(1, (aSize - 1) * bStride + 1);
        float[] a = Values(aSize, 22 + aSize), b = Values(bSize, 23 + bSize);
        double expected = 0;
        for (int i = 0; i < aSize; i++) expected += (double)a[i] * b[bOffset + i * bStride];
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(b);
        using var dOut = Filled(backend, 1, Sentinel);
        bool previous = AiDotNet.Tensors.Engines.AiDotNetEngine.DeterministicMode;
        AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(deterministic);
        try { backend.StridedDotProduct(da, db, dOut, aSize, bSize, bOffset, bStride); }
        finally { AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(previous); }
        float actual = backend.DownloadBuffer(dOut)[0];
        Assert.True(Math.Abs(expected - actual) <= 1e-4 * Math.Sqrt(aSize) * 4,
            $"StridedDotProduct({aSize}, +{bOffset}, x{bStride}, deterministic={deterministic}): expected {expected:R}, got {actual:R}");
    }
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(1000)]
    [InlineData(1_000_003)]
    public void ReduceMean_MatchesCpu(int size)
    {
        var backend = Backend();
        float[] x = Values(size, 24 + size);
        double expected = 0;
        for (int i = 0; i < size; i++) expected += x[i];
        expected /= size;
        using var dx = backend.AllocateBuffer(x);
        using var dOut = Filled(backend, 1, Sentinel);
        backend.ReduceMean(dx, dOut, size);
        float actual = backend.DownloadBuffer(dOut)[0];
        Assert.True(Math.Abs(expected - actual) <= 1e-5 * Math.Max(1.0, Math.Sqrt(size) * 2 / size * 100),
            $"ReduceMean({size}): expected {expected:R}, got {actual:R}");
    }
    [SkippableFact]
    public void Fma_MatchesCpu()
    {
        var backend = Backend();
        const int n = 1000;
        float[] a = Values(n, 9), b = Values(n, 10), c = Values(n, 11);
        var expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = (float)((double)a[i] * b[i] + c[i]);
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(b);
        using var dc = backend.AllocateBuffer(c);
        using var dOut = Filled(backend, n, Sentinel);
        backend.Fma(da, db, dc, dOut, n);
        AssertClose(expected, backend.DownloadBuffer(dOut), 1e-6, "Fma");
    }

    [SkippableTheory]
    [InlineData(1, 1, 1)]
    [InlineData(37, 20, 9)]
    public void RbfForward_MatchesCpu(int batch, int centers, int dim)
    {
        var backend = Backend();
        float[] x = Values(batch * dim, 12 + batch), cs = Values(centers * dim, 13 + centers);
        var eps = new float[centers];
        for (int c = 0; c < centers; c++) eps[c] = 0.1f + 0.05f * c;
        var expected = new float[batch * centers];
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < centers; c++)
            {
                double d2 = 0;
                for (int d = 0; d < dim; d++) { double diff = x[b * dim + d] - cs[c * dim + d]; d2 += diff * diff; }
                expected[b * centers + c] = (float)Math.Exp(-eps[c] * d2);
            }
        using var dx = backend.AllocateBuffer(x);
        using var dc = backend.AllocateBuffer(cs);
        using var de = backend.AllocateBuffer(eps);
        using var dOut = Filled(backend, batch * centers, Sentinel);
        backend.RbfForward(dx, dc, de, dOut, batch, centers, dim);
        AssertClose(expected, backend.DownloadBuffer(dOut), 1e-5, "RbfForward");
    }

    [SkippableFact]
    public void UpdateTraces_MatchesCpu()
    {
        var backend = Backend();
        const int n = 700;
        const float decay = 0.9f, threshold = 0.5f;
        float[] traces = Values(n, 14), input = Values(n, 15);
        var expTraces = new float[n];
        var expSpikes = new float[n];
        for (int i = 0; i < n; i++)
        {
            bool spike = input[i] > threshold;
            expSpikes[i] = spike ? 1f : 0f;
            expTraces[i] = decay * traces[i] + (spike ? 1f : 0f);
        }
        using var dt = backend.AllocateBuffer(traces);
        using var ds = Filled(backend, n, Sentinel);
        using var di = backend.AllocateBuffer(input);
        backend.UpdateTraces(dt, ds, di, decay, threshold, n);
        AssertClose(expTraces, backend.DownloadBuffer(dt), 1e-6, "UpdateTraces.traces");
        AssertClose(expSpikes, backend.DownloadBuffer(ds), 0, "UpdateTraces.spikes");
    }

    [SkippableTheory]
    [InlineData(1, 1)]
    [InlineData(23, 41)]
    public void StdpUpdate_MatchesCpu(int numPre, int numPost)
    {
        var backend = Backend();
        const float ltp = 0.01f, ltd = 0.012f, homeo = 0.001f, wMin = -1f, wMax = 1f;
        float[] w = Values(numPre * numPost, 16 + numPre), preT = Values(numPre, 17), postT = Values(numPost, 18);
        var rng = new Random(19);
        var preS = new float[numPre];
        var postS = new float[numPost];
        for (int i = 0; i < numPre; i++) preS[i] = rng.Next(2);
        for (int j = 0; j < numPost; j++) postS[j] = rng.Next(2);
        var expected = new float[w.Length];
        for (int i = 0; i < numPre; i++)
            for (int j = 0; j < numPost; j++)
            {
                float wij = w[i * numPost + j], dw = 0f;
                if (preS[i] > 0.5f && postT[j] > 0f) dw += ltp * postT[j];
                if (postS[j] > 0.5f && preT[i] > 0f) dw -= ltd * preT[i];
                dw += homeo * ((wMin + wMax) * 0.5f - wij);
                expected[i * numPost + j] = Math.Min(Math.Max(wij + dw, wMin), wMax);
            }
        using var dw0 = backend.AllocateBuffer(w);
        using var dPreT = backend.AllocateBuffer(preT);
        using var dPostT = backend.AllocateBuffer(postT);
        using var dPreS = backend.AllocateBuffer(preS);
        using var dPostS = backend.AllocateBuffer(postS);
        backend.StdpUpdate(dw0, dPreT, dPostT, dPreS, dPostS, ltp, ltd, homeo, wMin, wMax, numPre, numPost);
        AssertClose(expected, backend.DownloadBuffer(dw0), 1e-6, "StdpUpdate");
    }
}

[Collection("DirectGpuSerial")]
public sealed class HipRestoredKernelParityTests : RestoredKernelParityTestsBase, IClassFixture<HipTensorEngineTestFixture>
{
    public HipRestoredKernelParityTests(HipTensorEngineTestFixture fixture)
        : base(fixture.IsAvailable ? fixture.Engine?.GetBackend() : null, "HIP GPU backend not available") { }
}

[Collection("OpenClBackend")]
public sealed class OpenClRestoredKernelParityTests : RestoredKernelParityTestsBase, IClassFixture<OpenClBackendTestFixture>
{
    public OpenClRestoredKernelParityTests(OpenClBackendTestFixture fixture)
        : base(fixture.IsAvailable ? fixture.Backend : null, "OpenCL GPU backend not available") { }
}
#endif