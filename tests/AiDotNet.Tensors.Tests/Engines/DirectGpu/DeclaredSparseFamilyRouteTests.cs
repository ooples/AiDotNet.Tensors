using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The rest of the sparse kernels through declared programs (#1441) against the hand-written kernels on the device:
/// the warp, vec4 and bias layouts of CSR SpMM, SDDMM, and the weighted and unweighted edge scatter-add.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredSparseFamilyRouteTests
{
    private static IGpuBuffer Ints(CudaBackend backend, int[] data)
    {
        var buffer = backend.AllocateByteBuffer(data.Length * sizeof(int));
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i]) || float.IsInfinity(expected[i]))
            {
                Assert.True(expected[i].Equals(actual[i]), $"{what} index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
                continue;
            }
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"{what} index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
        }
    }

    private static CudaBackend OpenOrSkip()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var backend = CudaBackend.CreateOrThrow();
        backend.DeclaredCsrSpMMSwitch = true;
        if (!backend.DeclaredKernelsEnabled)
        {
            backend.Dispose();
            Skip.If(true, "The declared route is off for this process or this architecture has no released codegen kernels.");
        }
        return backend;
    }

    private static float[] Run(CudaBackend backend, bool declared, int size, Action<IGpuBuffer> call)
    {
        using var output = backend.AllocateBuffer(size);
        backend.DeclaredCsrSpMMSwitch = declared;
        long before = backend.DeclaredProgramLaunches;
        call(output);
        if (declared)
            Assert.True(backend.DeclaredProgramLaunches == before + 1, "The declared route declined on a device that supports it.");
        return backend.DownloadBuffer(output);
    }

    [SkippableTheory]
    [InlineData(64, 48, 16, 0.10)]
    [InlineData(17, 9, 4, 0.30)]
    [InlineData(128, 256, 32, 0.05)]
    public void CsrLayoutsAndSddmm_MatchTheHandWrittenKernels(int rows, int cols, int n, double density)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1441 + rows);
        var rowPtr = new int[rows + 1];
        var colList = new List<int>();
        var rowList = new List<int>();
        var valueList = new List<float>();
        for (int r = 0; r < rows; r++)
        {
            if (r % 5 != 3)
                for (int c = 0; c < cols; c++)
                    if (random.NextDouble() < density)
                    {
                        colList.Add(c);
                        rowList.Add(r);
                        valueList.Add((float)(random.NextDouble() * 4 - 2));
                    }
            rowPtr[r + 1] = colList.Count;
        }
        int nnz = colList.Count;
        var denseValues = new float[cols * n];
        for (int i = 0; i < denseValues.Length; i++) denseValues[i] = (float)(random.NextDouble() * 6 - 3);
        var biasValues = new float[n];
        for (int i = 0; i < n; i++) biasValues[i] = (float)(random.NextDouble() * 2 - 1);

        using var values = backend.AllocateBuffer(valueList.ToArray());
        using var colIdx = Ints(backend, colList.ToArray());
        using var rowIdx = Ints(backend, rowList.ToArray());
        using var rowPointers = Ints(backend, rowPtr);
        using var dense = backend.AllocateBuffer(denseValues);
        using var bias = backend.AllocateBuffer(biasValues);


        AssertClose(
            Run(backend, false, rows * n, o => backend.CsrSpMMWarp(values, colIdx, rowPointers, dense, o, rows, cols, n, nnz)),
            Run(backend, true, rows * n, o => backend.CsrSpMMWarp(values, colIdx, rowPointers, dense, o, rows, cols, n, nnz)),
            "warp");
        AssertClose(
            Run(backend, false, rows * n, o => backend.CsrSpMMVec4(values, colIdx, rowPointers, dense, o, rows, cols, n, nnz)),
            Run(backend, true, rows * n, o => backend.CsrSpMMVec4(values, colIdx, rowPointers, dense, o, rows, cols, n, nnz)),
            "vec4");
        AssertClose(
            Run(backend, false, rows * n, o => backend.CsrSpMMBias(values, colIdx, rowPointers, dense, bias, o, rows, cols, n, nnz)),
            Run(backend, true, rows * n, o => backend.CsrSpMMBias(values, colIdx, rowPointers, dense, bias, o, rows, cols, n, nnz)),
            "bias");

        // SDDMM at the same pattern: x [rows, n], y [cols, n].
        var xValues = new float[rows * n];
        for (int i = 0; i < xValues.Length; i++) xValues[i] = (float)(random.NextDouble() * 2 - 1);
        using var x = backend.AllocateBuffer(xValues);
        AssertClose(
            Run(backend, false, nnz, o => backend.CsrSddmm(rowIdx, colIdx, x, dense, o, nnz, n)),
            Run(backend, true, nnz, o => backend.CsrSddmm(rowIdx, colIdx, x, dense, o, nnz, n)),
            "sddmm");
    }

    [SkippableTheory]
    [InlineData(64, 48, 16, 0.10)]
    [InlineData(17, 9, 1, 0.30)]
    [InlineData(128, 256, 33, 0.05)]
    public void CsrSpMMDouble_MatchesTheHandWrittenKernel(int rows, int cols, int n, double density)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1446 + rows);
        var rowPtr = new int[rows + 1];
        var colList = new List<int>();
        var valueList = new List<double>();
        for (int r = 0; r < rows; r++)
        {
            if (r % 5 != 3)
                for (int c = 0; c < cols; c++)
                    if (random.NextDouble() < density) { colList.Add(c); valueList.Add(random.NextDouble() * 4 - 2); }
            rowPtr[r + 1] = colList.Count;
        }
        int nnz = colList.Count;
        var dense = new double[cols * n];
        for (int i = 0; i < dense.Length; i++) dense[i] = random.NextDouble() * 6 - 3;

        IGpuBuffer Doubles(double[] data)
        {
            var buffer = backend.AllocateByteBuffer(data.Length * sizeof(double));
            var bytes = new byte[data.Length * sizeof(double)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            backend.UploadByteBuffer(buffer, bytes);
            return buffer;
        }
        using var values = Doubles(valueList.ToArray());
        using var denseBuf = Doubles(dense);
        using var colIdx = Ints(backend, colList.ToArray());
        using var rowPointers = Ints(backend, rowPtr);
        var results = new double[2][];
        foreach (bool declared in new[] { false, true })
        {
            using var output = backend.AllocateByteBuffer(rows * n * sizeof(double));
            backend.DeclaredCsrSpMMSwitch = declared;
            long before = backend.DeclaredProgramLaunches;
            backend.CsrSpMMDouble(values, colIdx, rowPointers, denseBuf, output, rows, cols, n, nnz);
            if (declared) Assert.True(backend.DeclaredProgramLaunches == before + 1, "The declared fp64 route declined.");
            var bytes = backend.DownloadByteBuffer(output, rows * n * sizeof(double));
            results[declared ? 1 : 0] = new double[rows * n];
            Buffer.BlockCopy(bytes, 0, results[declared ? 1 : 0], 0, bytes.Length);
        }
        for (int i = 0; i < rows * n; i++)
            Assert.True(Math.Abs(results[0][i] - results[1][i]) <= 1e-12 * Math.Max(1, Math.Abs(results[0][i])),
                $"index {i}: hand-written {results[0][i]:R}, declared {results[1][i]:R}");
    }

    [SkippableTheory]
    [InlineData(64, 48, 16, 0.10)]
    [InlineData(17, 9, 4, 0.30)]
    [InlineData(200, 100, 8, 0.02)]
    public void CsrSegmentedReductions_MatchTheHandWrittenKernels(int rows, int k, int n, double density)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1444 + rows);
        var rowPtr = new List<float> { 0 };
        var cols = new List<float>();
        for (int r = 0; r < rows; r++)
        {
            if (r % 5 != 3)
                for (int c = 0; c < k; c++)
                    if (random.NextDouble() < density) cols.Add(c);
            rowPtr.Add(cols.Count);
        }
        cols.Add(0);   // a slot past the last row pointer belongs to no row
        var inputValues = new float[k * n];
        for (int i = 0; i < inputValues.Length; i++) inputValues[i] = (float)(random.NextDouble() * 6 - 3);
        inputValues[0] = float.NegativeInfinity;
        inputValues[1] = float.NaN;
        using var colIdx = backend.AllocateBuffer(cols.ToArray());
        using var rowPointers = backend.AllocateBuffer(rowPtr.ToArray());
        using var input = backend.AllocateBuffer(inputValues);

        AssertClose(
            Run(backend, false, rows * n, o => backend.CsrSegmentedMax(colIdx, rowPointers, input, o, rows, k, n)),
            Run(backend, true, rows * n, o => backend.CsrSegmentedMax(colIdx, rowPointers, input, o, rows, k, n)),
            "segmented max");
        AssertClose(
            Run(backend, false, rows * n, o => backend.CsrSegmentedMin(colIdx, rowPointers, input, o, rows, k, n)),
            Run(backend, true, rows * n, o => backend.CsrSegmentedMin(colIdx, rowPointers, input, o, rows, k, n)),
            "segmented min");
        inputValues[0] = 0.5f;
        inputValues[1] = -1.5f;
        using var finite = backend.AllocateBuffer(inputValues);
        AssertClose(
            Run(backend, false, rows * n, o => backend.CsrSegmentedStdDev(colIdx, rowPointers, finite, o, rows, k, n)),
            Run(backend, true, rows * n, o => backend.CsrSegmentedStdDev(colIdx, rowPointers, finite, o, rows, k, n)),
            "segmented stddev");
    }

    [SkippableTheory]
    [InlineData(50, 200, 16, false)]
    [InlineData(50, 200, 16, true)]
    [InlineData(300, 1000, 8, true)]
    [InlineData(7, 3, 33, false)]
    public void EdgeScatterAdd_MatchesTheHandWrittenKernel(int numNodes, int numEdges, int features, bool weighted)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1443 + numEdges);
        var src = new int[numEdges];
        var tgt = new int[numEdges];
        var w = new float[numEdges];
        for (int e = 0; e < numEdges; e++)
        {
            src[e] = random.Next(numNodes);
            tgt[e] = random.Next(Math.Max(1, numNodes - 1));   // the last node receives nothing
            w[e] = (float)(random.NextDouble() * 2 - 1);
        }
        var inputValues = new float[numNodes * features];
        for (int i = 0; i < inputValues.Length; i++) inputValues[i] = (float)(random.NextDouble() * 4 - 2);

        using var input = backend.AllocateBuffer(inputValues);
        using var srcBuf = Ints(backend, src);
        using var tgtBuf = Ints(backend, tgt);
        using var weights = backend.AllocateBuffer(w);
        var edgeValues = weighted ? weights : null;
        var expected = Run(backend, false, numNodes * features,
            o => backend.ScatterAddEdges(input, srcBuf, tgtBuf, edgeValues, o, numNodes, numEdges, features));
        var actual = Run(backend, true, numNodes * features,
            o => backend.ScatterAddEdges(input, srcBuf, tgtBuf, edgeValues, o, numNodes, numEdges, features));
        AssertClose(expected, actual, weighted ? "weighted edges" : "edges");
        for (int f = 0; f < features; f++) Assert.Equal(0f, actual[(numNodes - 1) * features + f]);
    }
}