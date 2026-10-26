using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The declared CSR SpMM (#1441) through CudaBackend.CsrSpMM, against the hand-written csr_spmm it replaces, to 1e-5
/// relative: the declared reduction need not add in csr_spmm's sequential order. The matrices include empty rows,
/// which must be exactly 0, and, with <c>nonFinite</c>, dense rows holding +Inf and NaN, which must reach only the
/// output rows whose entries reference them. Skips only where the declared route is unavailable by design; this is
/// the verification <see cref="CudaBackend.CsrSpMMDeclaredPromoted"/> waits for.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredCsrSpMMRouteTests
{
    [SkippableTheory]
    [InlineData(64, 48, 16, 0.10, false)]
    [InlineData(128, 256, 33, 0.05, false)]
    [InlineData(17, 9, 1, 0.30, false)]
    [InlineData(64, 48, 16, 0.10, true)]
    public void DriverOnlyDeclaredCsrSpMM_MatchesTheHandWrittenKernel(int rows, int cols, int n, double density,
        bool nonFinite)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = CudaBackend.CreateOrThrow();
        backend.DeclaredCsrSpMMSwitch = true;
        Skip.IfNot(backend.DeclaredKernelsEnabled,
            "The declared route is off for this process or this architecture has no released codegen kernels.");

        var random = RandomHelper.CreateSeededRandom(1441);
        var rowPtr = new int[rows + 1];
        var colList = new List<int>();
        var valueList = new List<float>();
        for (int r = 0; r < rows; r++)
        {
            if (r % 5 != 3)   // every fifth row stays empty
                for (int c = 0; c < cols; c++)
                    if (random.NextDouble() < density)
                    {
                        colList.Add(c);
                        valueList.Add((float)(random.NextDouble() * 4 - 2));
                    }
            rowPtr[r + 1] = colList.Count;
        }
        int nnz = colList.Count;
        Assert.True(nnz > 0);
        var denseValues = new float[cols * n];
        for (int i = 0; i < denseValues.Length; i++) denseValues[i] = (float)(random.NextDouble() * 6 - 3);
        // Dense row 0 holds +Inf and dense row 1 NaN; only the output rows with an entry in column 0 or 1 read them.
        if (nonFinite)
            for (int j = 0; j < n; j++)
            {
                denseValues[0 * n + j] = float.PositiveInfinity;
                denseValues[1 * n + j] = float.NaN;
            }

        using var values = backend.AllocateBuffer(valueList.ToArray());
        using var colIdx = Ints(backend, colList.ToArray());
        using var rowPointers = Ints(backend, rowPtr);
        using var dense = backend.AllocateBuffer(denseValues);
        using var outHand = backend.AllocateBuffer(rows * n);
        using var outDeclared = backend.AllocateBuffer(rows * n);

        backend.DeclaredCsrSpMMSwitch = false;
        backend.CsrSpMM(values, colIdx, rowPointers, dense, outHand, rows, cols, n, nnz);

        backend.DeclaredCsrSpMMSwitch = true;
        long before = backend.DeclaredProgramLaunches;
        backend.CsrSpMM(values, colIdx, rowPointers, dense, outDeclared, rows, cols, n, nnz);
        Assert.True(backend.DeclaredProgramLaunches == before + 1,
            "The declared CSR SpMM declined on a device that supports it.");

        var expected = backend.DownloadBuffer(outHand);
        var actual = backend.DownloadBuffer(outDeclared);
        Assert.Equal(expected.Length, actual.Length);
        for (int r = 0; r < rows; r++)
        {
            bool readsNonFinite = false;
            for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) readsNonFinite |= nonFinite && colList[p] <= 1;
            for (int j = 0; j < n; j++)
            {
                int i = r * n + j;
                if (rowPtr[r] == rowPtr[r + 1])
                {
                    Assert.True(actual[i] == 0f, $"row {r} is empty but its output {i} is {actual[i]:G9}");
                    continue;
                }
                if (!readsNonFinite)
                    Assert.True(!float.IsNaN(actual[i]) && !float.IsInfinity(actual[i]),
                        $"row {r} reads no non-finite dense value but its output {i} is {actual[i]:G9}");
                if (float.IsNaN(expected[i]) || float.IsInfinity(expected[i]))
                {
                    Assert.True(expected[i].Equals(actual[i]),
                        $"index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
                    continue;
                }
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                    $"index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
            }
        }
    }

    private static IGpuBuffer Ints(CudaBackend backend, int[] data)
    {
        var buffer = backend.AllocateByteBuffer(data.Length * sizeof(int));
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }
}
