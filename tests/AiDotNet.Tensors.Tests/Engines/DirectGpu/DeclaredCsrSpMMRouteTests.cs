using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The declared CSR SpMM (#1441) through CudaBackend.CsrSpMM, against the hand-written csr_spmm it replaces, to 1e-5
/// relative: the declared reduction need not add in csr_spmm's sequential order. The matrices include empty rows.
/// Skips only where the declared route is unavailable by design; this is the verification
/// <see cref="CudaBackend.CsrSpMMDeclaredPromoted"/> waits for.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredCsrSpMMRouteTests
{
    [SkippableTheory]
    [InlineData(64, 48, 16, 0.10)]
    [InlineData(128, 256, 33, 0.05)]
    [InlineData(17, 9, 1, 0.30)]
    public void DriverOnlyDeclaredCsrSpMM_MatchesTheHandWrittenKernel(int rows, int cols, int n, double density)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var backend = CudaBackend.CreateOrThrow();
        backend.DeclaredCsrSpMMSwitch = true;
        Skip.IfNot(backend.DeclaredKernelsEnabled,
            "The declared route is off for this process or this architecture has no released codegen kernels.");

        var random = RandomHelper.CreateSeededRandom(1441);
        var rowPtr = new int[rows + 1];
        var colList = new System.Collections.Generic.List<int>();
        var valueList = new System.Collections.Generic.List<float>();
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
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
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
