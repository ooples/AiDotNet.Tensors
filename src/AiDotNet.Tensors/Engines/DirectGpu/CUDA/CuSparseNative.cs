// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Raw P/Invoke bindings into <c>libcusparse</c>. Surfaces SpMM, SpMV,
/// and SpGEMM entry points. Cached <see cref="IsAvailable"/> probe so
/// callers can fall through to a CPU CSR implementation when the native
/// lib is missing.
/// </summary>
internal static class CuSparseNative
{
    private const string Lib = "cusparse64_12";

    public enum Status
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 2,
        InvalidValue = 3,
        ArchMismatch = 4,
        MappingError = 5,
        ExecutionFailed = 6,
        InternalError = 7,
        MatrixTypeNotSupported = 8,
        ZeroPivot = 9,
        NotSupported = 10,
        InsufficientResources = 11,
    }

    public enum DataType
    {
        R32F = 0,  // float
        R64F = 1,  // double
    }

    public enum IndexType
    {
        I32 = 1,
    }

    public enum SpMatDescr_Format { Csr = 1, Csc = 2, Coo = 3 }
    public enum DnMatDescr_Order { RowMajor = 0, ColMajor = 1 }

    public enum Operation
    {
        NonTranspose = 0,
        Transpose = 1,
        ConjugateTranspose = 2,
    }

    public enum SpMMAlg
    {
        Default = 0,
        CooAlg1 = 1,
        CooAlg2 = 2,
        CooAlg3 = 3,
        CsrAlg1 = 4,
        CooAlg4 = 5,
        CsrAlg2 = 6,
        CsrAlg3 = 12,
        BlockedEllAlg1 = 13,
    }

    [DllImport(Lib)] public static extern Status cusparseCreate(out IntPtr handle);
    [DllImport(Lib)] public static extern Status cusparseDestroy(IntPtr handle);

    [DllImport(Lib)]
    public static extern Status cusparseCreateCsr(out IntPtr descr,
        long rows, long cols, long nnz,
        IntPtr csrRowOffsets, IntPtr csrColInd, IntPtr csrValues,
        IndexType csrRowOffsetsType, IndexType csrColIndType,
        int idxBase, DataType valueType);

    [DllImport(Lib)]
    public static extern Status cusparseCreateDnMat(out IntPtr descr,
        long rows, long cols, long ld, IntPtr values, DataType valueType, DnMatDescr_Order order);

    [DllImport(Lib)] public static extern Status cusparseDestroySpMat(IntPtr descr);
    [DllImport(Lib)] public static extern Status cusparseDestroyDnMat(IntPtr descr);

    [DllImport(Lib)]
    public static extern Status cusparseSpMM_bufferSize(IntPtr handle, Operation opA, Operation opB,
        IntPtr alpha, IntPtr matA, IntPtr matB, IntPtr beta, IntPtr matC,
        DataType computeType, SpMMAlg alg, out ulong bufferSize);

    [DllImport(Lib)]
    public static extern Status cusparseSpMM(IntPtr handle, Operation opA, Operation opB,
        IntPtr alpha, IntPtr matA, IntPtr matB, IntPtr beta, IntPtr matC,
        DataType computeType, SpMMAlg alg, IntPtr externalBuffer);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var status = cusparseCreate(out var h);
            if (status == Status.Success) { cusparseDestroy(h); return true; }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
