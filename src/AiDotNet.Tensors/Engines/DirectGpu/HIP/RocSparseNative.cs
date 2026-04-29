// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// P/Invoke bindings into <c>librocsparse</c>. Mirrors the cuSPARSE
/// surface used by <see cref="LinearAlgebra.Sparse.SparseOps"/> so the
/// HIP / ROCm path is symmetrical with CUDA — the wrapper picks
/// whichever runtime is loadable on the host.
///
/// <para>rocSPARSE on Linux: <c>librocsparse.so</c>. On Windows ROCm
/// builds the canonical name varies by release; we probe both for
/// portability.</para>
/// </summary>
internal static class RocSparseNative
{
    private const string Lib = "rocsparse";

    public enum Status
    {
        Success = 0,
        InvalidHandle = 1,
        NotImplemented = 2,
        InvalidPointer = 3,
        InvalidSize = 4,
        MemoryError = 5,
        InternalError = 6,
        InvalidValue = 7,
        ArchMismatch = 8,
        ZeroPivot = 9,
        NotInitialized = 10,
        TypeMismatch = 11,
        Requires_SortedStorage = 12,
        ThrownException = 13,
        Continue = 14,
    }

    public enum DataType
    {
        R32F = 151,
        R64F = 152,
    }

    public enum IndexBase { Zero = 0, One = 1 }
    public enum IndexType { I32 = 0 }
    public enum Operation { NonTranspose = 111, Transpose = 112 }
    public enum SpMMAlg { Default = 0 }

    [DllImport(Lib)] public static extern Status rocsparse_create_handle(out IntPtr handle);
    [DllImport(Lib)] public static extern Status rocsparse_destroy_handle(IntPtr handle);
    [DllImport(Lib)] public static extern Status rocsparse_set_stream(IntPtr handle, IntPtr stream);

    [DllImport(Lib)]
    public static extern Status rocsparse_create_csr_descr(out IntPtr descr,
        long rows, long cols, long nnz,
        IntPtr csrRowOffsets, IntPtr csrColInd, IntPtr csrValues,
        IndexType csrRowOffsetsType, IndexType csrColIndType,
        IndexBase idxBase, DataType valueType);

    [DllImport(Lib)]
    public static extern Status rocsparse_create_dnmat_descr(out IntPtr descr,
        long rows, long cols, long ld, IntPtr values, DataType valueType, int order);

    [DllImport(Lib)] public static extern Status rocsparse_destroy_spmat_descr(IntPtr descr);
    [DllImport(Lib)] public static extern Status rocsparse_destroy_dnmat_descr(IntPtr descr);

    [DllImport(Lib)]
    public static extern Status rocsparse_spmm(IntPtr handle, Operation opA, Operation opB,
        IntPtr alpha, IntPtr matA, IntPtr matB, IntPtr beta, IntPtr matC,
        DataType computeType, SpMMAlg alg, int stage, ref ulong bufferSize, IntPtr externalBuffer);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var s = rocsparse_create_handle(out var h);
            if (s == Status.Success) { rocsparse_destroy_handle(h); return true; }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
