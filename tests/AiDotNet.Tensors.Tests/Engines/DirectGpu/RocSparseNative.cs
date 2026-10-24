// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Raw P/Invoke bindings into <c>librocsparse</c>, kept only as the baseline <see cref="GpuSpMMHipResidentBench"/> times
/// the managed HIP CSR kernels against (#515). The library itself no longer depends on rocSPARSE.
/// </summary>
/// <remarks>
/// Values and signatures are from the ROCm 6.4 rocSPARSE headers. A rocSPARSE build carries code only for the
/// architectures it was compiled for (the Windows 6.4 build: gfx803, 900, 906, 908, 942, 1030, 1100, 1102, 1200, 1201);
/// on any other GPU its kernels write nothing, and the bench detects that from its cross-check rather than timing it.
/// </remarks>
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

    // Values from rocsparse-types.h (ROCm 6.4). IndexType.I32 was declared 0, which rocSPARSE has no index type for,
    // so every descriptor this binding built was refused with InvalidValue (#515).
    public enum IndexBase { Zero = 0, One = 1 }
    public enum IndexType { U16 = 1, I32 = 2, I64 = 3 }
    public enum Operation { NonTranspose = 111, Transpose = 112 }
    public enum SpMMAlg { Default = 0 }
    public enum Order { Row = 0, Column = 1 }

    /// <summary>
    /// rocsparse_spmm's stages. All three are required, in order: the buffer size, then preprocess, then compute. The
    /// binding used to pass 0 and 2 as "size" and "compute", which are no stage and preprocess.
    /// </summary>
    public enum SpMMStage { BufferSize = 1, Preprocess = 2, Compute = 3 }

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
        long rows, long cols, long ld, IntPtr values, DataType valueType, Order order);

    [DllImport(Lib)] public static extern Status rocsparse_destroy_spmat_descr(IntPtr descr);
    [DllImport(Lib)] public static extern Status rocsparse_destroy_dnmat_descr(IntPtr descr);

    [DllImport(Lib)]
    public static extern Status rocsparse_spmm(IntPtr handle, Operation opA, Operation opB,
        IntPtr alpha, IntPtr matA, IntPtr matB, IntPtr beta, IntPtr matC,
        DataType computeType, SpMMAlg alg, SpMMStage stage, ref ulong bufferSize, IntPtr externalBuffer);

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
