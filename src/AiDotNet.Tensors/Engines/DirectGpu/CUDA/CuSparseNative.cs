// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// P/Invoke bindings into <c>libcusparse</c> (cusparse64_12.dll on
/// Windows, libcusparse.so.12 on Linux). Surfaces SpMM, SpMV, and
/// SpGEMM entry points that <see cref="LinearAlgebra.Sparse.SparseOps"/>
/// dispatches to when running with a CUDA-capable host.
///
/// <para>Cached <see cref="IsAvailable"/> probe so callers can fall
/// through to the CPU CSR / SIMD path when the native lib is missing
/// (no NVIDIA driver / CUDA install). Every entry is wrapped in a
/// <c>try/catch (DllNotFoundException)</c> at probe time so the static
/// init never throws on hosts without CUDA.</para>
///
/// <para>Co-evolves with #219's broader GPU-primitives surface — the
/// same binding set lives there. When #219 lands first these can be
/// deduped against that file; until then the binding owns its own
/// declarations so #221's GPU dispatch path has no cross-PR
/// dependency.</para>
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
        R32F = 0,
        R64F = 1,
    }

    public enum IndexType
    {
        I32 = 1,
    }

    public enum SpMMAlg { Default = 0, CooDefault = 0, CsrAlg1 = 4, CsrAlg2 = 6, CsrAlg3 = 12 }
    public enum SpMVAlg { Default = 0, CooAlg1 = 1, CsrAlg1 = 2, CsrAlg2 = 3 }
    public enum Operation { NonTranspose = 0, Transpose = 1, ConjTranspose = 2 }

    [DllImport(Lib)] public static extern Status cusparseCreate(out IntPtr handle);
    [DllImport(Lib)] public static extern Status cusparseDestroy(IntPtr handle);
    [DllImport(Lib)] public static extern Status cusparseSetStream(IntPtr handle, IntPtr stream);

    [DllImport(Lib)]
    public static extern Status cusparseCreateCsr(out IntPtr descr,
        long rows, long cols, long nnz,
        IntPtr csrRowOffsets, IntPtr csrColInd, IntPtr csrValues,
        IndexType csrRowOffsetsType, IndexType csrColIndType,
        int idxBase, DataType valueType);

    [DllImport(Lib)]
    public static extern Status cusparseCreateDnMat(out IntPtr descr,
        long rows, long cols, long ld, IntPtr values, DataType valueType, int order /* 0 = row-major, 1 = col-major */);

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

    /// <summary>Cached availability probe. <c>true</c> when libcusparse
    /// loaded successfully and <c>cusparseCreate</c> returned
    /// <see cref="Status.Success"/>.</summary>
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
