// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Raw P/Invoke bindings into <c>libcusolver</c> (cusolver64_11.dll).
/// Exposes the dense linear-algebra entry points: Cholesky (potrf),
/// QR (geqrf + orgqr), SVD (gesvdj/gesvd), symmetric eigen (syevj/syevd),
/// and LU (getrf + getrs). Used by the GPU implementation of
/// <see cref="DevicePrimitives.IDeviceLinalgOps"/>.
/// </summary>
internal static class CuSolverNative
{
    private const string Lib = "cusolver64_11";

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
    }

    public enum FillMode { Lower = 0, Upper = 1 }
    public enum Side { Left = 0, Right = 1 }
    public enum EigMode { NoVector = 0, Vector = 1 }

    [DllImport(Lib)] public static extern Status cusolverDnCreate(out IntPtr handle);
    [DllImport(Lib)] public static extern Status cusolverDnDestroy(IntPtr handle);

    // Cholesky.
    [DllImport(Lib)]
    public static extern Status cusolverDnSpotrf_bufferSize(IntPtr handle, FillMode uplo, int n, IntPtr A, int lda, out int lwork);

    [DllImport(Lib)]
    public static extern Status cusolverDnSpotrf(IntPtr handle, FillMode uplo, int n, IntPtr A, int lda, IntPtr work, int lwork, IntPtr devInfo);

    // QR.
    [DllImport(Lib)]
    public static extern Status cusolverDnSgeqrf_bufferSize(IntPtr handle, int m, int n, IntPtr A, int lda, out int lwork);

    [DllImport(Lib)]
    public static extern Status cusolverDnSgeqrf(IntPtr handle, int m, int n, IntPtr A, int lda, IntPtr tau, IntPtr work, int lwork, IntPtr devInfo);

    [DllImport(Lib)]
    public static extern Status cusolverDnSorgqr(IntPtr handle, int m, int n, int k, IntPtr A, int lda, IntPtr tau, IntPtr work, int lwork, IntPtr devInfo);

    // SVD (Jacobi).
    [DllImport(Lib)]
    public static extern Status cusolverDnSgesvdj_bufferSize(IntPtr handle, EigMode jobz, int econ, int m, int n,
        IntPtr A, int lda, IntPtr S, IntPtr U, int ldu, IntPtr V, int ldv, out int lwork, IntPtr gesvdjInfo);

    [DllImport(Lib)]
    public static extern Status cusolverDnSgesvdj(IntPtr handle, EigMode jobz, int econ, int m, int n,
        IntPtr A, int lda, IntPtr S, IntPtr U, int ldu, IntPtr V, int ldv,
        IntPtr work, int lwork, IntPtr devInfo, IntPtr gesvdjInfo);

    // Symmetric eigen (Jacobi).
    [DllImport(Lib)]
    public static extern Status cusolverDnSsyevj(IntPtr handle, EigMode jobz, FillMode uplo, int n,
        IntPtr A, int lda, IntPtr W, IntPtr work, int lwork, IntPtr devInfo, IntPtr syevjInfo);

    // LU.
    [DllImport(Lib)]
    public static extern Status cusolverDnSgetrf_bufferSize(IntPtr handle, int m, int n, IntPtr A, int lda, out int lwork);

    [DllImport(Lib)]
    public static extern Status cusolverDnSgetrf(IntPtr handle, int m, int n, IntPtr A, int lda, IntPtr work, IntPtr ipiv, IntPtr devInfo);

    [DllImport(Lib)]
    public static extern Status cusolverDnSgetrs(IntPtr handle, int op, int n, int nrhs, IntPtr A, int lda, IntPtr ipiv, IntPtr B, int ldb, IntPtr devInfo);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var status = cusolverDnCreate(out var h);
            if (status == Status.Success) { cusolverDnDestroy(h); return true; }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
