using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level wrapper over cuBLASLt's fused-epilogue matmul. Exposes
/// <c>D = alpha × A @ B + beta × C</c> with optional fused bias + ReLU
/// / GELU / Tanh-GELU in a single kernel launch.
///
/// <para><b>Why this matters:</b> transformer FFN blocks and attention
/// projections are (matmul + bias + activation) sequences — three
/// kernels as raw cuBLAS + elementwise ops. cuBLASLt fuses them into
/// one launch with one set of memory round-trips, which is frequently
/// a 1.5-2× end-to-end inference speedup on H100s.</para>
///
/// <para>Descriptor lifetime is per-call for simplicity; a future
/// optimization caches descriptors keyed by (dtype, layout, epilogue)
/// so a repeated call set reuses the handles.</para>
/// </summary>
public sealed class CuBlasLtMatmul : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    /// <summary>True iff libcublasLt can be loaded at runtime.</summary>
    public static bool IsAvailable
    {
        get
        {
            try
            {
                CuBlasLtNative.cublasLtCreate(out var h);
                CuBlasLtNative.cublasLtDestroy(h);
                return true;
            }
            catch { return false; }
        }
    }

    public CuBlasLtMatmul()
    {
        var status = CuBlasLtNative.cublasLtCreate(out _handle);
        if (status != CublasStatus.Success)
            throw new InvalidOperationException($"cublasLtCreate failed: {status}.");
    }

    /// <summary>
    /// Run <c>D = alpha × (op(A) @ op(B)) + beta × C</c> with fused
    /// epilogue. All pointers are device pointers; caller owns
    /// allocation + stream.
    /// </summary>
    /// <param name="aDev">Device pointer to A; shape (m, k) row-major
    /// when <paramref name="transA"/>=false.</param>
    /// <param name="bDev">Device pointer to B; shape (k, n) row-major
    /// when <paramref name="transB"/>=false.</param>
    /// <param name="cDev">Device pointer to C (often null → beta must be 0).</param>
    /// <param name="dDev">Device pointer to D (output).</param>
    /// <param name="biasDev">Device pointer to rank-1 bias of length n;
    /// null disables bias fusion (must match epilogue choice).</param>
    /// <param name="epilogue">Fused post-matmul op.</param>
    public void MatmulFused(
        IntPtr aDev, int m, int k, bool transA,
        IntPtr bDev, int n, bool transB,
        IntPtr cDev, IntPtr dDev,
        IntPtr biasDev,
        CublasLtEpilogue epilogue,
        float alpha = 1f, float beta = 0f,
        IntPtr workspace = default, ulong workspaceSizeInBytes = 0,
        IntPtr stream = default,
        CublasDataType dtype = CublasDataType.Float32,
        CublasComputeType computeType = CublasComputeType.Float32,
        CublasDataType? outputDtype = null)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuBlasLtMatmul));

        IntPtr opDesc = IntPtr.Zero, aDesc = IntPtr.Zero, bDesc = IntPtr.Zero, cDesc = IntPtr.Zero, dDesc = IntPtr.Zero;
        try
        {
            Check(CuBlasLtNative.cublasLtMatmulDescCreate(out opDesc, computeType, CublasDataType.Float32), "DescCreate");

            // Set transpose flags + epilogue + bias pointer if applicable.
            int tA = transA ? 1 : 0;
            int tB = transB ? 1 : 0;
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransA, ref tA, sizeof(int));
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransB, ref tB, sizeof(int));
            int epi = (int)epilogue;
            SetAttr(opDesc, CublasLtMatmulDescAttributes.Epilogue, ref epi, sizeof(int));
            if (biasDev != IntPtr.Zero)
            {
                long biasPtrRaw = biasDev.ToInt64();
                SetAttr(opDesc, CublasLtMatmulDescAttributes.EpilogueBiasPointer, ref biasPtrRaw, sizeof(long));
            }

            // Layout descriptors — cuBLAS is column-major natively,
            // so for row-major C# data we pass ld = inner stride and
            // flip the transposes downstream. Callers should arrange
            // data accordingly; we document the convention and leave
            // the choice to them rather than silently reinterpreting.
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out aDesc, dtype, (ulong)m, (ulong)k, transA ? k : m), "A layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out bDesc, dtype, (ulong)k, (ulong)n, transB ? n : k), "B layout");
            CublasDataType resultType = outputDtype ?? dtype;
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out cDesc, resultType, (ulong)m, (ulong)n, m), "C layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out dDesc, resultType, (ulong)m, (ulong)n, m), "D layout");

            Check(CuBlasLtNative.cublasLtMatmul(
                _handle, opDesc,
                ref alpha,
                aDev, aDesc, bDev, bDesc,
                ref beta,
                cDev == IntPtr.Zero ? dDev : cDev, cDesc,
                dDev, dDesc,
                IntPtr.Zero, // auto-select algo
                workspace, workspaceSizeInBytes,
                stream), "Matmul");
        }
        finally
        {
            if (dDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(dDesc);
            if (cDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(cDesc);
            if (bDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(bDesc);
            if (aDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(aDesc);
            if (opDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatmulDescDestroy(opDesc);
        }
    }

    /// <summary>
    /// Resident signed-INT8 GEMM with exact INT32 accumulation/output. The
    /// caller supplies already-packed device operands; scaling and nonlinear
    /// epilogues are intentionally separate because cuBLASLt's INT32 compute
    /// contract does not accept the FP32 per-channel scale ABI used by W8A8.
    /// Operands use standard column-major packing. Before transposition, A is
    /// <c>[m,k]</c> (or <c>[k,m]</c> when transposed), B is <c>[k,n]</c> (or
    /// <c>[n,k]</c> when transposed), and C/D are <c>[m,n]</c>. Device pointers,
    /// the reduction dimension, and the leading dimensions must satisfy the
    /// four-byte INT8 dot-product alignment contract.
    /// </summary>
    public void MatmulInt8ToInt32(
        IntPtr aDev, int m, int k, bool transA,
        IntPtr bDev, int n, bool transB,
        IntPtr dDev,
        IntPtr stream = default)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuBlasLtMatmul));
        ValidateInt8LayoutArguments(aDev, m, k, transA, bDev, n, transB, dDev);

        ulong aRows = checked((ulong)(transA ? k : m));
        ulong aColumns = checked((ulong)(transA ? m : k));
        long aLeadingDimension = checked((long)aRows);
        ulong bRows = checked((ulong)(transB ? n : k));
        ulong bColumns = checked((ulong)(transB ? k : n));
        long bLeadingDimension = checked((long)bRows);

        IntPtr opDesc = IntPtr.Zero, aDesc = IntPtr.Zero, bDesc = IntPtr.Zero;
        IntPtr cDesc = IntPtr.Zero, dDesc = IntPtr.Zero;
        try
        {
            Check(CuBlasLtNative.cublasLtMatmulDescCreate(
                out opDesc, CublasComputeType.Int32, CublasDataType.Int32), "INT8 DescCreate");
            int tA = transA ? 1 : 0;
            int tB = transB ? 1 : 0;
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransA, ref tA, sizeof(int));
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransB, ref tB, sizeof(int));
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out aDesc, CublasDataType.Int8, aRows, aColumns,
                aLeadingDimension), "INT8 A layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out bDesc, CublasDataType.Int8, bRows, bColumns,
                bLeadingDimension), "INT8 B layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out cDesc, CublasDataType.Int32, (ulong)m, (ulong)n, m),
                "INT32 C layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out dDesc, CublasDataType.Int32, (ulong)m, (ulong)n, m),
                "INT32 D layout");
            int alpha = 1, beta = 0;
            CublasStatus status = CuBlasLtNative.cublasLtMatmulInt32(
                _handle, opDesc, ref alpha,
                aDev, aDesc, bDev, bDesc, ref beta,
                dDev, cDesc, dDev, dDesc,
                IntPtr.Zero, IntPtr.Zero, 0, stream);
            if (status == CublasStatus.NotSupported)
                throw new NotSupportedException(
                    "cuBLASLt does not support this standard-column-major INT8 matmul on the active device.");
            Check(status, "INT8 Matmul");
        }
        finally
        {
            if (dDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(dDesc);
            if (cDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(cDesc);
            if (bDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(bDesc);
            if (aDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(aDesc);
            if (opDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatmulDescDestroy(opDesc);
        }
    }

    internal static void ValidateInt8LayoutArguments(
        IntPtr aDev, int m, int k, bool transA,
        IntPtr bDev, int n, bool transB,
        IntPtr dDev)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Matrix dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Matrix dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Matrix dimensions must be positive.");
        if ((m & 3) != 0 || (k & 3) != 0)
            throw new ArgumentException(
                $"Standard-layout INT8 matmul requires m and k to be multiples of four (m={m}, k={k}).");

        ValidateAlignedDevicePointer(aDev, nameof(aDev));
        ValidateAlignedDevicePointer(bDev, nameof(bDev));
        ValidateAlignedDevicePointer(dDev, nameof(dDev));

        int aLeadingDimension = transA ? k : m;
        int bLeadingDimension = transB ? n : k;
        if ((aLeadingDimension & 3) != 0 || (bLeadingDimension & 3) != 0)
            throw new ArgumentException(
                "Standard-layout INT8 operand leading dimensions must be multiples of four. " +
                $"Computed lda={aLeadingDimension}, ldb={bLeadingDimension} for " +
                $"transA={transA}, transB={transB}.");
    }

    private static void ValidateAlignedDevicePointer(IntPtr pointer, string paramName)
    {
        if (pointer == IntPtr.Zero)
            throw new ArgumentException("The device pointer cannot be null.", paramName);
        if ((pointer.ToInt64() & 3L) != 0)
            throw new ArgumentException("The device pointer must be at least four-byte aligned.", paramName);
    }

    private static void SetAttr<TAttr>(IntPtr desc, CublasLtMatmulDescAttributes attr, ref TAttr value, int sizeInBytes)
        where TAttr : unmanaged
    {
        unsafe
        {
            fixed (TAttr* p = &value)
            {
                Check(CuBlasLtNative.cublasLtMatmulDescSetAttribute(
                    desc, attr, (IntPtr)p, (ulong)sizeInBytes),
                    $"SetAttr {attr}");
            }
        }
    }

    private static void Check(CublasStatus status, string op)
    {
        if (status != CublasStatus.Success)
            throw new InvalidOperationException($"cublasLt {op} failed: {status}.");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != IntPtr.Zero)
        {
            try { CuBlasLtNative.cublasLtDestroy(_handle); } catch { }
            _handle = IntPtr.Zero;
        }
    }
}
