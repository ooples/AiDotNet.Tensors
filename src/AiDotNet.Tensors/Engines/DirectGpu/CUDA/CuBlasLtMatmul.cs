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
        CublasComputeType computeType = CublasComputeType.Float32)
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
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out cDesc, dtype, (ulong)m, (ulong)n, m), "C layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out dDesc, dtype, (ulong)m, (ulong)n, m), "D layout");

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
