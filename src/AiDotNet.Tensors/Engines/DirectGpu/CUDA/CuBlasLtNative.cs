using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// P/Invoke bindings for cuBLASLt — the fused-epilogue matmul API
/// shipped with CUDA 11+. Lets us fuse <c>(matmul + bias + GELU)</c>,
/// <c>(matmul + bias + ReLU)</c>, or <c>(matmul + bias)</c> into a
/// single kernel launch; on Hopper+ these route to the tensor-core
/// GEMM-with-epilogue hardware path.
///
/// <para>Only the subset of the API we actually use is bound here.
/// Missing entry points (algorithm selection heuristics, user-supplied
/// workspace sizing, scaling / amax outputs) can be added as needed
/// without breaking ABI.</para>
/// </summary>
public static class CuBlasLtNative
{
    private const string CuBlasLtLibrary = "cublasLt";

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtCreate")]
    public static extern CublasStatus cublasLtCreate(out IntPtr handle);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtDestroy")]
    public static extern CublasStatus cublasLtDestroy(IntPtr handle);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmul")]
    public static extern CublasStatus cublasLtMatmul(
        IntPtr handle,
        IntPtr computeDesc, ref float alpha,
        IntPtr A, IntPtr Adesc,
        IntPtr B, IntPtr Bdesc,
        ref float beta,
        IntPtr C, IntPtr Cdesc,
        IntPtr D, IntPtr Ddesc,
        IntPtr algo,
        IntPtr workspace, ulong workspaceSizeInBytes,
        IntPtr stream);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmul")]
    public static extern CublasStatus cublasLtMatmulInt32(
        IntPtr handle,
        IntPtr computeDesc, ref int alpha,
        IntPtr A, IntPtr Adesc,
        IntPtr B, IntPtr Bdesc,
        ref int beta,
        IntPtr C, IntPtr Cdesc,
        IntPtr D, IntPtr Ddesc,
        IntPtr algo,
        IntPtr workspace, ulong workspaceSizeInBytes,
        IntPtr stream);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmulDescCreate")]
    public static extern CublasStatus cublasLtMatmulDescCreate(
        out IntPtr matmulDesc, CublasComputeType computeType, CublasDataType scaleType);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmulDescDestroy")]
    public static extern CublasStatus cublasLtMatmulDescDestroy(IntPtr matmulDesc);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmulDescSetAttribute")]
    public static extern CublasStatus cublasLtMatmulDescSetAttribute(
        IntPtr matmulDesc, CublasLtMatmulDescAttributes attr,
        IntPtr valuePtr, ulong sizeInBytes);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatrixLayoutCreate")]
    public static extern CublasStatus cublasLtMatrixLayoutCreate(
        out IntPtr matLayout, CublasDataType type, ulong rows, ulong cols, long ld);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatrixLayoutDestroy")]
    public static extern CublasStatus cublasLtMatrixLayoutDestroy(IntPtr matLayout);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmulPreferenceCreate")]
    internal static extern CublasStatus cublasLtMatmulPreferenceCreate(out IntPtr pref);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmulPreferenceDestroy")]
    internal static extern CublasStatus cublasLtMatmulPreferenceDestroy(IntPtr pref);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmulPreferenceSetAttribute")]
    internal static extern CublasStatus cublasLtMatmulPreferenceSetAttribute(
        IntPtr pref, CublasLtMatmulPreferenceAttributes attr, IntPtr buf, ulong sizeInBytes);

    [DllImport(CuBlasLtLibrary, EntryPoint = "cublasLtMatmulAlgoGetHeuristic")]
    internal static extern unsafe CublasStatus cublasLtMatmulAlgoGetHeuristic(
        IntPtr handle, IntPtr operationDesc, IntPtr aDesc, IntPtr bDesc, IntPtr cDesc, IntPtr dDesc,
        IntPtr preference, int requestedAlgoCount, CublasLtMatmulHeuristicResult* results, out int returnAlgoCount);
}

/// <summary>cublasLtMatmulPreferenceAttributes_t (cublasLt.h, CUDA 13.1) -- subset.</summary>
internal enum CublasLtMatmulPreferenceAttributes
{
    /// <summary>uint64: the largest workspace an algorithm may use.</summary>
    MaxWorkspaceBytes = 1,
    /// <summary>uint32 mask of <see cref="CublasLtReductionScheme"/> an algorithm may use.</summary>
    ReductionSchemeMask = 3,
}

/// <summary>cublasLtReductionScheme_t (cublasLt.h) -- how a split-K algorithm combines its partial results.</summary>
[Flags]
internal enum CublasLtReductionScheme : uint
{
    None = 0,
    /// <summary>Partial results accumulated into the output with atomics: run-to-run nondeterministic.</summary>
    InPlace = 1,
    ComputeType = 2,
    OutputType = 4,
}

/// <summary>cublasLtMatmulHeuristicResult_t (cublasLt.h): a 64-byte algorithm, its workspace, status and waves.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CublasLtMatmulHeuristicResult
{
    public fixed ulong Algo[8];
    public UIntPtr WorkspaceSize;
    public CublasStatus State;
    public float WavesCount;
    public fixed int Reserved[4];
}

/// <summary>cuBLAS library status codes shared by cuBLAS and
/// cuBLASLt.</summary>
public enum CublasStatus
{
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16,
}

/// <summary>cuBLAS compute types.</summary>
public enum CublasComputeType
{
    Float16 = 64,
    Float16Pedantic = 65,
    Float32 = 68,
    Float32Pedantic = 69,
    Float32FastTf32 = 77,
    Float32FastTf32Pedantic = 78,
    Float32FastBF16 = 74,
    Float32Fast16F = 75,
    Float64 = 70,
    Int32 = 72,
}

/// <summary>cuBLAS element data types.</summary>
public enum CublasDataType
{
    Float16 = 2,
    Float32 = 0,
    Float64 = 1,
    BFloat16 = 14,
    Int8 = 3,
    Int32 = 10,
    Float8E4M3 = 28,
    Float8E5M2 = 29,
}

/// <summary>cuBLASLt matmul descriptor attributes — subset.</summary>
public enum CublasLtMatmulDescAttributes
{
    TransA = 3,
    TransB = 4,
    Epilogue = 7,
    EpilogueBiasPointer = 8,
}

/// <summary>
/// cuBLASLt epilogue codes. Each encodes a fused post-matmul
/// operation — applied element-wise to the <c>D</c> output tensor.
/// See cuBLAS programming guide, §cublasLtEpilogue_t.
/// </summary>
public enum CublasLtEpilogue
{
    /// <summary>D = alpha * (A @ B) + beta * C (no fusion).</summary>
    Default = 1,
    /// <summary>D = ReLU(alpha * (A @ B) + beta * C).</summary>
    ReLU = 2,
    /// <summary>D = alpha * (A @ B) + beta * C + bias (bias broadcast over rows).</summary>
    Bias = 4,
    /// <summary>D = ReLU(alpha * (A @ B) + beta * C + bias).</summary>
    ReLUBias = 6,
    /// <summary>D = GELU(alpha * (A @ B) + beta * C).</summary>
    GELU = 32,
    /// <summary>D = GELU(alpha * (A @ B) + beta * C + bias).</summary>
    GELUBias = 36,
    /// <summary>D = GELU_tanh_approx(...).</summary>
    GELUTanh = 33,
    /// <summary>D = GELU_tanh_approx(... + bias).</summary>
    GELUTanhBias = 37,
}
