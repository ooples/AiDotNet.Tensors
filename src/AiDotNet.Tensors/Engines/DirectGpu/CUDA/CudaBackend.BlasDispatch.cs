using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// #1265 (items 2-3 of #377 phase E): cuBLAS is optional. Every GEMM the backend issues goes through these helpers, which
/// keep cuBLAS's column-major calling convention: with cuBLAS loadable they call it; without it (no cuBLAS library on the
/// machine, or <c>AIDOTNET_DISABLE_CUBLAS=1</c>) the generated GEMM runs the same problem. cuBLAS loads lazily, on the
/// first GEMM, and its absence is normal: no exception, no warning.
/// </summary>
public sealed partial class CudaBackend : IGemmAutocastBackend
{
    // AIDOTNET_DISABLE_CUBLAS=1 runs every GEMM on the generated kernels, as on a machine without cuBLAS.
    private static readonly bool CublasDisabledByEnvironment =
        Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_CUBLAS") == "1";

    private int _cublasState;  // 0 not yet probed, 1 loadable, 2 not loadable
    private readonly ConcurrentDictionary<FallbackGemmKey, PtxSgemmKernel> _fallbackGemmKernels = new();
    private long _fallbackGemmDispatchCount;

    private readonly record struct FallbackGemmKey(PtxGemmConfiguration Configuration, PtxGemmLayout Layout, PtxGemmPrecision Precision);

    /// <summary>Tests: forces cuBLAS on (true) or off (false) for this backend; null follows the machine.</summary>
    internal bool? CublasEnabledOverride { get; set; }

    internal long FallbackGemmDispatchCount => Interlocked.Read(ref _fallbackGemmDispatchCount);

    /// <summary>
    /// Whether this backend's GEMMs use cuBLAS. The first query creates this thread's cuBLAS handle, which loads the
    /// library; a missing library (or entry point) means the generated kernels serve every GEMM from then on.
    /// </summary>
    public bool IsCublasAvailable
    {
        get
        {
            if (CublasEnabledOverride is bool forced) return forced;
            if (CublasDisabledByEnvironment || !IsAvailable) return false;
            int state = Volatile.Read(ref _cublasState);
            if (state != 0) return state == 1;
            bool loadable;
            try
            {
                _ = _cublasHandle;
                loadable = true;
            }
            catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException or
                                                 BadImageFormatException or TypeInitializationException)
            {
                loadable = false;
            }
            Interlocked.CompareExchange(ref _cublasState, loadable ? 1 : 2, 0);
            return Volatile.Read(ref _cublasState) == 1;
        }
    }

    // A GEMM calls cuBLAS when it is loadable and the user has not opted out (TensorCodecOptions.UseCublas).
    private bool UsesCublas => IsCublasAvailable && CudaDispatchPolicy.UseCublasForMatMul;

    // Matmul autocast (EnterGemmAutocast): the cuBLAS compute type every FP32 GEMM takes while a scope is open, 0 when
    // none is. Instance state, not thread-local: a compiled step's ops may run on the BLAS pool's threads, so every
    // thread's GEMMs on this backend take the open scope's type. The newest scope still open wins; scopes may close in
    // any order (two threads' scopes overlapping), and the type returns to FP32 once the last one closes. A scope that is
    // never disposed keeps its type in force, as an undisposed torch.autocast context would.
    private int _gemmAutocastComputeType;
    private readonly List<GemmAutocastScope> _gemmAutocastScopes = new();

    private int GemmAutocastComputeType => Volatile.Read(ref _gemmAutocastComputeType);

    // The open scope's 16-bit operand type when it casts the operands before the GEMM (CastOperands: CUDA_R_16F or
    // CUDA_R_16BF), 0 when it passes FP32 operands with a permissive compute type (FastComputeType) or none is open.
    private int _gemmAutocastCastType;
    private int GemmAutocastCastType => Volatile.Read(ref _gemmAutocastCastType);

    /// <summary>Tests: the 16-bit operand type autocast GEMMs cast to now (0: none, or the fast compute type).</summary>
    internal int GemmAutocastCastTypeForTest => GemmAutocastCastType;

    /// <summary>Tests: the compute type GEMMs take now (0: FP32).</summary>
    internal int GemmAutocastComputeTypeForTest => GemmAutocastComputeType;

    // An autocast scope is open (and cuBLAS serves GEMMs): the FP32-only generated and narrow GEMM kernels step aside so
    // every product reaches cuBLAS/cuBLASLt in the scope's compute type, as PyTorch's autocast sends matmuls to cuBLAS.
    private bool GemmAutocastActive => UsesCublas && GemmAutocastComputeType != 0;

    /// <summary>
    /// Opens a matmul autocast scope (PyTorch's <c>torch.autocast</c> for matmuls): until it is disposed every FP32 GEMM
    /// this backend issues through cuBLAS multiplies in <paramref name="precision"/> (FP16 or BF16 tensor cores, operands
    /// down-converted inside the kernel) and accumulates and writes FP32. Operands and results stay FP32 tensors, so
    /// nothing else in a step changes. A captured graph keeps the compute type its capture ran with. Without cuBLAS the
    /// generated FP32 GEMM runs unchanged.
    /// </summary>
    IDisposable IGemmAutocastBackend.EnterGemmAutocast(Gpu.PrecisionMode precision, Gpu.MatMulAutocastMode mode)
    {
        int compute = precision switch
        {
            Gpu.PrecisionMode.Float32 => 0,
            Gpu.PrecisionMode.Float16 => (int)CublasComputeType.Float32Fast16F,
            Gpu.PrecisionMode.BFloat16 => (int)CublasComputeType.Float32FastBF16,
            _ => throw new NotSupportedException($"Matmul autocast supports FP16 and BF16, not {precision}."),
        };
        int castType = compute == 0 || mode == Gpu.MatMulAutocastMode.FastComputeType ? 0
            : precision == Gpu.PrecisionMode.BFloat16 ? CuBlasNative.CUDA_R_16BF : CuBlasNative.CUDA_R_16F;
        var scope = new GemmAutocastScope(this, compute, castType);
        lock (_gemmAutocastScopes)
        {
            _gemmAutocastScopes.Add(scope);
            Volatile.Write(ref _gemmAutocastCastType, castType);
            Volatile.Write(ref _gemmAutocastComputeType, compute);
        }
        return scope;
    }

    private sealed class GemmAutocastScope : IDisposable
    {
        private CudaBackend? _owner;
        internal readonly int Compute;
        internal readonly int CastType;
        public GemmAutocastScope(CudaBackend owner, int compute, int castType)
        {
            _owner = owner;
            Compute = compute;
            CastType = castType;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _owner, null) is not { } owner) return;
            var scopes = owner._gemmAutocastScopes;
            lock (scopes)
            {
                scopes.Remove(this);
                var newest = scopes.Count == 0 ? null : scopes[scopes.Count - 1];
                Volatile.Write(ref owner._gemmAutocastCastType, newest?.CastType ?? 0);
                Volatile.Write(ref owner._gemmAutocastComputeType, newest?.Compute ?? 0);
            }
        }
    }

    /// <summary><c>cublasSgemm</c>, or the generated FP32 GEMM without cuBLAS.</summary>
    private void BlasSgemm(CublasOperation opA, CublasOperation opB, int m, int n, int k, ref float alpha,
        IntPtr a, int lda, IntPtr b, int ldb, ref float beta, IntPtr c, int ldc, string what)
    {
        if (UsesCublas)
        {
            if (GemmAutocastComputeType is int autocast and not 0)
            {
                // Autocast, casting the operands (the default): A and B rounded to 16 bits, multiplied and accumulated in
                // FP32 into the FP32 result.
                if (GemmAutocastCastType is int castType and not 0
                    && TryCastAutocastGemm(opA != CublasOperation.None, opB != CublasOperation.None, m, n, k, alpha, a,
                        lda, 0, b, ldb, 0, beta, c, ldc, 0, 1, castType, what))
                    return;
                // The fast compute type: FP32 operands and result, the product down-converted to FP16/BF16 inside the
                // kernel's tensor core loads where cuBLAS chooses to, and accumulated in FP32.
                unsafe
                {
                    float alphaValue = alpha, betaValue = beta;
                    CuBlasNative.CheckCublasStatus(
                        CuBlasNative.cublasGemmEx(_cublasHandle, opA, opB, m, n, k, (IntPtr)(&alphaValue),
                            a, CuBlasNative.CUDA_R_32F, lda, b, CuBlasNative.CUDA_R_32F, ldb, (IntPtr)(&betaValue),
                            c, CuBlasNative.CUDA_R_32F, ldc, autocast, CuBlasNative.CUBLAS_GEMM_DEFAULT), what + " (autocast)");
                }
                return;
            }
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSgemm(_cublasHandle, opA, opB, m, n, k, ref alpha, a, lda, b, ldb, ref beta, c, ldc), what);
            return;
        }
        FallbackGemm(PtxGemmPrecision.Fp32, opA, opB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1, 0, 0, 0, GemmLaunchStream);
    }

    /// <summary><c>cublasSgemmStridedBatched</c>, or the generated FP32 GEMM batched over grid z without cuBLAS.</summary>
    private void BlasSgemmStridedBatched(CublasOperation opA, CublasOperation opB, int m, int n, int k, ref float alpha,
        IntPtr a, int lda, long strideA, IntPtr b, int ldb, long strideB, ref float beta, IntPtr c, int ldc, long strideC,
        int batch, string what)
    {
        if (UsesCublas)
        {
            if (GemmAutocastComputeType is int autocast and not 0)
            {
                if (GemmAutocastCastType is int castType and not 0
                    && TryCastAutocastGemm(opA != CublasOperation.None, opB != CublasOperation.None, m, n, k, alpha, a,
                        lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batch, castType, what))
                    return;
                unsafe
                {
                    float alphaValue = alpha, betaValue = beta;
                    CuBlasNative.CheckCublasStatus(
                        CuBlasNative.cublasGemmStridedBatchedEx(_cublasHandle, opA, opB, m, n, k, (IntPtr)(&alphaValue),
                            a, CuBlasNative.CUDA_R_32F, lda, strideA, b, CuBlasNative.CUDA_R_32F, ldb, strideB,
                            (IntPtr)(&betaValue), c, CuBlasNative.CUDA_R_32F, ldc, strideC, batch, autocast,
                            CuBlasNative.CUBLAS_GEMM_DEFAULT), what + " (autocast)");
                }
                return;
            }
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSgemmStridedBatched(_cublasHandle, opA, opB, m, n, k, ref alpha, a, lda, strideA,
                    b, ldb, strideB, ref beta, c, ldc, strideC, batch), what);
            return;
        }
        FallbackGemm(PtxGemmPrecision.Fp32, opA, opB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch,
            strideA, strideB, strideC, GemmLaunchStream);
    }

    /// <summary><c>cublasDgemm</c>, or the generated FP64 GEMM without cuBLAS.</summary>
    private void BlasDgemm(CublasOperation opA, CublasOperation opB, int m, int n, int k, ref double alpha,
        IntPtr a, int lda, IntPtr b, int ldb, ref double beta, IntPtr c, int ldc, string what)
    {
        if (UsesCublas)
        {
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasDgemm(_cublasHandle, opA, opB, m, n, k, ref alpha, a, lda, b, ldb, ref beta, c, ldc), what);
            return;
        }
        FallbackGemm(PtxGemmPrecision.Fp64, opA, opB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1, 0, 0, 0, GemmLaunchStream);
    }

    /// <summary>
    /// <c>cublasGemmEx</c> with FP32 scalars (alpha and beta point to floats: a 32F compute type), or, without cuBLAS,
    /// the generated GEMM for its type combination: FP32, FP16 or BF16 operands with FP32 output, or FP16 throughout.
    /// </summary>
    private unsafe AiDotNet.Tensors.Engines.CublasStatus BlasGemmEx(CublasOperation opA, CublasOperation opB, int m, int n, int k, IntPtr alpha,
        IntPtr a, int aType, int lda, IntPtr b, int bType, int ldb, IntPtr beta, IntPtr c, int cType, int ldc,
        int computeType, int algorithm)
    {
        if (UsesCublas)
            return CuBlasNative.cublasGemmEx(_cublasHandle, opA, opB, m, n, k, alpha, a, aType, lda, b, bType, ldb,
                beta, c, cType, ldc, computeType, algorithm);
        if (aType != bType) throw new NotSupportedException($"GEMM with operand types {aType} and {bType} needs cuBLAS.");
        PtxGemmPrecision precision = (aType, cType) switch
        {
            (CuBlasNative.CUDA_R_32F, CuBlasNative.CUDA_R_32F) => PtxGemmPrecision.Fp32,
            (CuBlasNative.CUDA_R_16F, CuBlasNative.CUDA_R_32F) => PtxGemmPrecision.Fp16InFp32Out,
            (CuBlasNative.CUDA_R_16F, CuBlasNative.CUDA_R_16F) => PtxGemmPrecision.Fp16InFp16Out,
            (CuBlasNative.CUDA_R_16BF, CuBlasNative.CUDA_R_32F) => PtxGemmPrecision.Bf16InFp32Out,
            _ => throw new NotSupportedException($"GEMM with operand type {aType} and output type {cType} needs cuBLAS."),
        };
        FallbackGemm(precision, opA, opB, m, n, k, *(float*)alpha, a, lda, b, ldb, *(float*)beta, c, ldc, 1, 0, 0, 0,
            GemmLaunchStream);
        return AiDotNet.Tensors.Engines.CublasStatus.Success;
    }

    /// <summary>
    /// <c>cublasGemmStridedBatchedEx</c> with FP32 scalars, or, without cuBLAS, the generated GEMM for its type
    /// combination (see <see cref="BlasGemmEx"/>) over the same batch and strides.
    /// </summary>
    private unsafe AiDotNet.Tensors.Engines.CublasStatus BlasGemmStridedBatchedEx(CublasOperation opA, CublasOperation opB,
        int m, int n, int k, IntPtr alpha, IntPtr a, int aType, int lda, long strideA, IntPtr b, int bType, int ldb,
        long strideB, IntPtr beta, IntPtr c, int cType, int ldc, long strideC, int batch, int computeType, int algorithm)
    {
        if (UsesCublas)
            return CuBlasNative.cublasGemmStridedBatchedEx(_cublasHandle, opA, opB, m, n, k, alpha, a, aType, lda, strideA,
                b, bType, ldb, strideB, beta, c, cType, ldc, strideC, batch, computeType, algorithm);
        if (aType != bType) throw new NotSupportedException($"GEMM with operand types {aType} and {bType} needs cuBLAS.");
        PtxGemmPrecision precision = (aType, cType) switch
        {
            (CuBlasNative.CUDA_R_32F, CuBlasNative.CUDA_R_32F) => PtxGemmPrecision.Fp32,
            (CuBlasNative.CUDA_R_16F, CuBlasNative.CUDA_R_32F) => PtxGemmPrecision.Fp16InFp32Out,
            (CuBlasNative.CUDA_R_16F, CuBlasNative.CUDA_R_16F) => PtxGemmPrecision.Fp16InFp16Out,
            (CuBlasNative.CUDA_R_16BF, CuBlasNative.CUDA_R_32F) => PtxGemmPrecision.Bf16InFp32Out,
            _ => throw new NotSupportedException($"GEMM with operand type {aType} and output type {cType} needs cuBLAS."),
        };
        FallbackGemm(precision, opA, opB, m, n, k, *(float*)alpha, a, lda, b, ldb, *(float*)beta, c, ldc, batch,
            strideA, strideB, strideC, GemmLaunchStream);
        return AiDotNet.Tensors.Engines.CublasStatus.Success;
    }

    // cuBLAS's column-major C[m x n] (ldc) = alpha * opA(A) opB(B) + beta * C is the row-major problem
    // C[n x m] = opB(B)^T opA(A)^T on the same memory: the generated kernel's A is cuBLAS's B (transposed exactly when
    // opB is), its B is cuBLAS's A (transposed exactly when opA is), and M and N swap.
    private void FallbackGemm(PtxGemmPrecision precision, CublasOperation opA, CublasOperation opB, int m, int n, int k,
        double alpha, IntPtr a, int lda, IntPtr b, int ldb, double beta, IntPtr c, int ldc, int batch,
        long strideA, long strideB, long strideC, IntPtr stream)
    {
        if (m <= 0 || n <= 0 || batch <= 0) return;  // cuBLAS's quick return
        var layout = new PtxGemmLayout(TransposeA: opB != CublasOperation.None, TransposeB: opA != CublasOperation.None);
        int rowM = n, rowN = m;
        PtxSgemmKernel kernel = FallbackKernel(precision, layout, rowM, rowN, k, ldb, lda, ldc, strideB, strideA, strideC);
        int elementBytes = precision.OperandBytes(), outputBytes = precision.OutputBytes();
        // The grid's z extent caps one launch at 65,535 problems; larger batches launch in chunks.
        for (int first = 0; first < batch; first += 65535)
        {
            int count = Math.Min(65535, batch - first);
            IntPtr chunkA = Offset(b, first * strideB, elementBytes);
            IntPtr chunkB = Offset(a, first * strideA, elementBytes);
            IntPtr chunkC = Offset(c, first * strideC, outputBytes);
            lock (GpuDispatchLock)
            {
                if (precision == PtxGemmPrecision.Fp64)
                    kernel.LaunchDouble(chunkA, chunkB, chunkC, rowM, rowN, k, ldb, lda, ldc, alpha, beta, count,
                        strideB, strideA, strideC, stream);
                else
                    kernel.Launch(chunkA, chunkB, chunkC, rowM, rowN, k, ldb, lda, ldc, (float)alpha, (float)beta, count,
                        strideB, strideA, strideC, stream);
            }
        }
        Interlocked.Increment(ref _fallbackGemmDispatchCount);
    }

    private static IntPtr Offset(IntPtr pointer, long elements, int elementBytes) =>
        elements == 0 ? pointer : new IntPtr(pointer.ToInt64() + checked(elements * elementBytes));

    // FP32: the tuned seeds for this shape, the first that admits the actual leading dimensions and strides; every other
    // precision, and any problem no seed admits, the scalar register-staged kernel, which admits everything.
    private static readonly PtxGemmConfiguration ScalarFallbackConfiguration = new(64, 64, 16, 4, 4, 1, 0, true);

    private PtxSgemmKernel FallbackKernel(PtxGemmPrecision precision, PtxGemmLayout layout, int m, int n, int k,
        int lda, int ldb, int ldc, long strideA, long strideB, long strideC)
    {
        DirectPtxRuntime runtime = GetDirectPtxRuntime();
        PtxGemmConfiguration configuration = ScalarFallbackConfiguration;
        if (precision == PtxGemmPrecision.Fp32)
        {
            try
            {
                foreach (var seed in PtxGemmEvolution.Seeds(layout, m, n, Math.Max(1, k), PtxGemmEvolution.MultiprocessorCount(runtime),
                             runtime.ComputeCapabilityMajor))
                {
                    if (!PtxSgemmGenerator.Admits(seed, layout, m, n, k, lda, ldb, ldc, strideA, strideB, strideC)) continue;
                    configuration = seed;
                    break;
                }
            }
            catch (InvalidOperationException)
            {
                // No seed admits the packed problem either: the scalar kernel below.
            }
        }
        var key = new FallbackGemmKey(configuration, layout, precision);
        if (_fallbackGemmKernels.TryGetValue(key, out PtxSgemmKernel? cached)) return cached;
        if (IsStreamCapturing())
            throw new InvalidOperationException(
                $"The generated {precision} GEMM ({configuration.Name}, {layout.Name}) must run once before CUDA graph capture " +
                "when cuBLAS is unavailable: its module cannot be loaded inside a capture.");
        lock (_fallbackGemmKernels)
        {
            if (_fallbackGemmKernels.TryGetValue(key, out cached)) return cached;
            var kernel = new PtxSgemmKernel(runtime, configuration, layout, precision);
            _fallbackGemmKernels[key] = kernel;
            return kernel;
        }
    }

    private void DisposeFallbackGemmKernels()
    {
        foreach (var key in _fallbackGemmKernels.Keys)
            if (_fallbackGemmKernels.TryRemove(key, out PtxSgemmKernel? kernel)) kernel.Dispose();
    }
}
