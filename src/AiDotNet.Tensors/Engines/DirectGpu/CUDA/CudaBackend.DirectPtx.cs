#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // Resolve process-level feature switches once when the backend is built.
    // Reading environment variables in every dispatch would violate the
    // zero-allocation hot-path contract even though the PTX launch is resident.
    private readonly bool _directPtxAttentionOptedIn = DirectPtxFeatureGate.IsAttentionEnabled;
    private readonly bool _directPtxResidualRmsNormOptedIn =
        DirectPtxFeatureGate.IsResidualRmsNormEnabled;
    private readonly object _directPtxLock = new();
    private readonly DirectPtxKernelCache<DirectPtxAttentionKey, PtxOnlineFusedAttention128x64Kernel>
        _directPtxAttentionKernels = new(DirectPtxFeatureGate.CacheCapacity);
    private readonly DirectPtxPlanCache<DirectPtxAttentionPlanKey, int>
        _directPtxAttentionPlans = new(DirectPtxFeatureGate.CacheCapacity);
    private readonly DirectPtxKernelCache<DirectPtxResidualRmsNormKey, PtxFusedResidualRmsNormD64Kernel>
        _directPtxResidualRmsNormKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxDecodeKey, PtxFusedDecodeAttentionD64Kernel>
        _directPtxDecodeKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxPagedPrefillKey, PtxFusedPagedPrefillAttentionD64Kernel>
        _directPtxPagedPrefillKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxAttentionBackwardKey, PtxFusedAttentionBackwardD64Kernel>
        _directPtxAttentionBackwardKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxFlashAttentionBackwardKey, PtxFlashAttentionBackwardD64Kernel>
        _directPtxFlashAttentionBackwardKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxQkvRopeCacheKey, PtxFusedQkvRopeCacheD64Kernel>
        _directPtxQkvRopeCacheKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxCsrSpmmKey, PtxFusedCsrSpmmVec4F32Kernel>
        _directPtxCsrSpmmKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxSddmmKey, PtxFusedSddmmF32Kernel>
        _directPtxSddmmKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxCsrSpmmKey, PtxFusedCsrSpmmBiasVec4F32Kernel>
        _directPtxCsrSpmmBiasKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxCsrSpmmKey, PtxFusedCsrSpmmBiasVec4F32Kernel>
        _directPtxCsrSpmmBiasReluKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxCsrSpmmKey, PtxFusedCsrSpmmVec2F64Kernel>
        _directPtxCsrSpmmF64Kernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxGraphGatherKey, PtxGraphEdgeGatherVec4F32Kernel>
        _directPtxGraphGatherKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxGraphScatterKey, PtxGraphScatterAddDeterministicVec4F32Kernel>
        _directPtxGraphScatterKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxGraphScatterKey, PtxGraphScatterAddAtomicF32Kernel>
        _directPtxGraphScatterAtomicKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxSegmentReduction, PtxSegmentReduceDeterministicVec4F32Kernel>
        _directPtxSegmentReduceKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxCsrSegmentKey, PtxCsrSegmentReduceVec4F32Kernel>
        _directPtxCsrSegmentKernels = new(8);
    private readonly DirectPtxKernelCache<DirectPtxCsrBackwardTarget, PtxCsrSpmmBackwardF32Kernel>
        _directPtxCsrBackwardKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxSparseUtilityKey, PtxSparseUtilityF32Kernel>
        _directPtxSparseUtilityKernels = new(8);
    private readonly DirectPtxKernelCache<DirectPtxStructuredSparse2x4Key, PtxStructuredSparse2x4F32Kernel>
        _directPtxStructuredSparse2x4Kernels = new(16);
    private readonly DirectPtxKernelCache<int, PtxStructuredSparse2x4MmaSpF32Kernel>
        _directPtxStructuredSparse2x4MmaSpKernels = new(1);
    private readonly DirectPtxKernelCache<DirectPtxScalarScatterAddMode, PtxScatterAddScalarF32Kernel>
        _directPtxScalarScatterAddKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxScatterRowsOperation, PtxScatterRowsF32Kernel>
        _directPtxScatterRowsKernels = new(8);
    private readonly DirectPtxKernelCache<int, PtxScatterMaxRowsF32Kernel>
        _directPtxScatterMaxRowsKernels = new(1);
    private readonly DirectPtxKernelCache<int, PtxNeuralScatterMaxF32Kernel>
        _directPtxNeuralScatterMaxKernels = new(1);
    private readonly DirectPtxKernelCache<DirectPtxScatterBackwardRowsOperation, PtxScatterBackwardRowsF32Kernel>
        _directPtxScatterBackwardRowsKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxCapsuleRoutingOperation, PtxCapsuleRoutingF32Kernel>
        _directPtxCapsuleRoutingKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxCapsuleProjectionOperation, PtxCapsuleProjectionF32Kernel>
        _directPtxCapsuleProjectionKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxCapsuleSquashOperation, PtxCapsuleSquashF32Kernel>
        _directPtxCapsuleSquashKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxResidentScatterAuxOperation, PtxResidentScatterAuxF32Kernel>
        _directPtxResidentScatterAuxKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxResidentScatterSoftmaxOperation, PtxResidentScatterSoftmaxF32Kernel>
        _directPtxResidentScatterSoftmaxKernels = new(4);
    private readonly DirectPtxKernelCache<int, PtxUniformMeshLaplacianF32Kernel>
        _directPtxUniformMeshLaplacianKernels = new(1);
    private readonly DirectPtxKernelCache<DirectPtxSparseOptimizerKey, PtxSparseOptimizerF32Kernel>
        _directPtxSparseOptimizerKernels = new(32);
    private readonly DirectPtxKernelCache<DirectPtxFusedSparseLinearKey, PtxFusedSparseLinearF32Kernel>
        _directPtxFusedSparseLinearKernels = new(16);
    private readonly DirectPtxKernelCache<int, PtxTensorGatherRowsF32Kernel>
        _directPtxTensorGatherKernels = new(1);
    private readonly DirectPtxKernelCache<DirectPtxTensorScatterReduceMode, PtxTensorScatterReduceF32Kernel>
        _directPtxTensorScatterReduceKernels = new(4);
    private readonly DirectPtxKernelCache<DirectPtxTensorScatterHighLevelOperation, PtxTensorScatterHighLevelF32Kernel>
        _directPtxTensorScatterHighLevelKernels = new(2);
    private readonly DirectPtxKernelCache<DirectPtxMeshPoolOperation, PtxMeshPoolF32Kernel>
        _directPtxMeshPoolKernels = new(16);
    private readonly DirectPtxKernelCache<DirectPtxSparseEngineOperation, PtxSparseEngineF32Kernel>
        _directPtxSparseEngineKernels = new(22);
    private DirectPtxRuntime? _directPtxRuntime;

    /// <summary>The last opt-in direct-PTX initialization/launch failure, if fallback was required.</summary>
    internal string? DirectPtxLastError { get; private set; }
    internal long DirectPtxAttentionDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxAttentionDispatchCount);
    private long _directPtxAttentionDispatchCount;
    private long _directPtxResidualRmsNormDispatchCount;
    private long _directPtxDecodeDispatchCount;
    private long _directPtxPagedPrefillDispatchCount;
    private long _directPtxAttentionBackwardDispatchCount;
    private long _directPtxFlashAttentionBackwardDispatchCount;
    private long _directPtxQkvRopeCacheDispatchCount;
    private long _directPtxCsrSpmmDispatchCount;
    private long _directPtxSddmmDispatchCount;
    private long _directPtxCsrSpmmBiasDispatchCount;
    private long _directPtxCsrSpmmBiasReluDispatchCount;
    private long _directPtxCsrSpmmF64DispatchCount;
    private long _directPtxGraphGatherDispatchCount;
    private long _directPtxGraphScatterDispatchCount;
    private long _directPtxSegmentReduceDispatchCount;
    private long _directPtxCsrSegmentDispatchCount;
    private long _directPtxCsrBackwardDispatchCount;
    private long _directPtxSparseUtilityDispatchCount;
    private long _directPtxStructuredSparse2x4DispatchCount;
    private long _directPtxStructuredSparse2x4MmaSpDispatchCount;
    private long _directPtxScalarScatterAddDispatchCount;
    private long _directPtxScatterRowsDispatchCount;
    private long _directPtxScatterMaxRowsDispatchCount;
    private long _directPtxNeuralScatterMaxDispatchCount;
    private long _directPtxScatterBackwardRowsDispatchCount;
    private long _directPtxCapsuleRoutingDispatchCount;
    private long _directPtxCapsuleProjectionDispatchCount;
    private long _directPtxCapsuleSquashDispatchCount;
    private long _directPtxResidentScatterAuxDispatchCount;
    private long _directPtxResidentScatterSoftmaxDispatchCount;
    private long _directPtxUniformMeshLaplacianDispatchCount;
    private long _directPtxSparseOptimizerDispatchCount;
    private long _directPtxFusedSparseLinearDispatchCount;
    private long _directPtxTensorGatherDispatchCount;
    private long _directPtxTensorScatterReduceDispatchCount;
    private long _directPtxTensorScatterHighLevelDispatchCount;
    private long _directPtxMeshPoolDispatchCount;
    private long _directPtxSparseEngineDispatchCount;
    internal int DirectPtxCachedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxAttentionKernels.Count; }
    }

    internal bool IsDirectPtxAttentionEnabled =>
        _directPtxAttentionOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasValidatedOnlineAttention(_ccMajor, _ccMinor);

    internal bool IsDirectPtxResidualRmsNormEnabled =>
        _directPtxResidualRmsNormOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasValidatedOnlineAttention(_ccMajor, _ccMinor);

    internal bool IsDirectPtxFlashDecodeEnabled =>
        DirectPtxFeatureGate.IsFlashDecodeEnabled && IsAvailable &&
        DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) == DirectPtxArchitectureFamily.Ampere;

    internal bool IsDirectPtxPagedDecodeEnabled =>
        DirectPtxFeatureGate.IsPagedDecodeEnabled && IsAvailable &&
        DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) == DirectPtxArchitectureFamily.Ampere;

    internal bool IsDirectPtxPagedPrefillEnabled =>
        DirectPtxFeatureGate.IsPagedPrefillEnabled && IsAvailable &&
        DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) == DirectPtxArchitectureFamily.Ampere;

    internal bool IsDirectPtxAttentionBackwardEnabled =>
        DirectPtxFeatureGate.IsAttentionBackwardEnabled && IsAvailable &&
        DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) == DirectPtxArchitectureFamily.Ampere;

    internal bool IsDirectPtxFlashAttentionBackwardEnabled =>
        DirectPtxFeatureGate.IsFlashAttentionBackwardEnabled && IsAvailable &&
        DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) == DirectPtxArchitectureFamily.Ampere;

    internal bool IsDirectPtxQkvRopeCacheEnabled =>
        DirectPtxFeatureGate.IsQkvRopeCacheEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedQkvRopeCache(_ccMajor, _ccMinor);

    internal long DirectPtxResidualRmsNormDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxResidualRmsNormDispatchCount);

    internal long DirectPtxDecodeDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxDecodeDispatchCount);

    internal long DirectPtxPagedPrefillDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxPagedPrefillDispatchCount);

    internal long DirectPtxAttentionBackwardDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxAttentionBackwardDispatchCount);

    internal long DirectPtxFlashAttentionBackwardDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxFlashAttentionBackwardDispatchCount);

    internal long DirectPtxQkvRopeCacheDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxQkvRopeCacheDispatchCount);
    internal int DirectPtxQkvRopeCacheKernelCapacity => _directPtxQkvRopeCacheKernels.Capacity;
    internal int DirectPtxQkvRopeCachePinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxQkvRopeCacheKernels.PinnedCount; }
    }

    internal bool IsDirectPtxSparseGraphEnabled =>
        DirectPtxFeatureGate.IsSparseGraphEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedSparseGraph(_ccMajor, _ccMinor);
    internal long DirectPtxCsrSpmmDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCsrSpmmDispatchCount);
    internal int DirectPtxCsrSpmmKernelCapacity => _directPtxCsrSpmmKernels.Capacity;
    internal int DirectPtxCsrSpmmPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxCsrSpmmKernels.PinnedCount; }
    }
    internal long DirectPtxSddmmDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSddmmDispatchCount);
    internal int DirectPtxSddmmKernelCapacity => _directPtxSddmmKernels.Capacity;
    internal int DirectPtxSddmmPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxSddmmKernels.PinnedCount; }
    }
    internal long DirectPtxCsrSpmmBiasDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCsrSpmmBiasDispatchCount);
    internal long DirectPtxCsrSpmmBiasReluDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCsrSpmmBiasReluDispatchCount);
    internal long DirectPtxCsrSpmmF64DispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCsrSpmmF64DispatchCount);
    internal long DirectPtxGraphGatherDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxGraphGatherDispatchCount);
    internal long DirectPtxGraphScatterDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxGraphScatterDispatchCount);
    internal long DirectPtxSegmentReduceDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSegmentReduceDispatchCount);
    internal long DirectPtxCsrSegmentDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCsrSegmentDispatchCount);
    internal long DirectPtxCsrBackwardDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCsrBackwardDispatchCount);
    internal long DirectPtxSparseUtilityDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSparseUtilityDispatchCount);
    internal long DirectPtxStructuredSparse2x4DispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxStructuredSparse2x4DispatchCount);
    internal long DirectPtxStructuredSparse2x4MmaSpDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxStructuredSparse2x4MmaSpDispatchCount);
    internal long DirectPtxScalarScatterAddDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxScalarScatterAddDispatchCount);
    internal long DirectPtxScatterRowsDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxScatterRowsDispatchCount);
    internal long DirectPtxScatterMaxRowsDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxScatterMaxRowsDispatchCount);
    internal long DirectPtxNeuralScatterMaxDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxNeuralScatterMaxDispatchCount);
    internal long DirectPtxScatterBackwardRowsDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxScatterBackwardRowsDispatchCount);
    internal long DirectPtxCapsuleRoutingDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCapsuleRoutingDispatchCount);
    internal long DirectPtxCapsuleProjectionDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCapsuleProjectionDispatchCount);
    internal long DirectPtxCapsuleSquashDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCapsuleSquashDispatchCount);
    internal long DirectPtxResidentScatterAuxDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxResidentScatterAuxDispatchCount);
    internal long DirectPtxResidentScatterSoftmaxDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxResidentScatterSoftmaxDispatchCount);
    internal long DirectPtxUniformMeshLaplacianDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxUniformMeshLaplacianDispatchCount);
    internal long DirectPtxSparseOptimizerDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSparseOptimizerDispatchCount);
    internal long DirectPtxFusedSparseLinearDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxFusedSparseLinearDispatchCount);
    internal long DirectPtxTensorGatherDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxTensorGatherDispatchCount);
    internal long DirectPtxTensorScatterReduceDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxTensorScatterReduceDispatchCount);
    internal long DirectPtxTensorScatterHighLevelDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxTensorScatterHighLevelDispatchCount);
    internal long DirectPtxMeshPoolDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxMeshPoolDispatchCount);
    internal long DirectPtxSparseEngineDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSparseEngineDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 CSR SpMM golden specialization. The admitted ABI
    /// is CSR32 [1024,1024] with 16384 stored entries and contiguous B/C matrices
    /// with 64 columns. No shape or stride values cross the kernel boundary.
    /// </summary>
    internal bool TryDirectPtxCsrSpmmVec4F32(
        IGpuBuffer values,
        IGpuBuffer columnIndices,
        IGpuBuffer rowPointers,
        IGpuBuffer dense,
        IGpuBuffer output,
        int rows,
        int inner,
        int columns,
        int nonZeros)
    {
        if (!DirectPtxFeatureGate.IsSparseGraphEnabled)
        {
            DirectPtxLastError = "sparse-graph-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "sparse-graph-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "sparse-graph-architecture-not-implemented";
            return false;
        }
        if (!PtxFusedCsrSpmmVec4F32Kernel.SupportsShape(rows, inner, columns, nonZeros))
        {
            DirectPtxLastError = "csr-spmm-shape-not-implemented";
            return false;
        }
        if (values is null || columnIndices is null || rowPointers is null ||
            dense is null || output is null)
        {
            DirectPtxLastError = "csr-spmm-null-buffer";
            return false;
        }
        if (values.SizeInBytes != (long)nonZeros * sizeof(float) ||
            columnIndices.SizeInBytes != (long)nonZeros * sizeof(int) ||
            rowPointers.SizeInBytes != (long)(rows + 1) * sizeof(int) ||
            dense.SizeInBytes != (long)inner * columns * sizeof(float) ||
            output.SizeInBytes != (long)rows * columns * sizeof(float))
        {
            DirectPtxLastError = "csr-spmm-physical-extent-mismatch";
            return false;
        }
        if (values.Handle == IntPtr.Zero || columnIndices.Handle == IntPtr.Zero ||
            rowPointers.Handle == IntPtr.Zero || dense.Handle == IntPtr.Zero ||
            output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "csr-spmm-invalid-device-pointer";
            return false;
        }
        if ((((nuint)values.Handle | (nuint)columnIndices.Handle | (nuint)rowPointers.Handle |
              (nuint)dense.Handle | (nuint)output.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "csr-spmm-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, values) ||
            DirectPtxBuffersOverlap(output, columnIndices) ||
            DirectPtxBuffersOverlap(output, rowPointers) ||
            DirectPtxBuffersOverlap(output, dense))
        {
            DirectPtxLastError = "csr-spmm-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCsrSpmmKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX CSR SpMM must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedCsrSpmmVec4F32Kernel kernel = GetOrCreateCsrSpmmKernel(key);
                if (capturing && !_directPtxCsrSpmmKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX CSR SpMM module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(values, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(columnIndices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(rowPointers, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(dense, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCsrSpmmDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxCsrSpmmVec4F32(
        int rows,
        int inner,
        int columns,
        int nonZeros)
    {
        if (!DirectPtxFeatureGate.IsSparseGraphEnabled)
        {
            DirectPtxLastError = "sparse-graph-feature-disabled";
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasValidatedSparseGraph(_ccMajor, _ccMinor) ||
            !PtxFusedCsrSpmmVec4F32Kernel.SupportsShape(rows, inner, columns, nonZeros))
        {
            DirectPtxLastError = "csr-spmm-specialization-not-admitted";
            return false;
        }
        if (IsStreamCapturing())
        {
            DirectPtxLastError = "Direct PTX CSR SpMM prewarm is not capture-safe.";
            return false;
        }
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateCsrSpmmKernel(
                    new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedCsrSpmmVec4F32Kernel GetOrCreateCsrSpmmKernel(DirectPtxCsrSpmmKey key)
    {
        if (_directPtxCsrSpmmKernels.TryGetValue(key, out var existing)) return existing;
        return CreateAndCacheCsrSpmmKernelSlow(key);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedCsrSpmmVec4F32Kernel CreateAndCacheCsrSpmmKernelSlow(
        DirectPtxCsrSpmmKey key) =>
        _directPtxCsrSpmmKernels.GetOrAdd(
            key, () => new PtxFusedCsrSpmmVec4F32Kernel(_directPtxRuntime!));

    internal bool TryGetDirectPtxCsrSpmmAudit(
        int rows,
        int inner,
        int columns,
        int nonZeros,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxCsrSpmmKernels.TryGetValue(
                new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Attempts the exact fused FP32 CSR SpMM+bias specialization. Bias is
    /// consumed before the sparse reduction and output is written once.
    /// </summary>
    internal bool TryDirectPtxCsrSpmmBiasVec4F32(
        IGpuBuffer values,
        IGpuBuffer columnIndices,
        IGpuBuffer rowPointers,
        IGpuBuffer dense,
        IGpuBuffer bias,
        IGpuBuffer output,
        int rows,
        int inner,
        int columns,
        int nonZeros,
        bool fuseRelu = false)
    {
        if (!IsDirectPtxSparseGraphEnabled)
        {
            DirectPtxLastError = "csr-spmm-bias-specialization-not-enabled";
            return false;
        }
        if (!PtxFusedCsrSpmmBiasVec4F32Kernel.SupportsShape(rows, inner, columns, nonZeros))
        {
            DirectPtxLastError = "csr-spmm-bias-shape-not-implemented";
            return false;
        }
        if (values is null || columnIndices is null || rowPointers is null || dense is null ||
            bias is null || output is null)
        {
            DirectPtxLastError = "csr-spmm-bias-null-buffer";
            return false;
        }
        if (values.SizeInBytes != (long)nonZeros * sizeof(float) ||
            columnIndices.SizeInBytes != (long)nonZeros * sizeof(int) ||
            rowPointers.SizeInBytes != (long)(rows + 1) * sizeof(int) ||
            dense.SizeInBytes != (long)inner * columns * sizeof(float) ||
            bias.SizeInBytes != (long)columns * sizeof(float) ||
            output.SizeInBytes != (long)rows * columns * sizeof(float))
        {
            DirectPtxLastError = "csr-spmm-bias-physical-extent-mismatch";
            return false;
        }
        if (values.Handle == IntPtr.Zero || columnIndices.Handle == IntPtr.Zero ||
            rowPointers.Handle == IntPtr.Zero || dense.Handle == IntPtr.Zero ||
            bias.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "csr-spmm-bias-invalid-device-pointer";
            return false;
        }
        if ((((nuint)values.Handle | (nuint)columnIndices.Handle | (nuint)rowPointers.Handle |
              (nuint)dense.Handle | (nuint)bias.Handle | (nuint)output.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "csr-spmm-bias-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, values) ||
            DirectPtxBuffersOverlap(output, columnIndices) ||
            DirectPtxBuffersOverlap(output, rowPointers) ||
            DirectPtxBuffersOverlap(output, dense) || DirectPtxBuffersOverlap(output, bias))
        {
            DirectPtxLastError = "csr-spmm-bias-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros);
            DirectPtxKernelCache<DirectPtxCsrSpmmKey, PtxFusedCsrSpmmBiasVec4F32Kernel> cache =
                fuseRelu ? _directPtxCsrSpmmBiasReluKernels : _directPtxCsrSpmmBiasKernels;
            lock (_directPtxLock)
            {
                if (capturing && !cache.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX CSR SpMM+bias must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedCsrSpmmBiasVec4F32Kernel kernel = cache.GetOrAdd(
                    key, () => new PtxFusedCsrSpmmBiasVec4F32Kernel(_directPtxRuntime!, fuseRelu));
                if (capturing && !cache.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX CSR SpMM+bias module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(values, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(columnIndices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(rowPointers, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(dense, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[5]));
            }
            if (fuseRelu)
                System.Threading.Interlocked.Increment(ref _directPtxCsrSpmmBiasReluDispatchCount);
            else
                System.Threading.Interlocked.Increment(ref _directPtxCsrSpmmBiasDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxCsrSpmmBiasVec4F32(
        int rows, int inner, int columns, int nonZeros, bool fuseRelu = false)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing() ||
            !PtxFusedCsrSpmmBiasVec4F32Kernel.SupportsShape(rows, inner, columns, nonZeros))
            return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros);
                DirectPtxKernelCache<DirectPtxCsrSpmmKey, PtxFusedCsrSpmmBiasVec4F32Kernel> cache =
                    fuseRelu ? _directPtxCsrSpmmBiasReluKernels : _directPtxCsrSpmmBiasKernels;
                _ = cache.GetOrAdd(
                    key, () => new PtxFusedCsrSpmmBiasVec4F32Kernel(_directPtxRuntime!, fuseRelu));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxCsrSpmmBiasAudit(
        int rows, int inner, int columns, int nonZeros, out DirectPtxKernelAudit audit,
        bool fuseRelu = false)
    {
        lock (_directPtxLock)
        {
            DirectPtxKernelCache<DirectPtxCsrSpmmKey, PtxFusedCsrSpmmBiasVec4F32Kernel> cache =
                fuseRelu ? _directPtxCsrSpmmBiasReluKernels : _directPtxCsrSpmmBiasKernels;
            if (cache.TryGetValue(
                new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxCsrSpmmVec2F64(
        IGpuBuffer values,
        IGpuBuffer columnIndices,
        IGpuBuffer rowPointers,
        IGpuBuffer dense,
        IGpuBuffer output,
        int rows,
        int inner,
        int columns,
        int nonZeros)
    {
        if (!IsDirectPtxSparseGraphEnabled)
        {
            DirectPtxLastError = "csr-spmm-f64-specialization-not-enabled";
            return false;
        }
        if (!PtxFusedCsrSpmmVec2F64Kernel.SupportsShape(rows, inner, columns, nonZeros))
        {
            DirectPtxLastError = "csr-spmm-f64-shape-not-implemented";
            return false;
        }
        if (values is null || columnIndices is null || rowPointers is null ||
            dense is null || output is null)
        {
            DirectPtxLastError = "csr-spmm-f64-null-buffer";
            return false;
        }
        if (values.SizeInBytes != (long)nonZeros * sizeof(double) ||
            columnIndices.SizeInBytes != (long)nonZeros * sizeof(int) ||
            rowPointers.SizeInBytes != (long)(rows + 1) * sizeof(int) ||
            dense.SizeInBytes != (long)inner * columns * sizeof(double) ||
            output.SizeInBytes != (long)rows * columns * sizeof(double))
        {
            DirectPtxLastError = "csr-spmm-f64-physical-extent-mismatch";
            return false;
        }
        if (values.Handle == IntPtr.Zero || columnIndices.Handle == IntPtr.Zero ||
            rowPointers.Handle == IntPtr.Zero || dense.Handle == IntPtr.Zero ||
            output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "csr-spmm-f64-invalid-device-pointer";
            return false;
        }
        if ((((nuint)values.Handle | (nuint)columnIndices.Handle | (nuint)rowPointers.Handle |
              (nuint)dense.Handle | (nuint)output.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "csr-spmm-f64-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, values) ||
            DirectPtxBuffersOverlap(output, columnIndices) ||
            DirectPtxBuffersOverlap(output, rowPointers) || DirectPtxBuffersOverlap(output, dense))
        {
            DirectPtxLastError = "csr-spmm-f64-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCsrSpmmF64Kernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX FP64 CSR SpMM must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedCsrSpmmVec2F64Kernel kernel = _directPtxCsrSpmmF64Kernels.GetOrAdd(
                    key, () => new PtxFusedCsrSpmmVec2F64Kernel(_directPtxRuntime!));
                if (capturing && !_directPtxCsrSpmmF64Kernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX FP64 CSR SpMM module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(values, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(columnIndices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(rowPointers, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(dense, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCsrSpmmF64DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxCsrSpmmVec2F64(
        int rows, int inner, int columns, int nonZeros)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing() ||
            !PtxFusedCsrSpmmVec2F64Kernel.SupportsShape(rows, inner, columns, nonZeros))
            return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros);
                _ = _directPtxCsrSpmmF64Kernels.GetOrAdd(
                    key, () => new PtxFusedCsrSpmmVec2F64Kernel(_directPtxRuntime!));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxCsrSpmmF64Audit(
        int rows, int inner, int columns, int nonZeros, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxCsrSpmmF64Kernels.TryGetValue(
                new DirectPtxCsrSpmmKey(rows, inner, columns, nonZeros), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxGraphEdgeGatherVec4F32(
        IGpuBuffer nodeFeatures,
        IGpuBuffer edgeNodeIndices,
        IGpuBuffer edgeFeatures,
        int nodes,
        int edges,
        int features,
        bool gatherSource)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxGraphEdgeGatherVec4F32Kernel.SupportsShape(nodes, edges, features))
        {
            DirectPtxLastError = "graph-gather-specialization-not-admitted";
            return false;
        }
        if (nodeFeatures is null || edgeNodeIndices is null || edgeFeatures is null)
        {
            DirectPtxLastError = "graph-gather-null-buffer";
            return false;
        }
        if (nodeFeatures.SizeInBytes != (long)nodes * features * sizeof(float) ||
            edgeNodeIndices.SizeInBytes != (long)edges * sizeof(int) ||
            edgeFeatures.SizeInBytes != (long)edges * features * sizeof(float))
        {
            DirectPtxLastError = "graph-gather-physical-extent-mismatch";
            return false;
        }
        if (nodeFeatures.Handle == IntPtr.Zero || edgeNodeIndices.Handle == IntPtr.Zero ||
            edgeFeatures.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "graph-gather-invalid-device-pointer";
            return false;
        }
        if ((((nuint)nodeFeatures.Handle | (nuint)edgeNodeIndices.Handle |
              (nuint)edgeFeatures.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "graph-gather-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(edgeFeatures, nodeFeatures) ||
            DirectPtxBuffersOverlap(edgeFeatures, edgeNodeIndices))
        {
            DirectPtxLastError = "graph-gather-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxGraphGatherKey(gatherSource);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxGraphGatherKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX graph gather must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxGraphEdgeGatherVec4F32Kernel kernel = _directPtxGraphGatherKernels.GetOrAdd(
                    key, () => new PtxGraphEdgeGatherVec4F32Kernel(_directPtxRuntime!, gatherSource));
                if (capturing && !_directPtxGraphGatherKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX graph gather module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(nodeFeatures, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(edgeNodeIndices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(edgeFeatures, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxGraphGatherDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxGraphEdgeGather(bool gatherSource)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxGraphGatherKey(gatherSource);
                _ = _directPtxGraphGatherKernels.GetOrAdd(
                    key, () => new PtxGraphEdgeGatherVec4F32Kernel(_directPtxRuntime!, gatherSource));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxGraphScatterAddDeterministicVec4F32(
        IGpuBuffer input,
        IGpuBuffer sourceIndices,
        IGpuBuffer targetIndices,
        IGpuBuffer? edgeWeights,
        IGpuBuffer output,
        int nodes,
        int edges,
        int features)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxGraphScatterAddDeterministicVec4F32Kernel.SupportsShape(nodes, edges, features))
        {
            DirectPtxLastError = "graph-scatter-deterministic-specialization-not-admitted";
            return false;
        }
        if (input is null || sourceIndices is null || targetIndices is null || output is null)
        {
            DirectPtxLastError = "graph-scatter-deterministic-null-buffer";
            return false;
        }
        bool weighted = edgeWeights is not null;
        if (input.SizeInBytes != (long)nodes * features * sizeof(float) ||
            sourceIndices.SizeInBytes != (long)edges * sizeof(int) ||
            targetIndices.SizeInBytes != (long)edges * sizeof(int) ||
            (weighted && edgeWeights!.SizeInBytes != (long)edges * sizeof(float)) ||
            output.SizeInBytes != (long)nodes * features * sizeof(float))
        {
            DirectPtxLastError = "graph-scatter-deterministic-physical-extent-mismatch";
            return false;
        }
        if (input.Handle == IntPtr.Zero || sourceIndices.Handle == IntPtr.Zero ||
            targetIndices.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero ||
            (weighted && edgeWeights!.Handle == IntPtr.Zero))
        {
            DirectPtxLastError = "graph-scatter-deterministic-invalid-device-pointer";
            return false;
        }
        nuint pointers = (nuint)input.Handle | (nuint)sourceIndices.Handle |
            (nuint)targetIndices.Handle | (nuint)output.Handle;
        if (weighted) pointers |= (nuint)edgeWeights!.Handle;
        if ((pointers & 15u) != 0)
        {
            DirectPtxLastError = "graph-scatter-deterministic-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, input) ||
            DirectPtxBuffersOverlap(output, sourceIndices) ||
            DirectPtxBuffersOverlap(output, targetIndices) ||
            (weighted && DirectPtxBuffersOverlap(output, edgeWeights!)))
        {
            DirectPtxLastError = "graph-scatter-deterministic-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxGraphScatterKey(weighted);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxGraphScatterKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX graph scatter-add must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxGraphScatterAddDeterministicVec4F32Kernel kernel =
                    _directPtxGraphScatterKernels.GetOrAdd(
                        key, () => new PtxGraphScatterAddDeterministicVec4F32Kernel(
                            _directPtxRuntime!, weighted));
                if (capturing && !_directPtxGraphScatterKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX graph scatter-add module for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    DirectPtxTensorView inputView =
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]);
                    DirectPtxTensorView sourceView =
                        DirectPtxTensorView.Create(sourceIndices, kernel.Blueprint.Tensors[1]);
                    DirectPtxTensorView targetView =
                        DirectPtxTensorView.Create(targetIndices, kernel.Blueprint.Tensors[2]);
                    if (weighted)
                        kernel.LaunchWeighted(
                            inputView, sourceView, targetView,
                            DirectPtxTensorView.Create(edgeWeights!, kernel.Blueprint.Tensors[3]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
                    else
                        kernel.Launch(
                            inputView, sourceView, targetView,
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxGraphScatterDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxGraphScatterAddDeterministic(bool weighted)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxGraphScatterKey(weighted);
                _ = _directPtxGraphScatterKernels.GetOrAdd(
                    key, () => new PtxGraphScatterAddDeterministicVec4F32Kernel(
                        _directPtxRuntime!, weighted));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxGraphScatterAddAtomicF32(
        IGpuBuffer input,
        IGpuBuffer sourceIndices,
        IGpuBuffer targetIndices,
        IGpuBuffer? edgeWeights,
        IGpuBuffer output,
        int nodes,
        int edges,
        int features)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxGraphScatterAddAtomicF32Kernel.SupportsShape(nodes, edges, features))
        {
            DirectPtxLastError = "graph-scatter-atomic-specialization-not-admitted";
            return false;
        }
        if (input is null || sourceIndices is null || targetIndices is null || output is null)
        {
            DirectPtxLastError = "graph-scatter-atomic-null-buffer";
            return false;
        }
        bool weighted = edgeWeights is not null;
        if (input.SizeInBytes != (long)nodes * features * sizeof(float) ||
            sourceIndices.SizeInBytes != (long)edges * sizeof(int) ||
            targetIndices.SizeInBytes != (long)edges * sizeof(int) ||
            (weighted && edgeWeights!.SizeInBytes != (long)edges * sizeof(float)) ||
            output.SizeInBytes != (long)nodes * features * sizeof(float))
        {
            DirectPtxLastError = "graph-scatter-atomic-physical-extent-mismatch";
            return false;
        }
        if (input.Handle == IntPtr.Zero || sourceIndices.Handle == IntPtr.Zero ||
            targetIndices.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero ||
            (weighted && edgeWeights!.Handle == IntPtr.Zero))
        {
            DirectPtxLastError = "graph-scatter-atomic-invalid-device-pointer";
            return false;
        }
        nuint pointers = (nuint)input.Handle | (nuint)sourceIndices.Handle |
            (nuint)targetIndices.Handle | (nuint)output.Handle;
        if (weighted) pointers |= (nuint)edgeWeights!.Handle;
        if ((pointers & 15u) != 0)
        {
            DirectPtxLastError = "graph-scatter-atomic-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, input) ||
            DirectPtxBuffersOverlap(output, sourceIndices) ||
            DirectPtxBuffersOverlap(output, targetIndices) ||
            (weighted && DirectPtxBuffersOverlap(output, edgeWeights!)))
        {
            DirectPtxLastError = "graph-scatter-atomic-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxGraphScatterKey(weighted);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxGraphScatterAtomicKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX atomic graph scatter-add must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxGraphScatterAddAtomicF32Kernel kernel =
                    _directPtxGraphScatterAtomicKernels.GetOrAdd(
                        key, () => new PtxGraphScatterAddAtomicF32Kernel(
                            _directPtxRuntime!, weighted));
                if (capturing && !_directPtxGraphScatterAtomicKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX atomic graph scatter-add module for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    DirectPtxTensorView inputView =
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]);
                    DirectPtxTensorView sourceView =
                        DirectPtxTensorView.Create(sourceIndices, kernel.Blueprint.Tensors[1]);
                    DirectPtxTensorView targetView =
                        DirectPtxTensorView.Create(targetIndices, kernel.Blueprint.Tensors[2]);
                    if (weighted)
                        kernel.LaunchWeighted(
                            inputView, sourceView, targetView,
                            DirectPtxTensorView.Create(edgeWeights!, kernel.Blueprint.Tensors[3]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
                    else
                        kernel.Launch(
                            inputView, sourceView, targetView,
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxGraphScatterDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxGraphScatterAddAtomic(bool weighted)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxGraphScatterKey(weighted);
                _ = _directPtxGraphScatterAtomicKernels.GetOrAdd(
                    key, () => new PtxGraphScatterAddAtomicF32Kernel(
                        _directPtxRuntime!, weighted));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxSegmentReduceDeterministicVec4F32(
        IGpuBuffer input,
        IGpuBuffer segmentIds,
        IGpuBuffer? segmentSizes,
        IGpuBuffer output,
        int items,
        int segments,
        int features,
        DirectPtxSegmentReduction reduction)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxSegmentReduceDeterministicVec4F32Kernel.SupportsShape(items, segments, features))
        {
            DirectPtxLastError = "segment-reduce-specialization-not-admitted";
            return false;
        }
        bool mean = reduction == DirectPtxSegmentReduction.Mean;
        if (input is null || segmentIds is null || output is null || (mean && segmentSizes is null))
        {
            DirectPtxLastError = "segment-reduce-null-buffer";
            return false;
        }
        if (input.SizeInBytes != (long)items * features * sizeof(float) ||
            segmentIds.SizeInBytes != (long)items * sizeof(int) ||
            (mean && segmentSizes!.SizeInBytes != (long)segments * sizeof(int)) ||
            output.SizeInBytes != (long)segments * features * sizeof(float))
        {
            DirectPtxLastError = "segment-reduce-physical-extent-mismatch";
            return false;
        }
        if (input.Handle == IntPtr.Zero || segmentIds.Handle == IntPtr.Zero ||
            output.Handle == IntPtr.Zero || (mean && segmentSizes!.Handle == IntPtr.Zero))
        {
            DirectPtxLastError = "segment-reduce-invalid-device-pointer";
            return false;
        }
        nuint pointers = (nuint)input.Handle | (nuint)segmentIds.Handle | (nuint)output.Handle;
        if (mean) pointers |= (nuint)segmentSizes!.Handle;
        if ((pointers & 15u) != 0)
        {
            DirectPtxLastError = "segment-reduce-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, input) || DirectPtxBuffersOverlap(output, segmentIds) ||
            (mean && DirectPtxBuffersOverlap(output, segmentSizes!)))
        {
            DirectPtxLastError = "segment-reduce-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSegmentReduceKernels.TryGetValue(reduction, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX segmented reduction must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxSegmentReduceDeterministicVec4F32Kernel kernel =
                    _directPtxSegmentReduceKernels.GetOrAdd(
                        reduction, () => new PtxSegmentReduceDeterministicVec4F32Kernel(
                            _directPtxRuntime!, reduction));
                if (capturing && !_directPtxSegmentReduceKernels.Pin(reduction))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX segmented reduction module for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    DirectPtxTensorView inputView =
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]);
                    DirectPtxTensorView idsView =
                        DirectPtxTensorView.Create(segmentIds, kernel.Blueprint.Tensors[1]);
                    if (mean)
                        kernel.LaunchMean(
                            inputView, idsView,
                            DirectPtxTensorView.Create(segmentSizes!, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
                    else
                        kernel.Launch(
                            inputView, idsView,
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxSegmentReduceDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxSegmentReduce(DirectPtxSegmentReduction reduction)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxSegmentReduceKernels.GetOrAdd(
                    reduction, () => new PtxSegmentReduceDeterministicVec4F32Kernel(
                        _directPtxRuntime!, reduction));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxCsrSegmentReduceVec4F32(
        IGpuBuffer columnIndices,
        IGpuBuffer rowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int rows,
        int innerRows,
        int features,
        DirectPtxCsrSegmentReduction reduction,
        float epsilon = 1e-8f)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxCsrSegmentReduceVec4F32Kernel.SupportsShape(
                rows, innerRows, features, PtxCsrSegmentReduceVec4F32Kernel.NonZeros))
        {
            DirectPtxLastError = "csr-segment-reduce-specialization-not-admitted";
            return false;
        }
        if (columnIndices is null || rowPointers is null || input is null || output is null)
        {
            DirectPtxLastError = "csr-segment-reduce-null-buffer";
            return false;
        }
        if (columnIndices.SizeInBytes !=
                (long)PtxCsrSegmentReduceVec4F32Kernel.NonZeros * sizeof(float) ||
            rowPointers.SizeInBytes != (long)(rows + 1) * sizeof(float) ||
            input.SizeInBytes != (long)innerRows * features * sizeof(float) ||
            output.SizeInBytes != (long)rows * features * sizeof(float))
        {
            DirectPtxLastError = "csr-segment-reduce-physical-extent-mismatch";
            return false;
        }
        if (columnIndices.Handle == IntPtr.Zero || rowPointers.Handle == IntPtr.Zero ||
            input.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "csr-segment-reduce-invalid-device-pointer";
            return false;
        }
        if ((((nuint)columnIndices.Handle | (nuint)rowPointers.Handle |
              (nuint)input.Handle | (nuint)output.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "csr-segment-reduce-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, columnIndices) ||
            DirectPtxBuffersOverlap(output, rowPointers) || DirectPtxBuffersOverlap(output, input))
        {
            DirectPtxLastError = "csr-segment-reduce-alias-not-supported";
            return false;
        }
        if (reduction == DirectPtxCsrSegmentReduction.StdDev &&
            (!(epsilon >= 0f) || float.IsInfinity(epsilon)))
        {
            DirectPtxLastError = "csr-segment-reduce-invalid-epsilon";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxCsrSegmentKey(
                reduction,
                reduction == DirectPtxCsrSegmentReduction.StdDev
                    ? BitConverter.SingleToInt32Bits(epsilon) : 0);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCsrSegmentKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX CSR segmented reduction must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxCsrSegmentReduceVec4F32Kernel kernel = _directPtxCsrSegmentKernels.GetOrAdd(
                    key, () => new PtxCsrSegmentReduceVec4F32Kernel(
                        _directPtxRuntime!, reduction, epsilon));
                if (capturing && !_directPtxCsrSegmentKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX CSR segmented module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(columnIndices, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(rowPointers, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCsrSegmentDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxCsrSpmmBackwardF32(
        IGpuBuffer p0,
        IGpuBuffer p1,
        IGpuBuffer p2,
        IGpuBuffer gradOutput,
        IGpuBuffer output,
        int rows,
        int inner,
        int columns,
        int nonZeros,
        DirectPtxCsrBackwardTarget target)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxCsrSpmmBackwardF32Kernel.SupportsShape(rows, inner, columns, nonZeros))
        {
            DirectPtxLastError = "csr-backward-specialization-not-admitted";
            return false;
        }
        if (p0 is null || p1 is null || p2 is null || gradOutput is null || output is null)
        {
            DirectPtxLastError = "csr-backward-null-buffer";
            return false;
        }
        long p0Bytes = target == DirectPtxCsrBackwardTarget.DenseB
            ? (long)nonZeros * sizeof(float) : (long)nonZeros * sizeof(int);
        long p1Bytes = target == DirectPtxCsrBackwardTarget.DenseB
            ? (long)nonZeros * sizeof(int) : (long)(rows + 1) * sizeof(int);
        long p2Bytes = target == DirectPtxCsrBackwardTarget.DenseB
            ? (long)(rows + 1) * sizeof(int) : (long)inner * columns * sizeof(float);
        long outputBytes = target == DirectPtxCsrBackwardTarget.DenseB
            ? (long)inner * columns * sizeof(float) : (long)nonZeros * sizeof(float);
        if (p0.SizeInBytes != p0Bytes || p1.SizeInBytes != p1Bytes || p2.SizeInBytes != p2Bytes ||
            gradOutput.SizeInBytes != (long)rows * columns * sizeof(float) ||
            output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "csr-backward-physical-extent-mismatch";
            return false;
        }
        if (p0.Handle == IntPtr.Zero || p1.Handle == IntPtr.Zero || p2.Handle == IntPtr.Zero ||
            gradOutput.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "csr-backward-invalid-device-pointer";
            return false;
        }
        if ((((nuint)p0.Handle | (nuint)p1.Handle | (nuint)p2.Handle |
              (nuint)gradOutput.Handle | (nuint)output.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "csr-backward-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, p0) || DirectPtxBuffersOverlap(output, p1) ||
            DirectPtxBuffersOverlap(output, p2) || DirectPtxBuffersOverlap(output, gradOutput))
        {
            DirectPtxLastError = "csr-backward-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCsrBackwardKernels.TryGetValue(target, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX CSR backward must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxCsrSpmmBackwardF32Kernel kernel = _directPtxCsrBackwardKernels.GetOrAdd(
                    target, () => new PtxCsrSpmmBackwardF32Kernel(_directPtxRuntime!, target));
                if (capturing && !_directPtxCsrBackwardKernels.Pin(target))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX CSR backward module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(p1, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(p2, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCsrBackwardDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxSparseFill(IGpuBuffer output, int elements, bool negativeInfinity)
    {
        if (!IsDirectPtxSparseGraphEnabled || elements != PtxSparseUtilityF32Kernel.Elements ||
            output is null || output.SizeInBytes != (long)elements * sizeof(float) ||
            output.Handle == IntPtr.Zero || ((nuint)output.Handle & 15u) != 0)
            return false;
        DirectPtxSparseUtility utility = negativeInfinity
            ? DirectPtxSparseUtility.FillNegativeInfinity : DirectPtxSparseUtility.FillZero;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxSparseUtilityKey(utility, 0);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSparseUtilityKernels.TryGetValue(key, out _)) return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxSparseUtilityF32Kernel kernel = _directPtxSparseUtilityKernels.GetOrAdd(
                    key, () => new PtxSparseUtilityF32Kernel(_directPtxRuntime!, utility));
                if (capturing && !_directPtxSparseUtilityKernels.Pin(key)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchFill(DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[0]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxSparseUtilityDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxDegreeNormalize(
        IGpuBuffer input, IGpuBuffer degrees, IGpuBuffer output,
        int nodes, int features, float epsilon)
    {
        if (!IsDirectPtxSparseGraphEnabled || nodes != PtxSparseUtilityF32Kernel.Nodes ||
            features != PtxSparseUtilityF32Kernel.Features || input is null || degrees is null ||
            output is null || input.SizeInBytes != (long)nodes * features * sizeof(float) ||
            degrees.SizeInBytes != (long)nodes * sizeof(float) ||
            output.SizeInBytes != (long)nodes * features * sizeof(float))
            return false;
        return TryDirectPtxSparseUtilityCore(
            DirectPtxSparseUtility.DegreeNormalize, epsilon,
            input, degrees, null, null, null, output);
    }

    internal bool TryDirectPtxSymmetricDegreeNormalize(
        IGpuBuffer edgeValues, IGpuBuffer sourceIndices, IGpuBuffer targetIndices,
        IGpuBuffer sourceDegrees, IGpuBuffer targetDegrees, IGpuBuffer output,
        int nodes, int edges, float epsilon)
    {
        if (!IsDirectPtxSparseGraphEnabled || nodes != PtxSparseUtilityF32Kernel.Nodes ||
            edges != PtxSparseUtilityF32Kernel.Edges || edgeValues is null || sourceIndices is null ||
            targetIndices is null || sourceDegrees is null || targetDegrees is null || output is null ||
            edgeValues.SizeInBytes != (long)edges * sizeof(float) ||
            sourceIndices.SizeInBytes != (long)edges * sizeof(int) ||
            targetIndices.SizeInBytes != (long)edges * sizeof(int) ||
            sourceDegrees.SizeInBytes != (long)nodes * sizeof(float) ||
            targetDegrees.SizeInBytes != (long)nodes * sizeof(float) ||
            output.SizeInBytes != (long)edges * sizeof(float))
            return false;
        return TryDirectPtxSparseUtilityCore(
            DirectPtxSparseUtility.SymmetricDegreeNormalize, epsilon,
            edgeValues, sourceIndices, targetIndices, sourceDegrees, targetDegrees, output);
    }

    private bool TryDirectPtxSparseUtilityCore(
        DirectPtxSparseUtility utility,
        float epsilon,
        IGpuBuffer p0,
        IGpuBuffer p1,
        IGpuBuffer? p2,
        IGpuBuffer? p3,
        IGpuBuffer? p4,
        IGpuBuffer output)
    {
        if (!(epsilon >= 0f) || float.IsInfinity(epsilon)) return false;
        nuint pointers = (nuint)p0.Handle | (nuint)p1.Handle | (nuint)output.Handle;
        if (p2 is not null) pointers |= (nuint)p2.Handle;
        if (p3 is not null) pointers |= (nuint)p3.Handle;
        if (p4 is not null) pointers |= (nuint)p4.Handle;
        if (p0.Handle == IntPtr.Zero || p1.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero ||
            (p2 is not null && p2.Handle == IntPtr.Zero) || (p3 is not null && p3.Handle == IntPtr.Zero) ||
            (p4 is not null && p4.Handle == IntPtr.Zero) || (pointers & 15u) != 0)
            return false;
        if (DirectPtxBuffersOverlap(output, p0) || DirectPtxBuffersOverlap(output, p1) ||
            (p2 is not null && DirectPtxBuffersOverlap(output, p2)) ||
            (p3 is not null && DirectPtxBuffersOverlap(output, p3)) ||
            (p4 is not null && DirectPtxBuffersOverlap(output, p4)))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxSparseUtilityKey(utility, BitConverter.SingleToInt32Bits(epsilon));
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSparseUtilityKernels.TryGetValue(key, out _)) return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxSparseUtilityF32Kernel kernel = _directPtxSparseUtilityKernels.GetOrAdd(
                    key, () => new PtxSparseUtilityF32Kernel(_directPtxRuntime!, utility, epsilon));
                if (capturing && !_directPtxSparseUtilityKernels.Pin(key)) return false;
                lock (GpuDispatchLock)
                {
                    if (utility == DirectPtxSparseUtility.DegreeNormalize)
                        kernel.LaunchDegree(
                            DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(p1, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
                    else
                        kernel.LaunchSymmetric(
                            DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(p1, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(p2!, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(p3!, kernel.Blueprint.Tensors[3]),
                            DirectPtxTensorView.Create(p4!, kernel.Blueprint.Tensors[4]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[5]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxSparseUtilityDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxEnforce2x4(
        IGpuBuffer denseInput,
        IGpuBuffer sparseValues,
        IGpuBuffer sparseMetadata,
        int rows,
        int inner)
    {
        if (!PtxStructuredSparse2x4F32Kernel.SupportsMatrixShape(rows, inner) ||
            !HasExactBytes(denseInput, (long)rows * inner * sizeof(float)) ||
            !HasExactBytes(sparseValues, (long)rows * (inner / 2) * sizeof(float)) ||
            !HasExactBytes(sparseMetadata, (long)rows * (inner / 4)))
            return false;
        return TryDirectPtxStructuredSparse2x4Core(
            DirectPtxStructuredSparse2x4Operation.Enforce,
            denseInput, sparseValues, sparseMetadata, null, null, 1.0f, 0.0f);
    }

    internal bool TryDirectPtxDecompress2x4(
        IGpuBuffer sparseValues,
        IGpuBuffer sparseMetadata,
        IGpuBuffer denseOutput,
        int rows,
        int inner)
    {
        if (!PtxStructuredSparse2x4F32Kernel.SupportsMatrixShape(rows, inner) ||
            !HasExactBytes(sparseValues, (long)rows * (inner / 2) * sizeof(float)) ||
            !HasExactBytes(sparseMetadata, (long)rows * (inner / 4)) ||
            !HasExactBytes(denseOutput, (long)rows * inner * sizeof(float)))
            return false;
        return TryDirectPtxStructuredSparse2x4Core(
            DirectPtxStructuredSparse2x4Operation.Decompress,
            sparseValues, sparseMetadata, denseOutput, null, null, 1.0f, 0.0f);
    }

    internal bool TryDirectPtxSparseGemm2x4(
        IGpuBuffer sparseValues,
        IGpuBuffer sparseMetadata,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int rows,
        int columns,
        int inner,
        float alpha,
        float beta)
    {
        if (!PtxStructuredSparse2x4F32Kernel.SupportsGemmShape(rows, columns, inner) ||
            !float.IsFinite(alpha) || !float.IsFinite(beta) ||
            !HasExactBytes(sparseValues, (long)rows * (inner / 2) * sizeof(float)) ||
            !HasExactBytes(sparseMetadata, (long)rows * (inner / 4)) ||
            !HasExactBytes(denseB, (long)inner * columns * sizeof(float)) ||
            !HasExactBytes(output, (long)rows * columns * sizeof(float)))
            return false;
        return TryDirectPtxStructuredSparse2x4Core(
            DirectPtxStructuredSparse2x4Operation.Gemm,
            sparseValues, sparseMetadata, denseB, output, null, alpha, beta);
    }

    internal bool TryDirectPtxSparseGemmBiasRelu2x4(
        IGpuBuffer sparseValues,
        IGpuBuffer sparseMetadata,
        IGpuBuffer denseB,
        IGpuBuffer bias,
        IGpuBuffer output,
        int rows,
        int columns,
        int inner)
    {
        if (!PtxStructuredSparse2x4F32Kernel.SupportsGemmShape(rows, columns, inner) ||
            !HasExactBytes(sparseValues, (long)rows * (inner / 2) * sizeof(float)) ||
            !HasExactBytes(sparseMetadata, (long)rows * (inner / 4)) ||
            !HasExactBytes(denseB, (long)inner * columns * sizeof(float)) ||
            !HasExactBytes(bias, (long)columns * sizeof(float)) ||
            !HasExactBytes(output, (long)rows * columns * sizeof(float)))
            return false;
        return TryDirectPtxStructuredSparse2x4Core(
            DirectPtxStructuredSparse2x4Operation.GemmBiasRelu,
            sparseValues, sparseMetadata, denseB, bias, output, 1.0f, 0.0f);
    }

    internal bool TryDirectPtxSparse2x4MatMulBaseline(
        IGpuBuffer sparseValues,
        IGpuBuffer sparseMetadata,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int rows,
        int inner,
        int columns)
    {
        if (!PtxStructuredSparse2x4F32Kernel.SupportsGemmShape(rows, columns, inner) ||
            !HasExactBytes(sparseValues, (long)rows * (inner / 2) * sizeof(float)) ||
            !HasExactBytes(sparseMetadata, (long)rows * (inner / 4)) ||
            !HasExactBytes(denseB, (long)inner * columns * sizeof(float)) ||
            !HasExactBytes(output, (long)rows * columns * sizeof(float)))
            return false;
        return TryDirectPtxStructuredSparse2x4Core(
            DirectPtxStructuredSparse2x4Operation.MatMulBaseline,
            sparseValues, sparseMetadata, denseB, output, null, 1.0f, 0.0f);
    }

    internal bool TryDirectPtxSparse2x4MatMulMmaSp(
        IGpuBuffer sparseValues,
        IGpuBuffer sparseMetadata,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int rows,
        int inner,
        int columns)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxStructuredSparse2x4MmaSpF32Kernel.SupportsShape(rows, columns, inner) ||
            !HasExactBytes(sparseValues, (long)rows * (inner / 2) * sizeof(float)) ||
            !HasExactBytes(sparseMetadata, (long)rows * (inner / 4)) ||
            !HasExactBytes(denseB, (long)inner * columns * sizeof(float)) ||
            !HasExactBytes(output, (long)rows * columns * sizeof(float)))
            return false;
        nuint pointers = (nuint)sparseValues.Handle | (nuint)sparseMetadata.Handle |
            (nuint)denseB.Handle | (nuint)output.Handle;
        if (sparseValues.Handle == IntPtr.Zero || sparseMetadata.Handle == IntPtr.Zero ||
            denseB.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero ||
            (pointers & 15u) != 0 || DirectPtxBuffersOverlap(sparseValues, sparseMetadata) ||
            DirectPtxBuffersOverlap(sparseValues, denseB) ||
            DirectPtxBuffersOverlap(sparseMetadata, denseB) ||
            DirectPtxBuffersOverlap(output, sparseValues) ||
            DirectPtxBuffersOverlap(output, sparseMetadata) ||
            DirectPtxBuffersOverlap(output, denseB))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 0;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxStructuredSparse2x4MmaSpKernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxStructuredSparse2x4MmaSpF32Kernel kernel =
                    _directPtxStructuredSparse2x4MmaSpKernels.GetOrAdd(
                        key, () => new PtxStructuredSparse2x4MmaSpF32Kernel(_directPtxRuntime!));
                if (capturing && !_directPtxStructuredSparse2x4MmaSpKernels.Pin(key))
                    return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(sparseValues, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(sparseMetadata, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(denseB, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(
                ref _directPtxStructuredSparse2x4MmaSpDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxSparse2x4MatMulMmaSp()
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxStructuredSparse2x4MmaSpKernels.GetOrAdd(
                    0, () => new PtxStructuredSparse2x4MmaSpF32Kernel(_directPtxRuntime!));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxStructuredSparse2x4(
        DirectPtxStructuredSparse2x4Operation operation,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing() ||
            !float.IsFinite(alpha) || !float.IsFinite(beta))
            return false;
        try
        {
            EnsureContextCurrent();
            var key = new DirectPtxStructuredSparse2x4Key(
                operation, BitConverter.SingleToInt32Bits(alpha), BitConverter.SingleToInt32Bits(beta));
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxStructuredSparse2x4Kernels.GetOrAdd(
                    key, () => new PtxStructuredSparse2x4F32Kernel(
                        _directPtxRuntime!, operation, alpha, beta));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool TryDirectPtxStructuredSparse2x4Core(
        DirectPtxStructuredSparse2x4Operation operation,
        IGpuBuffer p0,
        IGpuBuffer p1,
        IGpuBuffer p2,
        IGpuBuffer? p3,
        IGpuBuffer? p4,
        float alpha,
        float beta)
    {
        if (!IsDirectPtxSparseGraphEnabled) return false;
        if (p0 is null || p1 is null || p2 is null ||
            p0.Handle == IntPtr.Zero || p1.Handle == IntPtr.Zero || p2.Handle == IntPtr.Zero ||
            (p3 is not null && p3.Handle == IntPtr.Zero) ||
            (p4 is not null && p4.Handle == IntPtr.Zero))
            return false;
        nuint pointers = (nuint)p0.Handle | (nuint)p1.Handle | (nuint)p2.Handle;
        if (p3 is not null) pointers |= (nuint)p3.Handle;
        if (p4 is not null) pointers |= (nuint)p4.Handle;
        if ((pointers & 15u) != 0) return false;
        if (DirectPtxBuffersOverlap(p0, p1) || DirectPtxBuffersOverlap(p0, p2) ||
            DirectPtxBuffersOverlap(p1, p2) ||
            (p3 is not null && (DirectPtxBuffersOverlap(p0, p3) ||
                DirectPtxBuffersOverlap(p1, p3) || DirectPtxBuffersOverlap(p2, p3))) ||
            (p4 is not null && (DirectPtxBuffersOverlap(p0, p4) ||
                DirectPtxBuffersOverlap(p1, p4) || DirectPtxBuffersOverlap(p2, p4) ||
                (p3 is not null && DirectPtxBuffersOverlap(p3, p4)))))
            return false;

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxStructuredSparse2x4Key(
                operation, BitConverter.SingleToInt32Bits(alpha), BitConverter.SingleToInt32Bits(beta));
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxStructuredSparse2x4Kernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxStructuredSparse2x4F32Kernel kernel =
                    _directPtxStructuredSparse2x4Kernels.GetOrAdd(
                        key, () => new PtxStructuredSparse2x4F32Kernel(
                            _directPtxRuntime!, operation, alpha, beta));
                if (capturing && !_directPtxStructuredSparse2x4Kernels.Pin(key)) return false;
                lock (GpuDispatchLock)
                {
                    if (operation is DirectPtxStructuredSparse2x4Operation.Enforce or
                        DirectPtxStructuredSparse2x4Operation.Decompress)
                        kernel.Launch(
                            DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(p1, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(p2, kernel.Blueprint.Tensors[2]));
                    else if (operation is DirectPtxStructuredSparse2x4Operation.Gemm or
                        DirectPtxStructuredSparse2x4Operation.MatMulBaseline)
                        kernel.LaunchGemm(
                            DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(p1, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(p2, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(p3!, kernel.Blueprint.Tensors[3]));
                    else
                        kernel.LaunchBiasRelu(
                            DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(p1, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(p2, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(p3!, kernel.Blueprint.Tensors[3]),
                            DirectPtxTensorView.Create(p4!, kernel.Blueprint.Tensors[4]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxStructuredSparse2x4DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool HasExactBytes(IGpuBuffer buffer, long bytes) =>
        buffer is not null && buffer.SizeInBytes == bytes;

    internal bool TryDirectPtxScatterAddScalar(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer destination,
        int sourceElements,
        int destinationElements,
        DirectPtxScalarScatterAddMode mode)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxScatterAddScalarF32Kernel.SupportsShape(sourceElements, destinationElements) ||
            !HasExactBytes(source, (long)sourceElements * sizeof(float)) ||
            !HasExactBytes(indices, (long)sourceElements * sizeof(int)) ||
            !HasExactBytes(destination, (long)destinationElements * sizeof(float)) ||
            source.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            destination.Handle == IntPtr.Zero ||
            ((((nuint)source.Handle | (nuint)indices.Handle | (nuint)destination.Handle) & 15u) != 0) ||
            DirectPtxBuffersOverlap(destination, source) ||
            DirectPtxBuffersOverlap(destination, indices))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxScalarScatterAddKernels.TryGetValue(mode, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxScatterAddScalarF32Kernel kernel = _directPtxScalarScatterAddKernels.GetOrAdd(
                    mode, () => new PtxScatterAddScalarF32Kernel(_directPtxRuntime!, mode));
                if (capturing && !_directPtxScalarScatterAddKernels.Pin(mode)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(destination, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxScalarScatterAddDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxScatterAddScalar(DirectPtxScalarScatterAddMode mode)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxScalarScatterAddKernels.GetOrAdd(
                    mode, () => new PtxScatterAddScalarF32Kernel(_directPtxRuntime!, mode));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxScatterAddRows(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer destination,
        int sourceRows,
        int destinationRows,
        int features,
        bool deterministic)
    {
        int sourceElements = checked(sourceRows * features);
        if (!ValidateDirectPtxScatterRows(
            source, indices, destination, null,
            sourceElements, destinationRows, features))
            return false;
        DirectPtxScatterRowsOperation operation = deterministic
            ? DirectPtxScatterRowsOperation.AddDeterministic
            : DirectPtxScatterRowsOperation.AddAtomic;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxScatterRowsKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxScatterRowsF32Kernel kernel = _directPtxScatterRowsKernels.GetOrAdd(
                    operation, () => new PtxScatterRowsF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxScatterRowsKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchThree(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(destination, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxScatterRowsDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxScatterMeanRows(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer destination,
        IGpuBuffer counts,
        int sourceElements,
        int destinationRows,
        int features,
        bool deterministic)
    {
        if (!ValidateDirectPtxScatterRows(
            source, indices, destination, counts,
            sourceElements, destinationRows, features))
            return false;
        DirectPtxScatterRowsOperation accumulate = deterministic
            ? DirectPtxScatterRowsOperation.MeanAccumulateDeterministic
            : DirectPtxScatterRowsOperation.MeanAccumulateAtomic;
        const DirectPtxScatterRowsOperation normalize = DirectPtxScatterRowsOperation.MeanNormalize;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing &&
                    (!_directPtxScatterRowsKernels.TryGetValue(accumulate, out _) ||
                     !_directPtxScatterRowsKernels.TryGetValue(normalize, out _)))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxScatterRowsF32Kernel accumulateKernel = _directPtxScatterRowsKernels.GetOrAdd(
                    accumulate, () => new PtxScatterRowsF32Kernel(_directPtxRuntime!, accumulate));
                PtxScatterRowsF32Kernel normalizeKernel = _directPtxScatterRowsKernels.GetOrAdd(
                    normalize, () => new PtxScatterRowsF32Kernel(_directPtxRuntime!, normalize));
                if (capturing &&
                    (!_directPtxScatterRowsKernels.Pin(accumulate) ||
                     !_directPtxScatterRowsKernels.Pin(normalize)))
                    return false;
                lock (GpuDispatchLock)
                {
                    accumulateKernel.LaunchFour(
                        DirectPtxTensorView.Create(source, accumulateKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, accumulateKernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(destination, accumulateKernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(counts, accumulateKernel.Blueprint.Tensors[3]));
                    normalizeKernel.LaunchNormalize(
                        DirectPtxTensorView.Create(destination, normalizeKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(counts, normalizeKernel.Blueprint.Tensors[1]));
                }
            }
            System.Threading.Interlocked.Add(ref _directPtxScatterRowsDispatchCount, 2);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxScatterRows(DirectPtxScatterRowsOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxScatterRowsKernels.GetOrAdd(
                    operation, () => new PtxScatterRowsF32Kernel(_directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool ValidateDirectPtxScatterRows(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer destination,
        IGpuBuffer? counts,
        int sourceElements,
        int destinationRows,
        int features)
    {
        if (!PtxScatterRowsF32Kernel.SupportsShape(sourceElements, destinationRows, features))
            return false;
        int sourceRows = sourceElements / features;
        if (!HasExactBytes(source, (long)sourceElements * sizeof(float)) ||
            !HasExactBytes(indices, (long)sourceRows * sizeof(int)) ||
            !HasExactBytes(destination, (long)destinationRows * features * sizeof(float)) ||
            (counts is not null && !HasExactBytes(counts, (long)destinationRows * sizeof(int))))
            return false;
        nuint pointers = (nuint)source.Handle | (nuint)indices.Handle | (nuint)destination.Handle;
        if (counts is not null) pointers |= (nuint)counts.Handle;
        if (source.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            destination.Handle == IntPtr.Zero || (counts is not null && counts.Handle == IntPtr.Zero) ||
            (pointers & 15u) != 0)
            return false;
        if (DirectPtxBuffersOverlap(destination, source) ||
            DirectPtxBuffersOverlap(destination, indices) ||
            (counts is not null && (DirectPtxBuffersOverlap(counts, source) ||
                DirectPtxBuffersOverlap(counts, indices) ||
                DirectPtxBuffersOverlap(counts, destination))))
            return false;
        return true;
    }

    internal bool TryDirectPtxScatterMaxRows(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer output,
        IGpuBuffer argmax,
        int sourceRows,
        int features,
        int destinationRows)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxScatterMaxRowsF32Kernel.SupportsShape(sourceRows, features, destinationRows) ||
            !HasExactBytes(source, (long)sourceRows * features * sizeof(float)) ||
            !HasExactBytes(indices, (long)sourceRows * sizeof(int)) ||
            !HasExactBytes(output, (long)destinationRows * features * sizeof(float)) ||
            !HasExactBytes(argmax, (long)destinationRows * features * sizeof(float)))
            return false;
        nuint pointers = (nuint)source.Handle | (nuint)indices.Handle |
            (nuint)output.Handle | (nuint)argmax.Handle;
        if (source.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            output.Handle == IntPtr.Zero || argmax.Handle == IntPtr.Zero || (pointers & 15u) != 0 ||
            DirectPtxBuffersOverlap(output, source) || DirectPtxBuffersOverlap(output, indices) ||
            DirectPtxBuffersOverlap(output, argmax) || DirectPtxBuffersOverlap(argmax, source) ||
            DirectPtxBuffersOverlap(argmax, indices))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 0;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxScatterMaxRowsKernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxScatterMaxRowsF32Kernel kernel = _directPtxScatterMaxRowsKernels.GetOrAdd(
                    key, () => new PtxScatterMaxRowsF32Kernel(_directPtxRuntime!));
                if (capturing && !_directPtxScatterMaxRowsKernels.Pin(key)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(argmax, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxScatterMaxRowsDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxScatterMaxRows()
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxScatterMaxRowsKernels.GetOrAdd(
                    0, () => new PtxScatterMaxRowsF32Kernel(_directPtxRuntime!));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxNeuralScatterMax(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer output,
        IGpuBuffer argmax,
        int sourceRows,
        int features,
        int destinationRows)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxNeuralScatterMaxF32Kernel.SupportsShape(
                sourceRows, features, destinationRows) ||
            !HasExactBytes(source, (long)sourceRows * features * sizeof(float)) ||
            !HasExactBytes(indices, (long)sourceRows * sizeof(int)) ||
            !HasExactBytes(output, (long)destinationRows * features * sizeof(float)) ||
            !HasExactBytes(argmax, (long)destinationRows * features * sizeof(int)))
            return false;
        nuint pointers = (nuint)source.Handle | (nuint)indices.Handle |
            (nuint)output.Handle | (nuint)argmax.Handle;
        if (source.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            output.Handle == IntPtr.Zero || argmax.Handle == IntPtr.Zero ||
            (pointers & 15u) != 0 || DirectPtxBuffersOverlap(source, indices) ||
            DirectPtxBuffersOverlap(output, source) || DirectPtxBuffersOverlap(output, indices) ||
            DirectPtxBuffersOverlap(output, argmax) || DirectPtxBuffersOverlap(argmax, source) ||
            DirectPtxBuffersOverlap(argmax, indices))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 0;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxNeuralScatterMaxKernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxNeuralScatterMaxF32Kernel kernel =
                    _directPtxNeuralScatterMaxKernels.GetOrAdd(
                        key, () => new PtxNeuralScatterMaxF32Kernel(_directPtxRuntime!));
                if (capturing && !_directPtxNeuralScatterMaxKernels.Pin(key)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(argmax, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxNeuralScatterMaxDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxNeuralScatterMax()
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxNeuralScatterMaxKernels.GetOrAdd(
                    0, () => new PtxNeuralScatterMaxF32Kernel(_directPtxRuntime!));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxScatterBackwardRows(
        IGpuBuffer gradOutput,
        IGpuBuffer indices,
        IGpuBuffer? counts,
        IGpuBuffer gradSource,
        int sourceRows,
        int features,
        int outputRows,
        DirectPtxScatterBackwardRowsOperation operation)
    {
        bool mean = operation == DirectPtxScatterBackwardRowsOperation.Mean;
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxScatterBackwardRowsF32Kernel.SupportsShape(sourceRows, features, outputRows) ||
            (mean && counts is null) ||
            !HasExactBytes(gradOutput, (long)outputRows * features * sizeof(float)) ||
            !HasExactBytes(indices, (long)sourceRows * sizeof(int)) ||
            (mean && !HasExactBytes(counts!, (long)outputRows * sizeof(int))) ||
            !HasExactBytes(gradSource, (long)sourceRows * features * sizeof(float)))
            return false;
        nuint pointers = (nuint)gradOutput.Handle | (nuint)indices.Handle | (nuint)gradSource.Handle;
        if (mean) pointers |= (nuint)counts!.Handle;
        if (gradOutput.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            gradSource.Handle == IntPtr.Zero || (mean && counts!.Handle == IntPtr.Zero) ||
            (pointers & 15u) != 0 || DirectPtxBuffersOverlap(gradSource, gradOutput) ||
            DirectPtxBuffersOverlap(gradSource, indices) ||
            (mean && DirectPtxBuffersOverlap(gradSource, counts!)))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxScatterBackwardRowsKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxScatterBackwardRowsF32Kernel kernel =
                    _directPtxScatterBackwardRowsKernels.GetOrAdd(
                        operation, () => new PtxScatterBackwardRowsF32Kernel(
                            _directPtxRuntime!, operation));
                if (capturing && !_directPtxScatterBackwardRowsKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                {
                    if (mean)
                        kernel.LaunchMean(
                            DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(counts!, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(gradSource, kernel.Blueprint.Tensors[3]));
                    else
                        kernel.LaunchAdd(
                            DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(gradSource, kernel.Blueprint.Tensors[2]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxScatterBackwardRowsDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxScatterBackwardRows(DirectPtxScatterBackwardRowsOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxScatterBackwardRowsKernels.GetOrAdd(
                    operation, () => new PtxScatterBackwardRowsF32Kernel(
                        _directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Attempts the exact pointer-only capsule routing reductions. The admitted
    /// ABI is B=32, input capsules=32, output capsules=10, dimension=16 with
    /// tightly packed row-major allocations and no runtime shape parameters.
    /// </summary>
    internal bool TryDirectPtxCapsuleRouting(
        IGpuBuffer first,
        IGpuBuffer second,
        IGpuBuffer output,
        int batch,
        int inputCapsules,
        int outputCapsules,
        int dimension,
        DirectPtxCapsuleRoutingOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxCapsuleRoutingF32Kernel.SupportsShape(
                batch, inputCapsules, outputCapsules, dimension))
            return false;

        long couplingBytes = (long)batch * inputCapsules * outputCapsules * sizeof(float);
        long predictionBytes = couplingBytes * dimension;
        long routedBytes = (long)batch * outputCapsules * dimension * sizeof(float);
        long firstBytes = operation == DirectPtxCapsuleRoutingOperation.WeightedSum
            ? couplingBytes : predictionBytes;
        long secondBytes = operation == DirectPtxCapsuleRoutingOperation.WeightedSum
            ? predictionBytes : routedBytes;
        long outputBytes = operation == DirectPtxCapsuleRoutingOperation.WeightedSum
            ? routedBytes : couplingBytes;
        if (!HasExactBytes(first, firstBytes) || !HasExactBytes(second, secondBytes) ||
            !HasExactBytes(output, outputBytes) || first.Handle == IntPtr.Zero ||
            second.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero ||
            ((((nuint)first.Handle | (nuint)second.Handle | (nuint)output.Handle) & 15u) != 0) ||
            DirectPtxBuffersOverlap(output, first) || DirectPtxBuffersOverlap(output, second))
            return false;

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCapsuleRoutingKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxCapsuleRoutingF32Kernel kernel = _directPtxCapsuleRoutingKernels.GetOrAdd(
                    operation, () => new PtxCapsuleRoutingF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxCapsuleRoutingKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(first, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(second, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCapsuleRoutingDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxCapsuleRouting(DirectPtxCapsuleRoutingOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxCapsuleRoutingKernels.GetOrAdd(
                    operation, () => new PtxCapsuleRoutingF32Kernel(_directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Attempts the exact capsule prediction/transform projection ABI. The two
    /// operations intentionally use distinct weight layout contracts even
    /// though their admitted allocations have the same byte length.
    /// </summary>
    internal bool TryDirectPtxCapsuleProjection(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer output,
        int batch,
        int inputCapsules,
        int inputDimension,
        int outputCapsules,
        int outputDimension,
        DirectPtxCapsuleProjectionOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxCapsuleProjectionF32Kernel.SupportsShape(
                batch, inputCapsules, inputDimension, outputCapsules, outputDimension) ||
            !HasExactBytes(input, (long)batch * inputCapsules * inputDimension * sizeof(float)) ||
            !HasExactBytes(weights,
                (long)inputCapsules * inputDimension * outputCapsules * outputDimension * sizeof(float)) ||
            !HasExactBytes(output,
                (long)batch * inputCapsules * outputCapsules * outputDimension * sizeof(float)) ||
            input.Handle == IntPtr.Zero || weights.Handle == IntPtr.Zero ||
            output.Handle == IntPtr.Zero ||
            ((((nuint)input.Handle | (nuint)weights.Handle | (nuint)output.Handle) & 15u) != 0) ||
            DirectPtxBuffersOverlap(output, input) || DirectPtxBuffersOverlap(output, weights))
            return false;

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCapsuleProjectionKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxCapsuleProjectionF32Kernel kernel = _directPtxCapsuleProjectionKernels.GetOrAdd(
                    operation, () => new PtxCapsuleProjectionF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxCapsuleProjectionKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCapsuleProjectionDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxCapsuleProjection(DirectPtxCapsuleProjectionOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxCapsuleProjectionKernels.GetOrAdd(
                    operation, () => new PtxCapsuleProjectionF32Kernel(_directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxCapsuleSquash(
        IGpuBuffer input,
        IGpuBuffer output,
        int capsules,
        int dimension,
        float epsilon)
    {
        const DirectPtxCapsuleSquashOperation operation = DirectPtxCapsuleSquashOperation.Forward;
        if (!ValidateDirectPtxCapsuleSquashBuffers(
                input, null, output, capsules, dimension, epsilon))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCapsuleSquashKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxCapsuleSquashF32Kernel kernel = _directPtxCapsuleSquashKernels.GetOrAdd(
                    operation, () => new PtxCapsuleSquashF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxCapsuleSquashKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchForward(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCapsuleSquashDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxCapsuleSquashBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer gradInput,
        int capsules,
        int dimension,
        float epsilon)
    {
        const DirectPtxCapsuleSquashOperation operation = DirectPtxCapsuleSquashOperation.Backward;
        if (!ValidateDirectPtxCapsuleSquashBuffers(
                gradOutput, input, gradInput, capsules, dimension, epsilon))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCapsuleSquashKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxCapsuleSquashF32Kernel kernel = _directPtxCapsuleSquashKernels.GetOrAdd(
                    operation, () => new PtxCapsuleSquashF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxCapsuleSquashKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchBackward(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradInput, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCapsuleSquashDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool ValidateDirectPtxCapsuleSquashBuffers(
        IGpuBuffer first,
        IGpuBuffer? second,
        IGpuBuffer output,
        int capsules,
        int dimension,
        float epsilon)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxCapsuleSquashF32Kernel.SupportsShape(capsules, dimension, epsilon))
            return false;
        long bytes = (long)capsules * dimension * sizeof(float);
        if (!HasExactBytes(first, bytes) || (second is not null && !HasExactBytes(second, bytes)) ||
            !HasExactBytes(output, bytes) || first.Handle == IntPtr.Zero ||
            (second is not null && second.Handle == IntPtr.Zero) || output.Handle == IntPtr.Zero)
            return false;
        nuint pointers = (nuint)first.Handle | (nuint)output.Handle;
        if (second is not null) pointers |= (nuint)second.Handle;
        return (pointers & 15u) == 0 && !DirectPtxBuffersOverlap(output, first) &&
            (second is null || !DirectPtxBuffersOverlap(output, second));
    }

    internal bool PrewarmDirectPtxCapsuleSquash(DirectPtxCapsuleSquashOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxCapsuleSquashKernels.GetOrAdd(
                    operation, () => new PtxCapsuleSquashF32Kernel(_directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxResidentScatterMeanRowsWithCounts(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer output,
        IGpuBuffer counts,
        int sourceRows,
        int features,
        int outputRows)
    {
        const DirectPtxResidentScatterAuxOperation operation =
            DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts;
        if (!IsDirectPtxSparseGraphEnabled || !ValidateDirectPtxResidentScatterAux(
                source, indices, output, counts, sourceRows, features, outputRows,
                countsAreOutputRows: true))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxResidentScatterAuxKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxResidentScatterAuxF32Kernel kernel = _directPtxResidentScatterAuxKernels.GetOrAdd(
                    operation, () => new PtxResidentScatterAuxF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxResidentScatterAuxKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchMean(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(counts, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxResidentScatterAuxDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxResidentScatterMaxBackwardRows(
        IGpuBuffer gradOutput,
        IGpuBuffer argmax,
        IGpuBuffer gradSource,
        int sourceRows,
        int features,
        int outputRows)
    {
        const DirectPtxResidentScatterAuxOperation operation =
            DirectPtxResidentScatterAuxOperation.MaxBackwardRows;
        if (!IsDirectPtxSparseGraphEnabled || !ValidateDirectPtxResidentScatterAux(
                gradSource, argmax, gradOutput, null, sourceRows, features, outputRows,
                countsAreOutputRows: false))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxResidentScatterAuxKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxResidentScatterAuxF32Kernel kernel = _directPtxResidentScatterAuxKernels.GetOrAdd(
                    operation, () => new PtxResidentScatterAuxF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxResidentScatterAuxKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchMaxBackward(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(argmax, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradSource, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxResidentScatterAuxDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool ValidateDirectPtxResidentScatterAux(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer grouped,
        IGpuBuffer? counts,
        int sourceRows,
        int features,
        int outputRows,
        bool countsAreOutputRows)
    {
        if (!PtxResidentScatterAuxF32Kernel.SupportsShape(sourceRows, features, outputRows))
            return false;
        long sourceBytes = (long)sourceRows * features * sizeof(float);
        long indexBytes = countsAreOutputRows
            ? (long)sourceRows * sizeof(int)
            : (long)outputRows * features * sizeof(int);
        long groupedBytes = (long)outputRows * features * sizeof(float);
        if (!HasExactBytes(source, sourceBytes) || !HasExactBytes(indices, indexBytes) ||
            !HasExactBytes(grouped, groupedBytes) ||
            (counts is not null && !HasExactBytes(counts, (long)outputRows * sizeof(float))) ||
            source.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            grouped.Handle == IntPtr.Zero || (counts is not null && counts.Handle == IntPtr.Zero))
            return false;
        nuint pointers = (nuint)source.Handle | (nuint)indices.Handle | (nuint)grouped.Handle;
        if (counts is not null) pointers |= (nuint)counts.Handle;
        if ((pointers & 15u) != 0 || DirectPtxBuffersOverlap(grouped, source) ||
            DirectPtxBuffersOverlap(grouped, indices) ||
            (!countsAreOutputRows && DirectPtxBuffersOverlap(source, indices)) ||
            (counts is not null && (DirectPtxBuffersOverlap(counts, source) ||
                DirectPtxBuffersOverlap(counts, indices) || DirectPtxBuffersOverlap(counts, grouped))))
            return false;
        return true;
    }

    internal bool PrewarmDirectPtxResidentScatterAux(DirectPtxResidentScatterAuxOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxResidentScatterAuxKernels.GetOrAdd(
                    operation, () => new PtxResidentScatterAuxF32Kernel(_directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxResidentScatterSoftmaxRows(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer output,
        int sourceRows,
        int features,
        int groups)
    {
        const DirectPtxResidentScatterSoftmaxOperation operation =
            DirectPtxResidentScatterSoftmaxOperation.Forward;
        if (!IsDirectPtxSparseGraphEnabled ||
            !ValidateDirectPtxResidentScatterSoftmax(
                source, null, indices, output, sourceRows, features, groups))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxResidentScatterSoftmaxKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxResidentScatterSoftmaxF32Kernel kernel =
                    _directPtxResidentScatterSoftmaxKernels.GetOrAdd(
                        operation, () => new PtxResidentScatterSoftmaxF32Kernel(
                            _directPtxRuntime!, operation));
                if (capturing && !_directPtxResidentScatterSoftmaxKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchForward(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxResidentScatterSoftmaxDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxResidentScatterSoftmaxBackwardRows(
        IGpuBuffer gradOutput,
        IGpuBuffer output,
        IGpuBuffer indices,
        IGpuBuffer gradSource,
        int sourceRows,
        int features,
        int groups)
    {
        const DirectPtxResidentScatterSoftmaxOperation operation =
            DirectPtxResidentScatterSoftmaxOperation.Backward;
        if (!IsDirectPtxSparseGraphEnabled ||
            !ValidateDirectPtxResidentScatterSoftmax(
                gradOutput, output, indices, gradSource, sourceRows, features, groups))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxResidentScatterSoftmaxKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxResidentScatterSoftmaxF32Kernel kernel =
                    _directPtxResidentScatterSoftmaxKernels.GetOrAdd(
                        operation, () => new PtxResidentScatterSoftmaxF32Kernel(
                            _directPtxRuntime!, operation));
                if (capturing && !_directPtxResidentScatterSoftmaxKernels.Pin(operation)) return false;
                lock (GpuDispatchLock)
                    kernel.LaunchBackward(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(gradSource, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxResidentScatterSoftmaxDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool ValidateDirectPtxResidentScatterSoftmax(
        IGpuBuffer first,
        IGpuBuffer? second,
        IGpuBuffer indices,
        IGpuBuffer output,
        int sourceRows,
        int features,
        int groups)
    {
        if (!PtxResidentScatterSoftmaxF32Kernel.SupportsShape(sourceRows, features, groups))
            return false;
        long rowBytes = (long)sourceRows * features * sizeof(float);
        if (!HasExactBytes(first, rowBytes) ||
            (second is not null && !HasExactBytes(second, rowBytes)) ||
            !HasExactBytes(indices, (long)sourceRows * sizeof(int)) ||
            !HasExactBytes(output, rowBytes) || first.Handle == IntPtr.Zero ||
            (second is not null && second.Handle == IntPtr.Zero) ||
            indices.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
            return false;
        nuint pointers = (nuint)first.Handle | (nuint)indices.Handle | (nuint)output.Handle;
        if (second is not null) pointers |= (nuint)second.Handle;
        return (pointers & 15u) == 0 && !DirectPtxBuffersOverlap(output, first) &&
            !DirectPtxBuffersOverlap(output, indices) &&
            (second is null || !DirectPtxBuffersOverlap(output, second));
    }

    internal bool PrewarmDirectPtxResidentScatterSoftmax(
        DirectPtxResidentScatterSoftmaxOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxResidentScatterSoftmaxKernels.GetOrAdd(
                    operation, () => new PtxResidentScatterSoftmaxF32Kernel(
                        _directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxUniformMeshLaplacian(
        IGpuBuffer faces,
        IGpuBuffer output,
        int faceCount,
        int vertexCount)
    {
        const int key = 0;
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxUniformMeshLaplacianF32Kernel.SupportsShape(faceCount, vertexCount) ||
            !HasExactBytes(faces, (long)faceCount * 3 * sizeof(int)) ||
            !HasExactBytes(output, (long)vertexCount * vertexCount * sizeof(float)) ||
            faces.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero ||
            ((((nuint)faces.Handle | (nuint)output.Handle) & 15u) != 0) ||
            DirectPtxBuffersOverlap(output, faces))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxUniformMeshLaplacianKernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxUniformMeshLaplacianF32Kernel kernel =
                    _directPtxUniformMeshLaplacianKernels.GetOrAdd(
                        key, () => new PtxUniformMeshLaplacianF32Kernel(_directPtxRuntime!));
                if (capturing && !_directPtxUniformMeshLaplacianKernels.Pin(key)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(faces, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxUniformMeshLaplacianDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxUniformMeshLaplacian()
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxUniformMeshLaplacianKernels.GetOrAdd(
                    0, () => new PtxUniformMeshLaplacianF32Kernel(_directPtxRuntime!));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxSparseOptimizer(
        IGpuBuffer parameter,
        IGpuBuffer indices,
        IGpuBuffer values,
        IGpuBuffer? state0,
        IGpuBuffer? state1,
        IGpuBuffer? state2,
        int nonZeros,
        DirectPtxSparseOptimizerKey key)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxSparseOptimizerF32Kernel.SupportsShape(parameter.Size, nonZeros) ||
            !PtxSparseOptimizerF32Kernel.SupportsConfiguration(key) ||
            !ValidateDirectPtxSparseOptimizerBuffers(
                parameter, indices, values, state0, state1, state2, key.Operation))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSparseOptimizerKernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxSparseOptimizerKernels.TryGetValue(
                    key, out PtxSparseOptimizerF32Kernel? kernel))
                {
                    kernel = _directPtxSparseOptimizerKernels.AddOrGetExisting(
                        key, new PtxSparseOptimizerF32Kernel(_directPtxRuntime, key));
                }
                if (capturing && !_directPtxSparseOptimizerKernels.Pin(key)) return false;
                int stateCount = PtxSparseOptimizerF32Kernel.GetStateCount(key.Operation);
                DirectPtxTensorView stateView0 = stateCount >= 1
                    ? DirectPtxTensorView.Create(state0!, kernel.Blueprint.Tensors[3]) : default;
                DirectPtxTensorView stateView1 = stateCount >= 2
                    ? DirectPtxTensorView.Create(state1!, kernel.Blueprint.Tensors[4]) : default;
                DirectPtxTensorView stateView2 = stateCount >= 3
                    ? DirectPtxTensorView.Create(state2!, kernel.Blueprint.Tensors[5]) : default;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(parameter, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(values, kernel.Blueprint.Tensors[2]),
                        stateView0, stateView1, stateView2);
            }
            System.Threading.Interlocked.Increment(ref _directPtxSparseOptimizerDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxSparseOptimizer(DirectPtxSparseOptimizerKey key)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing() ||
            !PtxSparseOptimizerF32Kernel.SupportsConfiguration(key))
            return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxSparseOptimizerKernels.TryGetValue(key, out _))
                    _directPtxSparseOptimizerKernels.AddOrGetExisting(
                        key, new PtxSparseOptimizerF32Kernel(_directPtxRuntime, key));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool ValidateDirectPtxSparseOptimizerBuffers(
        IGpuBuffer parameter,
        IGpuBuffer indices,
        IGpuBuffer values,
        IGpuBuffer? state0,
        IGpuBuffer? state1,
        IGpuBuffer? state2,
        DirectPtxSparseOptimizerOperation operation)
    {
        long parameterBytes = (long)PtxSparseOptimizerF32Kernel.ParameterElements * sizeof(float);
        long sparseBytes = (long)PtxSparseOptimizerF32Kernel.NonZeros * sizeof(float);
        int stateCount = PtxSparseOptimizerF32Kernel.GetStateCount(operation);
        if (!HasExactBytes(parameter, parameterBytes) || !HasExactBytes(indices, sparseBytes) ||
            !HasExactBytes(values, sparseBytes) || parameter.Handle == IntPtr.Zero ||
            indices.Handle == IntPtr.Zero || values.Handle == IntPtr.Zero ||
            (stateCount >= 1 && (!HasExactBytes(state0!, parameterBytes) || state0!.Handle == IntPtr.Zero)) ||
            (stateCount >= 2 && (!HasExactBytes(state1!, parameterBytes) || state1!.Handle == IntPtr.Zero)) ||
            (stateCount >= 3 && (!HasExactBytes(state2!, parameterBytes) || state2!.Handle == IntPtr.Zero)) ||
            (stateCount < 1 && state0 is not null) || (stateCount < 2 && state1 is not null) ||
            (stateCount < 3 && state2 is not null))
            return false;
        nuint pointers = (nuint)parameter.Handle | (nuint)indices.Handle | (nuint)values.Handle;
        if (stateCount >= 1) pointers |= (nuint)state0!.Handle;
        if (stateCount >= 2) pointers |= (nuint)state1!.Handle;
        if (stateCount >= 3) pointers |= (nuint)state2!.Handle;
        if ((pointers & 15u) != 0 || DirectPtxBuffersOverlap(parameter, indices) ||
            DirectPtxBuffersOverlap(parameter, values) || DirectPtxBuffersOverlap(indices, values) ||
            (stateCount >= 1 && (DirectPtxBuffersOverlap(state0!, parameter) ||
                DirectPtxBuffersOverlap(state0!, indices) || DirectPtxBuffersOverlap(state0!, values))) ||
            (stateCount >= 2 && (DirectPtxBuffersOverlap(state1!, parameter) ||
                DirectPtxBuffersOverlap(state1!, indices) || DirectPtxBuffersOverlap(state1!, values) ||
                DirectPtxBuffersOverlap(state1!, state0!))) ||
            (stateCount >= 3 && (DirectPtxBuffersOverlap(state2!, parameter) ||
                DirectPtxBuffersOverlap(state2!, indices) || DirectPtxBuffersOverlap(state2!, values) ||
                DirectPtxBuffersOverlap(state2!, state0!) || DirectPtxBuffersOverlap(state2!, state1!))))
            return false;
        return true;
    }

    internal bool TryDirectPtxFusedSparseLinear(
        IGpuBuffer input,
        IGpuBuffer packedCsr,
        IGpuBuffer values,
        IGpuBuffer bias,
        IGpuBuffer output,
        int batch,
        int inputFeatures,
        int outputFeatures,
        int nonZeros,
        int hasBias,
        int activation)
    {
        var key = new DirectPtxFusedSparseLinearKey(hasBias == 1, activation);
        if (!IsDirectPtxSparseGraphEnabled || (hasBias != 0 && hasBias != 1) ||
            !PtxFusedSparseLinearF32Kernel.SupportsShape(
                batch, inputFeatures, outputFeatures, nonZeros) ||
            !PtxFusedSparseLinearF32Kernel.SupportsActivation(activation) ||
            !ValidateDirectPtxFusedSparseLinearBuffers(
                input, packedCsr, values, bias, output, key.HasBias))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxFusedSparseLinearKernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxFusedSparseLinearKernels.TryGetValue(
                    key, out PtxFusedSparseLinearF32Kernel? kernel))
                {
                    kernel = _directPtxFusedSparseLinearKernels.AddOrGetExisting(
                        key, new PtxFusedSparseLinearF32Kernel(_directPtxRuntime, key));
                }
                if (capturing && !_directPtxFusedSparseLinearKernels.Pin(key)) return false;
                int outputContract = key.HasBias ? 4 : 3;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(packedCsr, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(values, kernel.Blueprint.Tensors[2]),
                        key.HasBias
                            ? DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[3])
                            : default,
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[outputContract]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxFusedSparseLinearDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxFusedSparseLinear(bool hasBias, int activation)
    {
        var key = new DirectPtxFusedSparseLinearKey(hasBias, activation);
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing() ||
            !PtxFusedSparseLinearF32Kernel.SupportsActivation(activation))
            return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxFusedSparseLinearKernels.TryGetValue(key, out _))
                    _directPtxFusedSparseLinearKernels.AddOrGetExisting(
                        key, new PtxFusedSparseLinearF32Kernel(_directPtxRuntime, key));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool ValidateDirectPtxFusedSparseLinearBuffers(
        IGpuBuffer input,
        IGpuBuffer packedCsr,
        IGpuBuffer values,
        IGpuBuffer bias,
        IGpuBuffer output,
        bool hasBias)
    {
        long inputBytes = (long)PtxFusedSparseLinearF32Kernel.Batch *
            PtxFusedSparseLinearF32Kernel.InputFeatures * sizeof(float);
        long packedBytes = (long)PtxFusedSparseLinearF32Kernel.PackedCsrElements * sizeof(int);
        long valuesBytes = (long)PtxFusedSparseLinearF32Kernel.NonZeros * sizeof(float);
        long biasBytes = (long)(hasBias ? PtxFusedSparseLinearF32Kernel.OutputFeatures : 1) * sizeof(float);
        long outputBytes = (long)PtxFusedSparseLinearF32Kernel.OutputElements * sizeof(float);
        if (!HasExactBytes(input, inputBytes) || !HasExactBytes(packedCsr, packedBytes) ||
            !HasExactBytes(values, valuesBytes) || !HasExactBytes(bias, biasBytes) ||
            !HasExactBytes(output, outputBytes) || input.Handle == IntPtr.Zero ||
            packedCsr.Handle == IntPtr.Zero || values.Handle == IntPtr.Zero ||
            bias.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
            return false;
        nuint pointers = (nuint)input.Handle | (nuint)packedCsr.Handle |
            (nuint)values.Handle | (nuint)bias.Handle | (nuint)output.Handle;
        if ((pointers & 15u) != 0 || DirectPtxBuffersOverlap(output, input) ||
            DirectPtxBuffersOverlap(output, packedCsr) || DirectPtxBuffersOverlap(output, values) ||
            DirectPtxBuffersOverlap(output, bias) || DirectPtxBuffersOverlap(input, packedCsr) ||
            DirectPtxBuffersOverlap(input, values) || DirectPtxBuffersOverlap(input, bias) ||
            DirectPtxBuffersOverlap(packedCsr, values) || DirectPtxBuffersOverlap(packedCsr, bias) ||
            DirectPtxBuffersOverlap(values, bias))
            return false;
        return true;
    }

    internal bool TryDirectPtxTensorGatherRows(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer output,
        int numIndices,
        int features)
    {
        const int key = 0;
        long sourceBytes = (long)PtxTensorGatherRowsF32Kernel.SourceRows *
            PtxTensorGatherRowsF32Kernel.Features * sizeof(float);
        long indexBytes = (long)PtxTensorGatherRowsF32Kernel.Indices * sizeof(int);
        long outputBytes = (long)PtxTensorGatherRowsF32Kernel.Indices *
            PtxTensorGatherRowsF32Kernel.Features * sizeof(float);
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxTensorGatherRowsF32Kernel.SupportsShape(numIndices, features) ||
            !HasExactBytes(source, sourceBytes) || !HasExactBytes(indices, indexBytes) ||
            !HasExactBytes(output, outputBytes) || source.Handle == IntPtr.Zero ||
            indices.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero ||
            ((((nuint)source.Handle | (nuint)indices.Handle | (nuint)output.Handle) & 15u) != 0) ||
            DirectPtxBuffersOverlap(output, source) || DirectPtxBuffersOverlap(output, indices) ||
            DirectPtxBuffersOverlap(source, indices))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxTensorGatherKernels.TryGetValue(key, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxTensorGatherKernels.TryGetValue(
                    key, out PtxTensorGatherRowsF32Kernel? kernel))
                    kernel = _directPtxTensorGatherKernels.AddOrGetExisting(
                        key, new PtxTensorGatherRowsF32Kernel(_directPtxRuntime));
                if (capturing && !_directPtxTensorGatherKernels.Pin(key)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxTensorGatherDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxTensorGatherRows()
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxTensorGatherKernels.TryGetValue(0, out _))
                    _directPtxTensorGatherKernels.AddOrGetExisting(
                        0, new PtxTensorGatherRowsF32Kernel(_directPtxRuntime));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxTensorScatterReduce(
        IGpuBuffer output,
        IGpuBuffer source,
        IGpuBuffer indices,
        int outer,
        int sourceDimension,
        int destinationDimension,
        int inner,
        int mode)
    {
        if (mode is < 0 or > 3) return false;
        var directMode = (DirectPtxTensorScatterReduceMode)mode;
        long sourceBytes = (long)PtxTensorScatterReduceF32Kernel.SourceElements * sizeof(float);
        long outputBytes = (long)PtxTensorScatterReduceF32Kernel.OutputElements * sizeof(float);
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxTensorScatterReduceF32Kernel.SupportsShape(
                outer, sourceDimension, destinationDimension, inner) ||
            !HasExactBytes(output, outputBytes) || !HasExactBytes(source, sourceBytes) ||
            !HasExactBytes(indices, sourceBytes) || output.Handle == IntPtr.Zero ||
            source.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            ((((nuint)output.Handle | (nuint)source.Handle | (nuint)indices.Handle) & 15u) != 0) ||
            DirectPtxBuffersOverlap(output, source) || DirectPtxBuffersOverlap(output, indices) ||
            DirectPtxBuffersOverlap(source, indices))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxTensorScatterReduceKernels.TryGetValue(directMode, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxTensorScatterReduceKernels.TryGetValue(
                    directMode, out PtxTensorScatterReduceF32Kernel? kernel))
                    kernel = _directPtxTensorScatterReduceKernels.AddOrGetExisting(
                        directMode, new PtxTensorScatterReduceF32Kernel(_directPtxRuntime, directMode));
                if (capturing && !_directPtxTensorScatterReduceKernels.Pin(directMode)) return false;
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxTensorScatterReduceDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxTensorScatterReduce(DirectPtxTensorScatterReduceMode mode)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxTensorScatterReduceKernels.TryGetValue(mode, out _))
                    _directPtxTensorScatterReduceKernels.AddOrGetExisting(
                        mode, new PtxTensorScatterReduceF32Kernel(_directPtxRuntime, mode));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxTensorScatterMean(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer output,
        IGpuBuffer counts,
        int sourceRows,
        int features,
        int outputRows)
    {
        const DirectPtxTensorScatterHighLevelOperation operation =
            DirectPtxTensorScatterHighLevelOperation.Mean;
        if (!IsDirectPtxSparseGraphEnabled ||
            !ValidateDirectPtxTensorScatterHighLevel(
                source, indices, output, counts, sourceRows, features, outputRows, operation))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxTensorScatterHighLevelKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxTensorScatterHighLevelF32Kernel kernel =
                    _directPtxTensorScatterHighLevelKernels.GetOrAdd(
                        operation, () => new PtxTensorScatterHighLevelF32Kernel(
                            _directPtxRuntime!, operation));
                if (capturing && !_directPtxTensorScatterHighLevelKernels.Pin(operation))
                    return false;
                lock (GpuDispatchLock)
                    kernel.LaunchMean(
                        DirectPtxTensorView.Create(source, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(counts, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxTensorScatterHighLevelDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxTensorScatterAddBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer indices,
        IGpuBuffer gradSource,
        int sourceRows,
        int features,
        int outputRows)
    {
        const DirectPtxTensorScatterHighLevelOperation operation =
            DirectPtxTensorScatterHighLevelOperation.AddBackward;
        if (!IsDirectPtxSparseGraphEnabled ||
            !ValidateDirectPtxTensorScatterHighLevel(
                gradSource, indices, gradOutput, null, sourceRows, features, outputRows, operation))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxTensorScatterHighLevelKernels.TryGetValue(operation, out _))
                    return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxTensorScatterHighLevelF32Kernel kernel =
                    _directPtxTensorScatterHighLevelKernels.GetOrAdd(
                        operation, () => new PtxTensorScatterHighLevelF32Kernel(
                            _directPtxRuntime!, operation));
                if (capturing && !_directPtxTensorScatterHighLevelKernels.Pin(operation))
                    return false;
                lock (GpuDispatchLock)
                    kernel.LaunchAddBackward(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(indices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradSource, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxTensorScatterHighLevelDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool ValidateDirectPtxTensorScatterHighLevel(
        IGpuBuffer source,
        IGpuBuffer indices,
        IGpuBuffer grouped,
        IGpuBuffer? counts,
        int sourceRows,
        int features,
        int outputRows,
        DirectPtxTensorScatterHighLevelOperation operation)
    {
        if (!PtxTensorScatterHighLevelF32Kernel.SupportsShape(
                sourceRows, features, outputRows))
            return false;
        long sourceBytes = (long)sourceRows * features * sizeof(float);
        long indexBytes = (long)sourceRows * sizeof(int);
        long groupedBytes = (long)outputRows * features * sizeof(float);
        bool mean = operation == DirectPtxTensorScatterHighLevelOperation.Mean;
        if (!HasExactBytes(source, sourceBytes) || !HasExactBytes(indices, indexBytes) ||
            !HasExactBytes(grouped, groupedBytes) ||
            (mean && (counts is null || !HasExactBytes(counts, (long)outputRows * sizeof(int)))) ||
            source.Handle == IntPtr.Zero || indices.Handle == IntPtr.Zero ||
            grouped.Handle == IntPtr.Zero || (mean && counts!.Handle == IntPtr.Zero))
            return false;
        nuint pointers = (nuint)source.Handle | (nuint)indices.Handle | (nuint)grouped.Handle;
        if (counts is not null) pointers |= (nuint)counts.Handle;
        if ((pointers & 15u) != 0 || DirectPtxBuffersOverlap(source, indices) ||
            DirectPtxBuffersOverlap(source, grouped) || DirectPtxBuffersOverlap(indices, grouped) ||
            (counts is not null && (DirectPtxBuffersOverlap(counts, source) ||
                DirectPtxBuffersOverlap(counts, indices) || DirectPtxBuffersOverlap(counts, grouped))))
            return false;
        return true;
    }

    internal bool PrewarmDirectPtxTensorScatterHighLevel(
        DirectPtxTensorScatterHighLevelOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxTensorScatterHighLevelKernels.GetOrAdd(
                    operation, () => new PtxTensorScatterHighLevelF32Kernel(
                        _directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool DirectPtxBuffersOverlap(IGpuBuffer left, IGpuBuffer right)
    {
        nuint leftStart = (nuint)left.Handle;
        nuint rightStart = (nuint)right.Handle;
        nuint leftEnd = checked(leftStart + (nuint)left.SizeInBytes);
        nuint rightEnd = checked(rightStart + (nuint)right.SizeInBytes);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    /// <summary>
    /// Attempts the exact FP32 SDDMM specialization for a 1024x1024 sampled
    /// product, K=64, and 16384 COO entries. The public CUDA SDDMM surface does
    /// not carry dense row counts, so exact allocation extents prove both row
    /// domains before the five-pointer kernel ABI is admitted.
    /// </summary>
    internal bool TryDirectPtxSddmmF32(
        IGpuBuffer rowIndices,
        IGpuBuffer columnIndices,
        IGpuBuffer x,
        IGpuBuffer y,
        IGpuBuffer output,
        int nonZeros,
        int inner)
    {
        if (!DirectPtxFeatureGate.IsSparseGraphEnabled)
        {
            DirectPtxLastError = "sddmm-feature-disabled";
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasValidatedSparseGraph(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "sddmm-architecture-not-implemented";
            return false;
        }
        if (!PtxFusedSddmmF32Kernel.SupportsShape(
            PtxFusedSddmmF32Kernel.Rows, PtxFusedSddmmF32Kernel.Columns, inner, nonZeros))
        {
            DirectPtxLastError = "sddmm-shape-not-implemented";
            return false;
        }
        if (rowIndices is null || columnIndices is null || x is null || y is null || output is null)
        {
            DirectPtxLastError = "sddmm-null-buffer";
            return false;
        }

        long patternBytes = (long)nonZeros * sizeof(int);
        long xBytes = (long)PtxFusedSddmmF32Kernel.Rows * inner * sizeof(float);
        long yBytes = (long)PtxFusedSddmmF32Kernel.Columns * inner * sizeof(float);
        long outputBytes = (long)nonZeros * sizeof(float);
        if (rowIndices.SizeInBytes != patternBytes || columnIndices.SizeInBytes != patternBytes ||
            x.SizeInBytes != xBytes || y.SizeInBytes != yBytes || output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "sddmm-physical-extent-mismatch";
            return false;
        }
        if (rowIndices.Handle == IntPtr.Zero || columnIndices.Handle == IntPtr.Zero ||
            x.Handle == IntPtr.Zero || y.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "sddmm-invalid-device-pointer";
            return false;
        }
        if ((((nuint)rowIndices.Handle | (nuint)columnIndices.Handle | (nuint)x.Handle |
              (nuint)y.Handle | (nuint)output.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "sddmm-alignment-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, rowIndices) ||
            DirectPtxBuffersOverlap(output, columnIndices) ||
            DirectPtxBuffersOverlap(output, x) || DirectPtxBuffersOverlap(output, y))
        {
            DirectPtxLastError = "sddmm-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxSddmmKey(nonZeros, inner);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSddmmKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = "Direct PTX SDDMM must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedSddmmF32Kernel kernel = _directPtxSddmmKernels.GetOrAdd(
                    key, () => new PtxFusedSddmmF32Kernel(_directPtxRuntime));
                if (capturing && !_directPtxSddmmKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX SDDMM module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(rowIndices, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(columnIndices, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(x, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(y, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxSddmmDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxSddmmF32(int nonZeros, int inner)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing() ||
            !PtxFusedSddmmF32Kernel.SupportsShape(
                PtxFusedSddmmF32Kernel.Rows, PtxFusedSddmmF32Kernel.Columns, inner, nonZeros))
            return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxSddmmKey(nonZeros, inner);
                _ = _directPtxSddmmKernels.GetOrAdd(
                    key, () => new PtxFusedSddmmF32Kernel(_directPtxRuntime));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxSddmmAudit(
        int nonZeros, int inner, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxSddmmKernels.TryGetValue(
                new DirectPtxSddmmKey(nonZeros, inner), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Attempts the baked FP32 decode-token D64 specialization that projects
    /// packed Q/K/V, adds bias, rotates Q/K, writes Q, and updates a dense KV
    /// cache in one launch. The exact physical ABI is checked before dispatch;
    /// the emitted PTX contains no shape, stride, layout, or position arguments.
    /// </summary>
    internal bool TryDirectPtxQkvRopeCacheD64(
        IGpuBuffer input,
        IGpuBuffer packedWeights,
        IGpuBuffer bias,
        IGpuBuffer cosine,
        IGpuBuffer sine,
        IGpuBuffer query,
        IGpuBuffer keyCache,
        IGpuBuffer valueCache,
        int heads,
        int cacheCapacity,
        int position)
    {
        if (!ValidateDirectPtxQkvRopeCacheEligibility(heads, cacheCapacity, position))
            return false;
        if (input is null || packedWeights is null || bias is null || cosine is null ||
            sine is null || query is null || keyCache is null || valueCache is null)
        {
            DirectPtxLastError = "qkv-rope-cache-null-buffer";
            return false;
        }

        int modelDimension = checked(heads * PtxFusedQkvRopeCacheD64Kernel.HeadDimension);
        long projectionElements = checked(3L * modelDimension);
        long cacheElements = checked((long)cacheCapacity * modelDimension);
        long ropeElements = checked((long)cacheCapacity *
            (PtxFusedQkvRopeCacheD64Kernel.HeadDimension / 2));
        if (input.SizeInBytes != (long)modelDimension * sizeof(float) ||
            packedWeights.SizeInBytes != projectionElements * modelDimension * sizeof(float) ||
            bias.SizeInBytes != projectionElements * sizeof(float) ||
            cosine.SizeInBytes != ropeElements * sizeof(float) ||
            sine.SizeInBytes != ropeElements * sizeof(float) ||
            query.SizeInBytes != (long)modelDimension * sizeof(float) ||
            keyCache.SizeInBytes != cacheElements * sizeof(float) ||
            valueCache.SizeInBytes != cacheElements * sizeof(float))
        {
            DirectPtxLastError = "qkv-rope-cache-physical-extent-mismatch";
            return false;
        }
        if (input.Handle == IntPtr.Zero || packedWeights.Handle == IntPtr.Zero ||
            bias.Handle == IntPtr.Zero || cosine.Handle == IntPtr.Zero ||
            sine.Handle == IntPtr.Zero || query.Handle == IntPtr.Zero ||
            keyCache.Handle == IntPtr.Zero || valueCache.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "qkv-rope-cache-invalid-device-pointer";
            return false;
        }
        if ((((nuint)input.Handle | (nuint)packedWeights.Handle | (nuint)bias.Handle |
              (nuint)cosine.Handle | (nuint)sine.Handle | (nuint)query.Handle |
              (nuint)keyCache.Handle | (nuint)valueCache.Handle) & 15u) != 0)
        {
            DirectPtxLastError = "qkv-rope-cache-alignment-mismatch";
            return false;
        }
        if (DirectPtxQkvRopeCacheOutputsOverlap(
            input, packedWeights, bias, cosine, sine, query, keyCache, valueCache))
        {
            DirectPtxLastError = "qkv-rope-cache-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxQkvRopeCacheKey(heads, cacheCapacity, position);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxQkvRopeCacheKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX QKV/RoPE/cache must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedQkvRopeCacheD64Kernel kernel = GetOrCreateQkvRopeCacheKernel(key);
                // A graph executable retains this CUfunction after capture.
                // cuModuleUnload invalidates function handles, so a captured
                // specialization must never be selected as an LRU victim.
                if (capturing && !_directPtxQkvRopeCacheKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX QKV/RoPE/cache module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(packedWeights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(cosine, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(sine, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.Create(query, kernel.Blueprint.Tensors[5]),
                        DirectPtxTensorView.Create(keyCache, kernel.Blueprint.Tensors[6]),
                        DirectPtxTensorView.Create(valueCache, kernel.Blueprint.Tensors[7]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxQkvRopeCacheDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedQkvRopeCacheD64Kernel GetOrCreateQkvRopeCacheKernel(
        DirectPtxQkvRopeCacheKey key)
    {
        if (_directPtxQkvRopeCacheKernels.TryGetValue(
            key, out PtxFusedQkvRopeCacheD64Kernel? existing))
            return existing;
        return CreateAndCacheQkvRopeCacheKernelSlow(key);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedQkvRopeCacheD64Kernel CreateAndCacheQkvRopeCacheKernelSlow(
        DirectPtxQkvRopeCacheKey key) =>
        _directPtxQkvRopeCacheKernels.GetOrAdd(key, () =>
            new PtxFusedQkvRopeCacheD64Kernel(
                _directPtxRuntime!, key.Heads, key.CacheCapacity, key.Position));

    internal bool PrewarmDirectPtxQkvRopeCacheD64(
        int heads,
        int cacheCapacity,
        int position)
    {
        if (!ValidateDirectPtxQkvRopeCacheEligibility(heads, cacheCapacity, position))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX QKV/RoPE/cache prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateQkvRopeCacheKernel(
                    new DirectPtxQkvRopeCacheKey(heads, cacheCapacity, position));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool ValidateDirectPtxQkvRopeCacheEligibility(
        int heads,
        int cacheCapacity,
        int position)
    {
        if (!DirectPtxFeatureGate.IsQkvRopeCacheEnabled)
        {
            DirectPtxLastError = "qkv-rope-cache-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "qkv-rope-cache-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedQkvRopeCache(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "qkv-rope-cache-architecture-not-implemented";
            return false;
        }
        if (heads is not (4 or 8 or 16))
        {
            DirectPtxLastError = "qkv-rope-cache-head-count-not-implemented";
            return false;
        }
        if (cacheCapacity is not (16 or 32 or 64 or 128))
        {
            DirectPtxLastError = "qkv-rope-cache-capacity-not-implemented";
            return false;
        }
        if (position < 0 || position >= cacheCapacity)
        {
            DirectPtxLastError = "qkv-rope-cache-position-out-of-range";
            return false;
        }
        return true;
    }

    private static bool DirectPtxQkvRopeCacheOutputsOverlap(
        IGpuBuffer input,
        IGpuBuffer packedWeights,
        IGpuBuffer bias,
        IGpuBuffer cosine,
        IGpuBuffer sine,
        IGpuBuffer query,
        IGpuBuffer keyCache,
        IGpuBuffer valueCache)
    {
        return Overlaps(query, keyCache) || Overlaps(query, valueCache) ||
            Overlaps(keyCache, valueCache) || IsInput(query) ||
            IsInput(keyCache) || IsInput(valueCache);

        bool IsInput(IGpuBuffer output) =>
            Overlaps(output, input) || Overlaps(output, packedWeights) ||
            Overlaps(output, bias) || Overlaps(output, cosine) || Overlaps(output, sine);

        static bool Overlaps(IGpuBuffer left, IGpuBuffer right)
        {
            nuint leftStart = (nuint)left.Handle;
            nuint rightStart = (nuint)right.Handle;
            nuint leftEnd = checked(leftStart + (nuint)left.SizeInBytes);
            nuint rightEnd = checked(rightStart + (nuint)right.SizeInBytes);
            return leftStart < rightEnd && rightStart < leftEnd;
        }
    }

    internal bool TryGetDirectPtxQkvRopeCacheAudit(
        int heads,
        int cacheCapacity,
        int position,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxQkvRopeCacheKey(heads, cacheCapacity, position);
            if (_directPtxQkvRopeCacheKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Attempts the canonical FP16-BHSD S in {16,32,64,128}, D=64 online attention
    /// specialization. All layout and extent checks happen here; the emitted
    /// PTX has no dtype, shape, stride, or storage-offset branches.
    /// </summary>
    internal bool TryDirectPtxOnlineAttention(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer outputFloat,
        IGpuBuffer softmaxStatsFloat,
        int batchHeads,
        float scale,
        bool isCausal,
        int sequenceLength = PtxOnlineFusedAttention128x64Kernel.DefaultSequenceLength)
    {
        return TryDirectPtxOnlineAttentionCore(
            queryHalf, keyHalf, valueHalf,
            gammaFloat: null, betaFloat: null,
            outputFloat, softmaxStatsFloat,
            batch: 1, queryHeads: batchHeads, keyValueHeads: batchHeads,
            querySequence: sequenceLength, keyValueSequence: sequenceLength,
            scale, isCausal, fuseLayerNormGelu: false, epsilon: 1e-5f,
            emitSoftmaxStats: true, causalQueryOffset: 0);
    }

    /// <summary>
    /// Attempts the baked dense-BHSD FP16 online-attention family for MHA, GQA,
    /// or MQA. Sq and Skv are independent specialization dimensions; the emitted
    /// PTX maps each query head to its KV head without runtime stride/layout checks.
    /// Causal alignment is a baked specialization value: offset zero implements
    /// FlashAttention's top-left convention; Skv-Sq implements SDPA bottom-right.
    /// </summary>
    internal bool TryDirectPtxOnlineAttentionFamily(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        bool emitSoftmaxStats = true,
        int causalQueryOffset = 0)
    {
        return TryDirectPtxOnlineAttentionCore(
            queryHalf, keyHalf, valueHalf,
            gammaFloat: null, betaFloat: null,
            outputFloat, softmaxStatsFloat,
            batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
            scale, isCausal, fuseLayerNormGelu: false, epsilon: 1e-5f,
            emitSoftmaxStats, causalQueryOffset);
    }

    /// <summary>
    /// Same online attention dataflow with a head-local LayerNorm(D=64),
    /// affine transform, and tanh-GELU fused before the single output store.
    /// </summary>
    internal bool TryDirectPtxOnlineAttentionLayerNormGelu(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer gammaFloat,
        IGpuBuffer betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer softmaxStatsFloat,
        int batchHeads,
        float scale,
        bool isCausal,
        float epsilon = 1e-5f,
        int sequenceLength = PtxOnlineFusedAttention128x64Kernel.DefaultSequenceLength)
    {
        return TryDirectPtxOnlineAttentionCore(
            queryHalf, keyHalf, valueHalf,
            gammaFloat, betaFloat,
            outputFloat, softmaxStatsFloat,
            batch: 1, queryHeads: batchHeads, keyValueHeads: batchHeads,
            querySequence: sequenceLength, keyValueSequence: sequenceLength,
            scale, isCausal, fuseLayerNormGelu: true, epsilon,
            emitSoftmaxStats: true, causalQueryOffset: 0);
    }

    private bool TryDirectPtxOnlineAttentionCore(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer? gammaFloat,
        IGpuBuffer? betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        bool emitSoftmaxStats,
        int causalQueryOffset)
    {
        if (!IsDirectPtxAttentionEnabled)
            return false;
        if (causalQueryOffset < -querySequence)
        {
            DirectPtxLastError = "causal-query-offset-outside-query-domain";
            return false;
        }
        if (!isCausal && causalQueryOffset != 0)
        {
            DirectPtxLastError = "causal-query-offset-without-causal-mask";
            return false;
        }
        // Precise fallback reason instead of an opaque swallowed NullReferenceException: LaunchAttentionKernel
        // dereferences softmaxStatsFloat! when emitSoftmaxStats is set, and gammaFloat!/betaFloat! when
        // fuseLayerNormGelu is set. Reject a null buffer here so DirectPtxLastError names the missing input.
        if (emitSoftmaxStats && softmaxStatsFloat is null)
        {
            DirectPtxLastError = "softmax-stats-buffer-null";
            return false;
        }
        if (fuseLayerNormGelu && (gammaFloat is null || betaFloat is null))
        {
            DirectPtxLastError = "layernorm-gamma-or-beta-null";
            return false;
        }

        DirectPtxEligibilityResult eligibility = DirectPtxAttentionEligibility.Evaluate(
            new DirectPtxAttentionRequest(
                DirectPtxArchitecture.Classify(_ccMajor, _ccMinor),
                _ccMajor,
                _ccMinor,
                DirectPtxPhysicalType.Float16,
                DirectPtxPhysicalLayout.Bhsd,
                Batch: batch,
                QueryHeads: queryHeads,
                KeyValueHeads: keyValueHeads,
                QuerySequence: querySequence,
                KeyValueSequence: keyValueSequence,
                HeadDimension: PtxOnlineFusedAttention128x64Kernel.HeadDimension,
                Mask: isCausal
                    ? causalQueryOffset == 0
                        ? DirectPtxAttentionMaskKind.CausalTopLeft
                        : DirectPtxAttentionMaskKind.CausalBottomRight
                    : DirectPtxAttentionMaskKind.None,
                Phase: DirectPtxAttentionPhase.Inference,
                DropoutProbability: 0,
                IsRagged: false,
                UsesPagedKv: false));
        if (!eligibility.IsEligible)
        {
            DirectPtxLastError = eligibility.Reason;
            return false;
        }
        try
        {
            bool capturing = IsStreamCapturing();
            // Establish this backend's context once on the calling thread.
            // DirectPtxRuntime detects it and skips redundant push/pop pairs.
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                var planKey = new DirectPtxAttentionPlanKey(
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    isCausal, causalQueryOffset, fuseLayerNormGelu,
                    emitSoftmaxStats,
                    BitConverter.SingleToInt32Bits(scale),
                    BitConverter.SingleToInt32Bits(epsilon));

                // Capture may launch only an already-loaded in-memory plan. No
                // disk lookup, module JIT, event allocation, or autotune occurs
                // while the stream is recording.
                if (capturing && (!_directPtxAttentionPlans.TryGetValue(planKey, out int captureWarps) ||
                    !_directPtxAttentionKernels.TryGetValue(
                        DirectPtxAttentionKey.FromPlan(planKey, captureWarps), out _)))
                {
                    DirectPtxLastError = "Direct PTX attention must be prewarmed before CUDA graph capture.";
                    return false;
                }

                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);

                if (!_directPtxAttentionPlans.TryGetValue(planKey, out int selectedWarps))
                    selectedWarps = ResolveAttentionPlanSlow(
                        planKey, queryHalf, keyHalf, valueHalf,
                        gammaFloat, betaFloat, outputFloat, softmaxStatsFloat,
                        scale, epsilon);

                PtxOnlineFusedAttention128x64Kernel kernel = GetOrCreateAttentionKernel(
                    planKey, selectedWarps, scale, epsilon);
                lock (GpuDispatchLock)
                    LaunchAttentionKernel(
                        kernel, queryHalf, keyHalf, valueHalf,
                        gammaFloat, betaFloat, outputFloat, softmaxStatsFloat);
            }
            System.Threading.Interlocked.Increment(ref _directPtxAttentionDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            // This is an experimental opt-in path. A bad shape/capability/JIT
            // must preserve the existing CUDA implementation as the fallback.
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    // Keep all closure-bearing autotune code out of the resident dispatch
    // method. C# creates a display object at method entry even when the branch
    // containing a captured lambda is not taken.
    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private int ResolveAttentionPlanSlow(
        DirectPtxAttentionPlanKey plan,
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer? gammaFloat,
        IGpuBuffer? betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat,
        float scale,
        float epsilon)
    {
        bool persisted = DirectPtxAttentionAutotuner.TryLoad(
            _directPtxRuntime!, plan.Batch, plan.QueryHeads, plan.KeyValueHeads,
            plan.QuerySequence, plan.KeyValueSequence, plan.IsCausal, plan.CausalQueryOffset,
            plan.FuseLayerNormGelu, plan.EmitSoftmaxStats, scale, epsilon, out int selectedWarps);
        int[] candidates = DirectPtxAttentionAutotuner.Candidates(plan.QuerySequence);
        if (!persisted) selectedWarps = candidates[0];

        if (!persisted && DirectPtxFeatureGate.IsAutotuneEnabled && candidates.Length > 1)
        {
            double bestMilliseconds = double.PositiveInfinity;
            int bestWarps = selectedWarps;
            lock (GpuDispatchLock)
            {
                foreach (int candidate in candidates)
                {
                    PtxOnlineFusedAttention128x64Kernel candidateKernel = GetOrCreateAttentionKernel(
                        plan, candidate, scale, epsilon);
                    float milliseconds = _directPtxRuntime!.MeasureKernelMilliseconds(
                        () => LaunchAttentionKernel(
                            candidateKernel, queryHalf, keyHalf, valueHalf,
                            gammaFloat, betaFloat, outputFloat, softmaxStatsFloat),
                        warmup: 3, iterations: 12);
                    if (milliseconds < bestMilliseconds)
                    {
                        bestMilliseconds = milliseconds;
                        bestWarps = candidate;
                    }
                }
            }
            selectedWarps = bestWarps;
            PtxOnlineFusedAttention128x64Kernel winner = GetOrCreateAttentionKernel(
                plan, selectedWarps, scale, epsilon);
            DirectPtxAttentionAutotuner.Store(
                _directPtxRuntime!, plan.Batch, plan.QueryHeads, plan.KeyValueHeads,
                plan.QuerySequence, plan.KeyValueSequence, plan.IsCausal, plan.CausalQueryOffset,
                plan.FuseLayerNormGelu, plan.EmitSoftmaxStats, scale, epsilon,
                selectedWarps, bestMilliseconds,
                winner.AttentionTflops((float)bestMilliseconds));
        }
        _directPtxAttentionPlans.Set(plan, selectedWarps);
        return selectedWarps;
    }

    private PtxOnlineFusedAttention128x64Kernel GetOrCreateAttentionKernel(
        DirectPtxAttentionPlanKey plan,
        int warpsPerBlock,
        float scale,
        float epsilon)
    {
        DirectPtxAttentionKey key = DirectPtxAttentionKey.FromPlan(plan, warpsPerBlock);
        if (_directPtxAttentionKernels.TryGetValue(key, out PtxOnlineFusedAttention128x64Kernel existing))
            return existing;
        return CreateAndCacheAttentionKernelSlow(key, plan, warpsPerBlock, scale, epsilon);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxOnlineFusedAttention128x64Kernel CreateAndCacheAttentionKernelSlow(
        DirectPtxAttentionKey key,
        DirectPtxAttentionPlanKey plan,
        int warpsPerBlock,
        float scale,
        float epsilon)
    {
        var created = new PtxOnlineFusedAttention128x64Kernel(
            _directPtxRuntime!, plan.Batch, plan.QueryHeads, plan.KeyValueHeads,
            plan.QuerySequence, plan.KeyValueSequence, plan.IsCausal,
            plan.FuseLayerNormGelu, scale, epsilon,
            plan.EmitSoftmaxStats, warpsPerBlock, plan.CausalQueryOffset);
        return _directPtxAttentionKernels.AddOrGetExisting(key, created);
    }

    private static void LaunchAttentionKernel(
        PtxOnlineFusedAttention128x64Kernel kernel,
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer? gammaFloat,
        IGpuBuffer? betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat)
    {
        DirectPtxTensorView gamma = default;
        DirectPtxTensorView beta = default;
        if (kernel.FuseLayerNormGelu)
        {
            gamma = DirectPtxTensorView.Create(gammaFloat!, kernel.Blueprint.Tensors[3]);
            beta = DirectPtxTensorView.Create(betaFloat!, kernel.Blueprint.Tensors[4]);
        }
        DirectPtxTensorView stats = kernel.EmitSoftmaxStats
            ? DirectPtxTensorView.Create(softmaxStatsFloat!, kernel.Blueprint.Tensors[6])
            : default;
        kernel.Launch(
            DirectPtxTensorView.Create(queryHalf, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.Create(keyHalf, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.Create(valueHalf, kernel.Blueprint.Tensors[2]),
            gamma,
            beta,
            DirectPtxTensorView.Create(outputFloat, kernel.Blueprint.Tensors[5]),
            stats);
    }

    internal bool TryDirectPtxFlashDecodeD64(
        IGpuBuffer query,
        IGpuBuffer key,
        IGpuBuffer value,
        IGpuBuffer output,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        float scale)
    {
        if (!IsDirectPtxFlashDecodeEnabled) return false;
        return TryDirectPtxDecodeCore(
            query, key, value, blockTable: null, output,
            isPaged: false, queryHeads, keyValueHeads, sequenceLength,
            blockSize: 0, scale);
    }

    internal bool TryDirectPtxPagedDecodeD64(
        IGpuBuffer query,
        IGpuBuffer keyPages,
        IGpuBuffer valuePages,
        IGpuBuffer blockTable,
        IGpuBuffer output,
        int queryHeads,
        int keyValueHeads,
        int blockSize,
        int sequenceLength,
        float scale)
    {
        if (!IsDirectPtxPagedDecodeEnabled) return false;
        return TryDirectPtxDecodeCore(
            query, keyPages, valuePages, blockTable, output,
            isPaged: true, queryHeads, keyValueHeads, sequenceLength,
            blockSize, scale);
    }

    private bool TryDirectPtxDecodeCore(
        IGpuBuffer query,
        IGpuBuffer key,
        IGpuBuffer value,
        IGpuBuffer? blockTable,
        IGpuBuffer output,
        bool isPaged,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        float scale)
    {
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
        {
            DirectPtxLastError = "decode-head-map-not-implemented";
            return false;
        }
        if (sequenceLength is not (16 or 32 or 64 or 128))
        {
            DirectPtxLastError = "decode-sequence-bucket-not-implemented";
            return false;
        }
        if (!float.IsFinite(scale))
        {
            DirectPtxLastError = "decode-scale-not-finite";
            return false;
        }
        if (isPaged && blockSize is not (16 or 32))
        {
            DirectPtxLastError = "paged-decode-block-size-not-implemented";
            return false;
        }

        long elementsPerBlock = isPaged
            ? checked((long)blockSize * keyValueHeads * PtxFusedDecodeAttentionD64Kernel.HeadDimension)
            : 0;
        if (key.SizeInBytes != value.SizeInBytes || key.SizeInBytes <= 0 ||
            (isPaged && (key.SizeInBytes % (elementsPerBlock * sizeof(float)) != 0)))
        {
            DirectPtxLastError = "decode-kv-physical-extent-mismatch";
            return false;
        }
        int poolBlocks = isPaged
            ? checked((int)(key.SizeInBytes / (elementsPerBlock * sizeof(float))))
            : 0;
        int logicalBlocks = isPaged ? (sequenceLength + blockSize - 1) / blockSize : 0;
        if (isPaged && (poolBlocks < logicalBlocks || blockTable is null ||
            blockTable.SizeInBytes != checked(logicalBlocks * sizeof(int))))
        {
            DirectPtxLastError = "paged-decode-table-or-pool-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var keyShape = new DirectPtxDecodeKey(
                isPaged, queryHeads, keyValueHeads, sequenceLength,
                blockSize, poolBlocks, BitConverter.SingleToInt32Bits(scale));
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxDecodeKernels.TryGetValue(keyShape, out _))
                {
                    DirectPtxLastError = "Direct PTX decode must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedDecodeAttentionD64Kernel kernel = GetOrCreateDecodeKernel(keyShape, scale);
                lock (GpuDispatchLock)
                {
                    if (isPaged)
                    {
                        kernel.LaunchPaged(
                            DirectPtxTensorView.Create(query, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(key, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(value, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(blockTable!, kernel.Blueprint.Tensors[3]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
                    }
                    else
                    {
                        kernel.LaunchDense(
                            DirectPtxTensorView.Create(query, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(key, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(value, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
                    }
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxDecodeDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedDecodeAttentionD64Kernel GetOrCreateDecodeKernel(
        DirectPtxDecodeKey key,
        float scale)
    {
        if (_directPtxDecodeKernels.TryGetValue(key, out PtxFusedDecodeAttentionD64Kernel? existing))
            return existing;
        return CreateAndCacheDecodeKernelSlow(key, scale);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedDecodeAttentionD64Kernel CreateAndCacheDecodeKernelSlow(
        DirectPtxDecodeKey key,
        float scale) =>
        _directPtxDecodeKernels.GetOrAdd(key, () =>
            new PtxFusedDecodeAttentionD64Kernel(
                _directPtxRuntime!, key.IsPaged, key.QueryHeads, key.KeyValueHeads,
                key.SequenceLength, key.BlockSize, key.PoolBlocks, scale));

    internal bool PrewarmDirectPtxDecodeD64(
        bool isPaged,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        int poolBlocks,
        float scale)
    {
        if (isPaged ? !IsDirectPtxPagedDecodeEnabled : !IsDirectPtxFlashDecodeEnabled)
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX decode prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxDecodeKey(
                    isPaged, queryHeads, keyValueHeads, sequenceLength,
                    blockSize, poolBlocks, BitConverter.SingleToInt32Bits(scale));
                _ = GetOrCreateDecodeKernel(key, scale);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxDecodeAudit(
        bool isPaged,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        int poolBlocks,
        float scale,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxDecodeKey(
                isPaged, queryHeads, keyValueHeads, sequenceLength,
                blockSize, poolBlocks, BitConverter.SingleToInt32Bits(scale));
            if (_directPtxDecodeKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxPagedPrefillD64(
        IGpuBuffer query,
        IGpuBuffer keyPages,
        IGpuBuffer valuePages,
        IGpuBuffer blockTable,
        IGpuBuffer output,
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        float scale)
    {
        if (!IsDirectPtxPagedPrefillEnabled) return false;
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
        {
            DirectPtxLastError = "paged-prefill-head-map-not-implemented";
            return false;
        }
        if (queryCount is not (2 or 4 or 8 or 16 or 32))
        {
            DirectPtxLastError = "paged-prefill-query-bucket-not-implemented";
            return false;
        }
        if (startPosition < 0 || checked(startPosition + queryCount) > 128)
        {
            DirectPtxLastError = "paged-prefill-key-domain-not-implemented";
            return false;
        }
        if (blockSize is not (16 or 32))
        {
            DirectPtxLastError = "paged-prefill-block-size-not-implemented";
            return false;
        }
        if (!float.IsFinite(scale))
        {
            DirectPtxLastError = "paged-prefill-scale-not-finite";
            return false;
        }

        int maximumKeyLength = checked(startPosition + queryCount);
        long elementsPerBlock = checked(
            (long)blockSize * keyValueHeads * PtxFusedPagedPrefillAttentionD64Kernel.HeadDimension);
        if (keyPages.SizeInBytes != valuePages.SizeInBytes || keyPages.SizeInBytes <= 0 ||
            keyPages.SizeInBytes % (elementsPerBlock * sizeof(float)) != 0)
        {
            DirectPtxLastError = "paged-prefill-kv-physical-extent-mismatch";
            return false;
        }
        int poolBlocks = checked((int)(keyPages.SizeInBytes / (elementsPerBlock * sizeof(float))));
        int logicalBlocks = (maximumKeyLength + blockSize - 1) / blockSize;
        if (poolBlocks < logicalBlocks || blockTable.SizeInBytes != checked(logicalBlocks * sizeof(int)))
        {
            DirectPtxLastError = "paged-prefill-table-or-pool-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxPagedPrefillKey(
                queryHeads, keyValueHeads, queryCount, startPosition,
                blockSize, poolBlocks, BitConverter.SingleToInt32Bits(scale));
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxPagedPrefillKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = "Direct PTX paged prefill must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedPagedPrefillAttentionD64Kernel kernel = GetOrCreatePagedPrefillKernel(key, scale);
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(query, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(keyPages, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(valuePages, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(blockTable, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxPagedPrefillDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedPagedPrefillAttentionD64Kernel GetOrCreatePagedPrefillKernel(
        DirectPtxPagedPrefillKey key,
        float scale)
    {
        if (_directPtxPagedPrefillKernels.TryGetValue(
            key, out PtxFusedPagedPrefillAttentionD64Kernel? existing))
            return existing;
        return CreateAndCachePagedPrefillKernelSlow(key, scale);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedPagedPrefillAttentionD64Kernel CreateAndCachePagedPrefillKernelSlow(
        DirectPtxPagedPrefillKey key,
        float scale) =>
        _directPtxPagedPrefillKernels.GetOrAdd(key, () =>
            new PtxFusedPagedPrefillAttentionD64Kernel(
                _directPtxRuntime!, key.QueryHeads, key.KeyValueHeads,
                key.QueryCount, key.StartPosition, key.BlockSize,
                key.PoolBlocks, scale));

    internal bool PrewarmDirectPtxPagedPrefillD64(
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        int poolBlocks,
        float scale)
    {
        if (!IsDirectPtxPagedPrefillEnabled) return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX paged-prefill prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxPagedPrefillKey(
                    queryHeads, keyValueHeads, queryCount, startPosition,
                    blockSize, poolBlocks, BitConverter.SingleToInt32Bits(scale));
                _ = GetOrCreatePagedPrefillKernel(key, scale);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxPagedPrefillAudit(
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        int poolBlocks,
        float scale,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxPagedPrefillKey(
                queryHeads, keyValueHeads, queryCount, startPosition,
                blockSize, poolBlocks, BitConverter.SingleToInt32Bits(scale));
            if (_directPtxPagedPrefillKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Deterministic FP32 D64 FlashAttention backward. The specialization
    /// recomputes probabilities from Q/K plus the forward LSE statistic and
    /// writes dQ/dK/dV without SxS intermediates, atomics, or dynamic strides.
    /// </summary>
    internal bool TryDirectPtxFlashAttentionBackwardD64(
        IGpuBuffer gradOutput,
        IGpuBuffer query,
        IGpuBuffer key,
        IGpuBuffer value,
        IGpuBuffer output,
        IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery,
        IGpuBuffer gradKey,
        IGpuBuffer gradValue,
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        int headDimension,
        float scale,
        bool isCausal,
        IGpuBuffer? attentionBias,
        int biasBatchStride = 0)
    {
        if (!IsDirectPtxFlashAttentionBackwardEnabled) return false;
        if (headDimension != PtxFlashAttentionBackwardD64Kernel.HeadDimension)
        {
            DirectPtxLastError = "flash-attention-backward-head-dimension-not-implemented";
            return false;
        }
        if (batch is <= 0 or > 16)
        {
            DirectPtxLastError = "flash-attention-backward-batch-not-implemented";
            return false;
        }
        if (heads is <= 0 or > 128)
        {
            DirectPtxLastError = "flash-attention-backward-head-count-not-implemented";
            return false;
        }
        if (querySequence is not (16 or 32 or 64 or 128) ||
            keyValueSequence is not (16 or 32 or 64 or 128))
        {
            DirectPtxLastError = "flash-attention-backward-sequence-bucket-not-implemented";
            return false;
        }
        if (!float.IsFinite(scale))
        {
            DirectPtxLastError = "flash-attention-backward-scale-not-finite";
            return false;
        }

        long queryElements = checked((long)batch * heads * querySequence * headDimension);
        long keyValueElements = checked((long)batch * heads * keyValueSequence * headDimension);
        long statsElements = checked((long)batch * heads * querySequence);
        int canonicalBiasBatchStride = checked(heads * querySequence * keyValueSequence);
        int bakedBiasBatchStride = -1;
        if (attentionBias is not null)
        {
            if (biasBatchStride != 0 && biasBatchStride != canonicalBiasBatchStride)
            {
                DirectPtxLastError = "flash-attention-backward-bias-stride-not-canonical";
                return false;
            }
            long biasElements = biasBatchStride == 0
                ? canonicalBiasBatchStride
                : checked((long)batch * canonicalBiasBatchStride);
            if (attentionBias.SizeInBytes != biasElements * sizeof(float))
            {
                DirectPtxLastError = "flash-attention-backward-bias-physical-extent-mismatch";
                return false;
            }
            bakedBiasBatchStride = biasBatchStride;
        }
        if (gradOutput.SizeInBytes != queryElements * sizeof(float) ||
            query.SizeInBytes != queryElements * sizeof(float) ||
            output.SizeInBytes != queryElements * sizeof(float) ||
            gradQuery.SizeInBytes != queryElements * sizeof(float) ||
            key.SizeInBytes != keyValueElements * sizeof(float) ||
            value.SizeInBytes != keyValueElements * sizeof(float) ||
            gradKey.SizeInBytes != keyValueElements * sizeof(float) ||
            gradValue.SizeInBytes != keyValueElements * sizeof(float) ||
            softmaxStats.SizeInBytes != statsElements * sizeof(float))
        {
            DirectPtxLastError = "flash-attention-backward-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var keyShape = new DirectPtxFlashAttentionBackwardKey(
                batch, heads, querySequence, keyValueSequence, isCausal,
                BitConverter.SingleToInt32Bits(scale), bakedBiasBatchStride);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxFlashAttentionBackwardKernels.TryGetValue(keyShape, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX FlashAttention backward must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFlashAttentionBackwardD64Kernel kernel =
                    GetOrCreateFlashAttentionBackwardKernel(keyShape, scale);
                DirectPtxTensorView? biasView = attentionBias is null
                    ? null
                    : DirectPtxTensorView.Create(attentionBias, kernel.Blueprint.Tensors[9]);
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(query, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(key, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(value, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.Create(softmaxStats, kernel.Blueprint.Tensors[5]),
                        DirectPtxTensorView.Create(gradQuery, kernel.Blueprint.Tensors[6]),
                        DirectPtxTensorView.Create(gradKey, kernel.Blueprint.Tensors[7]),
                        DirectPtxTensorView.Create(gradValue, kernel.Blueprint.Tensors[8]),
                        biasView);
            }
            System.Threading.Interlocked.Increment(ref _directPtxFlashAttentionBackwardDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFlashAttentionBackwardD64Kernel GetOrCreateFlashAttentionBackwardKernel(
        DirectPtxFlashAttentionBackwardKey key,
        float scale)
    {
        if (_directPtxFlashAttentionBackwardKernels.TryGetValue(
            key, out PtxFlashAttentionBackwardD64Kernel? existing))
            return existing;
        return CreateAndCacheFlashAttentionBackwardKernelSlow(key, scale);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFlashAttentionBackwardD64Kernel CreateAndCacheFlashAttentionBackwardKernelSlow(
        DirectPtxFlashAttentionBackwardKey key,
        float scale) =>
        _directPtxFlashAttentionBackwardKernels.GetOrAdd(key, () =>
            new PtxFlashAttentionBackwardD64Kernel(
                _directPtxRuntime!, key.Batch, key.Heads, key.QuerySequence,
                key.KeyValueSequence, scale, key.IsCausal, key.BiasBatchStride));

    internal bool PrewarmDirectPtxFlashAttentionBackwardD64(
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        int biasBatchStride = -1)
    {
        if (!IsDirectPtxFlashAttentionBackwardEnabled) return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX FlashAttention-backward prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxFlashAttentionBackwardKey(
                    batch, heads, querySequence, keyValueSequence, isCausal,
                    BitConverter.SingleToInt32Bits(scale), biasBatchStride);
                _ = GetOrCreateFlashAttentionBackwardKernel(key, scale);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxFlashAttentionBackwardAudits(
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        out DirectPtxKernelAudit gradQueryAudit,
        out DirectPtxKernelAudit gradKeyValueAudit,
        int biasBatchStride = -1)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFlashAttentionBackwardKey(
                batch, heads, querySequence, keyValueSequence, isCausal,
                BitConverter.SingleToInt32Bits(scale), biasBatchStride);
            if (_directPtxFlashAttentionBackwardKernels.TryGetValue(key, out var kernel))
            {
                gradQueryAudit = kernel.GradQueryAudit;
                gradKeyValueAudit = kernel.GradKeyValueAudit;
                return true;
            }
        }
        gradQueryAudit = null!;
        gradKeyValueAudit = null!;
        return false;
    }

    /// <summary>
    /// Fused deterministic FP32 D64 backward for APIs that provide the exact
    /// softmax probabilities. The specialization writes dQ/dK/dV directly and
    /// uses no global scratch, transposes, atomics, or runtime stride arguments.
    /// </summary>
    internal bool TryDirectPtxAttentionBackwardD64(
        IGpuBuffer gradOutput,
        IGpuBuffer query,
        IGpuBuffer key,
        IGpuBuffer value,
        IGpuBuffer probabilities,
        IGpuBuffer gradQuery,
        IGpuBuffer gradKey,
        IGpuBuffer gradValue,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        int headDimension,
        float scale)
    {
        if (!IsDirectPtxAttentionBackwardEnabled) return false;
        if (headDimension != PtxFusedAttentionBackwardD64Kernel.HeadDimension)
        {
            DirectPtxLastError = "attention-backward-head-dimension-not-implemented";
            return false;
        }
        if (batch is <= 0 or > 16)
        {
            DirectPtxLastError = "attention-backward-batch-not-implemented";
            return false;
        }
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0 ||
            checked((queryHeads / keyValueHeads) * querySequence) > 2048)
        {
            DirectPtxLastError = "attention-backward-head-map-not-implemented";
            return false;
        }
        if (querySequence is not (16 or 32 or 64 or 128) ||
            keyValueSequence is not (16 or 32 or 64 or 128))
        {
            DirectPtxLastError = "attention-backward-sequence-bucket-not-implemented";
            return false;
        }
        if (!float.IsFinite(scale))
        {
            DirectPtxLastError = "attention-backward-scale-not-finite";
            return false;
        }

        long queryElements = checked(
            (long)batch * queryHeads * querySequence * headDimension);
        long keyValueElements = checked(
            (long)batch * keyValueHeads * keyValueSequence * headDimension);
        long probabilityElements = checked(
            (long)batch * queryHeads * querySequence * keyValueSequence);
        if (gradOutput.SizeInBytes != queryElements * sizeof(float) ||
            query.SizeInBytes != queryElements * sizeof(float) ||
            gradQuery.SizeInBytes != queryElements * sizeof(float) ||
            key.SizeInBytes != keyValueElements * sizeof(float) ||
            value.SizeInBytes != keyValueElements * sizeof(float) ||
            gradKey.SizeInBytes != keyValueElements * sizeof(float) ||
            gradValue.SizeInBytes != keyValueElements * sizeof(float) ||
            probabilities.SizeInBytes != probabilityElements * sizeof(float))
        {
            DirectPtxLastError = "attention-backward-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var keyShape = new DirectPtxAttentionBackwardKey(
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                BitConverter.SingleToInt32Bits(scale));
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxAttentionBackwardKernels.TryGetValue(keyShape, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX attention backward must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedAttentionBackwardD64Kernel kernel =
                    GetOrCreateAttentionBackwardKernel(keyShape, scale);
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(query, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(key, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(value, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(probabilities, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.Create(gradQuery, kernel.Blueprint.Tensors[5]),
                        DirectPtxTensorView.Create(gradKey, kernel.Blueprint.Tensors[6]),
                        DirectPtxTensorView.Create(gradValue, kernel.Blueprint.Tensors[7]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxAttentionBackwardDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedAttentionBackwardD64Kernel GetOrCreateAttentionBackwardKernel(
        DirectPtxAttentionBackwardKey key,
        float scale)
    {
        if (_directPtxAttentionBackwardKernels.TryGetValue(
            key, out PtxFusedAttentionBackwardD64Kernel? existing))
            return existing;
        return CreateAndCacheAttentionBackwardKernelSlow(key, scale);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedAttentionBackwardD64Kernel CreateAndCacheAttentionBackwardKernelSlow(
        DirectPtxAttentionBackwardKey key,
        float scale) =>
        _directPtxAttentionBackwardKernels.GetOrAdd(key, () =>
            new PtxFusedAttentionBackwardD64Kernel(
                _directPtxRuntime!, key.Batch, key.QueryHeads, key.KeyValueHeads,
                key.QuerySequence, key.KeyValueSequence, scale));

    internal bool PrewarmDirectPtxAttentionBackwardD64(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale)
    {
        if (!IsDirectPtxAttentionBackwardEnabled) return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX attention-backward prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxAttentionBackwardKey(
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    BitConverter.SingleToInt32Bits(scale));
                _ = GetOrCreateAttentionBackwardKernel(key, scale);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxAttentionBackwardAudits(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        out DirectPtxKernelAudit rowDeltaAudit,
        out DirectPtxKernelAudit gradQueryAudit,
        out DirectPtxKernelAudit gradKeyValueAudit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxAttentionBackwardKey(
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                BitConverter.SingleToInt32Bits(scale));
            if (_directPtxAttentionBackwardKernels.TryGetValue(key, out var kernel))
            {
                rowDeltaAudit = kernel.RowDeltaAudit;
                gradQueryAudit = kernel.GradQueryAudit;
                gradKeyValueAudit = kernel.GradKeyValueAudit;
                return true;
            }
        }
        rowDeltaAudit = null!;
        gradQueryAudit = null!;
        gradKeyValueAudit = null!;
        return false;
    }

    /// <summary>
    /// Loads a persisted/default specialization outside capture. Stable caller-
    /// owned buffers may subsequently launch it while a CUDA graph is recording.
    /// Autotuning itself requires representative buffers and occurs on first use.
    /// </summary>
    internal bool PrewarmDirectPtxAttention(
        int batchHeads,
        int sequenceLength,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale,
        float epsilon = 1e-5f,
        bool emitSoftmaxStats = true)
    {
        return PrewarmDirectPtxAttentionFamily(
            batch: 1, queryHeads: batchHeads, keyValueHeads: batchHeads,
            querySequence: sequenceLength, keyValueSequence: sequenceLength,
            isCausal, fuseLayerNormGelu, scale, epsilon, emitSoftmaxStats,
            causalQueryOffset: 0);
    }

    internal bool PrewarmDirectPtxAttentionFamily(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale,
        float epsilon = 1e-5f,
        bool emitSoftmaxStats = true,
        int causalQueryOffset = 0)
    {
        if (!IsDirectPtxAttentionEnabled || IsStreamCapturing()) return false;
        if (causalQueryOffset < -querySequence)
        {
            DirectPtxLastError = "causal-query-offset-outside-query-domain";
            return false;
        }
        // Mirror the dispatch reject in TryDirectPtxOnlineAttentionCore: a non-zero causalQueryOffset
        // under a non-causal mask is never requested by dispatch (the mask derivation below collapses to
        // None when !isCausal), so without this a prewarmed plan keyed on (isCausal:false, offset!=0)
        // would be dead weight the dispatch path can never hit.
        if (!isCausal && causalQueryOffset != 0)
        {
            DirectPtxLastError = "causal-query-offset-without-causal-mask";
            return false;
        }
        DirectPtxEligibilityResult eligibility = DirectPtxAttentionEligibility.Evaluate(
            new DirectPtxAttentionRequest(
                DirectPtxArchitecture.Classify(_ccMajor, _ccMinor),
                _ccMajor, _ccMinor,
                DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Bhsd,
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                PtxOnlineFusedAttention128x64Kernel.HeadDimension,
                isCausal
                    ? causalQueryOffset == 0
                        ? DirectPtxAttentionMaskKind.CausalTopLeft
                        : DirectPtxAttentionMaskKind.CausalBottomRight
                    : DirectPtxAttentionMaskKind.None,
                DirectPtxAttentionPhase.Inference, 0, false, false));
        if (!eligibility.IsEligible)
        {
            DirectPtxLastError = eligibility.Reason;
            return false;
        }
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var plan = new DirectPtxAttentionPlanKey(
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    isCausal, causalQueryOffset, fuseLayerNormGelu, emitSoftmaxStats,
                    BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(epsilon));
                if (!_directPtxAttentionPlans.TryGetValue(plan, out int warps))
                {
                    if (!DirectPtxAttentionAutotuner.TryLoad(
                        _directPtxRuntime, batch, queryHeads, keyValueHeads,
                        querySequence, keyValueSequence, isCausal, causalQueryOffset,
                        fuseLayerNormGelu, emitSoftmaxStats, scale, epsilon, out warps))
                        warps = DirectPtxAttentionAutotuner.Candidates(querySequence)[0];
                    _directPtxAttentionPlans.Set(plan, warps);
                }
                _ = GetOrCreateAttentionKernel(plan, warps, scale, epsilon);
                DirectPtxLastError = null;
                return true;
            }
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Fuses a transformer residual addition and D=64 RMSNorm. Inputs, gamma,
    /// output, and saved RMS are FP32 canonical row-major allocations.
    /// </summary>
    internal bool TryDirectPtxFusedResidualRmsNorm(
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer gamma,
        IGpuBuffer output,
        IGpuBuffer savedRms,
        int rows,
        float epsilon = 1e-5f)
    {
        if (!IsDirectPtxResidualRmsNormEnabled || rows <= 0) return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                var key = new DirectPtxResidualRmsNormKey(
                    rows, BitConverter.SingleToInt32Bits(epsilon));
                if (capturing && !_directPtxResidualRmsNormKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = "Direct PTX residual RMSNorm must be prewarmed before CUDA graph capture.";
                    return false;
                }

                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedResidualRmsNormD64Kernel kernel;
                if (!_directPtxResidualRmsNormKernels.TryGetValue(key, out kernel))
                    kernel = CreateAndCacheResidualRmsNormKernel(key, rows, epsilon);
                lock (GpuDispatchLock)
                {
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(residual, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gamma, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(savedRms, kernel.Blueprint.Tensors[4]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxResidualRmsNormDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxFusedResidualRmsNorm(int rows, float epsilon = 1e-5f)
    {
        if (!IsDirectPtxResidualRmsNormEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxResidualRmsNormKey(rows, BitConverter.SingleToInt32Bits(epsilon));
                if (!_directPtxResidualRmsNormKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheResidualRmsNormKernel(key, rows, epsilon);
                return true;
            }
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedResidualRmsNormD64Kernel CreateAndCacheResidualRmsNormKernel(
        DirectPtxResidualRmsNormKey key,
        int rows,
        float epsilon)
    {
        var created = new PtxFusedResidualRmsNormD64Kernel(_directPtxRuntime!, rows, epsilon);
        return _directPtxResidualRmsNormKernels.AddOrGetExisting(key, created);
    }

    internal bool TryGetDirectPtxResidualRmsNormAudit(
        int rows,
        float epsilon,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxResidualRmsNormKey(rows, BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxResidualRmsNormKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxAttentionAudit(
        int batchHeads,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        int sequenceLength,
        out DirectPtxFunctionInfo info,
        out string jitInfoLog)
    {
        lock (_directPtxLock)
        {
            var plan = new DirectPtxAttentionPlanKey(
                1, batchHeads, batchHeads, sequenceLength, sequenceLength,
                isCausal, 0, fuseLayerNormGelu,
                true,
                BitConverter.SingleToInt32Bits(scale),
                BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxAttentionPlans.TryGetValue(plan, out int warps) &&
                _directPtxAttentionKernels.TryGetValue(
                    DirectPtxAttentionKey.FromPlan(plan, warps), out var kernel))
            {
                info = kernel.FunctionInfo;
                jitInfoLog = kernel.JitInfoLog;
                return true;
            }
        }
        info = default;
        jitInfoLog = string.Empty;
        return false;
    }

    internal bool TryGetDirectPtxAttentionKernelAudit(
        int batchHeads,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        int sequenceLength,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var plan = new DirectPtxAttentionPlanKey(
                1, batchHeads, batchHeads, sequenceLength, sequenceLength,
                isCausal, 0, fuseLayerNormGelu, true,
                BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxAttentionPlans.TryGetValue(plan, out int warps) &&
                _directPtxAttentionKernels.TryGetValue(
                    DirectPtxAttentionKey.FromPlan(plan, warps), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxAttentionKernelAuditFamily(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        bool emitSoftmaxStats,
        int causalQueryOffset,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var plan = new DirectPtxAttentionPlanKey(
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                isCausal, causalQueryOffset, fuseLayerNormGelu, emitSoftmaxStats,
                BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxAttentionPlans.TryGetValue(plan, out int warps) &&
                _directPtxAttentionKernels.TryGetValue(
                    DirectPtxAttentionKey.FromPlan(plan, warps), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxMeshPool(
        DirectPtxMeshPoolOperation operation,
        IGpuBuffer p0,
        IGpuBuffer? p1,
        IGpuBuffer? p2,
        IGpuBuffer? p3,
        int numEdges,
        int numKept,
        int channels,
        float temperature = 1f)
    {
        if (!IsDirectPtxSparseGraphEnabled ||
            !PtxMeshPoolF32Kernel.SupportsShape(operation, numEdges, numKept, channels, temperature))
            return false;
        int count = PtxMeshPoolF32Kernel.GetTensorCount(operation);
        if (p0 is null || (count > 1 && p1 is null) || (count > 2 && p2 is null) ||
            (count > 3 && p3 is null) ||
            !HasExactBytes(p0, PtxMeshPoolF32Kernel.GetRequiredBytes(operation, 0)) ||
            (count > 1 && !HasExactBytes(p1!, PtxMeshPoolF32Kernel.GetRequiredBytes(operation, 1))) ||
            (count > 2 && !HasExactBytes(p2!, PtxMeshPoolF32Kernel.GetRequiredBytes(operation, 2))) ||
            (count > 3 && !HasExactBytes(p3!, PtxMeshPoolF32Kernel.GetRequiredBytes(operation, 3))))
            return false;
        nuint pointerUnion = (nuint)p0.Handle |
            (count > 1 ? (nuint)p1!.Handle : 0u) |
            (count > 2 ? (nuint)p2!.Handle : 0u) |
            (count > 3 ? (nuint)p3!.Handle : 0u);
        if (p0.Handle == IntPtr.Zero || (count > 1 && p1!.Handle == IntPtr.Zero) ||
            (count > 2 && p2!.Handle == IntPtr.Zero) || (count > 3 && p3!.Handle == IntPtr.Zero) ||
            (pointerUnion & 15u) != 0 ||
            (count > 1 && DirectPtxBuffersOverlap(p0, p1!)) ||
            (count > 2 && (DirectPtxBuffersOverlap(p0, p2!) || DirectPtxBuffersOverlap(p1!, p2!))) ||
            (count > 3 && (DirectPtxBuffersOverlap(p0, p3!) || DirectPtxBuffersOverlap(p1!, p3!) ||
                           DirectPtxBuffersOverlap(p2!, p3!))))
            return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxMeshPoolKernels.TryGetValue(operation, out _)) return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxMeshPoolF32Kernel kernel = _directPtxMeshPoolKernels.GetOrAdd(
                    operation, () => new PtxMeshPoolF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxMeshPoolKernels.Pin(operation)) return false;
                Span<DirectPtxTensorView> views = stackalloc DirectPtxTensorView[4];
                views[0] = DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]);
                if (count > 1) views[1] = DirectPtxTensorView.Create(p1!, kernel.Blueprint.Tensors[1]);
                if (count > 2) views[2] = DirectPtxTensorView.Create(p2!, kernel.Blueprint.Tensors[2]);
                if (count > 3) views[3] = DirectPtxTensorView.Create(p3!, kernel.Blueprint.Tensors[3]);
                lock (GpuDispatchLock) kernel.Launch(views[..count]);
            }
            System.Threading.Interlocked.Increment(ref _directPtxMeshPoolDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxMeshPool(DirectPtxMeshPoolOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxMeshPoolKernels.GetOrAdd(
                    operation, () => new PtxMeshPoolF32Kernel(_directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxSparseEngine(
        DirectPtxSparseEngineOperation operation,
        IGpuBuffer p0,
        IGpuBuffer? p1 = null,
        IGpuBuffer? p2 = null,
        IGpuBuffer? p3 = null,
        IGpuBuffer? p4 = null,
        IGpuBuffer? p5 = null,
        IGpuBuffer? p6 = null)
    {
        if (!IsDirectPtxSparseGraphEnabled || p0 is null) return false;
        DirectPtxKernelBlueprint admission = PtxSparseEngineF32Kernel.GetAdmissionBlueprint(operation);
        int count = admission.Tensors.Count;
        bool Valid(IGpuBuffer? buffer, int index) => buffer is not null &&
            buffer.Handle != IntPtr.Zero && ((nuint)buffer.Handle & 15u) == 0 &&
            HasExactBytes(buffer, checked((long)admission.Tensors[index].RequiredBytes));
        if (!Valid(p0, 0) || (count > 1 && !Valid(p1, 1)) || (count > 2 && !Valid(p2, 2)) ||
            (count > 3 && !Valid(p3, 3)) || (count > 4 && !Valid(p4, 4)) ||
            (count > 5 && !Valid(p5, 5)) || (count > 6 && !Valid(p6, 6))) return false;
        bool Pair(IGpuBuffer left, IGpuBuffer? right) => right is not null && DirectPtxBuffersOverlap(left, right);
        if (Pair(p0, p1) || Pair(p0, p2) || Pair(p0, p3) || Pair(p0, p4) || Pair(p0, p5) || Pair(p0, p6) ||
            (p1 is not null && (Pair(p1, p2) || Pair(p1, p3) || Pair(p1, p4) || Pair(p1, p5) || Pair(p1, p6))) ||
            (p2 is not null && (Pair(p2, p3) || Pair(p2, p4) || Pair(p2, p5) || Pair(p2, p6))) ||
            (p3 is not null && (Pair(p3, p4) || Pair(p3, p5) || Pair(p3, p6))) ||
            (p4 is not null && (Pair(p4, p5) || Pair(p4, p6))) || (p5 is not null && Pair(p5, p6))) return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSparseEngineKernels.TryGetValue(operation, out _)) return false;
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxSparseEngineF32Kernel kernel = _directPtxSparseEngineKernels.GetOrAdd(
                    operation, () => new PtxSparseEngineF32Kernel(_directPtxRuntime!, operation));
                if (capturing && !_directPtxSparseEngineKernels.Pin(operation)) return false;
                Span<DirectPtxTensorView> views = stackalloc DirectPtxTensorView[7];
                views[0] = DirectPtxTensorView.Create(p0, kernel.Blueprint.Tensors[0]);
                if (count > 1) views[1] = DirectPtxTensorView.Create(p1!, kernel.Blueprint.Tensors[1]);
                if (count > 2) views[2] = DirectPtxTensorView.Create(p2!, kernel.Blueprint.Tensors[2]);
                if (count > 3) views[3] = DirectPtxTensorView.Create(p3!, kernel.Blueprint.Tensors[3]);
                if (count > 4) views[4] = DirectPtxTensorView.Create(p4!, kernel.Blueprint.Tensors[4]);
                if (count > 5) views[5] = DirectPtxTensorView.Create(p5!, kernel.Blueprint.Tensors[5]);
                if (count > 6) views[6] = DirectPtxTensorView.Create(p6!, kernel.Blueprint.Tensors[6]);
                lock (GpuDispatchLock) kernel.Launch(views[..count]);
            }
            System.Threading.Interlocked.Increment(ref _directPtxSparseEngineDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxSparseEngine(DirectPtxSparseEngineOperation operation)
    {
        if (!IsDirectPtxSparseGraphEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = _directPtxSparseEngineKernels.GetOrAdd(
                    operation, () => new PtxSparseEngineF32Kernel(_directPtxRuntime!, operation));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private void DisposeDirectPtxRuntime()
    {
        lock (_directPtxLock)
        {
            _directPtxAttentionKernels.Dispose();
            _directPtxAttentionPlans.Clear();
            _directPtxResidualRmsNormKernels.Dispose();
            _directPtxDecodeKernels.Dispose();
            _directPtxPagedPrefillKernels.Dispose();
            _directPtxAttentionBackwardKernels.Dispose();
            _directPtxFlashAttentionBackwardKernels.Dispose();
            _directPtxQkvRopeCacheKernels.Dispose();
            _directPtxCsrSpmmKernels.Dispose();
            _directPtxSddmmKernels.Dispose();
            _directPtxCsrSpmmBiasKernels.Dispose();
            _directPtxCsrSpmmBiasReluKernels.Dispose();
            _directPtxCsrSpmmF64Kernels.Dispose();
            _directPtxGraphGatherKernels.Dispose();
            _directPtxGraphScatterKernels.Dispose();
            _directPtxGraphScatterAtomicKernels.Dispose();
            _directPtxSegmentReduceKernels.Dispose();
            _directPtxCsrSegmentKernels.Dispose();
            _directPtxCsrBackwardKernels.Dispose();
            _directPtxSparseUtilityKernels.Dispose();
            _directPtxStructuredSparse2x4Kernels.Dispose();
            _directPtxStructuredSparse2x4MmaSpKernels.Dispose();
            _directPtxScalarScatterAddKernels.Dispose();
            _directPtxScatterRowsKernels.Dispose();
            _directPtxScatterMaxRowsKernels.Dispose();
            _directPtxNeuralScatterMaxKernels.Dispose();
            _directPtxScatterBackwardRowsKernels.Dispose();
            _directPtxCapsuleRoutingKernels.Dispose();
            _directPtxCapsuleProjectionKernels.Dispose();
            _directPtxCapsuleSquashKernels.Dispose();
            _directPtxResidentScatterAuxKernels.Dispose();
            _directPtxResidentScatterSoftmaxKernels.Dispose();
            _directPtxUniformMeshLaplacianKernels.Dispose();
            _directPtxSparseOptimizerKernels.Dispose();
            _directPtxFusedSparseLinearKernels.Dispose();
            _directPtxTensorGatherKernels.Dispose();
            _directPtxTensorScatterReduceKernels.Dispose();
            _directPtxTensorScatterHighLevelKernels.Dispose();
            _directPtxMeshPoolKernels.Dispose();
            _directPtxSparseEngineKernels.Dispose();
            _directPtxRuntime?.Dispose();
            _directPtxRuntime = null;
        }
    }

    private readonly record struct DirectPtxAttentionPlanKey(
        int Batch,
        int QueryHeads,
        int KeyValueHeads,
        int QuerySequence,
        int KeyValueSequence,
        bool IsCausal,
        int CausalQueryOffset,
        bool FuseLayerNormGelu,
        bool EmitSoftmaxStats,
        int ScaleBits,
        int EpsilonBits)
    {
        internal int BatchHeads => checked(Batch * QueryHeads);
    }

    private readonly record struct DirectPtxAttentionKey(
        DirectPtxAttentionPlanKey Plan,
        int WarpsPerBlock)
    {
        internal static DirectPtxAttentionKey FromPlan(
            DirectPtxAttentionPlanKey plan,
            int warpsPerBlock) => new(plan, warpsPerBlock);
    }

    private readonly record struct DirectPtxResidualRmsNormKey(int Rows, int EpsilonBits);
    private readonly record struct DirectPtxDecodeKey(
        bool IsPaged,
        int QueryHeads,
        int KeyValueHeads,
        int SequenceLength,
        int BlockSize,
        int PoolBlocks,
        int ScaleBits);
    private readonly record struct DirectPtxPagedPrefillKey(
        int QueryHeads,
        int KeyValueHeads,
        int QueryCount,
        int StartPosition,
        int BlockSize,
        int PoolBlocks,
        int ScaleBits);
    private readonly record struct DirectPtxAttentionBackwardKey(
        int Batch,
        int QueryHeads,
        int KeyValueHeads,
        int QuerySequence,
        int KeyValueSequence,
        int ScaleBits);
    private readonly record struct DirectPtxFlashAttentionBackwardKey(
        int Batch,
        int Heads,
        int QuerySequence,
        int KeyValueSequence,
        bool IsCausal,
        int ScaleBits,
        int BiasBatchStride);
    private readonly record struct DirectPtxQkvRopeCacheKey(
        int Heads,
        int CacheCapacity,
        int Position);
    private readonly record struct DirectPtxCsrSpmmKey(
        int Rows,
        int Inner,
        int Columns,
        int NonZeros);
    private readonly record struct DirectPtxSddmmKey(int NonZeros, int Inner);
    private readonly record struct DirectPtxGraphGatherKey(bool GatherSource);
    private readonly record struct DirectPtxGraphScatterKey(bool Weighted);
    private readonly record struct DirectPtxCsrSegmentKey(
        DirectPtxCsrSegmentReduction Reduction,
        int EpsilonBits);
    private readonly record struct DirectPtxSparseUtilityKey(
        DirectPtxSparseUtility Utility,
        int EpsilonBits);
    private readonly record struct DirectPtxStructuredSparse2x4Key(
        DirectPtxStructuredSparse2x4Operation Operation,
        int AlphaBits,
        int BetaBits);

}
#endif
