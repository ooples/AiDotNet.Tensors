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
    private readonly DirectPtxKernelCache<DirectPtxCastFp16Key, PtxFusedCastF32ToF16Kernel>
        _directPtxCastFp16Kernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private readonly DirectPtxKernelCache<DirectPtxCastFp32Key, PtxFusedCastF16ToF32Kernel>
        _directPtxCastFp32Kernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
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
    private long _directPtxCastFp16DispatchCount;
    private long _directPtxCastFp32DispatchCount;
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

    internal bool IsDirectPtxCastFp16Enabled =>
        DirectPtxFeatureGate.IsCastFp16Enabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedCastFp16(_ccMajor, _ccMinor);

    internal long DirectPtxCastFp16DispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCastFp16DispatchCount);
    internal int DirectPtxCastFp16PinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxCastFp16Kernels.PinnedCount; }
    }

    internal bool IsDirectPtxCastFp32Enabled =>
        DirectPtxFeatureGate.IsCastFp32Enabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedCastFp32(_ccMajor, _ccMinor);

    internal long DirectPtxCastFp32DispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxCastFp32DispatchCount);
    internal int DirectPtxCastFp32PinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxCastFp32Kernels.PinnedCount; }
    }

    internal int DirectPtxQkvRopeCacheKernelCapacity => _directPtxQkvRopeCacheKernels.Capacity;
    internal int DirectPtxQkvRopeCachePinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxQkvRopeCacheKernels.PinnedCount; }
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
        if (((PtxCompat.ToNuint(input.Handle) | PtxCompat.ToNuint(packedWeights.Handle) | PtxCompat.ToNuint(bias.Handle) |
              PtxCompat.ToNuint(cosine.Handle) | PtxCompat.ToNuint(sine.Handle) | PtxCompat.ToNuint(query.Handle) |
              PtxCompat.ToNuint(keyCache.Handle) | PtxCompat.ToNuint(valueCache.Handle)) & 15u) != 0)
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
            nuint leftStart = PtxCompat.ToNuint(left.Handle);
            nuint rightStart = PtxCompat.ToNuint(right.Handle);
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
                    PtxCompat.SingleToInt32Bits(scale),
                    PtxCompat.SingleToInt32Bits(epsilon));

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
        if (!PtxCompat.IsFinite(scale))
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
                blockSize, poolBlocks, PtxCompat.SingleToInt32Bits(scale));
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
                    blockSize, poolBlocks, PtxCompat.SingleToInt32Bits(scale));
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
                blockSize, poolBlocks, PtxCompat.SingleToInt32Bits(scale));
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
        if (!PtxCompat.IsFinite(scale))
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
                blockSize, poolBlocks, PtxCompat.SingleToInt32Bits(scale));
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
                    blockSize, poolBlocks, PtxCompat.SingleToInt32Bits(scale));
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
                blockSize, poolBlocks, PtxCompat.SingleToInt32Bits(scale));
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
        if (!PtxCompat.IsFinite(scale))
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
                PtxCompat.SingleToInt32Bits(scale), bakedBiasBatchStride);
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
                    PtxCompat.SingleToInt32Bits(scale), biasBatchStride);
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
                PtxCompat.SingleToInt32Bits(scale), biasBatchStride);
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
        if (!PtxCompat.IsFinite(scale))
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
                PtxCompat.SingleToInt32Bits(scale));
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
                    PtxCompat.SingleToInt32Bits(scale));
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
                PtxCompat.SingleToInt32Bits(scale));
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
                    PtxCompat.SingleToInt32Bits(scale), PtxCompat.SingleToInt32Bits(epsilon));
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
                    rows, PtxCompat.SingleToInt32Bits(epsilon));
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
                var key = new DirectPtxResidualRmsNormKey(rows, PtxCompat.SingleToInt32Bits(epsilon));
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
            var key = new DirectPtxResidualRmsNormKey(rows, PtxCompat.SingleToInt32Bits(epsilon));
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
                PtxCompat.SingleToInt32Bits(scale),
                PtxCompat.SingleToInt32Bits(epsilon));
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
                PtxCompat.SingleToInt32Bits(scale), PtxCompat.SingleToInt32Bits(epsilon));
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
                PtxCompat.SingleToInt32Bits(scale), PtxCompat.SingleToInt32Bits(epsilon));
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

    /// <summary>
    /// Attempts exact contiguous FP32-to-FP16 conversion of a [size] vector.
    /// Shape validation happens before dispatch; the PTX ABI receives only
    /// input/output pointers.
    /// </summary>
    internal bool TryDirectPtxCastFp16(
        IGpuBuffer input,
        IGpuBuffer output,
        int size)
    {
        if (input is null || output is null)
        {
            DirectPtxLastError = "cast-fp16-null-buffer";
            return false;
        }
        if (!DirectPtxFeatureGate.IsCastFp16Enabled)
        {
            DirectPtxLastError = "cast-fp16-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "cast-fp16-cuda-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedCastFp16(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "cast-fp16-architecture-not-validated";
            return false;
        }
        if (!PtxFusedCastF32ToF16Kernel.IsSupportedShape(size))
        {
            DirectPtxLastError = "cast-fp16-shape-not-implemented";
            return false;
        }
        if (!PtxFusedCastF32ToF16Kernel.IsPromotedShape(size) &&
            !DirectPtxFeatureGate.CastFp16ExperimentOverride)
        {
            DirectPtxLastError = "cast-fp16-performance-gate-not-met";
            return false;
        }

        if (input.SizeInBytes != checked((long)size * sizeof(float)) ||
            output.SizeInBytes != checked((long)size * 2))
        {
            DirectPtxLastError = "cast-fp16-physical-extent-mismatch";
            return false;
        }
        if (input.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "cast-fp16-invalid-device-pointer";
            return false;
        }
        if (((PtxCompat.ToNuint(input.Handle) | PtxCompat.ToNuint(output.Handle)) & 15u) != 0)
        {
            DirectPtxLastError = "cast-fp16-alignment-mismatch";
            return false;
        }
        if (DirectPtxCastBuffersOverlap(input, output))
        {
            DirectPtxLastError = "cast-fp16-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxCastFp16Key(size);
            lock (_directPtxLock)
            {
                if (!_directPtxCastFp16Kernels.TryGetValue(
                    key, out PtxFusedCastF32ToF16Kernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX cast must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheCastFp16KernelSlow(key);
                }
                if (capturing && !_directPtxCastFp16Kernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX cast module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCastFp16DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static bool DirectPtxCastBuffersOverlap(IGpuBuffer input, IGpuBuffer output)
    {
        nuint inputStart = PtxCompat.ToNuint(input.Handle);
        nuint outputStart = PtxCompat.ToNuint(output.Handle);
        nuint inputEnd = checked(inputStart + (nuint)input.SizeInBytes);
        nuint outputEnd = checked(outputStart + (nuint)output.SizeInBytes);
        return inputStart < outputEnd && outputStart < inputEnd;
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedCastF32ToF16Kernel CreateAndCacheCastFp16KernelSlow(
        DirectPtxCastFp16Key key) =>
        _directPtxCastFp16Kernels.GetOrAdd(key, () =>
            new PtxFusedCastF32ToF16Kernel(_directPtxRuntime!, key.Size));

    internal bool PrewarmDirectPtxCastFp16(int size)
    {
        if (!DirectPtxFeatureGate.IsCastFp16Enabled)
        {
            DirectPtxLastError = "cast-fp16-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "cast-fp16-cuda-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedCastFp16(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "cast-fp16-architecture-not-validated";
            return false;
        }
        if (!PtxFusedCastF32ToF16Kernel.IsSupportedShape(size))
        {
            DirectPtxLastError = "cast-fp16-shape-not-implemented";
            return false;
        }
        if (!PtxFusedCastF32ToF16Kernel.IsPromotedShape(size) &&
            !DirectPtxFeatureGate.CastFp16ExperimentOverride)
        {
            DirectPtxLastError = "cast-fp16-performance-gate-not-met";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX cast prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxCastFp16Key(size);
                if (!_directPtxCastFp16Kernels.TryGetValue(key, out _))
                    _ = CreateAndCacheCastFp16KernelSlow(key);
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
    /// Widening FP16-to-FP32 cast (issue #845). Mirrors the narrowing FP32-to-FP16
    /// route: same closed shape family, same alignment and aliasing rules, same
    /// fail-closed reason strings, with the input/output byte extents swapped.
    /// </summary>
    internal bool TryDirectPtxCastFp32(
        IGpuBuffer input,
        IGpuBuffer output,
        int size)
    {
        if (input is null || output is null)
        {
            DirectPtxLastError = "cast-fp32-null-buffer";
            return false;
        }
        if (!DirectPtxFeatureGate.IsCastFp32Enabled)
        {
            DirectPtxLastError = "cast-fp32-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "cast-fp32-cuda-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedCastFp32(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "cast-fp32-architecture-not-validated";
            return false;
        }
        if (!PtxFusedCastF16ToF32Kernel.IsSupportedShape(size))
        {
            DirectPtxLastError = "cast-fp32-shape-not-implemented";
            return false;
        }
        if (!PtxFusedCastF16ToF32Kernel.IsPromotedShape(size) &&
            !DirectPtxFeatureGate.CastFp32ExperimentOverride)
        {
            DirectPtxLastError = "cast-fp32-performance-gate-not-met";
            return false;
        }

        // Widening: 2 bytes in per element, 4 bytes out.
        if (input.SizeInBytes != checked((long)size * 2) ||
            output.SizeInBytes != checked((long)size * sizeof(float)))
        {
            DirectPtxLastError = "cast-fp32-physical-extent-mismatch";
            return false;
        }
        if (input.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "cast-fp32-invalid-device-pointer";
            return false;
        }
        if (((PtxCompat.ToNuint(input.Handle) | PtxCompat.ToNuint(output.Handle)) & 15u) != 0)
        {
            DirectPtxLastError = "cast-fp32-alignment-mismatch";
            return false;
        }
        if (DirectPtxCastBuffersOverlap(input, output))
        {
            DirectPtxLastError = "cast-fp32-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxCastFp32Key(size);
            lock (_directPtxLock)
            {
                if (!_directPtxCastFp32Kernels.TryGetValue(
                    key, out PtxFusedCastF16ToF32Kernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX widening cast must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheCastFp32KernelSlow(key);
                }
                if (capturing && !_directPtxCastFp32Kernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX widening cast module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxCastFp32DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedCastF16ToF32Kernel CreateAndCacheCastFp32KernelSlow(
        DirectPtxCastFp32Key key)
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException(
                "The direct-PTX runtime must be initialized before creating a widening cast kernel.");
        return _directPtxCastFp32Kernels.GetOrAdd(key, () =>
            new PtxFusedCastF16ToF32Kernel(runtime, key.Size));
    }

    internal bool PrewarmDirectPtxCastFp32(int size)
    {
        if (!DirectPtxFeatureGate.IsCastFp32Enabled)
        {
            DirectPtxLastError = "cast-fp32-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "cast-fp32-cuda-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedCastFp32(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "cast-fp32-architecture-not-validated";
            return false;
        }
        if (!PtxFusedCastF16ToF32Kernel.IsSupportedShape(size))
        {
            DirectPtxLastError = "cast-fp32-shape-not-implemented";
            return false;
        }
        if (!PtxFusedCastF16ToF32Kernel.IsPromotedShape(size) &&
            !DirectPtxFeatureGate.CastFp32ExperimentOverride)
        {
            DirectPtxLastError = "cast-fp32-performance-gate-not-met";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX widening cast prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxCastFp32Key(size);
                if (!_directPtxCastFp32Kernels.TryGetValue(key, out _))
                    _ = CreateAndCacheCastFp32KernelSlow(key);
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

    internal bool TryGetDirectPtxCastFp32Audit(int size, out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxCastFp32Key(size);
            if (_directPtxCastFp32Kernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    internal bool TryGetDirectPtxCastFp16Audit(int size, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxCastFp16Key(size);
            if (_directPtxCastFp16Kernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
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
            _directPtxCastFp16Kernels.Dispose();
            _directPtxCastFp32Kernels.Dispose();
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
    private readonly record struct DirectPtxCastFp16Key(int Size);

    private readonly record struct DirectPtxCastFp32Key(int Size);

}
