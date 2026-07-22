using System;
using System.Collections.Generic;
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
    private readonly DirectPtxKernelCache<DirectPtxResidualRmsNormKey, PtxFusedResidualBiasLayerNormGeluD64Kernel>
        _directPtxResidualLayerNormGeluKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxRowNormalizationKey, PtxRowNormalizationD64Kernel>
        _directPtxRowNormalizationKernels = new(DirectPtxFeatureGate.CacheCapacity);
    private readonly DirectPtxKernelCache<DirectPtxChannelNormalizationKey, PtxChannelNormalizationD64Kernel>
        _directPtxChannelNormalizationKernels = new(DirectPtxFeatureGate.CacheCapacity);
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
    private readonly DirectPtxKernelCache<DirectPtxFusedLinearKey, PtxFusedLinearGeluM1Kernel>
        _directPtxFusedLinearKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxFusedLinearKey, PtxFusedLinearGeluFp16M1Kernel>
        _directPtxMixedLinearKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxFusedLinearKey, PtxFusedLinearGeluFp16M16Kernel>
        _directPtxMixedLinearM16Kernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxFusedLinearKey, PtxFusedLinearGeluW8A8M1Kernel>
        _directPtxQuantizedLinearKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private DirectPtxRuntime? _directPtxRuntime;
    // A captured CUDA graph retains CUfunction handles. Track every cache pin
    // acquired while recording so destroying/updating the graph can release the
    // corresponding module references instead of exhausting the bounded LRUs.
    private List<Action>? _directPtxPendingGraphUnpins;
    private readonly Dictionary<IntPtr, List<Action>> _directPtxGraphUnpins = new();

    private void BeginDirectPtxGraphCapture()
    {
        lock (_directPtxLock)
        {
            if (_directPtxPendingGraphUnpins is not null)
                throw new InvalidOperationException("A direct-PTX graph capture is already active.");
            _directPtxPendingGraphUnpins = new List<Action>();
        }
    }

    private bool PinDirectPtxKernel<TKey, TKernel>(
        DirectPtxKernelCache<TKey, TKernel> cache,
        TKey key)
        where TKey : notnull
        where TKernel : class, IDisposable
    {
        lock (_directPtxLock)
        {
            if (_directPtxPendingGraphUnpins is null || !cache.Pin(key))
                return false;
            _directPtxPendingGraphUnpins.Add(() => cache.Unpin(key));
            return true;
        }
    }

    private void AbortDirectPtxGraphCapture()
    {
        lock (_directPtxLock)
        {
            if (_directPtxPendingGraphUnpins is null) return;
            ReleaseDirectPtxPinActions(_directPtxPendingGraphUnpins);
            _directPtxPendingGraphUnpins = null;
        }
    }

    private void CommitDirectPtxGraphCapture(IntPtr graphExec, bool replaceExisting)
    {
        lock (_directPtxLock)
        {
            List<Action> pins = _directPtxPendingGraphUnpins ?? new List<Action>();
            _directPtxPendingGraphUnpins = null;
            if (replaceExisting && _directPtxGraphUnpins.TryGetValue(graphExec, out List<Action>? oldPins))
                ReleaseDirectPtxPinActions(oldPins);
            if (pins.Count != 0)
                _directPtxGraphUnpins[graphExec] = pins;
            else if (replaceExisting)
                _directPtxGraphUnpins.Remove(graphExec);
        }
    }

    private void ReleaseDirectPtxGraphPins(IntPtr graphExec)
    {
        lock (_directPtxLock)
        {
            if (!_directPtxGraphUnpins.TryGetValue(graphExec, out List<Action>? pins)) return;
            _directPtxGraphUnpins.Remove(graphExec);
            ReleaseDirectPtxPinActions(pins);
        }
    }

    private static void ReleaseDirectPtxPinActions(List<Action> pins)
    {
        for (int i = pins.Count - 1; i >= 0; i--)
            pins[i]();
    }

    /// <summary>The last opt-in direct-PTX initialization/launch failure, if fallback was required.</summary>
    internal string? DirectPtxLastError { get; private set; }
    internal long DirectPtxAttentionDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxAttentionDispatchCount);
    private long _directPtxAttentionDispatchCount;
    private long _directPtxResidualRmsNormDispatchCount;
    private long _directPtxResidualLayerNormGeluDispatchCount;
    private long _directPtxRowNormalizationDispatchCount;
    private long _directPtxChannelNormalizationDispatchCount;
    private long _directPtxDecodeDispatchCount;
    private long _directPtxPagedPrefillDispatchCount;
    private long _directPtxAttentionBackwardDispatchCount;
    private long _directPtxFlashAttentionBackwardDispatchCount;
    private long _directPtxQkvRopeCacheDispatchCount;
    private long _directPtxFusedLinearDispatchCount;
    private long _directPtxMixedLinearDispatchCount;
    private long _directPtxQuantizedLinearDispatchCount;
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

    internal bool IsDirectPtxResidualLayerNormGeluEnabled =>
        DirectPtxFeatureGate.IsResidualLayerNormGeluEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(_ccMajor, _ccMinor);

    internal bool IsDirectPtxNormalizationEnabled =>
        DirectPtxFeatureGate.IsNormalizationEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(_ccMajor, _ccMinor);

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

    internal bool IsDirectPtxFusedLinearEnabled =>
        DirectPtxFeatureGate.IsFusedLinearEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedFusedLinear(_ccMajor, _ccMinor);

    internal bool IsDirectPtxMixedLinearEnabled =>
        DirectPtxFeatureGate.IsMixedPrecisionLinearEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedMixedLinear(_ccMajor, _ccMinor);

    internal bool IsDirectPtxQuantizedLinearEnabled =>
        DirectPtxFeatureGate.IsQuantizedLinearEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedQuantizedLinear(_ccMajor, _ccMinor);

    internal long DirectPtxResidualRmsNormDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxResidualRmsNormDispatchCount);
    internal long DirectPtxResidualLayerNormGeluDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxResidualLayerNormGeluDispatchCount);
    internal long DirectPtxRowNormalizationDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxRowNormalizationDispatchCount);
    internal long DirectPtxChannelNormalizationDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxChannelNormalizationDispatchCount);
    internal int DirectPtxRowNormalizationPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRowNormalizationKernels.PinnedCount; }
    }
    internal int DirectPtxChannelNormalizationPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxChannelNormalizationKernels.PinnedCount; }
    }
    internal int DirectPtxResidualLayerNormGeluPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxResidualLayerNormGeluKernels.PinnedCount; }
    }

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
    internal long DirectPtxFusedLinearDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxFusedLinearDispatchCount);
    internal int DirectPtxFusedLinearPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxFusedLinearKernels.PinnedCount; }
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
                if (capturing && !PinDirectPtxKernel(_directPtxQkvRopeCacheKernels, key))
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

    internal long DirectPtxMixedLinearDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxMixedLinearDispatchCount);

    internal long DirectPtxQuantizedLinearDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxQuantizedLinearDispatchCount);
    internal int DirectPtxMixedLinearPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxMixedLinearKernels.PinnedCount; }
    }
    internal int DirectPtxMixedLinearM16PinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxMixedLinearM16Kernels.PinnedCount; }
    }
    internal int DirectPtxQuantizedLinearPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxQuantizedLinearKernels.PinnedCount; }
    }

    /// <summary>
    /// Attempts the exact contiguous FP16-input/weight, FP32-accumulate M=1
    /// linear + FP32 bias + tanh-GELU specialization.
    /// </summary>
    internal bool TryDirectPtxFusedLinearGeluFp16M1(
        IGpuBuffer inputHalf,
        IGpuBuffer outputMajorWeightsHalf,
        IGpuBuffer biasFloat,
        IGpuBuffer outputFloat,
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxMixedLinearEnabled) return false;
        if (!PtxFusedLinearGeluFp16M1Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "mixed-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluFp16M1Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride)
        {
            DirectPtxLastError = "mixed-linear-performance-gate-not-met";
            return false;
        }
        long inputBytes = checked((long)inputFeatures * sizeof(ushort));
        long weightBytes = checked((long)inputFeatures * outputFeatures * sizeof(ushort));
        long outputBytes = checked((long)outputFeatures * sizeof(float));
        if (inputHalf.SizeInBytes != inputBytes ||
            outputMajorWeightsHalf.SizeInBytes != weightBytes ||
            biasFloat.SizeInBytes != outputBytes || outputFloat.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "mixed-linear-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            lock (_directPtxLock)
            {
                if (!_directPtxMixedLinearKernels.TryGetValue(
                    key, out PtxFusedLinearGeluFp16M1Kernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX mixed linear must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheMixedLinearKernelSlow(key);
                }
                if (capturing && !PinDirectPtxKernel(_directPtxMixedLinearKernels, key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX mixed-linear module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(inputHalf, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(outputMajorWeightsHalf, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(biasFloat, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(outputFloat, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxMixedLinearDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedLinearGeluFp16M1Kernel CreateAndCacheMixedLinearKernelSlow(
        DirectPtxFusedLinearKey key) =>
        _directPtxMixedLinearKernels.GetOrAdd(key, () =>
            new PtxFusedLinearGeluFp16M1Kernel(
                _directPtxRuntime!, key.InputFeatures, key.OutputFeatures));

    internal bool PrewarmDirectPtxFusedLinearGeluFp16M1(
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxMixedLinearEnabled) return false;
        if (!PtxFusedLinearGeluFp16M1Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "mixed-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluFp16M1Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride)
        {
            DirectPtxLastError = "mixed-linear-performance-gate-not-met";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX mixed linear prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
                if (!_directPtxMixedLinearKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheMixedLinearKernelSlow(key);
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

    internal bool TryGetDirectPtxMixedLinearAudit(
        int inputFeatures,
        int outputFeatures,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            if (_directPtxMixedLinearKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Attempts the exact contiguous symmetric W8A8 M=1 projection with S32
    /// DP4A accumulation and a fused FP32 dequant/bias/tanh-GELU epilogue.
    /// </summary>
    internal bool TryDirectPtxFusedLinearGeluW8A8M1(
        IGpuBuffer inputInt8,
        IGpuBuffer outputMajorWeightsInt8,
        IGpuBuffer activationScaleFloat,
        IGpuBuffer weightScalesFloat,
        IGpuBuffer biasFloat,
        IGpuBuffer outputFloat,
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxQuantizedLinearEnabled) return false;
        if (!PtxFusedLinearGeluW8A8M1Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "quantized-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluW8A8M1Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.QuantizedLinearExperimentOverride)
        {
            DirectPtxLastError = "quantized-linear-performance-gate-not-met";
            return false;
        }
        long outputBytes = checked((long)outputFeatures * sizeof(float));
        if (inputInt8.SizeInBytes != inputFeatures ||
            outputMajorWeightsInt8.SizeInBytes != checked((long)inputFeatures * outputFeatures) ||
            activationScaleFloat.SizeInBytes != sizeof(float) ||
            weightScalesFloat.SizeInBytes != outputBytes ||
            biasFloat.SizeInBytes != outputBytes || outputFloat.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "quantized-linear-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            lock (_directPtxLock)
            {
                if (!_directPtxQuantizedLinearKernels.TryGetValue(
                    key, out PtxFusedLinearGeluW8A8M1Kernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX W8A8 linear must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheQuantizedLinearKernelSlow(key);
                }
                if (capturing && !PinDirectPtxKernel(_directPtxQuantizedLinearKernels, key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX W8A8-linear module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(inputInt8, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(outputMajorWeightsInt8, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(activationScaleFloat, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(weightScalesFloat, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(biasFloat, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.Create(outputFloat, kernel.Blueprint.Tensors[5]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxQuantizedLinearDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedLinearGeluW8A8M1Kernel CreateAndCacheQuantizedLinearKernelSlow(
        DirectPtxFusedLinearKey key) =>
        _directPtxQuantizedLinearKernels.GetOrAdd(key, () =>
            new PtxFusedLinearGeluW8A8M1Kernel(
                _directPtxRuntime!, key.InputFeatures, key.OutputFeatures));

    internal bool PrewarmDirectPtxFusedLinearGeluW8A8M1(
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxQuantizedLinearEnabled) return false;
        if (!PtxFusedLinearGeluW8A8M1Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "quantized-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluW8A8M1Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.QuantizedLinearExperimentOverride)
        {
            DirectPtxLastError = "quantized-linear-performance-gate-not-met";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX W8A8 linear prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
                if (!_directPtxQuantizedLinearKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheQuantizedLinearKernelSlow(key);
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

    internal bool TryGetDirectPtxQuantizedLinearAudit(
        int inputFeatures,
        int outputFeatures,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            if (_directPtxQuantizedLinearKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Attempts the exact contiguous M=16 FP16-input/weight, FP32-accumulate
    /// Tensor Core linear + FP32 bias + tanh-GELU specialization.
    /// </summary>
    internal bool TryDirectPtxFusedLinearGeluFp16M16(
        IGpuBuffer inputHalf,
        IGpuBuffer outputMajorWeightsHalf,
        IGpuBuffer biasFloat,
        IGpuBuffer outputFloat,
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxMixedLinearEnabled) return false;
        if (!PtxFusedLinearGeluFp16M16Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "mixed-linear-m16-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluFp16M16Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride)
        {
            DirectPtxLastError = "mixed-linear-m16-performance-gate-not-met";
            return false;
        }
        long inputBytes = checked((long)PtxFusedLinearGeluFp16M16Kernel.Rows *
            inputFeatures * sizeof(ushort));
        long weightBytes = checked((long)inputFeatures * outputFeatures * sizeof(ushort));
        long biasBytes = checked((long)outputFeatures * sizeof(float));
        long outputBytes = checked((long)PtxFusedLinearGeluFp16M16Kernel.Rows *
            outputFeatures * sizeof(float));
        if (inputHalf.SizeInBytes != inputBytes ||
            outputMajorWeightsHalf.SizeInBytes != weightBytes ||
            biasFloat.SizeInBytes != biasBytes || outputFloat.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "mixed-linear-m16-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            lock (_directPtxLock)
            {
                if (!_directPtxMixedLinearM16Kernels.TryGetValue(
                    key, out PtxFusedLinearGeluFp16M16Kernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX M=16 mixed linear must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheMixedLinearM16KernelSlow(key);
                }
                if (capturing && !PinDirectPtxKernel(_directPtxMixedLinearM16Kernels, key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX M=16 mixed-linear module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(inputHalf, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(outputMajorWeightsHalf, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(biasFloat, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(outputFloat, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxMixedLinearDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedLinearGeluFp16M16Kernel CreateAndCacheMixedLinearM16KernelSlow(
        DirectPtxFusedLinearKey key) =>
        _directPtxMixedLinearM16Kernels.GetOrAdd(key, () =>
            new PtxFusedLinearGeluFp16M16Kernel(
                _directPtxRuntime!, key.InputFeatures, key.OutputFeatures));

    internal bool PrewarmDirectPtxFusedLinearGeluFp16M16(
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxMixedLinearEnabled) return false;
        if (!PtxFusedLinearGeluFp16M16Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "mixed-linear-m16-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluFp16M16Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride)
        {
            DirectPtxLastError = "mixed-linear-m16-performance-gate-not-met";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX M=16 mixed linear prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
                if (!_directPtxMixedLinearM16Kernels.TryGetValue(key, out _))
                    _ = CreateAndCacheMixedLinearM16KernelSlow(key);
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

    internal bool TryGetDirectPtxMixedLinearM16Audit(
        int inputFeatures,
        int outputFeatures,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            if (_directPtxMixedLinearM16Kernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Attempts the exact contiguous FP32 M=1 linear + bias + tanh-GELU
    /// specialization. K and N live in the loaded module, not in the launch ABI.
    /// </summary>
    internal bool TryDirectPtxFusedLinearGeluM1(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer bias,
        IGpuBuffer output,
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxFusedLinearGeluM1Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "fused-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluM1Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "fused-linear-performance-gate-not-met";
            return false;
        }
        long inputBytes = checked((long)inputFeatures * sizeof(float));
        long outputBytes = checked((long)outputFeatures * sizeof(float));
        long weightBytes = checked((long)inputFeatures * outputFeatures * sizeof(float));
        if (input.SizeInBytes != inputBytes || weights.SizeInBytes != weightBytes ||
            bias.SizeInBytes != outputBytes || output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "fused-linear-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            lock (_directPtxLock)
            {
                if (!_directPtxFusedLinearKernels.TryGetValue(
                    key, out PtxFusedLinearGeluM1Kernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX fused linear must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheFusedLinearKernelSlow(key);
                }
                // CUDA graph executables retain the CUfunction after capture.
                // Pin its module so later specialization churn cannot unload it.
                if (capturing && !PinDirectPtxKernel(_directPtxFusedLinearKernels, key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX fused-linear module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxFusedLinearDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFusedLinearGeluM1Kernel GetOrCreateFusedLinearKernel(
        DirectPtxFusedLinearKey key)
    {
        if (_directPtxFusedLinearKernels.TryGetValue(
            key, out PtxFusedLinearGeluM1Kernel? existing))
            return existing;
        return CreateAndCacheFusedLinearKernelSlow(key);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedLinearGeluM1Kernel CreateAndCacheFusedLinearKernelSlow(
        DirectPtxFusedLinearKey key) =>
        _directPtxFusedLinearKernels.GetOrAdd(key, () =>
            new PtxFusedLinearGeluM1Kernel(
                _directPtxRuntime!, key.InputFeatures, key.OutputFeatures));

    internal bool PrewarmDirectPtxFusedLinearGeluM1(
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxFusedLinearEnabled)
            return false;
        if (!PtxFusedLinearGeluM1Kernel.IsSupportedShape(inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "fused-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluM1Kernel.IsPromotedShape(inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "fused-linear-performance-gate-not-met";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX fused linear prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateFusedLinearKernel(
                    new DirectPtxFusedLinearKey(inputFeatures, outputFeatures));
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

    internal bool TryGetDirectPtxFusedLinearAudit(
        int inputFeatures,
        int outputFeatures,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFusedLinearKey(inputFeatures, outputFeatures);
            if (_directPtxFusedLinearKernels.TryGetValue(key, out var kernel))
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

    /// <summary>
    /// Attempts the exact contiguous FP32 D=64 transformer boundary
    /// GELU(LayerNorm(input + residual + bias)). The direct kernel reads each
    /// activation once, retains both values owned by a lane in registers, and
    /// performs no intermediate global-memory stores.
    /// </summary>
    internal bool TryDirectPtxFusedResidualBiasLayerNormGeluD64(
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer preNormBias,
        IGpuBuffer gamma,
        IGpuBuffer beta,
        IGpuBuffer output,
        int rows,
        float epsilon = 1e-5f)
    {
        if (!DirectPtxFeatureGate.IsResidualLayerNormGeluEnabled)
        {
            DirectPtxLastError = "residual-layernorm-gelu-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "residual-layernorm-gelu-cuda-unavailable";
            return false;
        }
        if (DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) !=
            DirectPtxArchitectureFamily.Ampere)
        {
            DirectPtxLastError = "residual-layernorm-gelu-architecture-not-validated";
            return false;
        }
        if (!PtxFusedResidualBiasLayerNormGeluD64Kernel.IsSupportedRows(rows))
        {
            DirectPtxLastError = "residual-layernorm-gelu-shape-not-implemented";
            return false;
        }
        if (!PtxFusedResidualBiasLayerNormGeluD64Kernel.IsPromotedRows(rows) &&
            !DirectPtxFeatureGate.NormalizationExperimentOverride)
        {
            DirectPtxLastError = "residual-layernorm-gelu-performance-gate-not-met";
            return false;
        }

        long matrixBytes = checked((long)rows * PtxFusedResidualBiasLayerNormGeluD64Kernel.Dimension * sizeof(float));
        long vectorBytes = PtxFusedResidualBiasLayerNormGeluD64Kernel.Dimension * sizeof(float);
        if (input.SizeInBytes != matrixBytes || residual.SizeInBytes != matrixBytes ||
            preNormBias.SizeInBytes != vectorBytes || gamma.SizeInBytes != vectorBytes ||
            beta.SizeInBytes != vectorBytes || output.SizeInBytes != matrixBytes)
        {
            DirectPtxLastError = "residual-layernorm-gelu-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxResidualRmsNormKey(rows, PtxCompat.SingleToInt32Bits(epsilon));
            lock (_directPtxLock)
            {
                if (!_directPtxResidualLayerNormGeluKernels.TryGetValue(
                    key, out PtxFusedResidualBiasLayerNormGeluD64Kernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX residual LayerNorm+GELU must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheResidualLayerNormGeluKernelSlow(key);
                }
                if (capturing && !PinDirectPtxKernel(_directPtxResidualLayerNormGeluKernels, key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX residual LayerNorm+GELU module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(residual, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(preNormBias, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(gamma, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(beta, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[5]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxResidualLayerNormGeluDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedResidualBiasLayerNormGeluD64Kernel
        CreateAndCacheResidualLayerNormGeluKernelSlow(DirectPtxResidualRmsNormKey key) =>
        _directPtxResidualLayerNormGeluKernels.GetOrAdd(key, () =>
            new PtxFusedResidualBiasLayerNormGeluD64Kernel(
                _directPtxRuntime!, key.Rows, PtxCompat.Int32BitsToSingle(key.EpsilonBits)));

    internal bool PrewarmDirectPtxFusedResidualBiasLayerNormGeluD64(
        int rows,
        float epsilon = 1e-5f)
    {
        if (!DirectPtxFeatureGate.IsResidualLayerNormGeluEnabled)
        {
            DirectPtxLastError = "residual-layernorm-gelu-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "residual-layernorm-gelu-cuda-unavailable";
            return false;
        }
        if (DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) !=
            DirectPtxArchitectureFamily.Ampere)
        {
            DirectPtxLastError = "residual-layernorm-gelu-architecture-not-validated";
            return false;
        }
        if (!PtxFusedResidualBiasLayerNormGeluD64Kernel.IsSupportedRows(rows))
        {
            DirectPtxLastError = "residual-layernorm-gelu-shape-not-implemented";
            return false;
        }
        if (!PtxFusedResidualBiasLayerNormGeluD64Kernel.IsPromotedRows(rows) &&
            !DirectPtxFeatureGate.NormalizationExperimentOverride)
        {
            DirectPtxLastError = "residual-layernorm-gelu-performance-gate-not-met";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX residual LayerNorm+GELU prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxResidualRmsNormKey(
                    rows, PtxCompat.SingleToInt32Bits(epsilon));
                if (!_directPtxResidualLayerNormGeluKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheResidualLayerNormGeluKernelSlow(key);
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

    internal bool TryGetDirectPtxResidualLayerNormGeluAudit(
        int rows,
        float epsilon,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxResidualRmsNormKey(rows, PtxCompat.SingleToInt32Bits(epsilon));
            if (_directPtxResidualLayerNormGeluKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxBatchNormUnit64(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize,
        float epsilon, float momentum, bool training)
    {
        if (batch != PtxChannelNormalizationD64Kernel.BatchNormBatch ||
            channels != PtxChannelNormalizationD64Kernel.BatchNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.BatchNormSpatial)
            return RejectDirectPtxNormalizationShape();

        return training
            ? TryDirectPtxChannelNormalization(
                DirectPtxChannelNormalizationOperation.BatchNormTraining,
                epsilon, momentum, input, gamma, beta, runningMean, runningVar,
                output, saveMean, saveInvVar)
            : TryDirectPtxChannelNormalization(
                DirectPtxChannelNormalizationOperation.BatchNormInference,
                epsilon, momentum, input, gamma, beta, runningMean, runningVar, output);
    }

    internal bool TryDirectPtxFusedBatchNormActivationUnit64(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar,
        int batch, int channels, int spatialSize, float epsilon,
        FusedActivationType activation)
    {
        if (batch != PtxChannelNormalizationD64Kernel.BatchNormBatch ||
            channels != PtxChannelNormalizationD64Kernel.BatchNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.BatchNormSpatial)
            return RejectDirectPtxNormalizationShape();

        DirectPtxChannelNormalizationOperation? operation = activation switch
        {
            FusedActivationType.ReLU => DirectPtxChannelNormalizationOperation.BatchNormRelu,
            FusedActivationType.GELU => DirectPtxChannelNormalizationOperation.BatchNormGelu,
            FusedActivationType.Sigmoid => DirectPtxChannelNormalizationOperation.BatchNormSigmoid,
            FusedActivationType.Tanh => DirectPtxChannelNormalizationOperation.BatchNormTanh,
            _ => null
        };
        if (!operation.HasValue)
        {
            DirectPtxLastError = "normalization-activation-not-implemented";
            return false;
        }
        return TryDirectPtxChannelNormalization(
            operation.Value, epsilon, 0f, input, gamma, beta, runningMean, runningVar, output);
    }

    internal bool TryDirectPtxBatchNormBackwardUnit64(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.BatchNormBatch ||
            channels != PtxChannelNormalizationD64Kernel.BatchNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.BatchNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalization(
            DirectPtxChannelNormalizationOperation.BatchNormBackward,
            epsilon, 0f, gradOutput, input, gamma, saveMean, saveInvVar,
            gradInput, gradGamma, gradBeta);
    }

    internal bool TryDirectPtxResidualBatchNormReluUnit64(
        IGpuBuffer input, IGpuBuffer residual, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.BatchNormBatch ||
            channels != PtxChannelNormalizationD64Kernel.BatchNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.BatchNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalization(
            DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu,
            epsilon, 0f, input, residual, gamma, beta, runningMean, runningVar, output);
    }

    internal bool TryDirectPtxGroupNormUnit64(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveVariance,
        int batch, int groups, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.GroupNormBatch ||
            groups != PtxChannelNormalizationD64Kernel.GroupNormGroups ||
            channels != PtxChannelNormalizationD64Kernel.GroupNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.GroupNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalization(
            DirectPtxChannelNormalizationOperation.GroupNormForward,
            epsilon, 0f, input, gamma, beta, output, saveMean, saveVariance);
    }

    internal bool TryDirectPtxGroupNormSwishUnit64(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        int batch, int groups, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.GroupNormBatch ||
            groups != PtxChannelNormalizationD64Kernel.GroupNormGroups ||
            channels != PtxChannelNormalizationD64Kernel.GroupNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.GroupNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalization(
            DirectPtxChannelNormalizationOperation.GroupNormSwish,
            epsilon, 0f, input, gamma, beta, output);
    }

    internal bool TryDirectPtxAddGroupNormUnit64(
        IGpuBuffer left, IGpuBuffer right, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta,
        int batch, int groups, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.GroupNormBatch ||
            groups != PtxChannelNormalizationD64Kernel.GroupNormGroups ||
            channels != PtxChannelNormalizationD64Kernel.GroupNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.GroupNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalization(
            DirectPtxChannelNormalizationOperation.AddGroupNorm,
            epsilon, 0f, left, right, gamma, beta, output);
    }

    internal bool TryDirectPtxInstanceNormUnit64(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.InstanceNormBatch ||
            channels != PtxChannelNormalizationD64Kernel.InstanceNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.InstanceNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalization(
            DirectPtxChannelNormalizationOperation.InstanceNormForward,
            epsilon, 0f, input, gamma, beta, output, saveMean, saveInvVar);
    }

    internal bool TryDirectPtxInstanceNormBackwardUnit64(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.InstanceNormBatch ||
            channels != PtxChannelNormalizationD64Kernel.InstanceNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.InstanceNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalizationPair(
            DirectPtxChannelNormalizationOperation.InstanceNormBackwardInput,
            DirectPtxChannelNormalizationOperation.InstanceNormGradParameters,
            epsilon,
            gradOutput, input, gamma, saveMean, saveInvVar, gradInput,
            gradOutput, input, saveMean, saveInvVar, gradGamma, gradBeta);
    }

    internal bool TryDirectPtxGroupNormBackwardUnit64(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer mean, IGpuBuffer variance,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int groups, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.GroupNormBatch ||
            groups != PtxChannelNormalizationD64Kernel.GroupNormGroups ||
            channels != PtxChannelNormalizationD64Kernel.GroupNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.GroupNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalizationPair(
            DirectPtxChannelNormalizationOperation.GroupNormBackwardInput,
            DirectPtxChannelNormalizationOperation.GroupNormGradParameters,
            epsilon,
            gradOutput, input, gamma, mean, variance, gradInput,
            gradOutput, input, mean, variance, gradGamma, gradBeta);
    }

    private bool TryDirectPtxChannelNormalizationPair(
        DirectPtxChannelNormalizationOperation inputOperation,
        DirectPtxChannelNormalizationOperation parameterOperation,
        float epsilon,
        IGpuBuffer inputTensor0, IGpuBuffer inputTensor1, IGpuBuffer inputTensor2,
        IGpuBuffer inputTensor3, IGpuBuffer inputTensor4, IGpuBuffer inputTensor5,
        IGpuBuffer parameterTensor0, IGpuBuffer parameterTensor1, IGpuBuffer parameterTensor2,
        IGpuBuffer parameterTensor3, IGpuBuffer parameterTensor4, IGpuBuffer parameterTensor5)
    {
        if (!IsDirectPtxChannelNormalizationAdmitted(inputOperation) ||
            !IsDirectPtxChannelNormalizationAdmitted(parameterOperation))
            return false;
        try
        {
            Span<DirectPtxTensorView> inputViews = stackalloc DirectPtxTensorView[6];
            Span<DirectPtxTensorView> parameterViews = stackalloc DirectPtxTensorView[6];
            int inputCount = PrepareDirectPtxChannelViews(
                inputOperation, inputViews, out _,
                inputTensor0, inputTensor1, inputTensor2, inputTensor3, inputTensor4, inputTensor5);
            int parameterCount = PrepareDirectPtxChannelViews(
                parameterOperation, parameterViews, out _,
                parameterTensor0, parameterTensor1, parameterTensor2,
                parameterTensor3, parameterTensor4, parameterTensor5);

            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var inputKey = DirectPtxChannelNormalizationKey.Create(inputOperation, epsilon, 0f);
            var parameterKey = DirectPtxChannelNormalizationKey.Create(parameterOperation, epsilon, 0f);
            lock (_directPtxLock)
            {
                if (capturing &&
                    (!_directPtxChannelNormalizationKernels.TryGetValue(inputKey, out _) ||
                     !_directPtxChannelNormalizationKernels.TryGetValue(parameterKey, out _)))
                {
                    DirectPtxLastError =
                        "Direct PTX channel normalization must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxChannelNormalizationKernels.TryGetValue(inputKey, out var inputKernel))
                    inputKernel = CreateAndCacheChannelNormalizationKernelSlow(inputKey);
                if (!_directPtxChannelNormalizationKernels.TryGetValue(parameterKey, out var parameterKernel))
                    parameterKernel = CreateAndCacheChannelNormalizationKernelSlow(parameterKey);
                if (capturing &&
                    (!PinDirectPtxKernel(_directPtxChannelNormalizationKernels, inputKey) ||
                     !PinDirectPtxKernel(_directPtxChannelNormalizationKernels, parameterKey)))
                    throw new InvalidOperationException(
                        "Could not pin both direct-PTX normalization-backward modules for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    inputKernel.LaunchPrevalidated(inputViews.Slice(0, inputCount));
                    parameterKernel.LaunchPrevalidated(parameterViews.Slice(0, parameterCount));
                }
            }
            System.Threading.Interlocked.Add(ref _directPtxChannelNormalizationDispatchCount, 2);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"channel-normalization-{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxFp16GroupNormSwishUnit64(
        IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output,
        int batch, int groups, int channels, int spatialSize, float epsilon)
    {
        if (batch != PtxChannelNormalizationD64Kernel.GroupNormBatch ||
            groups != PtxChannelNormalizationD64Kernel.GroupNormGroups ||
            channels != PtxChannelNormalizationD64Kernel.GroupNormChannels ||
            spatialSize != PtxChannelNormalizationD64Kernel.GroupNormSpatial)
            return RejectDirectPtxNormalizationShape();
        return TryDirectPtxChannelNormalization(
            DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish,
            epsilon, 0f, input, gamma, beta, output);
    }

    private bool RejectDirectPtxNormalizationShape()
    {
        DirectPtxLastError = "normalization-shape-not-implemented";
        return false;
    }

    private bool TryDirectPtxChannelNormalization(
        DirectPtxChannelNormalizationOperation operation,
        float epsilon,
        float momentum,
        IGpuBuffer tensor0,
        IGpuBuffer tensor1,
        IGpuBuffer? tensor2 = null,
        IGpuBuffer? tensor3 = null,
        IGpuBuffer? tensor4 = null,
        IGpuBuffer? tensor5 = null,
        IGpuBuffer? tensor6 = null,
        IGpuBuffer? tensor7 = null)
    {
        if (!IsDirectPtxChannelNormalizationAdmitted(operation))
            return false;

        try
        {
            Span<DirectPtxTensorView> views = stackalloc DirectPtxTensorView[8];
            int count = PrepareDirectPtxChannelViews(
                operation, views, out _,
                tensor0, tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7);
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            DirectPtxChannelNormalizationKey key =
                DirectPtxChannelNormalizationKey.Create(operation, epsilon, momentum);
            lock (_directPtxLock)
            {
                if (!_directPtxChannelNormalizationKernels.TryGetValue(key, out var kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX channel normalization must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheChannelNormalizationKernelSlow(key);
                }
                if (capturing && !PinDirectPtxKernel(_directPtxChannelNormalizationKernels, key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX channel-normalization module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.LaunchPrevalidated(views.Slice(0, count));
            }
            System.Threading.Interlocked.Increment(ref _directPtxChannelNormalizationDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"channel-normalization-{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool IsDirectPtxChannelNormalizationAdmitted(
        DirectPtxChannelNormalizationOperation operation)
    {
        if (!DirectPtxFeatureGate.IsNormalizationEnabled)
        {
            DirectPtxLastError = "normalization-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "normalization-cuda-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "normalization-architecture-not-validated";
            return false;
        }
        if (!PtxChannelNormalizationD64Kernel.IsPromoted(operation) &&
            !DirectPtxFeatureGate.NormalizationExperimentOverride)
        {
            DirectPtxLastError = "normalization-performance-gate-not-met";
            return false;
        }
        return true;
    }

    private static int PrepareDirectPtxChannelViews(
        DirectPtxChannelNormalizationOperation operation,
        Span<DirectPtxTensorView> views,
        out DirectPtxKernelBlueprint blueprint,
        IGpuBuffer tensor0,
        IGpuBuffer tensor1,
        IGpuBuffer? tensor2 = null,
        IGpuBuffer? tensor3 = null,
        IGpuBuffer? tensor4 = null,
        IGpuBuffer? tensor5 = null,
        IGpuBuffer? tensor6 = null,
        IGpuBuffer? tensor7 = null)
    {
        blueprint = PtxChannelNormalizationD64Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);
        int count = blueprint.Tensors.Count;
        if (views.Length < count)
            throw new ArgumentException("The channel-normalization view span is too small.", nameof(views));
        views[0] = DirectPtxTensorView.Create(tensor0, blueprint.Tensors[0]);
        views[1] = DirectPtxTensorView.Create(tensor1, blueprint.Tensors[1]);
        if (count > 2) views[2] = DirectPtxTensorView.Create(
            tensor2 ?? throw new ArgumentNullException(nameof(tensor2)), blueprint.Tensors[2]);
        if (count > 3) views[3] = DirectPtxTensorView.Create(
            tensor3 ?? throw new ArgumentNullException(nameof(tensor3)), blueprint.Tensors[3]);
        if (count > 4) views[4] = DirectPtxTensorView.Create(
            tensor4 ?? throw new ArgumentNullException(nameof(tensor4)), blueprint.Tensors[4]);
        if (count > 5) views[5] = DirectPtxTensorView.Create(
            tensor5 ?? throw new ArgumentNullException(nameof(tensor5)), blueprint.Tensors[5]);
        if (count > 6) views[6] = DirectPtxTensorView.Create(
            tensor6 ?? throw new ArgumentNullException(nameof(tensor6)), blueprint.Tensors[6]);
        if (count > 7) views[7] = DirectPtxTensorView.Create(
            tensor7 ?? throw new ArgumentNullException(nameof(tensor7)), blueprint.Tensors[7]);
        PtxChannelNormalizationD64Kernel.ValidateTensors(
            blueprint, views.Slice(0, count),
            PtxChannelNormalizationD64Kernel.GetEntryPoint(operation));
        return count;
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxChannelNormalizationD64Kernel CreateAndCacheChannelNormalizationKernelSlow(
        DirectPtxChannelNormalizationKey key) =>
        _directPtxChannelNormalizationKernels.GetOrAdd(key, () =>
            new PtxChannelNormalizationD64Kernel(
                _directPtxRuntime!, key.Operation,
                PtxCompat.Int32BitsToSingle(key.EpsilonBits),
                PtxCompat.Int32BitsToSingle(key.MomentumBits)));

    internal bool PrewarmDirectPtxChannelNormalization(
        DirectPtxChannelNormalizationOperation operation,
        float epsilon = 1e-5f,
        float momentum = 0.1f)
    {
        if (!IsDirectPtxChannelNormalizationAdmitted(operation))
            return false;
        if (IsStreamCapturing())
        {
            DirectPtxLastError = "channel-normalization-prewarm-during-capture";
            return false;
        }
        try
        {
            EnsureContextCurrent();
            DirectPtxChannelNormalizationKey key =
                DirectPtxChannelNormalizationKey.Create(operation, epsilon, momentum);
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxChannelNormalizationKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheChannelNormalizationKernelSlow(key);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"channel-normalization-{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxChannelNormalizationAudit(
        DirectPtxChannelNormalizationOperation operation,
        float epsilon,
        float momentum,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            DirectPtxChannelNormalizationKey key =
                DirectPtxChannelNormalizationKey.Create(operation, epsilon, momentum);
            if (_directPtxChannelNormalizationKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxLayerNormD64(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int rows, float epsilon) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.LayerNormForward, rows, epsilon,
            input, gamma, beta, output, saveMean, saveInvVar);

    internal bool TryDirectPtxLayerNormBackwardD64(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int rows, float epsilon)
    {
        if (!GpuDeterminism.IsActive && rows == 8_192)
        {
            return TryDirectPtxRowNormalization(
                DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic,
                rows, epsilon,
                gradOutput, input, gamma, saveMean, saveInvVar, gradInput,
                gradGamma, gradBeta);
        }
        return TryDirectPtxRowNormalizationPair(
            DirectPtxRowNormalizationOperation.LayerNormBackwardInput,
            DirectPtxRowNormalizationOperation.LayerNormGradParameters,
            rows, epsilon,
            gradOutput, input, gamma, saveMean, saveInvVar, gradInput,
            gradOutput, input, saveMean, saveInvVar, gradGamma, gradBeta);
    }

    internal bool TryDirectPtxFp16LayerNormD64(
        IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output,
        IGpuBuffer mean, IGpuBuffer variance, int rows, float epsilon) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.Fp16LayerNormForward,
            rows, epsilon, input, gamma, beta, output, mean, variance);

    internal bool TryDirectPtxFp16LayerNormBackwardD64(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer mean, IGpuBuffer invVar, IGpuBuffer gradInput, int rows) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.Fp16LayerNormBackwardInput,
            rows, 1e-5f, gradOutput, input, gamma, mean, invVar, gradInput);

    internal bool TryDirectPtxFp16LayerNormGradParametersD64(
        IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer mean, IGpuBuffer invVar,
        IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters,
            rows, 1e-5f, gradOutput, input, mean, invVar, gradGamma, gradBeta);

    internal bool TryDirectPtxRmsNormD64(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int rows, float epsilon) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.RmsNormForward, rows, epsilon,
            input, gamma, output, saveRms);

    internal bool TryDirectPtxRmsNormBackwardD64(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int rows, float epsilon)
    {
        if (!GpuDeterminism.IsActive && rows == 8_192)
        {
            return TryDirectPtxRowNormalization(
                DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic,
                rows, epsilon,
                gradOutput, input, gamma, saveRms, gradInput, gradGamma);
        }
        return TryDirectPtxRowNormalizationPair(
            DirectPtxRowNormalizationOperation.RmsNormBackwardInput,
            DirectPtxRowNormalizationOperation.RmsNormGradGamma,
            rows, epsilon,
            gradOutput, input, gamma, saveRms, gradInput, null,
            gradOutput, input, saveRms, gradGamma, null, null);
    }

    internal bool TryDirectPtxNormAxisD64(
        IGpuBuffer input, IGpuBuffer output, int rows) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.NormAxis, rows, 1e-5f,
            input, output);

    internal bool TryDirectPtxNormalizeL2D64(
        IGpuBuffer input, IGpuBuffer output, int rows) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.NormalizeL2, rows, 1e-5f,
            input, output);

    private bool TryDirectPtxRowNormalizationPair(
        DirectPtxRowNormalizationOperation inputOperation,
        DirectPtxRowNormalizationOperation parameterOperation,
        int rows,
        float epsilon,
        IGpuBuffer inputTensor0,
        IGpuBuffer inputTensor1,
        IGpuBuffer? inputTensor2,
        IGpuBuffer? inputTensor3,
        IGpuBuffer? inputTensor4,
        IGpuBuffer? inputTensor5,
        IGpuBuffer parameterTensor0,
        IGpuBuffer parameterTensor1,
        IGpuBuffer? parameterTensor2,
        IGpuBuffer? parameterTensor3,
        IGpuBuffer? parameterTensor4,
        IGpuBuffer? parameterTensor5)
    {
        parameterOperation = PtxRowNormalizationD64Kernel.SelectFastOperation(
            parameterOperation, GpuDeterminism.IsActive, rows);
        if (!IsDirectPtxRowNormalizationAdmitted(inputOperation, rows) ||
            !IsDirectPtxRowNormalizationAdmitted(parameterOperation, rows))
            return false;
        try
        {
            Span<DirectPtxTensorView> inputViews = stackalloc DirectPtxTensorView[6];
            Span<DirectPtxTensorView> parameterViews = stackalloc DirectPtxTensorView[6];
            int inputCount = PrepareDirectPtxRowViews(
                inputOperation, rows, inputViews,
                inputTensor0, inputTensor1, inputTensor2, inputTensor3, inputTensor4, inputTensor5);
            int parameterCount = PrepareDirectPtxRowViews(
                parameterOperation, rows, parameterViews,
                parameterTensor0, parameterTensor1, parameterTensor2,
                parameterTensor3, parameterTensor4, parameterTensor5);

            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var inputKey = new DirectPtxRowNormalizationKey(
                inputOperation, rows, PtxCompat.SingleToInt32Bits(epsilon));
            var parameterKey = new DirectPtxRowNormalizationKey(
                parameterOperation, rows, PtxCompat.SingleToInt32Bits(epsilon));
            lock (_directPtxLock)
            {
                bool hasInput = _directPtxRowNormalizationKernels.TryGetValue(inputKey, out _);
                bool hasParameters = _directPtxRowNormalizationKernels.TryGetValue(parameterKey, out _);
                if (capturing && (!hasInput || !hasParameters))
                {
                    DirectPtxLastError =
                        "Direct PTX normalization backward pair must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!hasInput)
                    _ = CreateAndCacheRowNormalizationKernelSlow(inputKey);
                if (!hasParameters)
                    _ = CreateAndCacheRowNormalizationKernelSlow(parameterKey);
                _directPtxRowNormalizationKernels.TryGetValue(inputKey, out var inputKernel);
                _directPtxRowNormalizationKernels.TryGetValue(parameterKey, out var parameterKernel);
                if (capturing &&
                    (!PinDirectPtxKernel(_directPtxRowNormalizationKernels, inputKey) ||
                     !PinDirectPtxKernel(_directPtxRowNormalizationKernels, parameterKey)))
                    throw new InvalidOperationException(
                        "Could not pin both direct-PTX normalization-backward modules for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    inputKernel!.LaunchPrevalidated(inputViews.Slice(0, inputCount));
                    if (PtxRowNormalizationD64Kernel.IsAtomicParameterGradient(
                            parameterOperation))
                        ClearDirectPtxWriteOutputs(parameterViews.Slice(0, parameterCount));
                    parameterKernel!.LaunchPrevalidated(parameterViews.Slice(0, parameterCount));
                }
            }
            System.Threading.Interlocked.Add(ref _directPtxRowNormalizationDispatchCount, 2);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"normalization-{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private void ClearDirectPtxWriteOutputs(ReadOnlySpan<DirectPtxTensorView> views)
    {
        for (int i = 0; i < views.Length; i++)
        {
            DirectPtxTensorView view = views[i];
            if ((view.Access & DirectPtxTensorAccess.Write) == 0)
                continue;
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuMemsetD8Async(
                    view.Pointer, 0, checked((ulong)view.ByteLength), _stream),
                "cuMemsetD8Async(direct PTX normalization accumulation output)");
        }
    }

    internal bool TryDirectPtxNormBackwardD64(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer norm,
        IGpuBuffer gradInput, int rows) =>
        TryDirectPtxRowNormalization(
            DirectPtxRowNormalizationOperation.NormBackward, rows, 1e-5f,
            gradOutput, input, norm, gradInput);

    internal bool TryDirectPtxReduceNormL2D64(
        IGpuBuffer input, IGpuBuffer output, int rows)
    {
        DirectPtxRowNormalizationOperation operation =
            PtxRowNormalizationD64Kernel.SelectFastOperation(
                DirectPtxRowNormalizationOperation.ReduceNormL2,
                GpuDeterminism.IsActive, rows);
        return TryDirectPtxRowNormalization(operation, rows, 1e-5f, input, output);
    }

    private bool TryDirectPtxRowNormalization(
        DirectPtxRowNormalizationOperation operation,
        int rows,
        float epsilon,
        IGpuBuffer tensor0,
        IGpuBuffer tensor1,
        IGpuBuffer? tensor2 = null,
        IGpuBuffer? tensor3 = null,
        IGpuBuffer? tensor4 = null,
        IGpuBuffer? tensor5 = null,
        IGpuBuffer? tensor6 = null,
        IGpuBuffer? tensor7 = null)
    {
        if (!IsDirectPtxRowNormalizationAdmitted(operation, rows))
            return false;

        try
        {
            Span<DirectPtxTensorView> views = stackalloc DirectPtxTensorView[8];
            int count = PrepareDirectPtxRowViews(
                operation, rows, views, tensor0, tensor1,
                tensor2, tensor3, tensor4, tensor5, tensor6, tensor7);
            Span<DirectPtxTensorView> admitted = views.Slice(0, count);

            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxRowNormalizationKey(
                operation, rows, PtxCompat.SingleToInt32Bits(epsilon));
            lock (_directPtxLock)
            {
                if (!_directPtxRowNormalizationKernels.TryGetValue(key, out var kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX normalization must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheRowNormalizationKernelSlow(key);
                }
                if (capturing && !PinDirectPtxKernel(_directPtxRowNormalizationKernels, key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX normalization module for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    if (operation == DirectPtxRowNormalizationOperation.ReduceNormL2Atomic)
                        ClearDirectPtxWriteOutputs(admitted);
                    else if (operation ==
                             DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic)
                        ClearDirectPtxWriteOutputs(admitted.Slice(6, 2));
                    else if (operation ==
                             DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic)
                        ClearDirectPtxWriteOutputs(admitted.Slice(5, 1));
                    kernel.LaunchPrevalidated(admitted);
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxRowNormalizationDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"normalization-{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool IsDirectPtxRowNormalizationAdmitted(
        DirectPtxRowNormalizationOperation operation,
        int rows)
    {
        if (!DirectPtxFeatureGate.IsNormalizationEnabled)
        {
            DirectPtxLastError = "normalization-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "normalization-cuda-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "normalization-architecture-not-validated";
            return false;
        }
        if (!PtxRowNormalizationD64Kernel.IsSupportedRows(rows))
        {
            DirectPtxLastError = "normalization-shape-not-implemented";
            return false;
        }
        if (!PtxRowNormalizationD64Kernel.IsPromoted(operation, rows) &&
            !DirectPtxFeatureGate.NormalizationExperimentOverride)
        {
            DirectPtxLastError = "normalization-performance-gate-not-met";
            return false;
        }
        return true;
    }

    private static int PrepareDirectPtxRowViews(
        DirectPtxRowNormalizationOperation operation,
        int rows,
        Span<DirectPtxTensorView> views,
        IGpuBuffer tensor0,
        IGpuBuffer tensor1,
        IGpuBuffer? tensor2,
        IGpuBuffer? tensor3,
        IGpuBuffer? tensor4,
        IGpuBuffer? tensor5,
        IGpuBuffer? tensor6 = null,
        IGpuBuffer? tensor7 = null)
    {
        DirectPtxKernelBlueprint blueprint =
            PtxRowNormalizationD64Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere, operation, rows);
        int count = blueprint.Tensors.Count;
        views[0] = DirectPtxTensorView.Create(tensor0, blueprint.Tensors[0]);
        views[1] = DirectPtxTensorView.Create(tensor1, blueprint.Tensors[1]);
        if (count > 2)
            views[2] = DirectPtxTensorView.Create(
                tensor2 ?? throw new ArgumentNullException(nameof(tensor2)), blueprint.Tensors[2]);
        if (count > 3)
            views[3] = DirectPtxTensorView.Create(
                tensor3 ?? throw new ArgumentNullException(nameof(tensor3)), blueprint.Tensors[3]);
        if (count > 4)
            views[4] = DirectPtxTensorView.Create(
                tensor4 ?? throw new ArgumentNullException(nameof(tensor4)), blueprint.Tensors[4]);
        if (count > 5)
            views[5] = DirectPtxTensorView.Create(
                tensor5 ?? throw new ArgumentNullException(nameof(tensor5)), blueprint.Tensors[5]);
        if (count > 6)
            views[6] = DirectPtxTensorView.Create(
                tensor6 ?? throw new ArgumentNullException(nameof(tensor6)), blueprint.Tensors[6]);
        if (count > 7)
            views[7] = DirectPtxTensorView.Create(
                tensor7 ?? throw new ArgumentNullException(nameof(tensor7)), blueprint.Tensors[7]);
        PtxRowNormalizationD64Kernel.ValidateTensors(
            blueprint, views.Slice(0, count),
            PtxRowNormalizationD64Kernel.GetEntryPoint(operation));
        return count;
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxRowNormalizationD64Kernel CreateAndCacheRowNormalizationKernelSlow(
        DirectPtxRowNormalizationKey key) =>
        _directPtxRowNormalizationKernels.GetOrAdd(key, () =>
            new PtxRowNormalizationD64Kernel(
                _directPtxRuntime!, key.Operation, key.Rows,
                PtxCompat.Int32BitsToSingle(key.EpsilonBits)));

    internal bool PrewarmDirectPtxRowNormalization(
        DirectPtxRowNormalizationOperation operation,
        int rows,
        float epsilon = 1e-5f)
    {
        if (!IsDirectPtxRowNormalizationAdmitted(operation, rows))
            return false;
        if (IsStreamCapturing())
        {
            DirectPtxLastError = "normalization-prewarm-during-capture";
            return false;
        }

        try
        {
            EnsureContextCurrent();
            var key = new DirectPtxRowNormalizationKey(
                operation, rows, PtxCompat.SingleToInt32Bits(epsilon));
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxRowNormalizationKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheRowNormalizationKernelSlow(key);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"normalization-{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxRowNormalizationAudit(
        DirectPtxRowNormalizationOperation operation,
        int rows,
        float epsilon,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxRowNormalizationKey(
                operation, rows, PtxCompat.SingleToInt32Bits(epsilon));
            if (_directPtxRowNormalizationKernels.TryGetValue(key, out var kernel))
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

    private void DisposeDirectPtxRuntime()
    {
        lock (_directPtxLock)
        {
            _directPtxPendingGraphUnpins = null;
            _directPtxGraphUnpins.Clear();
            _directPtxAttentionKernels.Dispose();
            _directPtxAttentionPlans.Clear();
            _directPtxResidualRmsNormKernels.Dispose();
            _directPtxResidualLayerNormGeluKernels.Dispose();
            _directPtxRowNormalizationKernels.Dispose();
            _directPtxChannelNormalizationKernels.Dispose();
            _directPtxDecodeKernels.Dispose();
            _directPtxPagedPrefillKernels.Dispose();
            _directPtxAttentionBackwardKernels.Dispose();
            _directPtxFlashAttentionBackwardKernels.Dispose();
            _directPtxQkvRopeCacheKernels.Dispose();
            _directPtxFusedLinearKernels.Dispose();
            _directPtxMixedLinearKernels.Dispose();
            _directPtxMixedLinearM16Kernels.Dispose();
            _directPtxQuantizedLinearKernels.Dispose();
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
    private readonly record struct DirectPtxRowNormalizationKey(
        DirectPtxRowNormalizationOperation Operation,
        int Rows,
        int EpsilonBits);
    private readonly record struct DirectPtxChannelNormalizationKey(
        DirectPtxChannelNormalizationOperation Operation,
        int EpsilonBits,
        int MomentumBits)
    {
        internal static DirectPtxChannelNormalizationKey Create(
            DirectPtxChannelNormalizationOperation operation,
            float epsilon,
            float momentum) => new(
                operation,
                PtxCompat.SingleToInt32Bits(epsilon),
                PtxCompat.SingleToInt32Bits(momentum));
    }
    private readonly record struct DirectPtxFusedLinearKey(
        int InputFeatures,
        int OutputFeatures);
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

}
