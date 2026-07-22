#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly bool _directPtxRngDropoutOptedIn =
        DirectPtxFeatureGate.IsRngDropoutEnabled;
    private readonly DirectPtxKernelCache<DirectPtxRngDropoutKey, PtxFusedPhiloxDropoutF32Kernel>
        _directPtxRngDropoutKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxRngDropoutDispatchCount;

    internal bool IsDirectPtxRngDropoutEnabled =>
        _directPtxRngDropoutOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor);

    internal long DirectPtxRngDropoutDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxRngDropoutDispatchCount);

    internal int DirectPtxRngDropoutKernelCapacity => _directPtxRngDropoutKernels.Capacity;

    internal int DirectPtxRngDropoutCachedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRngDropoutKernels.Count; }
    }

    internal int DirectPtxRngDropoutPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRngDropoutKernels.PinnedCount; }
    }

    /// <summary>
    /// Attempts the exact-shape FP32 Philox dropout specialization. Seed is the
    /// 64-bit Philox key; subsequence is the high 64-bit counter; counterOffset
    /// selects the first four-word group. Public Dropout uses subsequence and
    /// counter offset zero, preserving repeatability for an explicit seed.
    /// </summary>
    internal bool TryDirectPtxRngDropoutF32(
        IGpuBuffer input,
        IGpuBuffer output,
        IGpuBuffer mask,
        int elementCount,
        float dropoutRate,
        ulong seed,
        ulong subsequence = 0,
        ulong counterOffset = 0)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!IsAvailable)
        {
            DirectPtxLastError = "rng-dropout-cuda-backend-unavailable";
            return false;
        }
        if (input is null || output is null || mask is null)
        {
            DirectPtxLastError = "rng-dropout-null-buffer";
            return false;
        }
        if (!DirectPtxRngDropoutAdmission.TryValidate(
            input.Handle, input.SizeInBytes,
            output.Handle, output.SizeInBytes,
            mask.Handle, mask.SizeInBytes,
            elementCount, dropoutRate, _ccMajor, _ccMinor,
            out DirectPtxRngDropoutParameters parameters,
            out string? rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxRngDropoutKey(elementCount);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxRngDropoutKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX RNG/dropout must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedPhiloxDropoutF32Kernel kernel = GetOrCreateRngDropoutKernel(key);
                if (capturing && !_directPtxRngDropoutKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX RNG/dropout module for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(mask, kernel.Blueprint.Tensors[2]),
                        seed, subsequence, counterOffset,
                        parameters.KeepThreshold, parameters.InverseKeep);
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxRngDropoutDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxRngDropoutF32(int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!IsAvailable)
        {
            DirectPtxLastError = "rng-dropout-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "rng-dropout-exact-sm-not-supported";
            return false;
        }
        if (!PtxFusedPhiloxDropoutF32Kernel.IsSupportedElementCount(elementCount))
        {
            DirectPtxLastError = "rng-dropout-exact-shape-not-supported";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX RNG/dropout prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateRngDropoutKernel(new DirectPtxRngDropoutKey(elementCount));
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

    internal bool TryGetDirectPtxRngDropoutAudit(
        int elementCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxRngDropoutKernels.TryGetValue(
                new DirectPtxRngDropoutKey(elementCount), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private PtxFusedPhiloxDropoutF32Kernel GetOrCreateRngDropoutKernel(
        DirectPtxRngDropoutKey key)
    {
        if (_directPtxRngDropoutKernels.TryGetValue(
            key, out PtxFusedPhiloxDropoutF32Kernel? existing))
            return existing;
        return CreateAndCacheRngDropoutKernelSlow(key);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedPhiloxDropoutF32Kernel CreateAndCacheRngDropoutKernelSlow(
        DirectPtxRngDropoutKey key) =>
        _directPtxRngDropoutKernels.GetOrAdd(
            key, () => new PtxFusedPhiloxDropoutF32Kernel(_directPtxRuntime!, key.ElementCount));

    private readonly record struct DirectPtxRngDropoutKey(int ElementCount);
}
#endif
