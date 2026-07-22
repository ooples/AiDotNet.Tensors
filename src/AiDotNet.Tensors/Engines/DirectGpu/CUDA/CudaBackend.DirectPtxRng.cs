#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IPhiloxBiasDropoutBackend, ICategoricalSamplingBackend,
    IGumbelSoftmaxBackwardBackend, IPhiloxRReluBackend, ISeededGumbelSoftmaxBackend
{
    private readonly bool _directPtxRngDropoutOptedIn =
        DirectPtxFeatureGate.IsRngDropoutEnabled;
    private readonly DirectPtxKernelCache<DirectPtxRngDropoutKey, PtxFusedPhiloxDropoutF32Kernel>
        _directPtxRngDropoutKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxRngFillKey, PtxPhiloxFillF32Kernel>
        _directPtxRngFillKernels = new(Math.Max(9, DirectPtxFeatureGate.CacheCapacity));
    private readonly DirectPtxKernelCache<int, PtxDropoutBackwardF32Kernel>
        _directPtxDropoutBackwardKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxFusedGumbelSoftmax32F32Kernel>
        _directPtxGumbelSoftmaxKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxFusedImportanceSampling64F32Kernel>
        _directPtxImportanceSamplingKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxFusedBiasPhiloxDropout256F32Kernel>
        _directPtxBiasDropoutKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxFusedDdimStepF32Kernel>
        _directPtxDdimKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxPhiloxCategorical32F32Kernel>
        _directPtxCategoricalKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxGumbelSoftmaxBackward32F32Kernel>
        _directPtxGumbelBackwardKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxFusedPhiloxRreluF32Kernel>
        _directPtxFusedRreluKernels = new(Math.Max(3, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxRreluKey, PtxRreluF32Kernel>
        _directPtxRreluKernels = new(Math.Max(6, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxRngDropoutDispatchCount;
    private long _directPtxRngFillDispatchCount;

    internal bool IsDirectPtxRngDropoutEnabled =>
        _directPtxRngDropoutOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor);

    internal long DirectPtxRngDropoutDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxRngDropoutDispatchCount);
    internal long DirectPtxRngFillDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxRngFillDispatchCount);

    internal int DirectPtxRngDropoutKernelCapacity => _directPtxRngDropoutKernels.Capacity;

    internal int DirectPtxRngDropoutCachedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRngDropoutKernels.Count; }
    }

    internal int DirectPtxRngDropoutPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRngDropoutKernels.PinnedCount; }
    }

    internal int DirectPtxRngFillPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRngFillKernels.PinnedCount; }
    }

    public bool TryFusedPhiloxRRelu(
        IGpuBuffer input,
        IGpuBuffer noise,
        IGpuBuffer output,
        int elementCount,
        float lower,
        float upper,
        ulong seed)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!ValidateExactRreluBuffer(input, elementCount, "input", out string? rejection) ||
            !ValidateExactRreluBuffer(noise, elementCount, "saved-noise", out rejection) ||
            !ValidateExactRreluBuffer(output, elementCount, "output", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (!float.IsFinite(lower) || !float.IsFinite(upper) || lower > upper)
        {
            DirectPtxLastError = "rng-rrelu-bounds-not-supported";
            return false;
        }
        if (input.Handle == noise.Handle || input.Handle == output.Handle ||
            noise.Handle == output.Handle)
        {
            DirectPtxLastError = "rng-rrelu-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxFusedRreluKernels.TryGetValue(elementCount, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX fused RReLU must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedPhiloxRreluF32Kernel kernel =
                    GetOrCreateFusedRreluKernel(elementCount);
                if (capturing && !_directPtxFusedRreluKernels.Pin(elementCount))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX fused-RReLU module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(noise, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        seed, 0, 0, lower, upper);
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

    internal bool TryDirectPtxRreluF32(
        IGpuBuffer input,
        IGpuBuffer noise,
        IGpuBuffer output,
        int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!ValidateExactRreluBuffer(input, elementCount, "input", out string? rejection) ||
            !ValidateExactRreluBuffer(noise, elementCount, "saved-noise", out rejection) ||
            !ValidateExactRreluBuffer(output, elementCount, "output", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (input.Handle == noise.Handle || input.Handle == output.Handle ||
            noise.Handle == output.Handle)
        {
            DirectPtxLastError = "rng-rrelu-alias-not-supported";
            return false;
        }
        return TryLaunchDirectPtxRrelu(
            DirectPtxRreluKind.Forward, elementCount,
            input, noise, output, fourth: null);
    }

    internal bool TryDirectPtxRreluBackwardF32(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer noise,
        IGpuBuffer gradInput,
        int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!ValidateExactRreluBuffer(gradOutput, elementCount, "grad-output", out string? rejection) ||
            !ValidateExactRreluBuffer(input, elementCount, "input", out rejection) ||
            !ValidateExactRreluBuffer(noise, elementCount, "saved-noise", out rejection) ||
            !ValidateExactRreluBuffer(gradInput, elementCount, "grad-input", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (gradInput.Handle == gradOutput.Handle || gradInput.Handle == input.Handle ||
            gradInput.Handle == noise.Handle)
        {
            DirectPtxLastError = "rng-rrelu-backward-alias-not-supported";
            return false;
        }
        return TryLaunchDirectPtxRrelu(
            DirectPtxRreluKind.Backward, elementCount,
            gradOutput, input, noise, gradInput);
    }

    private bool TryLaunchDirectPtxRrelu(
        DirectPtxRreluKind kind,
        int elementCount,
        IGpuBuffer first,
        IGpuBuffer second,
        IGpuBuffer third,
        IGpuBuffer? fourth)
    {
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxRreluKey(kind, elementCount);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxRreluKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX saved-noise RReLU must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxRreluF32Kernel kernel = GetOrCreateRreluKernel(key);
                if (capturing && !_directPtxRreluKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX saved-noise RReLU module for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    if (kind == DirectPtxRreluKind.Forward)
                    {
                        kernel.LaunchForward(
                            DirectPtxTensorView.Create(first, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(second, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(third, kernel.Blueprint.Tensors[2]));
                    }
                    else
                    {
                        kernel.LaunchBackward(
                            DirectPtxTensorView.Create(first, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(second, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(third, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(fourth!, kernel.Blueprint.Tensors[3]));
                    }
                }
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

    internal bool PrewarmDirectPtxFusedRreluF32(int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable || IsStreamCapturing() ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxFusedPhiloxRreluF32Kernel.IsSupportedElementCount(elementCount))
        {
            DirectPtxLastError = "rng-rrelu-fused-prewarm-not-eligible";
            return false;
        }
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateFusedRreluKernel(elementCount);
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

    internal bool PrewarmDirectPtxRreluF32(DirectPtxRreluKind kind, int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable || IsStreamCapturing() ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxRreluF32Kernel.IsSupportedElementCount(elementCount))
        {
            DirectPtxLastError = "rng-rrelu-prewarm-not-eligible";
            return false;
        }
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxRreluKey(kind, elementCount);
                _ = GetOrCreateRreluKernel(key);
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

    internal bool TryGetDirectPtxFusedRreluAudit(
        int elementCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxFusedRreluKernels.TryGetValue(elementCount, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxRreluAudit(
        DirectPtxRreluKind kind,
        int elementCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxRreluKernels.TryGetValue(
                    new DirectPtxRreluKey(kind, elementCount), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxDropoutBackwardF32(
        IGpuBuffer gradOutput,
        IGpuBuffer mask,
        IGpuBuffer gradInput,
        int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!ValidateExactRngVector(gradOutput, elementCount, "grad-output", out string? rejection) ||
            !ValidateExactRngVector(mask, elementCount, "saved-mask", out rejection) ||
            !ValidateExactRngVector(gradInput, elementCount, "grad-input", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (gradInput.Handle == gradOutput.Handle || gradInput.Handle == mask.Handle)
        {
            DirectPtxLastError = "rng-dropout-backward-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxDropoutBackwardKernels.TryGetValue(elementCount, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX dropout backward must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxDropoutBackwardF32Kernel kernel =
                    GetOrCreateDropoutBackwardKernel(elementCount);
                if (capturing && !_directPtxDropoutBackwardKernels.Pin(elementCount))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX dropout-backward module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(mask, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradInput, kernel.Blueprint.Tensors[2]));
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

    internal bool PrewarmDirectPtxDropoutBackwardF32(int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxDropoutBackwardF32Kernel.IsSupportedElementCount(elementCount))
        {
            DirectPtxLastError = "rng-dropout-backward-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX dropout-backward prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDropoutBackwardKernel(elementCount);
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

    internal bool TryDirectPtxGumbelSoftmaxF32(
        IGpuBuffer logits,
        IGpuBuffer output,
        int outerSize,
        int innerSize,
        float temperature,
        ulong seed,
        ulong subsequence = 0,
        ulong counterOffset = 0)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!IsAvailable)
        {
            DirectPtxLastError = "rng-gumbel-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "rng-gumbel-exact-sm-not-supported";
            return false;
        }
        if (!PtxFusedGumbelSoftmax32F32Kernel.IsSupportedShape(outerSize, innerSize))
        {
            DirectPtxLastError = "rng-gumbel-exact-shape-not-supported";
            return false;
        }
        if (!float.IsFinite(temperature) || temperature <= 0f)
        {
            DirectPtxLastError = "rng-gumbel-temperature-not-supported";
            return false;
        }
        int elements = checked(outerSize * innerSize);
        if (!ValidateExactRngMatrix(logits, elements, "logits", out string? rejection) ||
            !ValidateExactRngMatrix(output, elements, "output", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (logits.Handle == output.Handle)
        {
            DirectPtxLastError = "rng-gumbel-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxGumbelSoftmaxKernels.TryGetValue(outerSize, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX Gumbel-softmax must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedGumbelSoftmax32F32Kernel kernel =
                    GetOrCreateGumbelSoftmaxKernel(outerSize);
                if (capturing && !_directPtxGumbelSoftmaxKernels.Pin(outerSize))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX Gumbel-softmax module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(logits, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]),
                        seed, subsequence, counterOffset, temperature);
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

    internal bool PrewarmDirectPtxGumbelSoftmaxF32(int outerSize, int innerSize)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxFusedGumbelSoftmax32F32Kernel.IsSupportedShape(outerSize, innerSize))
        {
            DirectPtxLastError = "rng-gumbel-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX Gumbel-softmax prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateGumbelSoftmaxKernel(outerSize);
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

    internal bool TryGetDirectPtxGumbelSoftmaxAudit(
        int outerSize,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxGumbelSoftmaxKernels.TryGetValue(outerSize, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxImportanceSamplingF32(
        IGpuBuffer tValues,
        IGpuBuffer weights,
        IGpuBuffer output,
        int numRays,
        int numCoarseSamples,
        int numFineSamples,
        uint seed)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!IsAvailable)
        {
            DirectPtxLastError = "rng-importance-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "rng-importance-exact-sm-not-supported";
            return false;
        }
        if (!PtxFusedImportanceSampling64F32Kernel.IsSupportedShape(
                numRays, numCoarseSamples, numFineSamples))
        {
            DirectPtxLastError = "rng-importance-exact-shape-not-supported";
            return false;
        }
        int elements = checked(numRays * PtxFusedImportanceSampling64F32Kernel.Samples);
        if (!ValidateExactImportanceBuffer(tValues, elements, "t-values", out string? rejection) ||
            !ValidateExactImportanceBuffer(weights, elements, "weights", out rejection) ||
            !ValidateExactImportanceBuffer(output, elements, "output", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (output.Handle == tValues.Handle || output.Handle == weights.Handle)
        {
            DirectPtxLastError = "rng-importance-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxImportanceSamplingKernels.TryGetValue(numRays, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX importance sampling must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedImportanceSampling64F32Kernel kernel =
                    GetOrCreateImportanceSamplingKernel(numRays);
                if (capturing && !_directPtxImportanceSamplingKernels.Pin(numRays))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX importance-sampling module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(tValues, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        seed, 0, 0);
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

    internal bool PrewarmDirectPtxImportanceSamplingF32(
        int numRays,
        int numCoarseSamples,
        int numFineSamples)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxFusedImportanceSampling64F32Kernel.IsSupportedShape(
                numRays, numCoarseSamples, numFineSamples))
        {
            DirectPtxLastError = "rng-importance-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX importance-sampling prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateImportanceSamplingKernel(numRays);
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

    internal bool TryGetDirectPtxImportanceSamplingAudit(
        int numRays,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxImportanceSamplingKernels.TryGetValue(numRays, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    public bool TryFusedBiasPhiloxDropout(
        IGpuBuffer input,
        IGpuBuffer output,
        IGpuBuffer bias,
        IGpuBuffer mask,
        int rows,
        int cols,
        float dropoutRate,
        ulong seed)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!IsAvailable)
        {
            DirectPtxLastError = "rng-bias-dropout-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "rng-bias-dropout-exact-sm-not-supported";
            return false;
        }
        if (!PtxFusedBiasPhiloxDropout256F32Kernel.IsSupportedShape(rows, cols))
        {
            DirectPtxLastError = "rng-bias-dropout-exact-shape-not-supported";
            return false;
        }
        if (!float.IsFinite(dropoutRate) || dropoutRate <= 0f || dropoutRate >= 1f)
        {
            DirectPtxLastError = "rng-bias-dropout-rate-not-supported";
            return false;
        }
        int elements = checked(rows * cols);
        if (!ValidateExactBiasDropoutBuffer(input, elements, "input", out string? rejection) ||
            !ValidateExactBiasDropoutBuffer(output, elements, "output", out rejection) ||
            !ValidateExactBiasDropoutBuffer(mask, elements, "mask", out rejection) ||
            !ValidateExactBiasDropoutBuffer(bias, cols, "bias", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (output.Handle == input.Handle || output.Handle == bias.Handle ||
            output.Handle == mask.Handle || mask.Handle == input.Handle || mask.Handle == bias.Handle)
        {
            DirectPtxLastError = "rng-bias-dropout-alias-not-supported";
            return false;
        }

        double keepProbability = 1d - dropoutRate;
        ulong threshold64 = (ulong)Math.Floor(keepProbability * 4_294_967_296d);
        if (threshold64 is 0 or > uint.MaxValue)
        {
            DirectPtxLastError = "rng-bias-dropout-threshold-not-supported";
            return false;
        }
        float inverseKeep = 1f / (1f - dropoutRate);
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxBiasDropoutKernels.TryGetValue(rows, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX bias-dropout must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedBiasPhiloxDropout256F32Kernel kernel =
                    GetOrCreateBiasDropoutKernel(rows);
                if (capturing && !_directPtxBiasDropoutKernels.Pin(rows))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX bias-dropout module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(mask, kernel.Blueprint.Tensors[3]),
                        seed, 0, 0, (uint)threshold64, inverseKeep);
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

    internal bool PrewarmDirectPtxBiasDropoutF32(int rows, int cols)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxFusedBiasPhiloxDropout256F32Kernel.IsSupportedShape(rows, cols))
        {
            DirectPtxLastError = "rng-bias-dropout-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX bias-dropout prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateBiasDropoutKernel(rows);
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

    internal bool TryGetDirectPtxBiasDropoutAudit(
        int rows,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxBiasDropoutKernels.TryGetValue(rows, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxFusedDdimStepF32(
        IGpuBuffer xT,
        IGpuBuffer epsilon,
        IGpuBuffer output,
        int elementCount,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!IsAvailable)
        {
            DirectPtxLastError = "rng-ddim-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "rng-ddim-exact-sm-not-supported";
            return false;
        }
        if (!PtxFusedDdimStepF32Kernel.IsSupportedElementCount(elementCount))
        {
            DirectPtxLastError = "rng-ddim-exact-shape-not-supported";
            return false;
        }
        if (!(alphaBarT > 0f && alphaBarT <= 1f) ||
            !(alphaBarTMinus1 >= 0f && alphaBarTMinus1 <= 1f))
        {
            DirectPtxLastError = "rng-ddim-schedule-not-supported";
            return false;
        }
        if (!ValidateExactDdimBuffer(xT, elementCount, "x-t", out string? rejection) ||
            !ValidateExactDdimBuffer(epsilon, elementCount, "epsilon", out rejection) ||
            !ValidateExactDdimBuffer(output, elementCount, "output", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (output.Handle == xT.Handle || output.Handle == epsilon.Handle)
        {
            DirectPtxLastError = "rng-ddim-alias-not-supported";
            return false;
        }

        double sqrtAt = Math.Sqrt(alphaBarT);
        double sqrtAtMinus1 = Math.Sqrt(alphaBarTMinus1);
        float xCoefficient = (float)(sqrtAtMinus1 / sqrtAt);
        float epsilonCoefficient = (float)(
            Math.Sqrt(1d - alphaBarTMinus1) -
            Math.Sqrt(1d - alphaBarT) * sqrtAtMinus1 / sqrtAt);
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxDdimKernels.TryGetValue(elementCount, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX DDIM step must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedDdimStepF32Kernel kernel = GetOrCreateDdimKernel(elementCount);
                if (capturing && !_directPtxDdimKernels.Pin(elementCount))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX DDIM module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(xT, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(epsilon, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]),
                        xCoefficient, epsilonCoefficient);
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

    internal bool PrewarmDirectPtxFusedDdimStepF32(int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxFusedDdimStepF32Kernel.IsSupportedElementCount(elementCount))
        {
            DirectPtxLastError = "rng-ddim-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX DDIM prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDdimKernel(elementCount);
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

    internal bool TryGetDirectPtxFusedDdimStepAudit(
        int elementCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxDdimKernels.TryGetValue(elementCount, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    public bool CanCategoricalSample(int rows, int classes) =>
        _directPtxRngDropoutOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) &&
        PtxPhiloxCategorical32F32Kernel.IsSupportedShape(rows, classes);

    public bool TryCategoricalSample(
        IGpuBuffer probabilities,
        IGpuBuffer oneHot,
        int rows,
        int classes,
        ulong seed)
    {
        if (!CanCategoricalSample(rows, classes))
        {
            DirectPtxLastError = "rng-categorical-specialization-not-supported";
            return false;
        }
        int elements = checked(rows * classes);
        if (!ValidateExactCategoricalBuffer(probabilities, elements, "probabilities", out string? rejection) ||
            !ValidateExactCategoricalBuffer(oneHot, elements, "one-hot", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (probabilities.Handle == oneHot.Handle)
        {
            DirectPtxLastError = "rng-categorical-alias-not-supported";
            return false;
        }
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxCategoricalKernels.TryGetValue(rows, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX categorical sampling must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxPhiloxCategorical32F32Kernel kernel = GetOrCreateCategoricalKernel(rows);
                if (capturing && !_directPtxCategoricalKernels.Pin(rows))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX categorical module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(probabilities, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(oneHot, kernel.Blueprint.Tensors[1]),
                        seed, 0, 0);
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

    internal bool PrewarmDirectPtxCategoricalF32(int rows, int classes)
    {
        if (!CanCategoricalSample(rows, classes))
        {
            DirectPtxLastError = "rng-categorical-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX categorical prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateCategoricalKernel(rows);
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

    internal bool TryGetDirectPtxCategoricalAudit(
        int rows,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxCategoricalKernels.TryGetValue(rows, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    public bool CanGumbelSoftmaxBackward(int rows, int classes) =>
        _directPtxRngDropoutOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) &&
        PtxGumbelSoftmaxBackward32F32Kernel.IsSupportedShape(rows, classes);

    public bool TryGumbelSoftmaxBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer softOutput,
        IGpuBuffer gradInput,
        int rows,
        int classes,
        float temperature)
    {
        if (!CanGumbelSoftmaxBackward(rows, classes))
        {
            DirectPtxLastError = "rng-gumbel-backward-specialization-not-supported";
            return false;
        }
        if (!float.IsFinite(temperature) || temperature <= 0f)
        {
            DirectPtxLastError = "rng-gumbel-backward-temperature-not-supported";
            return false;
        }
        int elements = checked(rows * classes);
        if (!ValidateExactGumbelBackwardBuffer(gradOutput, elements, "grad-output", out string? rejection) ||
            !ValidateExactGumbelBackwardBuffer(softOutput, elements, "soft-output", out rejection) ||
            !ValidateExactGumbelBackwardBuffer(gradInput, elements, "grad-input", out rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (gradInput.Handle == gradOutput.Handle || gradInput.Handle == softOutput.Handle)
        {
            DirectPtxLastError = "rng-gumbel-backward-alias-not-supported";
            return false;
        }
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxGumbelBackwardKernels.TryGetValue(rows, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX Gumbel-softmax backward must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxGumbelSoftmaxBackward32F32Kernel kernel =
                    GetOrCreateGumbelBackwardKernel(rows);
                if (capturing && !_directPtxGumbelBackwardKernels.Pin(rows))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX Gumbel-backward module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(softOutput, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradInput, kernel.Blueprint.Tensors[2]),
                        temperature);
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

    internal bool PrewarmDirectPtxGumbelSoftmaxBackwardF32(int rows, int classes)
    {
        if (!CanGumbelSoftmaxBackward(rows, classes))
        {
            DirectPtxLastError = "rng-gumbel-backward-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX Gumbel-backward prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateGumbelBackwardKernel(rows);
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

    internal bool TryGetDirectPtxGumbelSoftmaxBackwardAudit(
        int rows,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxGumbelBackwardKernels.TryGetValue(rows, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxDropoutBackwardAudit(
        int elementCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxDropoutBackwardKernels.TryGetValue(elementCount, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryDirectPtxPhiloxFillF32(
        IGpuBuffer output,
        int elementCount,
        DirectPtxPhiloxFillKind kind,
        float first,
        float second,
        ulong seed,
        ulong subsequence = 0,
        ulong counterOffset = 0)
    {
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!ValidateRngFillOutput(output, elementCount, kind, out string? rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (kind == DirectPtxPhiloxFillKind.BernoulliMask ||
            !float.IsFinite(first) || !float.IsFinite(second) ||
            (kind == DirectPtxPhiloxFillKind.Uniform && second <= first) ||
            (kind == DirectPtxPhiloxFillKind.Normal && second < 0f))
        {
            DirectPtxLastError = "rng-fill-numerical-contract-not-supported";
            return false;
        }
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxRngFillKey(kind, elementCount);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxRngFillKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = "Direct PTX RNG fill must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxPhiloxFillF32Kernel kernel = GetOrCreateRngFillKernel(key);
                if (capturing && !_directPtxRngFillKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX RNG fill module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.LaunchRange(
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[0]),
                        seed, subsequence, counterOffset, first, second);
            }
            System.Threading.Interlocked.Increment(ref _directPtxRngFillDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxPhiloxMaskF32(
        IGpuBuffer output,
        int elementCount,
        uint threshold,
        float scale,
        ulong seed,
        ulong subsequence = 0,
        ulong counterOffset = 0)
    {
        return TryDirectPtxPhiloxMaskF32(
            output, elementCount, DirectPtxPhiloxFillKind.BernoulliMask,
            threshold, scale, seed, subsequence, counterOffset);
    }

    internal bool TryDirectPtxPhiloxMaskF32(
        IGpuBuffer output,
        int elementCount,
        DirectPtxPhiloxFillKind kind,
        uint threshold,
        float scale,
        ulong seed,
        ulong subsequence = 0,
        ulong counterOffset = 0)
    {
        if (kind is not (DirectPtxPhiloxFillKind.BernoulliMask or
                         DirectPtxPhiloxFillKind.DropThresholdMask))
        {
            DirectPtxLastError = "rng-mask-kind-not-supported";
            return false;
        }
        if (!_directPtxRngDropoutOptedIn) return false;
        if (!ValidateRngFillOutput(output, elementCount, kind, out string? rejection))
        {
            DirectPtxLastError = rejection;
            return false;
        }
        if (threshold == 0 || !float.IsFinite(scale) || scale < 0f)
        {
            DirectPtxLastError = "rng-mask-numerical-contract-not-supported";
            return false;
        }
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxRngFillKey(kind, elementCount);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxRngFillKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = "Direct PTX RNG mask must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxPhiloxFillF32Kernel kernel = GetOrCreateRngFillKernel(key);
                if (capturing && !_directPtxRngFillKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX RNG mask module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.LaunchMask(
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[0]),
                        seed, subsequence, counterOffset, threshold, scale);
            }
            System.Threading.Interlocked.Increment(ref _directPtxRngFillDispatchCount);
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

    internal bool PrewarmDirectPtxRngFillF32(
        DirectPtxPhiloxFillKind kind,
        int elementCount)
    {
        if (!_directPtxRngDropoutOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor) ||
            !PtxPhiloxFillF32Kernel.IsSupportedElementCount(elementCount))
        {
            DirectPtxLastError = "rng-fill-prewarm-not-eligible";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX RNG fill prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateRngFillKernel(new DirectPtxRngFillKey(kind, elementCount));
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

    internal bool TryGetDirectPtxRngFillAudit(
        DirectPtxPhiloxFillKind kind,
        int elementCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxRngFillKernels.TryGetValue(
                new DirectPtxRngFillKey(kind, elementCount), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private bool ValidateRngFillOutput(
        IGpuBuffer output,
        int elementCount,
        DirectPtxPhiloxFillKind kind,
        out string? rejection)
    {
        if (!IsAvailable)
        {
            rejection = "rng-fill-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            rejection = "rng-fill-exact-sm-not-supported";
            return false;
        }
        if (kind is not (DirectPtxPhiloxFillKind.Uniform or
                         DirectPtxPhiloxFillKind.Normal or
                         DirectPtxPhiloxFillKind.BernoulliMask or
                         DirectPtxPhiloxFillKind.DropThresholdMask) ||
            !PtxPhiloxFillF32Kernel.IsSupportedElementCount(elementCount))
        {
            rejection = "rng-fill-exact-shape-or-kind-not-supported";
            return false;
        }
        if (output is null || output.Handle == IntPtr.Zero)
        {
            rejection = "rng-fill-null-or-invalid-output";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (output.Size != elementCount || output.SizeInBytes != bytes)
        {
            rejection = "rng-fill-physical-extent-mismatch";
            return false;
        }
        if (((nuint)output.Handle & 15u) != 0)
        {
            rejection = "rng-fill-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private bool ValidateExactRngVector(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (!IsAvailable)
        {
            rejection = "rng-dropout-backward-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            rejection = "rng-dropout-backward-exact-sm-not-supported";
            return false;
        }
        if (!PtxDropoutBackwardF32Kernel.IsSupportedElementCount(elementCount))
        {
            rejection = "rng-dropout-backward-exact-shape-not-supported";
            return false;
        }
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-dropout-backward-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-dropout-backward-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 15u) != 0)
        {
            rejection = $"rng-dropout-backward-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private bool ValidateExactRngMatrix(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-gumbel-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-gumbel-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 3u) != 0)
        {
            rejection = $"rng-gumbel-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private static bool ValidateExactImportanceBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-importance-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-importance-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 3u) != 0)
        {
            rejection = $"rng-importance-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private static bool ValidateExactBiasDropoutBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-bias-dropout-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-bias-dropout-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 15u) != 0)
        {
            rejection = $"rng-bias-dropout-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private static bool ValidateExactDdimBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-ddim-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-ddim-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 15u) != 0)
        {
            rejection = $"rng-ddim-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private static bool ValidateExactCategoricalBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-categorical-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-categorical-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 3u) != 0)
        {
            rejection = $"rng-categorical-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private static bool ValidateExactGumbelBackwardBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-gumbel-backward-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-gumbel-backward-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 3u) != 0)
        {
            rejection = $"rng-gumbel-backward-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private bool ValidateExactRreluBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string role,
        out string? rejection)
    {
        if (!IsAvailable)
        {
            rejection = "rng-rrelu-cuda-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(_ccMajor, _ccMinor))
        {
            rejection = "rng-rrelu-exact-sm-not-supported";
            return false;
        }
        if (!PtxFusedPhiloxRreluF32Kernel.IsSupportedElementCount(elementCount))
        {
            rejection = "rng-rrelu-exact-shape-not-supported";
            return false;
        }
        if (buffer is null || buffer.Handle == IntPtr.Zero)
        {
            rejection = $"rng-rrelu-{role}-invalid";
            return false;
        }
        long bytes = checked((long)elementCount * sizeof(float));
        if (buffer.Size != elementCount || buffer.SizeInBytes != bytes)
        {
            rejection = $"rng-rrelu-{role}-extent-mismatch";
            return false;
        }
        if (((nuint)buffer.Handle & 15u) != 0)
        {
            rejection = $"rng-rrelu-{role}-alignment-mismatch";
            return false;
        }
        rejection = null;
        return true;
    }

    private PtxFusedPhiloxRreluF32Kernel GetOrCreateFusedRreluKernel(int elementCount)
    {
        if (_directPtxFusedRreluKernels.TryGetValue(
                elementCount, out PtxFusedPhiloxRreluF32Kernel? existing))
            return existing;
        return CreateAndCacheFusedRreluKernelSlow(elementCount);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedPhiloxRreluF32Kernel CreateAndCacheFusedRreluKernelSlow(int elementCount) =>
        _directPtxFusedRreluKernels.GetOrAdd(
            elementCount,
            () => new PtxFusedPhiloxRreluF32Kernel(_directPtxRuntime!, elementCount));

    private PtxRreluF32Kernel GetOrCreateRreluKernel(DirectPtxRreluKey key)
    {
        if (_directPtxRreluKernels.TryGetValue(key, out PtxRreluF32Kernel? existing))
            return existing;
        return CreateAndCacheRreluKernelSlow(key);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxRreluF32Kernel CreateAndCacheRreluKernelSlow(DirectPtxRreluKey key) =>
        _directPtxRreluKernels.GetOrAdd(
            key, () => new PtxRreluF32Kernel(_directPtxRuntime!, key.Kind, key.ElementCount));

    private PtxDropoutBackwardF32Kernel GetOrCreateDropoutBackwardKernel(int elementCount)
    {
        if (_directPtxDropoutBackwardKernels.TryGetValue(
                elementCount, out PtxDropoutBackwardF32Kernel? existing))
            return existing;
        return CreateAndCacheDropoutBackwardKernelSlow(elementCount);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxDropoutBackwardF32Kernel CreateAndCacheDropoutBackwardKernelSlow(int elementCount) =>
        _directPtxDropoutBackwardKernels.GetOrAdd(
            elementCount, () => new PtxDropoutBackwardF32Kernel(_directPtxRuntime!, elementCount));

    private PtxFusedGumbelSoftmax32F32Kernel GetOrCreateGumbelSoftmaxKernel(int rows)
    {
        if (_directPtxGumbelSoftmaxKernels.TryGetValue(
                rows, out PtxFusedGumbelSoftmax32F32Kernel? existing))
            return existing;
        return CreateAndCacheGumbelSoftmaxKernelSlow(rows);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedGumbelSoftmax32F32Kernel CreateAndCacheGumbelSoftmaxKernelSlow(int rows) =>
        _directPtxGumbelSoftmaxKernels.GetOrAdd(
            rows, () => new PtxFusedGumbelSoftmax32F32Kernel(_directPtxRuntime!, rows));

    private PtxFusedImportanceSampling64F32Kernel GetOrCreateImportanceSamplingKernel(int rays)
    {
        if (_directPtxImportanceSamplingKernels.TryGetValue(
                rays, out PtxFusedImportanceSampling64F32Kernel? existing))
            return existing;
        return CreateAndCacheImportanceSamplingKernelSlow(rays);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedImportanceSampling64F32Kernel CreateAndCacheImportanceSamplingKernelSlow(
        int rays) =>
        _directPtxImportanceSamplingKernels.GetOrAdd(
            rays, () => new PtxFusedImportanceSampling64F32Kernel(_directPtxRuntime!, rays));

    private PtxFusedBiasPhiloxDropout256F32Kernel GetOrCreateBiasDropoutKernel(int rows)
    {
        if (_directPtxBiasDropoutKernels.TryGetValue(
                rows, out PtxFusedBiasPhiloxDropout256F32Kernel? existing))
            return existing;
        return CreateAndCacheBiasDropoutKernelSlow(rows);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedBiasPhiloxDropout256F32Kernel CreateAndCacheBiasDropoutKernelSlow(int rows) =>
        _directPtxBiasDropoutKernels.GetOrAdd(
            rows, () => new PtxFusedBiasPhiloxDropout256F32Kernel(_directPtxRuntime!, rows));

    private PtxFusedDdimStepF32Kernel GetOrCreateDdimKernel(int elementCount)
    {
        if (_directPtxDdimKernels.TryGetValue(
                elementCount, out PtxFusedDdimStepF32Kernel? existing))
            return existing;
        return CreateAndCacheDdimKernelSlow(elementCount);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedDdimStepF32Kernel CreateAndCacheDdimKernelSlow(int elementCount) =>
        _directPtxDdimKernels.GetOrAdd(
            elementCount, () => new PtxFusedDdimStepF32Kernel(_directPtxRuntime!, elementCount));

    private PtxPhiloxCategorical32F32Kernel GetOrCreateCategoricalKernel(int rows)
    {
        if (_directPtxCategoricalKernels.TryGetValue(
                rows, out PtxPhiloxCategorical32F32Kernel? existing))
            return existing;
        return CreateAndCacheCategoricalKernelSlow(rows);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxPhiloxCategorical32F32Kernel CreateAndCacheCategoricalKernelSlow(int rows) =>
        _directPtxCategoricalKernels.GetOrAdd(
            rows, () => new PtxPhiloxCategorical32F32Kernel(_directPtxRuntime!, rows));

    private PtxGumbelSoftmaxBackward32F32Kernel GetOrCreateGumbelBackwardKernel(int rows)
    {
        if (_directPtxGumbelBackwardKernels.TryGetValue(
                rows, out PtxGumbelSoftmaxBackward32F32Kernel? existing))
            return existing;
        return CreateAndCacheGumbelBackwardKernelSlow(rows);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxGumbelSoftmaxBackward32F32Kernel CreateAndCacheGumbelBackwardKernelSlow(int rows) =>
        _directPtxGumbelBackwardKernels.GetOrAdd(
            rows, () => new PtxGumbelSoftmaxBackward32F32Kernel(_directPtxRuntime!, rows));

    private PtxPhiloxFillF32Kernel GetOrCreateRngFillKernel(DirectPtxRngFillKey key)
    {
        if (_directPtxRngFillKernels.TryGetValue(key, out PtxPhiloxFillF32Kernel? existing))
            return existing;
        return CreateAndCacheRngFillKernelSlow(key);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxPhiloxFillF32Kernel CreateAndCacheRngFillKernelSlow(DirectPtxRngFillKey key) =>
        _directPtxRngFillKernels.GetOrAdd(
            key, () => new PtxPhiloxFillF32Kernel(_directPtxRuntime!, key.Kind, key.ElementCount));

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedPhiloxDropoutF32Kernel CreateAndCacheRngDropoutKernelSlow(
        DirectPtxRngDropoutKey key) =>
        _directPtxRngDropoutKernels.GetOrAdd(
            key, () => new PtxFusedPhiloxDropoutF32Kernel(_directPtxRuntime!, key.ElementCount));

    private readonly record struct DirectPtxRngDropoutKey(int ElementCount);
    private readonly record struct DirectPtxRngFillKey(
        DirectPtxPhiloxFillKind Kind,
        int ElementCount);
    private readonly record struct DirectPtxRreluKey(
        DirectPtxRreluKind Kind,
        int ElementCount);
}
#endif
