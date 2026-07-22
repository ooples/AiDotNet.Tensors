using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly bool _directPtxConvolutionOptedIn =
        DirectPtxFeatureGate.IsConvolutionEnabled;
    private readonly DirectPtxKernelCache<int, PtxFusedConv2DNchwK1Kernel>
        _directPtxConvolutionKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConvolutionDispatchCount;

    internal bool IsDirectPtxConvolutionEnabled =>
        _directPtxConvolutionOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor);

    internal long DirectPtxConvolutionDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConvolutionDispatchCount);

    internal int DirectPtxConvolutionPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxConvolutionKernels.PinnedCount; }
    }

    /// <summary>
    /// Attempts the exact FP32 NCHW 1x1 convolution + bias + ReLU experiment.
    /// Every unsupported contract fails closed before module lookup so the
    /// caller can execute the established cuDNN/NVRTC composition.
    /// </summary>
    internal bool TryDirectPtxFusedConv2DBiasRelu(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer bias,
        IGpuBuffer output,
        DirectPtxConvolutionShape shape)
    {
        string? rejection = DirectPtxConvolutionEligibility.Validate(
            _directPtxConvolutionOptedIn, IsAvailable, _ccMajor, _ccMinor,
            shape, input, weights, bias, output);
        if (rejection is not null)
        {
            DirectPtxLastError = rejection;
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConvolutionKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX convolution must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedConv2DNchwK1Kernel kernel = GetOrCreateDirectPtxConvolutionKernel();
                if (capturing && !_directPtxConvolutionKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX convolution module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConvolutionDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxFusedConv2DBiasRelu()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX convolution prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConvolutionKernel();
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

    internal bool TryGetDirectPtxConvolutionAudit(out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConvolutionKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private PtxFusedConv2DNchwK1Kernel GetOrCreateDirectPtxConvolutionKernel()
    {
        if (_directPtxConvolutionKernels.TryGetValue(
                1, out PtxFusedConv2DNchwK1Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConvolutionKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedConv2DNchwK1Kernel CreateAndCacheDirectPtxConvolutionKernelSlow() =>
        _directPtxConvolutionKernels.GetOrAdd(
            1, () => new PtxFusedConv2DNchwK1Kernel(_directPtxRuntime!));

    private readonly DirectPtxKernelCache<int, PtxFusedDepthwiseConv2D3x3F32Kernel>
        _directPtxDepthwiseConvKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxDepthwiseConvDispatchCount;

    internal long DirectPtxDepthwiseConvDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxDepthwiseConvDispatchCount);

    internal int DirectPtxDepthwiseConvPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxDepthwiseConvKernels.PinnedCount; }
    }

    /// <summary>
    /// Attempts the exact FP32 NCHW depthwise 3x3 (stride 1, pad 1) convolution
    /// experiment. Every unsupported contract fails closed before module lookup
    /// so the caller can execute the established cuDNN/NVRTC composition.
    /// </summary>
    internal bool TryDirectPtxDepthwiseConv2D3x3(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer output,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxFusedDepthwiseConv2D3x3F32Kernel.Shape)
        {
            DirectPtxLastError = "depthwise-conv-shape-not-implemented";
            return false;
        }
        if (input is null || weights is null || output is null)
        {
            DirectPtxLastError = "depthwise-conv-null-buffer";
            return false;
        }
        if (input.SizeInBytes != PtxFusedDepthwiseConv2D3x3F32Kernel.InputBytes ||
            weights.SizeInBytes != PtxFusedDepthwiseConv2D3x3F32Kernel.WeightBytes ||
            output.SizeInBytes != PtxFusedDepthwiseConv2D3x3F32Kernel.OutputBytes)
        {
            DirectPtxLastError = "depthwise-conv-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxDepthwiseConvKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX depthwise convolution must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedDepthwiseConv2D3x3F32Kernel kernel = GetOrCreateDirectPtxDepthwiseConvKernel();
                if (capturing && !_directPtxDepthwiseConvKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX depthwise convolution module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxDepthwiseConvDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxDepthwiseConv2D3x3()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX depthwise convolution prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxDepthwiseConvKernel();
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

    internal bool TryGetDirectPtxDepthwiseConvAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxDepthwiseConvKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxFusedDepthwiseConv2D3x3F32Kernel GetOrCreateDirectPtxDepthwiseConvKernel()
    {
        if (_directPtxDepthwiseConvKernels.TryGetValue(
                1, out PtxFusedDepthwiseConv2D3x3F32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxDepthwiseConvKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedDepthwiseConv2D3x3F32Kernel CreateAndCacheDirectPtxDepthwiseConvKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxDepthwiseConvKernels.GetOrAdd(
            1, () => new PtxFusedDepthwiseConv2D3x3F32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxDepthwiseConv2D3x3BackwardInputF32Kernel>
        _directPtxDepthwiseConvBwdInputKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxDepthwiseConvBwdInputDispatchCount;

    internal long DirectPtxDepthwiseConvBwdInputDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxDepthwiseConvBwdInputDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCHW depthwise 3x3 (stride 1, pad 1) backward-input
    /// gradient experiment. Fails closed on any unsupported contract so the caller
    /// runs the established composition.
    /// </summary>
    internal bool TryDirectPtxDepthwiseConv2D3x3BackwardInput(
        IGpuBuffer gradOutput,
        IGpuBuffer weights,
        IGpuBuffer gradInput,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxDepthwiseConv2D3x3BackwardInputF32Kernel.Shape)
        {
            DirectPtxLastError = "depthwise-conv-bwd-input-shape-not-implemented";
            return false;
        }
        if (gradOutput is null || weights is null || gradInput is null)
        {
            DirectPtxLastError = "depthwise-conv-bwd-input-null-buffer";
            return false;
        }
        if (gradOutput.SizeInBytes != PtxDepthwiseConv2D3x3BackwardInputF32Kernel.GradOutputBytes ||
            weights.SizeInBytes != PtxDepthwiseConv2D3x3BackwardInputF32Kernel.WeightBytes ||
            gradInput.SizeInBytes != PtxDepthwiseConv2D3x3BackwardInputF32Kernel.GradInputBytes)
        {
            DirectPtxLastError = "depthwise-conv-bwd-input-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxDepthwiseConvBwdInputKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX depthwise backward-input must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxDepthwiseConv2D3x3BackwardInputF32Kernel kernel =
                    GetOrCreateDirectPtxDepthwiseConvBwdInputKernel();
                if (capturing && !_directPtxDepthwiseConvBwdInputKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX depthwise backward-input module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradInput, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxDepthwiseConvBwdInputDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxDepthwiseConv2D3x3BackwardInput()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX depthwise backward-input prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxDepthwiseConvBwdInputKernel();
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

    internal bool TryGetDirectPtxDepthwiseConvBwdInputAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxDepthwiseConvBwdInputKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxDepthwiseConv2D3x3BackwardInputF32Kernel GetOrCreateDirectPtxDepthwiseConvBwdInputKernel()
    {
        if (_directPtxDepthwiseConvBwdInputKernels.TryGetValue(
                1, out PtxDepthwiseConv2D3x3BackwardInputF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxDepthwiseConvBwdInputKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxDepthwiseConv2D3x3BackwardInputF32Kernel CreateAndCacheDirectPtxDepthwiseConvBwdInputKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxDepthwiseConvBwdInputKernels.GetOrAdd(
            1, () => new PtxDepthwiseConv2D3x3BackwardInputF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxDepthwiseConv2D3x3BackwardWeightF32Kernel>
        _directPtxDepthwiseConvBwdWeightKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxDepthwiseConvBwdWeightDispatchCount;

    internal long DirectPtxDepthwiseConvBwdWeightDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxDepthwiseConvBwdWeightDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCHW depthwise 3x3 (stride 1, pad 1) backward-weight
    /// gradient experiment. Fails closed on any unsupported contract so the caller
    /// runs the established composition.
    /// </summary>
    internal bool TryDirectPtxDepthwiseConv2D3x3BackwardWeight(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer gradWeight,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.Shape)
        {
            DirectPtxLastError = "depthwise-conv-bwd-weight-shape-not-implemented";
            return false;
        }
        if (gradOutput is null || input is null || gradWeight is null)
        {
            DirectPtxLastError = "depthwise-conv-bwd-weight-null-buffer";
            return false;
        }
        if (gradOutput.SizeInBytes != PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.GradOutputBytes ||
            input.SizeInBytes != PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.InputBytes ||
            gradWeight.SizeInBytes != PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.GradWeightBytes)
        {
            DirectPtxLastError = "depthwise-conv-bwd-weight-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxDepthwiseConvBwdWeightKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX depthwise backward-weight must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxDepthwiseConv2D3x3BackwardWeightF32Kernel kernel =
                    GetOrCreateDirectPtxDepthwiseConvBwdWeightKernel();
                if (capturing && !_directPtxDepthwiseConvBwdWeightKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX depthwise backward-weight module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradWeight, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxDepthwiseConvBwdWeightDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxDepthwiseConv2D3x3BackwardWeight()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX depthwise backward-weight prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxDepthwiseConvBwdWeightKernel();
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

    internal bool TryGetDirectPtxDepthwiseConvBwdWeightAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxDepthwiseConvBwdWeightKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxDepthwiseConv2D3x3BackwardWeightF32Kernel GetOrCreateDirectPtxDepthwiseConvBwdWeightKernel()
    {
        if (_directPtxDepthwiseConvBwdWeightKernels.TryGetValue(
                1, out PtxDepthwiseConv2D3x3BackwardWeightF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxDepthwiseConvBwdWeightKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxDepthwiseConv2D3x3BackwardWeightF32Kernel CreateAndCacheDirectPtxDepthwiseConvBwdWeightKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxDepthwiseConvBwdWeightKernels.GetOrAdd(
            1, () => new PtxDepthwiseConv2D3x3BackwardWeightF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxConv2DBackwardBiasF32Kernel>
        _directPtxConvBwdBiasKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConvBwdBiasDispatchCount;

    internal long DirectPtxConvBwdBiasDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConvBwdBiasDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCHW convolution bias-gradient experiment
    /// (dBias[k] = sum_{y,x} dOut[k,y,x]) for the golden-slice output geometry.
    /// Fails closed on any unsupported contract so the caller runs the established
    /// reduction.
    /// </summary>
    internal bool TryDirectPtxConv2DBackwardBias(
        IGpuBuffer gradOutput,
        IGpuBuffer gradBias,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxConv2DBackwardBiasF32Kernel.Shape)
        {
            DirectPtxLastError = "conv-bwd-bias-shape-not-implemented";
            return false;
        }
        if (gradOutput is null || gradBias is null)
        {
            DirectPtxLastError = "conv-bwd-bias-null-buffer";
            return false;
        }
        if (gradOutput.SizeInBytes != PtxConv2DBackwardBiasF32Kernel.GradOutputBytes ||
            gradBias.SizeInBytes != PtxConv2DBackwardBiasF32Kernel.GradBiasBytes)
        {
            DirectPtxLastError = "conv-bwd-bias-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConvBwdBiasKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX conv backward-bias must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv2DBackwardBiasF32Kernel kernel = GetOrCreateDirectPtxConvBwdBiasKernel();
                if (capturing && !_directPtxConvBwdBiasKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX conv backward-bias module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(gradBias, kernel.Blueprint.Tensors[1]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConvBwdBiasDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxConv2DBackwardBias()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX conv backward-bias prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConvBwdBiasKernel();
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

    internal bool TryGetDirectPtxConvBwdBiasAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConvBwdBiasKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxConv2DBackwardBiasF32Kernel GetOrCreateDirectPtxConvBwdBiasKernel()
    {
        if (_directPtxConvBwdBiasKernels.TryGetValue(
                1, out PtxConv2DBackwardBiasF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConvBwdBiasKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv2DBackwardBiasF32Kernel CreateAndCacheDirectPtxConvBwdBiasKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxConvBwdBiasKernels.GetOrAdd(
            1, () => new PtxConv2DBackwardBiasF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxConv2DNchwK1BackwardInputF32Kernel>
        _directPtxConvBwdInputKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConvBwdInputDispatchCount;

    internal long DirectPtxConvBwdInputDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConvBwdInputDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCHW 1x1 convolution backward-input gradient
    /// experiment (dIn[c,y,x] = sum_k W[k,c] * dOut[k,y,x]). Fails closed on any
    /// unsupported contract so the caller runs the established composition.
    /// </summary>
    internal bool TryDirectPtxConv2DBackwardInput(
        IGpuBuffer gradOutput,
        IGpuBuffer weights,
        IGpuBuffer gradInput,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxConv2DNchwK1BackwardInputF32Kernel.Shape)
        {
            DirectPtxLastError = "conv-bwd-input-shape-not-implemented";
            return false;
        }
        if (gradOutput is null || weights is null || gradInput is null)
        {
            DirectPtxLastError = "conv-bwd-input-null-buffer";
            return false;
        }
        if (gradOutput.SizeInBytes != PtxConv2DNchwK1BackwardInputF32Kernel.GradOutputBytes ||
            weights.SizeInBytes != PtxConv2DNchwK1BackwardInputF32Kernel.WeightBytes ||
            gradInput.SizeInBytes != PtxConv2DNchwK1BackwardInputF32Kernel.GradInputBytes)
        {
            DirectPtxLastError = "conv-bwd-input-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConvBwdInputKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX conv backward-input must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv2DNchwK1BackwardInputF32Kernel kernel = GetOrCreateDirectPtxConvBwdInputKernel();
                if (capturing && !_directPtxConvBwdInputKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX conv backward-input module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradInput, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConvBwdInputDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxConv2DBackwardInput()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX conv backward-input prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConvBwdInputKernel();
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

    internal bool TryGetDirectPtxConvBwdInputAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConvBwdInputKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxConv2DNchwK1BackwardInputF32Kernel GetOrCreateDirectPtxConvBwdInputKernel()
    {
        if (_directPtxConvBwdInputKernels.TryGetValue(
                1, out PtxConv2DNchwK1BackwardInputF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConvBwdInputKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv2DNchwK1BackwardInputF32Kernel CreateAndCacheDirectPtxConvBwdInputKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxConvBwdInputKernels.GetOrAdd(
            1, () => new PtxConv2DNchwK1BackwardInputF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxConv2DNchwK1BackwardWeightF32Kernel>
        _directPtxConvBwdWeightKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConvBwdWeightDispatchCount;

    internal long DirectPtxConvBwdWeightDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConvBwdWeightDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCHW 1x1 convolution backward-weight gradient
    /// experiment (dW[k,c] = sum_{y,x} dOut[k,y,x] * in[c,y,x]). Fails closed on any
    /// unsupported contract so the caller runs the established composition.
    /// </summary>
    internal bool TryDirectPtxConv2DBackwardWeight(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer gradWeight,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxConv2DNchwK1BackwardWeightF32Kernel.Shape)
        {
            DirectPtxLastError = "conv-bwd-weight-shape-not-implemented";
            return false;
        }
        if (gradOutput is null || input is null || gradWeight is null)
        {
            DirectPtxLastError = "conv-bwd-weight-null-buffer";
            return false;
        }
        if (gradOutput.SizeInBytes != PtxConv2DNchwK1BackwardWeightF32Kernel.GradOutputBytes ||
            input.SizeInBytes != PtxConv2DNchwK1BackwardWeightF32Kernel.InputBytes ||
            gradWeight.SizeInBytes != PtxConv2DNchwK1BackwardWeightF32Kernel.GradWeightBytes)
        {
            DirectPtxLastError = "conv-bwd-weight-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConvBwdWeightKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX conv backward-weight must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv2DNchwK1BackwardWeightF32Kernel kernel = GetOrCreateDirectPtxConvBwdWeightKernel();
                if (capturing && !_directPtxConvBwdWeightKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX conv backward-weight module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradWeight, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConvBwdWeightDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxConv2DBackwardWeight()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX conv backward-weight prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConvBwdWeightKernel();
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

    internal bool TryGetDirectPtxConvBwdWeightAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConvBwdWeightKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxConv2DNchwK1BackwardWeightF32Kernel GetOrCreateDirectPtxConvBwdWeightKernel()
    {
        if (_directPtxConvBwdWeightKernels.TryGetValue(
                1, out PtxConv2DNchwK1BackwardWeightF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConvBwdWeightKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv2DNchwK1BackwardWeightF32Kernel CreateAndCacheDirectPtxConvBwdWeightKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxConvBwdWeightKernels.GetOrAdd(
            1, () => new PtxConv2DNchwK1BackwardWeightF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxConv1DNclForwardF32Kernel>
        _directPtxConv1DKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConv1DDispatchCount;

    internal long DirectPtxConv1DDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConv1DDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCL 1D convolution experiment (kernel 3, stride 1,
    /// pad 1). Fails closed on any unsupported contract so the caller runs the
    /// established composition.
    /// </summary>
    internal bool TryDirectPtxConv1D(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer output,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxConv1DNclForwardF32Kernel.Shape)
        {
            DirectPtxLastError = "conv1d-shape-not-implemented";
            return false;
        }
        if (input is null || weights is null || output is null)
        {
            DirectPtxLastError = "conv1d-null-buffer";
            return false;
        }
        if (input.SizeInBytes != PtxConv1DNclForwardF32Kernel.InputBytes ||
            weights.SizeInBytes != PtxConv1DNclForwardF32Kernel.WeightBytes ||
            output.SizeInBytes != PtxConv1DNclForwardF32Kernel.OutputBytes)
        {
            DirectPtxLastError = "conv1d-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConv1DKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX Conv1D must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv1DNclForwardF32Kernel kernel = GetOrCreateDirectPtxConv1DKernel();
                if (capturing && !_directPtxConv1DKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX Conv1D module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConv1DDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxConv1D()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX Conv1D prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConv1DKernel();
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

    internal bool TryGetDirectPtxConv1DAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConv1DKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxConv1DNclForwardF32Kernel GetOrCreateDirectPtxConv1DKernel()
    {
        if (_directPtxConv1DKernels.TryGetValue(
                1, out PtxConv1DNclForwardF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConv1DKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv1DNclForwardF32Kernel CreateAndCacheDirectPtxConv1DKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxConv1DKernels.GetOrAdd(
            1, () => new PtxConv1DNclForwardF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxConv1DNclBackwardInputF32Kernel>
        _directPtxConv1DBwdInputKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConv1DBwdInputDispatchCount;

    internal long DirectPtxConv1DBwdInputDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConv1DBwdInputDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCL 1D convolution backward-input gradient
    /// experiment (dIn[ci,l] = sum_co sum_k W[co,ci,k] * dOut[co, l-k+1]). Fails
    /// closed on any unsupported contract so the caller runs the established path.
    /// </summary>
    internal bool TryDirectPtxConv1DBackwardInput(
        IGpuBuffer gradOutput,
        IGpuBuffer weights,
        IGpuBuffer gradInput,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxConv1DNclBackwardInputF32Kernel.Shape)
        {
            DirectPtxLastError = "conv1d-bwd-input-shape-not-implemented";
            return false;
        }
        if (gradOutput is null || weights is null || gradInput is null)
        {
            DirectPtxLastError = "conv1d-bwd-input-null-buffer";
            return false;
        }
        if (gradOutput.SizeInBytes != PtxConv1DNclBackwardInputF32Kernel.GradOutputBytes ||
            weights.SizeInBytes != PtxConv1DNclBackwardInputF32Kernel.WeightBytes ||
            gradInput.SizeInBytes != PtxConv1DNclBackwardInputF32Kernel.GradInputBytes)
        {
            DirectPtxLastError = "conv1d-bwd-input-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConv1DBwdInputKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX Conv1D backward-input must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv1DNclBackwardInputF32Kernel kernel = GetOrCreateDirectPtxConv1DBwdInputKernel();
                if (capturing && !_directPtxConv1DBwdInputKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX Conv1D backward-input module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradInput, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConv1DBwdInputDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxConv1DBackwardInput()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX Conv1D backward-input prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConv1DBwdInputKernel();
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

    internal bool TryGetDirectPtxConv1DBwdInputAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConv1DBwdInputKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxConv1DNclBackwardInputF32Kernel GetOrCreateDirectPtxConv1DBwdInputKernel()
    {
        if (_directPtxConv1DBwdInputKernels.TryGetValue(
                1, out PtxConv1DNclBackwardInputF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConv1DBwdInputKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv1DNclBackwardInputF32Kernel CreateAndCacheDirectPtxConv1DBwdInputKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxConv1DBwdInputKernels.GetOrAdd(
            1, () => new PtxConv1DNclBackwardInputF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxConv1DNclBackwardWeightF32Kernel>
        _directPtxConv1DBwdWeightKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConv1DBwdWeightDispatchCount;

    internal long DirectPtxConv1DBwdWeightDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConv1DBwdWeightDispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCL 1D convolution backward-weight gradient
    /// experiment (dW[co,ci,k] = sum_l dOut[co,l] * in[ci, l+k-1]). Fails closed on
    /// any unsupported contract so the caller runs the established path.
    /// </summary>
    internal bool TryDirectPtxConv1DBackwardWeight(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer gradWeight,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxConv1DNclBackwardWeightF32Kernel.Shape)
        {
            DirectPtxLastError = "conv1d-bwd-weight-shape-not-implemented";
            return false;
        }
        if (gradOutput is null || input is null || gradWeight is null)
        {
            DirectPtxLastError = "conv1d-bwd-weight-null-buffer";
            return false;
        }
        if (gradOutput.SizeInBytes != PtxConv1DNclBackwardWeightF32Kernel.GradOutputBytes ||
            input.SizeInBytes != PtxConv1DNclBackwardWeightF32Kernel.InputBytes ||
            gradWeight.SizeInBytes != PtxConv1DNclBackwardWeightF32Kernel.GradWeightBytes)
        {
            DirectPtxLastError = "conv1d-bwd-weight-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConv1DBwdWeightKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX Conv1D backward-weight must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv1DNclBackwardWeightF32Kernel kernel = GetOrCreateDirectPtxConv1DBwdWeightKernel();
                if (capturing && !_directPtxConv1DBwdWeightKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX Conv1D backward-weight module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradWeight, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConv1DBwdWeightDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxConv1DBackwardWeight()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX Conv1D backward-weight prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConv1DBwdWeightKernel();
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

    internal bool TryGetDirectPtxConv1DBwdWeightAudit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConv1DBwdWeightKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxConv1DNclBackwardWeightF32Kernel GetOrCreateDirectPtxConv1DBwdWeightKernel()
    {
        if (_directPtxConv1DBwdWeightKernels.TryGetValue(
                1, out PtxConv1DNclBackwardWeightF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConv1DBwdWeightKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv1DNclBackwardWeightF32Kernel CreateAndCacheDirectPtxConv1DBwdWeightKernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxConv1DBwdWeightKernels.GetOrAdd(
            1, () => new PtxConv1DNclBackwardWeightF32Kernel(runtime));
    }

    private readonly DirectPtxKernelCache<int, PtxConv2DNchw3x3ForwardF32Kernel>
        _directPtxConv2D3x3Kernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxConv2D3x3DispatchCount;

    internal long DirectPtxConv2D3x3DispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConv2D3x3DispatchCount);

    /// <summary>
    /// Attempts the exact FP32 NCHW general 3x3 convolution experiment (stride 1,
    /// pad 1). Fails closed on any unsupported contract so the caller runs the
    /// established cuDNN/NVRTC composition.
    /// </summary>
    internal bool TryDirectPtxConv2D3x3(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer output,
        DirectPtxConvolutionShape shape)
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.BackendUnavailable;
            return false;
        }
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        if (shape != PtxConv2DNchw3x3ForwardF32Kernel.Shape)
        {
            DirectPtxLastError = "conv2d-3x3-shape-not-implemented";
            return false;
        }
        if (input is null || weights is null || output is null)
        {
            DirectPtxLastError = "conv2d-3x3-null-buffer";
            return false;
        }
        if (input.SizeInBytes != PtxConv2DNchw3x3ForwardF32Kernel.InputBytes ||
            weights.SizeInBytes != PtxConv2DNchw3x3ForwardF32Kernel.WeightBytes ||
            output.SizeInBytes != PtxConv2DNchw3x3ForwardF32Kernel.OutputBytes)
        {
            DirectPtxLastError = "conv2d-3x3-exact-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxConv2D3x3Kernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX Conv2D 3x3 must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv2DNchw3x3ForwardF32Kernel kernel = GetOrCreateDirectPtxConv2D3x3Kernel();
                if (capturing && !_directPtxConv2D3x3Kernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX Conv2D 3x3 module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxConv2D3x3DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxConv2D3x3()
    {
        if (!_directPtxConvolutionOptedIn)
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.FeatureDisabled;
            return false;
        }
        if (!IsAvailable || !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxConvolutionEligibility.ArchitectureNotImplemented;
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX Conv2D 3x3 prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateDirectPtxConv2D3x3Kernel();
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

    internal bool TryGetDirectPtxConv2D3x3Audit(out DirectPtxKernelAudit? audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxConv2D3x3Kernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null;
        return false;
    }

    private PtxConv2DNchw3x3ForwardF32Kernel GetOrCreateDirectPtxConv2D3x3Kernel()
    {
        if (_directPtxConv2D3x3Kernels.TryGetValue(
                1, out PtxConv2DNchw3x3ForwardF32Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConv2D3x3KernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv2DNchw3x3ForwardF32Kernel CreateAndCacheDirectPtxConv2D3x3KernelSlow()
    {
        DirectPtxRuntime runtime = _directPtxRuntime ??
            throw new InvalidOperationException("The direct-PTX runtime is not initialized.");
        return _directPtxConv2D3x3Kernels.GetOrAdd(
            1, () => new PtxConv2DNchw3x3ForwardF32Kernel(runtime));
    }
}
