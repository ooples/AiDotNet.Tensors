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
}
