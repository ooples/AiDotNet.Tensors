using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // The promoted register-blocked specialization: ResNet c64 1x1
    // (N32/C64/56x56/K64), which beats cuDNN's best by ~1.60x on SM86.
    private static readonly Conv2DRegBlockShape RegBlockedC64Shape =
        new(32, 64, 64, 3136, 64, 64, 16, 4, 4);

    private readonly DirectPtxKernelCache<int, PtxConv2DNchwK1RegBlockedKernel>
        _directPtxRegBlockedConvKernels = new(Math.Max(2, DirectPtxFeatureGate.CacheCapacity / 4));
    private long _directPtxRegBlockedConvDispatchCount;

    internal long DirectPtxRegBlockedConvDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxRegBlockedConvDispatchCount);

    internal int DirectPtxRegBlockedConvPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRegBlockedConvKernels.PinnedCount; }
    }

    /// <summary>
    /// Attempts the promoted register-blocked FP32 NCHW 1x1 conv + bias + ReLU
    /// (the ResNet c64 specialization). Every other contract fails closed before
    /// module lookup so the caller runs the established cuDNN/NVRTC composition.
    /// </summary>
    internal bool TryDirectPtxRegBlockedConv2DBiasRelu(
        IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, IGpuBuffer output,
        int batch, int inChannels, int height, int width, int outChannels)
    {
        if (!_directPtxConvolutionOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "Register-blocked convolution is not enabled/available.";
            return false;
        }

        Conv2DRegBlockShape s = RegBlockedC64Shape;
        if (batch != s.Batch || inChannels != s.InputChannels ||
            (long)height * width != s.Spatial || outChannels != s.OutputChannels)
        {
            DirectPtxLastError = "Contract is not the promoted register-blocked specialization.";
            return false;
        }
        if (input is null || weights is null || bias is null || output is null ||
            input.Handle == IntPtr.Zero || weights.Handle == IntPtr.Zero ||
            bias.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
        {
            DirectPtxLastError = "Register-blocked convolution requires four non-null device buffers.";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            const int key = 1;
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxRegBlockedConvKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Register-blocked convolution must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxConv2DNchwK1RegBlockedKernel kernel = GetOrCreateRegBlockedConvKernel();
                if (capturing && !_directPtxRegBlockedConvKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the register-blocked convolution module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxRegBlockedConvDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxRegBlockedConv2D()
    {
        if (!_directPtxConvolutionOptedIn || !IsAvailable ||
            !DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "Register-blocked convolution is not enabled/available.";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Register-blocked convolution prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateRegBlockedConvKernel();
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

    internal bool TryGetDirectPtxRegBlockedConvAudit(out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxRegBlockedConvKernels.TryGetValue(1, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private PtxConv2DNchwK1RegBlockedKernel GetOrCreateRegBlockedConvKernel()
    {
        if (_directPtxRegBlockedConvKernels.TryGetValue(1, out var existing))
            return existing;
        return CreateAndCacheRegBlockedConvKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv2DNchwK1RegBlockedKernel CreateAndCacheRegBlockedConvKernelSlow() =>
        _directPtxRegBlockedConvKernels.GetOrAdd(
            1, () => new PtxConv2DNchwK1RegBlockedKernel(_directPtxRuntime!, RegBlockedC64Shape));
}
