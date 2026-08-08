using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private const int DirectPtxConvolutionKernelKey = 1;
    private readonly bool _directPtxConvolutionOptedIn =
        DirectPtxFeatureGate.IsConvolutionEnabled;
    private readonly DirectPtxKernelCache<int, PtxFusedConv2DNchwK1Kernel>
        _directPtxConvolutionKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<int, PtxConv2DNchwK1TiledKernel>
        _directPtxTiledConvolutionKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private DirectPtxConvolutionVariant? _directPtxConvolutionPlan;
    private long _directPtxConvolutionDispatchCount;

    internal bool IsDirectPtxConvolutionEnabled =>
        _directPtxConvolutionOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor);

    internal long DirectPtxConvolutionDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxConvolutionDispatchCount);

    internal int DirectPtxConvolutionPinnedKernelCount
    {
        get
        {
            lock (_directPtxLock)
                return _directPtxConvolutionKernels.PinnedCount +
                    _directPtxTiledConvolutionKernels.PinnedCount;
        }
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
            lock (_directPtxLock)
            {
                if (capturing &&
                    (!_directPtxConvolutionPlan.HasValue ||
                     !IsDirectPtxConvolutionKernelLoaded(_directPtxConvolutionPlan.Value)))
                {
                    DirectPtxLastError =
                        "Direct PTX convolution must be prewarmed before CUDA graph capture.";
                    return false;
                }

                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                DirectPtxConvolutionVariant selected = _directPtxConvolutionPlan ??=
                    ResolveDirectPtxConvolutionPlanSlow(input, weights, bias, output, shape);
                EnsureDirectPtxConvolutionKernelLoaded(selected);
                if (capturing && !PinDirectPtxConvolutionKernel(selected))
                    throw new InvalidOperationException(
                        "Could not pin the selected direct-PTX convolution module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    LaunchDirectPtxConvolution(selected, input, weights, bias, output);
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
                if (!_directPtxConvolutionPlan.HasValue)
                {
                    DirectPtxConvolutionVariant selected = DirectPtxConvolutionVariant.Direct;
                    if (DirectPtxFeatureGate.IsAutotuneEnabled)
                        DirectPtxConvolutionAutotuner.TryLoad(
                            _directPtxRuntime,
                            PtxFusedConv2DNchwK1Kernel.Batch,
                            PtxFusedConv2DNchwK1Kernel.OutputChannels,
                            PtxFusedConv2DNchwK1Kernel.InputChannels,
                            PtxFusedConv2DNchwK1Kernel.SpatialElements,
                            out selected);
                    _directPtxConvolutionPlan = selected;
                }
                EnsureDirectPtxConvolutionKernelLoaded(_directPtxConvolutionPlan.Value);
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
            if (_directPtxConvolutionPlan is { } selected)
            {
                if (selected.IsTiled &&
                    _directPtxTiledConvolutionKernels.TryGetValue(
                        selected.Tile, out PtxConv2DNchwK1TiledKernel? tiled))
                {
                    audit = tiled.Audit;
                    return true;
                }
                if (!selected.IsTiled &&
                    _directPtxConvolutionKernels.TryGetValue(
                        DirectPtxConvolutionKernelKey,
                        out PtxFusedConv2DNchwK1Kernel? direct))
                {
                    audit = direct.Audit;
                    return true;
                }
            }
        }
        audit = null!;
        return false;
    }

    // Keep closure-bearing measurement code off the resident dispatch path.
    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private DirectPtxConvolutionVariant ResolveDirectPtxConvolutionPlanSlow(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer bias,
        IGpuBuffer output,
        DirectPtxConvolutionShape shape)
    {
        long operations = checked(
            2L * shape.Batch * shape.OutputChannels * shape.InputChannels *
            shape.OutputHeight * shape.OutputWidth);
        return DirectPtxConvolutionAutotuner.Resolve(
            _directPtxRuntime!,
            shape.Batch,
            shape.OutputChannels,
            shape.InputChannels,
            checked(shape.OutputHeight * shape.OutputWidth),
            candidate =>
            {
                EnsureDirectPtxConvolutionKernelLoaded(candidate);
                return GpuAutotuneMeasurement.AdaptiveStableGflops(
                    launchesPerSample =>
                    {
                        lock (GpuDispatchLock)
                            return _directPtxRuntime!.MeasureKernelSamples(
                                () => LaunchDirectPtxConvolution(
                                    candidate, input, weights, bias, output),
                                warmup: 3, samples: 20, launchesPerSample);
                    },
                    operations);
            },
            DirectPtxFeatureGate.IsAutotuneEnabled);
    }

    private bool IsDirectPtxConvolutionKernelLoaded(DirectPtxConvolutionVariant selected) =>
        selected.IsTiled
            ? _directPtxTiledConvolutionKernels.TryGetValue(selected.Tile, out _)
            : _directPtxConvolutionKernels.TryGetValue(DirectPtxConvolutionKernelKey, out _);

    private void EnsureDirectPtxConvolutionKernelLoaded(DirectPtxConvolutionVariant selected)
    {
        if (selected.IsTiled)
            _ = GetOrCreateDirectPtxTiledConvolutionKernel(selected.Tile);
        else
            _ = GetOrCreateDirectPtxConvolutionKernel();
    }

    private bool PinDirectPtxConvolutionKernel(DirectPtxConvolutionVariant selected) =>
        selected.IsTiled
            ? _directPtxTiledConvolutionKernels.Pin(selected.Tile)
            : _directPtxConvolutionKernels.Pin(DirectPtxConvolutionKernelKey);

    private void LaunchDirectPtxConvolution(
        DirectPtxConvolutionVariant selected,
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer bias,
        IGpuBuffer output)
    {
        if (selected.IsTiled)
        {
            PtxConv2DNchwK1TiledKernel kernel =
                GetOrCreateDirectPtxTiledConvolutionKernel(selected.Tile);
            kernel.Launch(
                DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
            return;
        }

        PtxFusedConv2DNchwK1Kernel direct = GetOrCreateDirectPtxConvolutionKernel();
        direct.Launch(
            DirectPtxTensorView.Create(input, direct.Blueprint.Tensors[0]),
            DirectPtxTensorView.Create(weights, direct.Blueprint.Tensors[1]),
            DirectPtxTensorView.Create(bias, direct.Blueprint.Tensors[2]),
            DirectPtxTensorView.Create(output, direct.Blueprint.Tensors[3]));
    }

    private PtxFusedConv2DNchwK1Kernel GetOrCreateDirectPtxConvolutionKernel()
    {
        if (_directPtxConvolutionKernels.TryGetValue(
                DirectPtxConvolutionKernelKey,
                out PtxFusedConv2DNchwK1Kernel? existing))
            return existing;
        return CreateAndCacheDirectPtxConvolutionKernelSlow();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedConv2DNchwK1Kernel CreateAndCacheDirectPtxConvolutionKernelSlow()
    {
        var created = new PtxFusedConv2DNchwK1Kernel(_directPtxRuntime!);
        return _directPtxConvolutionKernels.AddOrGetExisting(
            DirectPtxConvolutionKernelKey, created);
    }

    private PtxConv2DNchwK1TiledKernel GetOrCreateDirectPtxTiledConvolutionKernel(int tile)
    {
        if (_directPtxTiledConvolutionKernels.TryGetValue(
                tile, out PtxConv2DNchwK1TiledKernel? existing))
            return existing;
        return CreateAndCacheDirectPtxTiledConvolutionKernelSlow(tile);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxConv2DNchwK1TiledKernel CreateAndCacheDirectPtxTiledConvolutionKernelSlow(int tile)
    {
        var shape = new Conv2DTiledShape(
            PtxFusedConv2DNchwK1Kernel.Batch,
            PtxFusedConv2DNchwK1Kernel.OutputChannels,
            PtxFusedConv2DNchwK1Kernel.InputChannels,
            PtxFusedConv2DNchwK1Kernel.SpatialElements,
            tile);
        var created = new PtxConv2DNchwK1TiledKernel(_directPtxRuntime!, shape);
        return _directPtxTiledConvolutionKernels.AddOrGetExisting(tile, created);
    }
}
