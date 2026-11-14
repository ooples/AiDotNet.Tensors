// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Autocast 2D convolution (#1549): under an <see cref="AutocastScope"/> of <see cref="PrecisionMode.Float16"/> or
/// <see cref="PrecisionMode.BFloat16"/>, the convolution forward, input gradient and weight gradient run in that 16-bit
/// precision on a backend whose <see cref="IGpuMixedPrecisionConvBackend"/> does the math natively (cuDNN tensor cores
/// on CUDA), as PyTorch autocast runs conv in FP16. Before this the scope never reached a convolution: every autocast
/// conv ran in FP32, and an FP16 conv network trained slower than in FP32 against PyTorch's AMP.
/// </summary>
/// <remarks>
/// <para>
/// The FP32 operands are converted into 16-bit transient scratch, the convolution runs on 16-bit tensors with FP32
/// accumulation (cuDNN's compute type), and the 16-bit result is widened into the op's usual FP32 output buffer. So
/// every tensor the rest of the engine sees stays FP32 -- bias, activation, gradient accumulation and the optimizer are
/// unchanged -- and the weights are converted on every call, so an updated weight is never served stale.
/// </para>
/// <para>
/// Capture-safe: the 16-bit operands come from <see cref="AllocateTransientScratch"/> (stable per-action slots in a
/// compiled step, nothing allocated during a CUDA graph capture) and the conversions are plain kernel launches. cuDNN
/// sizes its workspace during the step's eager pre-pass, before capture.
/// </para>
/// <para>
/// Not taken where it would not pay or would be wrong: outside autocast or for an op autocast keeps in FP32; on a
/// backend whose mixed-precision conv only stores 16-bit around an FP32 convolution (the portable path); under a
/// recording or delegating backend (a deferred graph records ops, while the mixed-precision surface would run them
/// immediately); and when <c>AIDOTNET_AUTOCAST_CONV=0</c>.
/// </para>
/// </remarks>
public partial class DirectGpuTensorEngine
{
    private static readonly bool s_autocastConvEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_AUTOCAST_CONV") != "0";

    /// <summary>Tests: forces the autocast conv route off (false) or follows the default (null).</summary>
    internal static bool? AutocastConvOverride { get; set; }

    private long _autocastConvForwardCount, _autocastConvBackwardDataCount, _autocastConvBackwardFilterCount;

    /// <summary>Autocast convolutions this engine ran in 16-bit: forward, input gradient, weight gradient (test hook).</summary>
    internal (long Forward, long BackwardData, long BackwardFilter) AutocastConvDispatchCounts =>
        (Interlocked.Read(ref _autocastConvForwardCount), Interlocked.Read(ref _autocastConvBackwardDataCount),
            Interlocked.Read(ref _autocastConvBackwardFilterCount));

    /// <summary>The resolved 16-bit route for one autocast convolution.</summary>
    internal readonly struct AutocastConvRoute
    {
        internal AutocastConvRoute(IGpuMixedPrecisionConvBackend conv, IGenericFftBackend convert,
            GpuMixedPrecisionDataType dataType, FftElementType storage)
        {
            Conv = conv;
            Convert = convert;
            DataType = dataType;
            Storage = storage;
        }

        internal IGpuMixedPrecisionConvBackend Conv { get; }
        internal IGenericFftBackend Convert { get; }
        internal GpuMixedPrecisionDataType DataType { get; }
        internal FftElementType Storage { get; }
    }

    /// <summary>
    /// Records the exact convolution precision executed by an eager taped forward pass so
    /// <see cref="Autodiff.BackwardFunctions{T}.Conv2DBackward"/> reuses the forward pass's route even if
    /// <see cref="AutocastScope"/> has exited before backward, and stays in FP32 if the forward pass ran in or
    /// declined to FP32.
    /// </summary>
    internal sealed class RecordedConvPrecision
    {
        internal static readonly RecordedConvPrecision Fp32 = new(hasRoute: false, route: default);

        private RecordedConvPrecision(bool hasRoute, AutocastConvRoute route)
        {
            HasRoute = hasRoute;
            Route = route;
        }

        internal RecordedConvPrecision(AutocastConvRoute route)
            : this(hasRoute: true, route)
        {
        }

        internal bool HasRoute { get; }
        internal AutocastConvRoute Route { get; }
    }

    [ThreadStatic] private static RecordedConvPrecision? t_lastForwardConvPrecision;
    [ThreadStatic] private static RecordedConvPrecision? t_backwardConvPrecisionOverride;

    internal readonly struct BackwardConvPrecisionScope : IDisposable
    {
        private readonly RecordedConvPrecision? _prior;
        private readonly bool _active;

        internal BackwardConvPrecisionScope(RecordedConvPrecision? precision)
        {
            _prior = t_backwardConvPrecisionOverride;
            t_backwardConvPrecisionOverride = precision;
            _active = precision is not null;
        }

        public void Dispose()
        {
            if (_active) t_backwardConvPrecisionOverride = _prior;
        }
    }

    private static RecordedConvPrecision ConsumeLastForwardConvPrecision()
    {
        var recorded = t_lastForwardConvPrecision ?? RecordedConvPrecision.Fp32;
        t_lastForwardConvPrecision = null;
        return recorded;
    }

    internal static BackwardConvPrecisionScope PushBackwardConvPrecision(object[]? savedState, IEngine engine)
    {
        if (savedState is not null)
        {
            for (int i = savedState.Length - 1; i >= 0; i--)
            {
                if (savedState[i] is RecordedConvPrecision recorded)
                    return new BackwardConvPrecisionScope(recorded);
            }
        }
        // Eager CpuEngine / fallback forward nodes record [stride, padding, dilation] without RecordedConvPrecision;
        // keep FP32 if an autocast scope is opened later around backward. Symbolic GraphMode / CompiledTrainingPlan
        // nodes run inside ResidentStepActive and resolve the compiled step's active AutocastScope.
        if (!(engine.TryGet<DirectGpuTensorEngine>(out var residentGpu) && residentGpu.ResidentStepActive))
            return new BackwardConvPrecisionScope(RecordedConvPrecision.Fp32);
        return default;
    }

    /// <summary>
    /// True when this engine would route a convolution under an active <paramref name="precision"/> autocast scope onto
    /// native 16-bit convolution math (used by parity harness baseline selection).
    /// </summary>
    internal bool SupportsEffectiveAutocastConv(PrecisionMode precision)
    {
        if (!(AutocastConvOverride ?? s_autocastConvEnabled)) return false;
        if (GpuExecutionPolicyScope.EffectivePolicyFor(typeof(float)).AccuracyMode == GpuAccuracyMode.PreserveInputType)
            return false;
        if (!TryGetBackendForQuery(out var backend) || backend is DelegatingGpuBackend) return false;
        if (backend is not INativeReducedPrecisionConv { NativeReducedPrecisionMath: true }
            || backend.MixedPrecisionConv is not { } conv) return false;
        if (backend is not IGenericFftBackend convert) return false;
        return precision switch
        {
            PrecisionMode.Float16 => conv.SupportsHalfConv && convert.SupportsFftElementType(FftElementType.Float16),
            PrecisionMode.BFloat16 => conv.SupportsBFloat16Conv && convert.SupportsFftElementType(FftElementType.BFloat16),
            _ => false,
        };
    }

    /// <summary>
    /// The 16-bit route for a convolution under the current autocast scope, or false to keep FP32. Read when the op is
    /// issued (not inside a deferred launch), so a launch that runs later still uses the precision of its scope.
    /// </summary>
    private static bool TryResolveAutocastConvRoute(IDirectGpuBackend backend, out AutocastConvRoute route)
        // Inside a backward that replays a recorded precision, a forward convolution it issues (the input gradient,
        // or any convolution of a create-graph backward) runs in that precision too, not the scope open now.
        => t_backwardConvPrecisionOverride is not null
            ? TryResolveBackwardAutocastConvRoute(backend, out route)
            : TryResolveAmbientAutocastConvRoute(backend, out route);

    private static bool TryResolveAmbientAutocastConvRoute(IDirectGpuBackend backend, out AutocastConvRoute route)
    {
        route = default;
        if (!(AutocastConvOverride ?? s_autocastConvEnabled)) return false;
        if (!AutocastScope.ShouldAutocast("Conv2D")) return false;
        // An explicit PreserveInputType policy keeps every op in its input type, as GpuPrecisionPlanner does for the
        // other autocast ops.
        if (GpuExecutionPolicyScope.EffectivePolicyFor(typeof(float)).AccuracyMode == GpuAccuracyMode.PreserveInputType)
            return false;
        // A wrapper (the deferred-graph recorder) records ops; its mixed-precision surface would run them now.
        if (backend is DelegatingGpuBackend) return false;
        if (backend is not INativeReducedPrecisionConv { NativeReducedPrecisionMath: true }
            || backend.MixedPrecisionConv is not { } conv) return false;
        if (backend is not IGenericFftBackend convert) return false;
        switch (AutocastScope.ActivePrecision)
        {
            // The conversions run in the backend's 16-bit storage kernels; without them the convolution stays FP32.
            case PrecisionMode.Float16 when conv.SupportsHalfConv && convert.SupportsFftElementType(FftElementType.Float16):
                route = new AutocastConvRoute(conv, convert, GpuMixedPrecisionDataType.Half, FftElementType.Float16);
                return true;
            case PrecisionMode.BFloat16 when conv.SupportsBFloat16Conv && convert.SupportsFftElementType(FftElementType.BFloat16):
                route = new AutocastConvRoute(conv, convert, GpuMixedPrecisionDataType.BFloat16, FftElementType.BFloat16);
                return true;
            default:
                return false;
        }
    }

    private static bool TryResolveBackwardAutocastConvRoute(IDirectGpuBackend backend, out AutocastConvRoute route)
    {
        if (t_backwardConvPrecisionOverride is { } recorded)
        {
            if (!recorded.HasRoute
                || backend is DelegatingGpuBackend
                || !ReferenceEquals(recorded.Route.Conv, backend.MixedPrecisionConv))
            {
                route = default;
                return false;
            }
            route = recorded.Route;
            return true;
        }
        return TryResolveAmbientAutocastConvRoute(backend, out route);
    }

    // Floats of scratch that hold `elements` 16-bit values.
    private static int HalfStorageFloats(int elements) => checked((elements + 1) / 2);

    /// <summary>Tests: makes the 16-bit convolutions fail as a cuDNN error would (null: no fault).</summary>
    internal static string? AutocastConvInjectedFault { get; set; }

    // A 16-bit convolution cuDNN or the conversions declined (an algorithm or data type this cuDNN does not support
    // for the shape, a failed workspace allocation): the caller runs the FP32 GPU convolution instead, as before #1549,
    // rather than the error reaching the user or moving the op to the host. Anything else propagates.
    private static bool IsAutocastConvDecline(Exception ex)
        => ex is NotSupportedException
           || (ex is InvalidOperationException && ex.Message.IndexOf("cuDNN", StringComparison.Ordinal) >= 0);

    private bool DeclineAutocastConv(string operation, Exception ex)
    {
        if (ThrowOnGpuKernelFallback) return false;
        GpuLaunchProbe.OnFallback(operation, ex);
        return true;
    }

    private static void ThrowIfAutocastConvFaultInjected()
    {
        if (AutocastConvInjectedFault is { } fault)
            throw new InvalidOperationException("cuDNN " + fault + " failed: injected");
    }

    /// <summary>
    /// Y = conv(X, W) in the autocast precision, written as FP32 into <paramref name="output"/> (fully). False when the
    /// route does not apply or the 16-bit convolution declined; the caller then runs the FP32 convolution, which
    /// rewrites the whole output.
    /// </summary>
    private bool TryAutocastConv2DForward(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer kernel,
        IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight,
        int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH,
        int dilationW)
        => TryAutocastConv2DForward(backend, input, kernel, output, batch, inChannels, inHeight, inWidth, outChannels,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, out _);

    private bool TryAutocastConv2DForward(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer kernel,
        IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight,
        int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH,
        int dilationW, out AutocastConvRoute executedRoute)
    {
        executedRoute = default;
        if (!TryResolveAutocastConvRoute(backend, out var route)) return false;
        int inputCount = checked(batch * inChannels * inHeight * inWidth);
        int kernelCount = checked(outChannels * inChannels * kernelH * kernelW);
        int outputCount = checked(batch * outChannels * outHeight * outWidth);
        OwnedBuffer input16 = default, kernel16 = default, output16 = default;
        try
        {
            input16 = AllocateTransientScratch(backend, HalfStorageFloats(inputCount), fullyWritten: true);
            kernel16 = AllocateTransientScratch(backend, HalfStorageFloats(kernelCount), fullyWritten: true);
            output16 = AllocateTransientScratch(backend, HalfStorageFloats(outputCount), fullyWritten: true);
            route.Convert.ConvertFloatToFftStorage(input, input16.Buffer, inputCount, route.Storage);
            route.Convert.ConvertFloatToFftStorage(kernel, kernel16.Buffer, kernelCount, route.Storage);
            ThrowIfAutocastConvFaultInjected();
            route.Conv.Conv2DMixed(input16.Buffer, kernel16.Buffer, output16.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, route.DataType);
            route.Convert.ConvertFftStorageToFloat(output16.Buffer, output, outputCount, route.Storage);
        }
        catch (Exception ex) when (IsAutocastConvDecline(ex) && DeclineAutocastConv("AutocastConv2DForward", ex))
        {
            return false;
        }
        finally
        {
            output16.DisposeAfterQueuedUse(backend);
            kernel16.DisposeAfterQueuedUse(backend);
            input16.DisposeAfterQueuedUse(backend);
        }
        executedRoute = route;
        Interlocked.Increment(ref _autocastConvForwardCount);
        return true;
    }

    /// <summary>
    /// dX = conv_backward_data(dY, W) in the route's precision, written as FP32 into <paramref name="gradInput"/>. False
    /// when the 16-bit convolution declined; the caller then runs the FP32 one.
    /// </summary>
    private bool TryAutocastConv2DBackwardData(IDirectGpuBackend backend, in AutocastConvRoute route, IGpuBuffer kernel,
        IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH,
        int padW, int dilationH, int dilationW)
    {
        int inputCount = checked(batch * inChannels * inHeight * inWidth);
        int kernelCount = checked(outChannels * inChannels * kernelH * kernelW);
        int outputCount = checked(batch * outChannels * outHeight * outWidth);
        OwnedBuffer kernel16 = default, gradOutput16 = default, gradInput16 = default;
        try
        {
            kernel16 = AllocateTransientScratch(backend, HalfStorageFloats(kernelCount), fullyWritten: true);
            gradOutput16 = AllocateTransientScratch(backend, HalfStorageFloats(outputCount), fullyWritten: true);
            gradInput16 = AllocateTransientScratch(backend, HalfStorageFloats(inputCount), fullyWritten: true);
            route.Convert.ConvertFloatToFftStorage(kernel, kernel16.Buffer, kernelCount, route.Storage);
            route.Convert.ConvertFloatToFftStorage(gradOutput, gradOutput16.Buffer, outputCount, route.Storage);
            ThrowIfAutocastConvFaultInjected();
            route.Conv.Conv2DBackwardDataMixed(kernel16.Buffer, gradOutput16.Buffer, gradInput16.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, route.DataType);
            route.Convert.ConvertFftStorageToFloat(gradInput16.Buffer, gradInput, inputCount, route.Storage);
        }
        catch (Exception ex) when (IsAutocastConvDecline(ex) && DeclineAutocastConv("AutocastConv2DBackwardData", ex))
        {
            return false;
        }
        finally
        {
            gradInput16.DisposeAfterQueuedUse(backend);
            gradOutput16.DisposeAfterQueuedUse(backend);
            kernel16.DisposeAfterQueuedUse(backend);
        }
        Interlocked.Increment(ref _autocastConvBackwardDataCount);
        return true;
    }

    /// <summary>
    /// dW = conv_backward_filter(X, dY) in the route's precision, written as FP32 into <paramref name="gradKernel"/>.
    /// False when the 16-bit convolution declined; the caller then runs the FP32 one.
    /// </summary>
    private bool TryAutocastConv2DBackwardFilter(IDirectGpuBackend backend, in AutocastConvRoute route, IGpuBuffer input,
        IGpuBuffer gradOutput, IGpuBuffer gradKernel, int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH,
        int padW, int dilationH, int dilationW)
    {
        int inputCount = checked(batch * inChannels * inHeight * inWidth);
        int kernelCount = checked(outChannels * inChannels * kernelH * kernelW);
        int outputCount = checked(batch * outChannels * outHeight * outWidth);
        OwnedBuffer input16 = default, gradOutput16 = default, gradKernel16 = default;
        try
        {
            input16 = AllocateTransientScratch(backend, HalfStorageFloats(inputCount), fullyWritten: true);
            gradOutput16 = AllocateTransientScratch(backend, HalfStorageFloats(outputCount), fullyWritten: true);
            gradKernel16 = AllocateTransientScratch(backend, HalfStorageFloats(kernelCount), fullyWritten: true);
            route.Convert.ConvertFloatToFftStorage(input, input16.Buffer, inputCount, route.Storage);
            route.Convert.ConvertFloatToFftStorage(gradOutput, gradOutput16.Buffer, outputCount, route.Storage);
            ThrowIfAutocastConvFaultInjected();
            route.Conv.Conv2DBackwardFilterMixed(input16.Buffer, gradOutput16.Buffer, gradKernel16.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, route.DataType);
            route.Convert.ConvertFftStorageToFloat(gradKernel16.Buffer, gradKernel, kernelCount, route.Storage);
        }
        catch (Exception ex) when (IsAutocastConvDecline(ex) && DeclineAutocastConv("AutocastConv2DBackwardFilter", ex))
        {
            return false;
        }
        finally
        {
            gradKernel16.DisposeAfterQueuedUse(backend);
            gradOutput16.DisposeAfterQueuedUse(backend);
            input16.DisposeAfterQueuedUse(backend);
        }
        Interlocked.Increment(ref _autocastConvBackwardFilterCount);
        return true;
    }
}
