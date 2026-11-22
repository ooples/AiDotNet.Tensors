// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
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
        private readonly NarrowedOperandShare? _share;

        internal BackwardConvPrecisionScope(RecordedConvPrecision? precision, bool shareNarrowedOperands = false)
        {
            _prior = t_backwardConvPrecisionOverride;
            t_backwardConvPrecisionOverride = precision;
            _active = precision is not null;
            _share = shareNarrowedOperands ? NarrowedOperandShare.Enter() : null;
        }

        public void Dispose()
        {
            _share?.Exit();
            if (_active) t_backwardConvPrecisionOverride = _prior;
        }
    }

    /// <summary>
    /// The 16-bit channels-last copies one convolution's backward narrowed, for its other pass to read instead of
    /// narrowing the same operand again: the data and filter gradients both read dY (a transposed convolution's input
    /// and kernel gradients too). Lives for one backward function, which reads its operands and writes only its
    /// gradients, so a copy cannot go stale inside it; and one backward function is one compiled action, whose transient
    /// scratch rents are distinct slots, so a shared copy is not overwritten by the other pass's scratch either. An owned
    /// (uncompiled) copy is released when the scope ends.
    /// </summary>
    private sealed class NarrowedOperandShare
    {
        [ThreadStatic] private static NarrowedOperandShare? t_current;

        private readonly NarrowedOperandShare? _prior;
        private readonly List<(IGpuBuffer Source, int Outer, int Channels, int Spatial, int Type, OwnedBuffer Copy,
            IDirectGpuBackend Backend)> _copies = new(2);

        private NarrowedOperandShare(NarrowedOperandShare? prior) => _prior = prior;

        internal static NarrowedOperandShare? Current => t_current;

        internal static NarrowedOperandShare Enter() => t_current = new NarrowedOperandShare(t_current);

        internal void Exit()
        {
            t_current = _prior;
            foreach (var copy in _copies) copy.Copy.DisposeAfterQueuedUse(copy.Backend);
            _copies.Clear();
        }

        internal IGpuBuffer? Find(IGpuBuffer source, int outer, int channels, int spatial, int type)
        {
            foreach (var copy in _copies)
                if (ReferenceEquals(copy.Source, source) && copy.Outer == outer && copy.Channels == channels
                    && copy.Spatial == spatial && copy.Type == type)
                    return copy.Copy.Buffer;
            return null;
        }

        // Takes the copy over from the pass that narrowed it: the pass's own release becomes a no-op.
        internal void Adopt(IGpuBuffer source, int outer, int channels, int spatial, int type, OwnedBuffer copy,
            IDirectGpuBackend backend)
        {
            var kept = new OwnedBuffer(copy.Buffer, copy.OwnsBuffer);
            copy.RelinquishOwnership();
            _copies.Add((source, outer, channels, spatial, type, kept, backend));
        }
    }

    // Narrows an operand into its pass's scratch, or hands back the copy another pass of the same backward narrowed.
    // An operand already channels-last (an activation a channels-last plan keeps so) is only cast: a source buffer has
    // one layout, so a shared copy is the same whichever way it was made.
    private static IGpuBuffer NarrowOperand(IChannelsLastMixedConvBackend channelsLast, IDirectGpuBackend backend,
        IGpuBuffer source, OwnedBuffer scratch, (int Outer, int Channels, int Spatial) shape,
        GpuMixedPrecisionDataType type, in AutocastConvRoute route = default, bool sourceChannelsLast = false)
    {
        var share = NarrowedOperandShare.Current;
        if (share?.Find(source, shape.Outer, shape.Channels, shape.Spatial, (int)type) is { } shared)
        {
            Interlocked.Increment(ref s_sharedNarrowings);
            return shared;
        }
        if (sourceChannelsLast)
            route.Convert.ConvertFloatToFftStorage(source, scratch.Buffer, checked(shape.Outer * shape.Channels * shape.Spatial),
                route.Storage);
        else
            channelsLast.NarrowToChannelsLast(source, scratch.Buffer, shape.Outer, shape.Channels, shape.Spatial, type);
        share?.Adopt(source, shape.Outer, shape.Channels, shape.Spatial, (int)type, scratch, backend);
        return scratch.Buffer;
    }

    /// <summary>Narrowings skipped because another pass of the same backward had narrowed that operand.</summary>
    internal static long SharedNarrowings => Interlocked.Read(ref s_sharedNarrowings);
    private static long s_sharedNarrowings;

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
                    return new BackwardConvPrecisionScope(recorded, shareNarrowedOperands: true);
            }
        }
        // Eager CpuEngine / fallback forward nodes record [stride, padding, dilation] without RecordedConvPrecision;
        // keep FP32 if an autocast scope is opened later around backward. Symbolic GraphMode / CompiledTrainingPlan
        // nodes run inside ResidentStepActive and resolve the compiled step's active AutocastScope.
        if (!(engine.TryGet<DirectGpuTensorEngine>(out var residentGpu) && residentGpu.ResidentStepActive))
            return new BackwardConvPrecisionScope(RecordedConvPrecision.Fp32, shareNarrowedOperands: true);
        return new BackwardConvPrecisionScope(null, shareNarrowedOperands: true);
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
    private static bool TryResolveAutocastConvRoute(IDirectGpuBackend backend, out AutocastConvRoute route,
        string operation = "Conv2D")
        // Inside a backward that replays a recorded precision, a forward convolution it issues (the input gradient,
        // or any convolution of a create-graph backward) runs in that precision too, not the scope open now.
        => t_backwardConvPrecisionOverride is not null
            ? TryResolveBackwardAutocastConvRoute(backend, out route, operation)
            : TryResolveAmbientAutocastConvRoute(backend, out route, operation);

    private static bool TryResolveAmbientAutocastConvRoute(IDirectGpuBackend backend, out AutocastConvRoute route,
        string operation = "Conv2D")
    {
        route = default;
        if (!(AutocastConvOverride ?? s_autocastConvEnabled)) return false;
        if (!AutocastScope.ShouldAutocast(operation)) return false;
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

    private static bool TryResolveBackwardAutocastConvRoute(IDirectGpuBackend backend, out AutocastConvRoute route,
        string operation = "Conv2D")
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
        return TryResolveAmbientAutocastConvRoute(backend, out route, operation);
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
            if (!TryAutocastConvChannelsLast(backend, route, 0, input, kernel, output, input16, kernel16,
                    output16.Buffer, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH,
                    kernelW, strideH, strideW, padH, padW, dilationH, dilationW))
            {
                route.Convert.ConvertFloatToFftStorage(input, input16.Buffer, inputCount, route.Storage);
                route.Convert.ConvertFloatToFftStorage(kernel, kernel16.Buffer, kernelCount, route.Storage);
                ThrowIfAutocastConvFaultInjected();
                route.Conv.Conv2DMixed(input16.Buffer, kernel16.Buffer, output16.Buffer,
                    batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                    kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, route.DataType);
                route.Convert.ConvertFftStorageToFloat(output16.Buffer, output, outputCount, route.Storage);
            }
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
            if (!TryAutocastConvChannelsLast(backend, route, 1, kernel, gradOutput, gradInput, kernel16,
                    gradOutput16, gradInput16.Buffer, batch, inChannels, inHeight, inWidth, outChannels, outHeight,
                    outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW))
            {
                route.Convert.ConvertFloatToFftStorage(kernel, kernel16.Buffer, kernelCount, route.Storage);
                route.Convert.ConvertFloatToFftStorage(gradOutput, gradOutput16.Buffer, outputCount, route.Storage);
                ThrowIfAutocastConvFaultInjected();
                route.Conv.Conv2DBackwardDataMixed(kernel16.Buffer, gradOutput16.Buffer, gradInput16.Buffer,
                    batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                    kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, route.DataType);
                route.Convert.ConvertFftStorageToFloat(gradInput16.Buffer, gradInput, inputCount, route.Storage);
            }
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
            if (!TryAutocastConvChannelsLast(backend, route, 2, input, gradOutput, gradKernel, input16,
                    gradOutput16, gradKernel16.Buffer, batch, inChannels, inHeight, inWidth, outChannels, outHeight,
                    outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW))
            {
                route.Convert.ConvertFloatToFftStorage(input, input16.Buffer, inputCount, route.Storage);
                route.Convert.ConvertFloatToFftStorage(gradOutput, gradOutput16.Buffer, outputCount, route.Storage);
                ThrowIfAutocastConvFaultInjected();
                route.Conv.Conv2DBackwardFilterMixed(input16.Buffer, gradOutput16.Buffer, gradKernel16.Buffer,
                    batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                    kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, route.DataType);
                route.Convert.ConvertFftStorageToFloat(gradKernel16.Buffer, gradKernel, kernelCount, route.Storage);
            }
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

    // Shapes whose channels-last 16-bit convolution the backend declined (no kernel for that layout): later calls go
    // straight to the NCHW route instead of converting their operands first.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        ((int, int, int, int, int, int, int), (int, int, int, int, int, int, int), (int, int, int)), byte>
        s_channelsLastAutocastDeclined = new();

    /// <summary>
    /// One autocast convolution pass staged channels-last (<see cref="IChannelsLastMixedConvBackend"/>): the FP32 NCHW
    /// operands are narrowed into 16-bit NHWC scratch (filters [k, r, s, c]) in one pass each, the convolution runs on
    /// the layout the backend's 16-bit kernels read natively, and the result is widened back into FP32 NCHW. Operands
    /// follow the backend's pass convention (0: x, w -> y; 1: w, dy -> dx; 2: x, dy -> dw); the 16-bit scratch is the
    /// NCHW route's. False when the backend does not stage channels-last or declined the shape; the NCHW route then
    /// reconverts the operands into the same scratch.
    /// </summary>
    /// <summary>Whether an autocast scope active now runs this backend's convolutions channels-last in 16 bits.</summary>
    private static bool AutocastRunsChannelsLast(IDirectGpuBackend backend)
        => TryResolveAutocastConvRoute(backend, out var route)
           && route.Conv is IChannelsLastMixedConvBackend channelsLast
           && channelsLast.SupportsChannelsLastMixedConv(route.DataType);

    /// <summary>Autocast channels-last passes run (tests, diagnostics).</summary>
    internal static long AutocastChannelsLastLayoutPasses => Interlocked.Read(ref s_autocastChannelsLastLayoutPasses);
    private static long s_autocastChannelsLastLayoutPasses;

    /// <summary>
    /// One pass of a channels-last plan's convolution in the autocast precision: each activation operand cast in its
    /// layout (<paramref name="inputNhwc"/> for x / dx, <paramref name="outputNhwc"/> for y / dy), the filter narrowed and
    /// transposed, the FP32 result written in its operand's layout (a filter gradient as [k, c, r, s]). False -- nothing
    /// written -- when no 16-bit route applies or it declined; the caller runs the FP32 pass.
    /// </summary>
    private bool TryAutocastConvolutionChannelsLast(IDirectGpuBackend backend, int pass, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer c, in ChannelsLastConvGeometry g, bool inputNhwc, bool outputNhwc)
    {
        if (pass is < 0 or > 2
            || !TryResolveAutocastConvRoute(backend, out var route)
            || route.Conv is not IChannelsLastMixedConvBackend channelsLast
            || !channelsLast.SupportsChannelsLastMixedConv(route.DataType))
            return false;
        var key = ((pass, g.Batch, g.Channels, g.Height, g.Width, g.OutChannels, g.OutHeight),
            (g.OutWidth, g.KernelH, g.KernelW, g.StrideH, g.StrideW, g.PadH, g.PadW), (g.DilationH, g.DilationW, (int)route.DataType));
        if (s_channelsLastAutocastDeclined.ContainsKey(key)) return false;
        var x = (Outer: g.Batch, Channels: g.Channels, Spatial: g.Height * g.Width);
        var w = (Outer: g.OutChannels, Channels: g.Channels, Spatial: g.KernelH * g.KernelW);
        var y = (Outer: g.Batch, Channels: g.OutChannels, Spatial: g.OutHeight * g.OutWidth);
        // Per pass: the operands' shapes and whether each is a channels-last activation (a filter never is).
        var (aShape, aNhwc, bShape, bNhwc, cShape, cNhwc) = pass switch
        {
            0 => (x, inputNhwc, w, false, y, outputNhwc),
            1 => (w, false, y, outputNhwc, x, inputNhwc),
            _ => (x, inputNhwc, y, outputNhwc, w, false),
        };
        var type = route.DataType;
        OwnedBuffer a16 = default, b16 = default, c16 = default;
        try
        {
            a16 = AllocateTransientScratch(backend, HalfStorageFloats(aShape.Outer * aShape.Channels * aShape.Spatial), fullyWritten: true);
            b16 = AllocateTransientScratch(backend, HalfStorageFloats(bShape.Outer * bShape.Channels * bShape.Spatial), fullyWritten: true);
            int cCount = checked(cShape.Outer * cShape.Channels * cShape.Spatial);
            c16 = AllocateTransientScratch(backend, HalfStorageFloats(cCount), fullyWritten: true);
            var aNarrow = NarrowOperand(channelsLast, backend, a, a16, aShape, type, route, aNhwc);
            var bNarrow = NarrowOperand(channelsLast, backend, b, b16, bShape, type, route, bNhwc);
            ThrowIfAutocastConvFaultInjected();
            if (!channelsLast.TryConv2DMixedChannelsLast(pass, aNarrow, bNarrow, c16.Buffer, g.Batch, g.Channels, g.Height,
                    g.Width, g.OutChannels, g.OutHeight, g.OutWidth, g.KernelH, g.KernelW, g.StrideH, g.StrideW, g.PadH,
                    g.PadW, g.DilationH, g.DilationW, type, out bool shapeUnsupported))
            {
                if (shapeUnsupported) s_channelsLastAutocastDeclined.TryAdd(key, 0);
                return false;
            }
            if (cNhwc) route.Convert.ConvertFftStorageToFloat(c16.Buffer, c, cCount, route.Storage);
            else channelsLast.WidenFromChannelsLast(c16.Buffer, c, cShape.Outer, cShape.Channels, cShape.Spatial, type);
        }
        catch (Exception ex) when (IsAutocastConvDecline(ex) && DeclineAutocastConv("AutocastConvChannelsLast", ex))
        {
            return false;
        }
        finally
        {
            c16.DisposeAfterQueuedUse(backend);
            b16.DisposeAfterQueuedUse(backend);
            a16.DisposeAfterQueuedUse(backend);
        }
        Interlocked.Increment(ref s_autocastChannelsLastLayoutPasses);
        return true;
    }

    private static bool TryAutocastConvChannelsLast(IDirectGpuBackend backend, in AutocastConvRoute route, int pass,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, OwnedBuffer a16Scratch, OwnedBuffer b16Scratch, IGpuBuffer c16, int batch, int inChannels, int inHeight,
        int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW,
        int padH, int padW, int dilationH, int dilationW)
    {
        if (route.Conv is not IChannelsLastMixedConvBackend channelsLast
            || !channelsLast.SupportsChannelsLastMixedConv(route.DataType))
            return false;
        var key = ((pass, batch, inChannels, inHeight, inWidth, outChannels, outHeight),
            (outWidth, kernelH, kernelW, strideH, strideW, padH, padW), (dilationH, dilationW, (int)route.DataType));
        if (s_channelsLastAutocastDeclined.ContainsKey(key)) return false;

        // Each operand as [outer][channels][spatial]: an activation's outer extent is the batch, a filter's its k.
        var x = (Outer: batch, Channels: inChannels, Spatial: inHeight * inWidth);
        var w = (Outer: outChannels, Channels: inChannels, Spatial: kernelH * kernelW);
        var y = (Outer: batch, Channels: outChannels, Spatial: outHeight * outWidth);
        var (aShape, bShape, cShape) = pass switch
        {
            0 => (x, w, y),
            1 => (w, y, x),
            _ => (x, y, w),
        };
        var type = route.DataType;
        // Shared only inside a convolution's backward function (outside one there is no share).
        var a16 = NarrowOperand(channelsLast, backend, a, a16Scratch, aShape, type);
        var b16 = NarrowOperand(channelsLast, backend, b, b16Scratch, bShape, type);
        ThrowIfAutocastConvFaultInjected();
        if (!channelsLast.TryConv2DMixedChannelsLast(pass, a16, b16, c16, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW,
                type, out bool shapeUnsupported))
        {
            if (shapeUnsupported) s_channelsLastAutocastDeclined.TryAdd(key, 0);
            return false;
        }
        channelsLast.WidenFromChannelsLast(c16, c, cShape.Outer, cShape.Channels, cShape.Spatial, type);
        return true;
    }

    // A transposed convolution x [n, ci, h, w] -> y [n, co, oh, ow] with kernel [ci, co, kh, kw] is the data gradient of
    // the convolution y -> x whose filter is that kernel read as [k = ci, c = co, kh, kw]. Its three passes are that
    // convolution's with the roles swapped: forward = the convolution's data gradient, input gradient = the
    // convolution's forward, kernel gradient = the convolution's filter gradient with input and output gradient
    // exchanged. The output padding is implicit in oh / ow (cuDNN reads the extents it is given).

    /// <summary>y = conv_transpose(x, W) in the autocast precision, written as FP32 into <paramref name="output"/>.</summary>
    private bool TryAutocastConvTranspose2DForward(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer kernel,
        IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight,
        int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
        out AutocastConvRoute executedRoute)
    {
        executedRoute = default;
        if (!TryResolveAutocastConvRoute(backend, out var route, "ConvTranspose2D")) return false;
        if (!TryAutocastConv2DBackwardData(backend, route, kernel, gradOutput: input, gradInput: output, batch,
                inChannels: outChannels, inHeight: outHeight, inWidth: outWidth, outChannels: inChannels,
                outHeight: inHeight, outWidth: inWidth, kernelH, kernelW, strideH, strideW, padH, padW, 1, 1))
            return false;
        executedRoute = route;
        return true;
    }

    /// <summary>dX of a transposed convolution (the convolution's forward of dY) in the autocast precision.</summary>
    private bool TryAutocastConvTranspose2DBackwardInput(IDirectGpuBackend backend, IGpuBuffer gradOutput,
        IGpuBuffer kernel, IGpuBuffer gradInput, int batch, int inChannels, int inHeight, int inWidth, int outChannels,
        int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (!TryResolveBackwardAutocastConvRoute(backend, out var route, "ConvTranspose2D")) return false;
        using var replay = new BackwardConvPrecisionScope(new RecordedConvPrecision(route));
        return TryAutocastConv2DForward(backend, gradOutput, kernel, gradInput, batch, inChannels: outChannels,
            inHeight: outHeight, inWidth: outWidth, outChannels: inChannels, outHeight: inHeight, outWidth: inWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, 1, 1);
    }

    /// <summary>dW of a transposed convolution (the convolution's filter gradient, x and dY exchanged).</summary>
    private bool TryAutocastConvTranspose2DBackwardKernel(IDirectGpuBackend backend, IGpuBuffer input,
        IGpuBuffer gradOutput, IGpuBuffer gradKernel, int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH,
        int padW)
    {
        if (!TryResolveBackwardAutocastConvRoute(backend, out var route, "ConvTranspose2D")) return false;
        return TryAutocastConv2DBackwardFilter(backend, route, input: gradOutput, gradOutput: input, gradKernel, batch,
            inChannels: outChannels, inHeight: outHeight, inWidth: outWidth, outChannels: inChannels,
            outHeight: inHeight, outWidth: inWidth, kernelH, kernelW, strideH, strideW, padH, padW, 1, 1);
    }
}
