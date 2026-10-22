// Copyright (c) AiDotNet. All rights reserved.
// #1369: Winograd F(2x2, 3x3) forward and input gradient and F(3x3, 2x2) weight gradient on Metal, through the shared
// WinogradF23Pipeline: the tile transforms (Metal Shading Language emitted from WinogradF23KernelTemplate) around
// BatchedGemm. Backward convolution gets tuned slots here for the first time, with the direct kernels it always ran as
// the references, so Winograd is served only on the shape classes where it measures faster.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL instance, whose WinogradF23PortTests run on a real device.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IWinogradF23Transforms, IWinogradFusedFilterGradient
{
    private const string WinogradLibraryName = "Winograd";
    private const string WinogradFusedLibraryName = "WinogradFused";
    private IntPtr _winogradLibrary, _winogradFusedLibrary;
    private bool? _winogradReady, _winogradFusedReady;
    private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardDataSlot, _conv2DBackwardFilterSlot;

    // The library, compiled on first need. A compile the driver rejects turns three tuned candidates off, and says why.
    private bool WinogradKernelsReady
    {
        get
        {
            if (_winogradReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_winogradReady is null)
                {
                    try
                    {
                        _winogradLibrary = _shaderLibrary.CompileLibrary(WinogradLibraryName,
                            WinogradF23KernelTemplate.GetModuleSource(WinogradKernelDialect.Metal,
                                WinogradKernelGroup.Base));
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal Winograd kernels unavailable: {ex.Message}");
                        _winogradLibrary = IntPtr.Zero;
                    }
                    _winogradReady = _winogradLibrary != IntPtr.Zero;
                }
                return _winogradReady.Value;
            }
        }
    }

    internal TunedKernelSlot<TunedConv2DBackwardArgs> Conv2DBackwardDataSlot
    {
        get
        {
            if (_conv2DBackwardDataSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _conv2DBackwardDataSlot ??= new TunedKernelSlot<TunedConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardData,
                    TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, new ITunedKernelCandidate<TunedConv2DBackwardArgs>[]
                    {
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("metal.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                            true, _ => true, (in TunedConv2DBackwardArgs a) => LaunchDxDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("metal.conv2d_dx.winograd_f23", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinograd(a)) LaunchDxDirect(a); }),
                        // The stride-2 input gradients of ConvDgradS2KernelTemplate (#1476 ported): the 3 x 3 four-phase kernel and
                        // the 1 x 1 kernel written straight into dX with its zero phases.
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("metal.conv2d_dx.s2k3", TunedKernelOrigin.Builtin,
                            true, shape => PortableConvDgradS2.K3Enabled && PortableConvDgradS2.AdmitsK3Shape(shape) && DgradS2Ready,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardInputS2K3(a)) LaunchDxDirect(a); }),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("metal.conv2d_dx.s2k1", TunedKernelOrigin.Builtin,
                            true, shape => PortableConvDgradS2.K1Enabled && PortableConvDgradS2.AdmitsK1Shape(shape) && DgradS2Ready,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardInputS2K1(a)) LaunchDxDirect(a); }),
                    });
                return _conv2DBackwardDataSlot;
            }
        }
    }

    internal TunedKernelSlot<TunedConv2DBackwardArgs> Conv2DBackwardFilterSlot
    {
        get
        {
            if (_conv2DBackwardFilterSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _conv2DBackwardFilterSlot ??= new TunedKernelSlot<TunedConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardFilter,
                    TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, new ITunedKernelCandidate<TunedConv2DBackwardArgs>[]
                    {
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("metal.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                            true, _ => true, (in TunedConv2DBackwardArgs a) => LaunchDwDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("metal.conv2d_dw.winograd_f32", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinograd(a)) LaunchDwDirect(a); }),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("metal.conv2d_dw.winograd_fused", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && FusedFilterGradientReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradFused(a)) LaunchDwDirect(a); }),
                    });
                return _conv2DBackwardFilterSlot;
            }
        }
    }

    private BackendTunedKernelHarness<TunedConv2DBackwardArgs> BackwardHarness() =>
        new(this, ConvolutionTolerance, () => IsAvailable, MeasureTunedKernelMilliseconds,
            a => new[] { (a.Output, a.OutputCount) },
            a => SameBuffer(a.Output, a.First) || SameBuffer(a.Output, a.Second));

    // Per-route entry points, for the slot candidates above and for the tests, which run one route regardless of what
    // the slot would pick. False when the route declines the shape (or its library did not compile).
    internal bool TryConv2DForwardWinograd(in TunedConv2DArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryForward(this, this, a);

    internal bool TryConv2DBackwardDataWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardData(this, this, a);

    internal bool TryConv2DBackwardFilterWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardFilter(this, this, a);

    // The fused weight gradient: one dispatch of 256-wide threadgroups (plus the finish when the tiles split). Its
    // thread roles need all 256 in one group, so a pipeline the compiler capped below that (register pressure) declines.
    internal bool TryConv2DBackwardFilterWinogradFused(in TunedConv2DBackwardArgs a, int splits = 0)
        => FusedFilterGradientReady && WinogradF23Pipeline.TryBackwardFilterFused(this, this, a, splits);

    // The fused kernels compile as their own library, so a driver that rejects them - they are the only ones here
    // declaring threadgroup memory - drops this candidate alone and leaves the transforms' library serving the
    // forward, input-gradient and unfused weight-gradient routes.
    private bool FusedLibraryReady
    {
        get
        {
            if (_winogradFusedReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_winogradFusedReady is null)
                {
                    try
                    {
                        _winogradFusedLibrary = _shaderLibrary.CompileLibrary(WinogradFusedLibraryName,
                            WinogradF23KernelTemplate.GetModuleSource(WinogradKernelDialect.Metal,
                                WinogradKernelGroup.Fused));
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            $"Metal fused Winograd weight-gradient kernels unavailable: {ex.Message}");
                        _winogradFusedLibrary = IntPtr.Zero;
                    }
                    _winogradFusedReady = _winogradFusedLibrary != IntPtr.Zero;
                }
                return _winogradFusedReady.Value;
            }
        }
    }

    private bool FusedFilterGradientReady
        => WinogradKernelsReady
           && FusedLibraryReady
           && GetPipeline(WinogradFusedLibraryName, _winogradFusedLibrary, WinogradF23KernelTemplate.FusedFilterGradient)
               .MaxTotalThreadsPerThreadgroup >= WinogradF23KernelTemplate.FusedGroupSize
           && GetPipeline(WinogradFusedLibraryName, _winogradFusedLibrary, WinogradF23KernelTemplate.FusedFilterGradientDirect)
               .MaxTotalThreadsPerThreadgroup >= WinogradF23KernelTemplate.FusedGroupSize;

    int IWinogradFusedFilterGradient.FusedFilterGradientGroupTarget => ComputeUnits;

    void IWinogradFusedFilterGradient.FusedFilterGradient(bool direct, IGpuBuffer x, IGpuBuffer res, IGpuBuffer dy,
        int groups, int[] ints)
    {
        var pipeline = GetPipeline(WinogradFusedLibraryName, _winogradFusedLibrary,
            direct ? WinogradF23KernelTemplate.FusedFilterGradientDirect : WinogradF23KernelTemplate.FusedFilterGradient);
        var fusedBuffers = new[] { x, res, dy };
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < fusedBuffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)fusedBuffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(fusedBuffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)WinogradF23KernelTemplate.FusedGroupSize, 1, 1));
    }

    void IWinogradFusedFilterGradient.FusedFilterGradientFinish(IGpuBuffer partials, IGpuBuffer dw, int kc, int splits)
        => TryDispatchConvPoolMetal(_winogradFusedLibrary, WinogradFusedLibraryName,
            WinogradF23KernelTemplate.FusedFilterGradientFinish, checked((int)(9L * kc)),
            new[] { partials, dw }, new[] { kc, splits });

    private static int[] BackwardInts(in TunedConv2DBackwardArgs a) => new[]
    {
        a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW,
        a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW,
    };

    // The direct kernels: one thread per input element (input gradient) or per weight (weight gradient).
    // The direct reference kernels, the fallback whenever the Winograd route declines a shape: a thread count past int
    // overflows instead of wrapping to a non-positive count that dispatches nothing, and a failed dispatch throws rather
    // than leaving the gradient unwritten (as Conv2DDirectReference does).
    private void LaunchDxDirect(in TunedConv2DBackwardArgs a)
    {
        int threads = checked(a.Batch * a.InChannels * a.InHeight * a.InWidth);
        if (threads == 0) return; // an empty tensor: nothing to compute
        if (!TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv2d_backward_input",
                threads, new[] { a.First, a.Second, a.Output }, BackwardInts(a)))
            throw new GpuCapabilityUnavailableException("The Metal conv2d_backward_input kernel is unavailable.");
    }

    private void LaunchDwDirect(in TunedConv2DBackwardArgs a)
    {
        int threads = checked(a.OutChannels * a.InChannels * a.KernelH * a.KernelW);
        if (threads == 0) return; // an empty tensor: nothing to compute
        if (!TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv2d_backward_weights",
                threads, new[] { a.First, a.Second, a.Output }, BackwardInts(a)))
            throw new GpuCapabilityUnavailableException("The Metal conv2d_backward_weights kernel is unavailable.");
    }

    private void RunWinograd(string name, long threads, IGpuBuffer read, IGpuBuffer write, params int[] ints)
        => TryDispatchConvPoolMetal(_winogradLibrary, WinogradLibraryName, name, checked((int)threads),
            new[] { read, write }, ints);

    void IWinogradF23Transforms.Filter(IGpuBuffer w, IGpuBuffer u, int k, int c, bool flip)
        => RunWinograd(flip ? WinogradF23KernelTemplate.FilterFlip : WinogradF23KernelTemplate.Filter, (long)k * c, w, u, k, c);

    void IWinogradF23Transforms.Input(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH, int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.Input, (long)c * n * tilesH * tilesW, x, v, n, c, h, w, pad, tilesH, tilesW);

    void IWinogradF23Transforms.Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH, int tilesW,
        IGpuBuffer? bias, int epilogue)
        => TryDispatchConvPoolMetal(_winogradLibrary, WinogradLibraryName, WinogradF23KernelTemplate.Output,
            checked((int)((long)k * n * tilesH * tilesW)), new[] { m, y, bias ?? y },
            new[] { n, k, outH, outW, tilesH, tilesW, epilogue });

    void IWinogradF23Transforms.GradOutputTiles(IGpuBuffer dy, IGpuBuffer u, int n, int k, int outH, int outW, int tilesH,
        int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.GradOutputTiles, (long)k * n * tilesH * tilesW, dy, u, n, k, outH, outW,
            tilesH, tilesW);

    void IWinogradF23Transforms.InputTilesTileMajor(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad,
        int tilesH, int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.InputTilesTileMajor, (long)c * n * tilesH * tilesW, x, v, n, c, h, w, pad,
            tilesH, tilesW);

    void IWinogradF23Transforms.FilterGradient(IGpuBuffer m, IGpuBuffer dw, int k, int c, int splits)
        => RunWinograd(WinogradF23KernelTemplate.FilterGradient, (long)k * c, m, dw, k, c, splits);
}