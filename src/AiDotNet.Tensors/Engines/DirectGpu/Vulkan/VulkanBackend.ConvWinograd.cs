using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1371: Winograd F(2x2, 3x3) forward and input gradient and F(3x3, 2x2) weight gradient on Vulkan, through the shared
// WinogradF23Pipeline: the tile transforms below (GLSL emitted from WinogradF23KernelTemplate) around BatchedGemm.
// Backward convolution gets tuned slots here for the first time, with the direct shaders it always ran as the
// references, so Winograd is served only on the shape classes where it measures faster.
public sealed unsafe partial class VulkanBackend : IWinogradF23Transforms, IWinogradFusedFilterGradient
{
    private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardDataSlot, _conv2DBackwardFilterSlot;

    // One shader per kernel, generated once: the pipeline cache is keyed by the source text.
    private static readonly Dictionary<string, string> s_winogradShaders = BuildWinogradShaders();

    private static Dictionary<string, string> BuildWinogradShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in WinogradF23KernelTemplate.GetKernelNames())
            shaders[name] = WinogradF23KernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _winogradReady, _winogradFusedReady;

    // Every Winograd pipeline, built once. A shader the driver rejects would otherwise throw from inside a tuned slot
    // on the first 3x3 convolution; this turns it into a candidate that is not offered, and says why.
    private bool WinogradKernelsReady => WinogradGroupReady(WinogradKernelGroup.Base, ref _winogradReady,
        "Vulkan Winograd kernels");

    // The fused weight gradient's shaders are checked on their own: they are the only ones here that declare shared
    // memory and a fixed 256-wide group, so a driver that rejects one of them must drop only the fused candidate and
    // leave the forward, input-gradient and unfused weight-gradient routes offered.
    private bool FusedFilterGradientReady => WinogradKernelsReady
        && WinogradGroupReady(WinogradKernelGroup.Fused, ref _winogradFusedReady,
            "Vulkan fused Winograd weight-gradient kernels");

    private bool WinogradGroupReady(WinogradKernelGroup group, ref bool? cached, string label)
    {
        if (cached is { } ready) return ready;
        lock (_tunedSlotsLock)
        {
            if (cached is null)
            {
                bool ok = true;
                try
                {
                    foreach (var name in WinogradF23KernelTemplate.GetKernelNames(group))
                    {
                        var spec = WinogradF23KernelTemplate.Get(name);
                        if (GetOrCreateGlslPipeline(s_winogradShaders[name], spec.Buffers.Length,
                                (uint)(spec.Ints.Length * sizeof(uint))) is null)
                        {
                            System.Diagnostics.Trace.TraceWarning($"{label} unavailable: '{name}' did not build.");
                            ok = false;
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.TraceWarning($"{label} unavailable: {ex.Message}");
                    ok = false;
                }
                cached = ok;
            }
            return cached.Value;
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
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("vulkan.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                            true, _ => true, (in TunedConv2DBackwardArgs a) => LaunchDxDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("vulkan.conv2d_dx.winograd_f23", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinograd(a)) LaunchDxDirect(a); }),
                        // The stride-2 input gradients of ConvDgradS2KernelTemplate (#1476 ported): the 3 x 3 four-phase kernel and
                        // the 1 x 1 kernel written straight into dX with its zero phases.
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("vulkan.conv2d_dx.s2k3", TunedKernelOrigin.Builtin,
                            true, shape => PortableConvDgradS2.K3Enabled && PortableConvDgradS2.AdmitsK3Shape(shape) && DgradS2Ready,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardInputS2K3(a)) LaunchDxDirect(a); }),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("vulkan.conv2d_dx.s2k1", TunedKernelOrigin.Builtin,
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
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("vulkan.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                            true, _ => true, (in TunedConv2DBackwardArgs a) => LaunchDwDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("vulkan.conv2d_dw.winograd_f32", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinograd(a)) LaunchDwDirect(a); }),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("vulkan.conv2d_dw.winograd_fused", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && FusedFilterGradientReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradFused(a)) LaunchDwDirect(a); }),
                    });
                return _conv2DBackwardFilterSlot;
            }
        }
    }

    private BackendTunedKernelHarness<TunedConv2DBackwardArgs> BackwardHarness() =>
        new(this, ConvolutionTolerance, () => CanMeasureTunedKernels, MeasureTunedKernelMilliseconds,
            a => new[] { (a.Output, a.OutputCount) },
            a => ReferenceEquals(a.Output, a.First) || ReferenceEquals(a.Output, a.Second));

    // Per-route entry points, for the slot candidates above and for the tests, which run one route regardless of what
    // the slot would pick. False when the route declines the shape (or its shaders did not build); nothing is launched
    // then.
    internal bool TryConv2DForwardWinograd(in TunedConv2DArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryForward(this, this, a);

    internal bool TryConv2DBackwardDataWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardData(this, this, a);

    internal bool TryConv2DBackwardFilterWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardFilter(this, this, a);

    // The fused weight gradient: one dispatch of 256-wide groups (plus the finish when the tiles split).
    internal bool TryConv2DBackwardFilterWinogradFused(in TunedConv2DBackwardArgs a, int splits = 0)
        => FusedFilterGradientReady && WinogradF23Pipeline.TryBackwardFilterFused(this, this, a, splits);

    // Core Vulkan reports no multiprocessor count (ComputeUnits here is the work-group invocation limit), so the splits
    // aim at 64 groups, a mid-range discrete GPU's SM / CU count; the tuned slot still measures the result.
    private const int FusedFilterGradientGroups = 64;

    int IWinogradFusedFilterGradient.FusedFilterGradientGroupTarget => FusedFilterGradientGroups;

    void IWinogradFusedFilterGradient.FusedFilterGradient(bool direct, IGpuBuffer x, IGpuBuffer res, IGpuBuffer dy,
        int groups, int[] ints)
    {
        var fusedPush = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) fusedPush[i] = (uint)ints[i];
        string name = direct ? WinogradF23KernelTemplate.FusedFilterGradientDirect : WinogradF23KernelTemplate.FusedFilterGradient;
        TryDispatchConvPoolGlsl(s_winogradShaders[name], fusedPush, groups * WinogradF23KernelTemplate.FusedGroupSize, x, res, dy);
    }

    void IWinogradFusedFilterGradient.FusedFilterGradientFinish(IGpuBuffer partials, IGpuBuffer dw, int kc, int splits)
        => RunWinograd(WinogradF23KernelTemplate.FusedFilterGradientFinish, 9L * kc, partials, dw, kc, splits);

    private static uint[] BackwardPush(in TunedConv2DBackwardArgs a) => new uint[]
    {
        (uint)a.Batch, (uint)a.InChannels, (uint)a.InHeight, (uint)a.InWidth, (uint)a.OutChannels, (uint)a.OutHeight,
        (uint)a.OutWidth, (uint)a.KernelH, (uint)a.KernelW, (uint)a.StrideH, (uint)a.StrideW, (uint)a.PadH, (uint)a.PadW,
        (uint)a.DilationH, (uint)a.DilationW,
    };

    // The direct shaders: one invocation per input element (input gradient) or per weight (weight gradient).
    // The direct reference kernels, the fallback whenever the Winograd route declines a shape: a thread count past int
    // overflows instead of wrapping, and a failed dispatch throws rather than leaving the gradient unwritten (as
    // Conv2DDirectReference does).
    private void LaunchDxDirect(in TunedConv2DBackwardArgs a)
    {
        int threads = checked(a.Batch * a.InChannels * a.InHeight * a.InWidth);
        if (threads == 0) return; // an empty tensor: nothing to compute
        if (!TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv2DBackwardInput, BackwardPush(a), threads, a.First, a.Second, a.Output))
            throw new GpuCapabilityUnavailableException("The Vulkan conv2d backward-input shader is unavailable.");
    }

    private void LaunchDwDirect(in TunedConv2DBackwardArgs a)
    {
        int threads = checked(a.OutChannels * a.InChannels * a.KernelH * a.KernelW);
        if (threads == 0) return; // an empty tensor: nothing to compute
        if (!TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv2DBackwardKernel, BackwardPush(a), threads, a.First, a.Second, a.Output))
            throw new GpuCapabilityUnavailableException("The Vulkan conv2d backward-kernel shader is unavailable.");
    }

    private void RunWinograd(string name, long threads, IGpuBuffer read, IGpuBuffer write, params int[] ints)
    {
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = (uint)ints[i];
        TryDispatchConvPoolGlsl(s_winogradShaders[name], push, checked((int)threads), read, write);
    }

    void IWinogradF23Transforms.Filter(IGpuBuffer w, IGpuBuffer u, int k, int c, bool flip)
        => RunWinograd(flip ? WinogradF23KernelTemplate.FilterFlip : WinogradF23KernelTemplate.Filter, (long)k * c, w, u, k, c);

    void IWinogradF23Transforms.Input(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH, int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.Input, (long)c * n * tilesH * tilesW, x, v, n, c, h, w, pad, tilesH, tilesW);

    void IWinogradF23Transforms.Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH, int tilesW,
        IGpuBuffer? bias, int epilogue)
        => TryDispatchConvPoolGlsl(s_winogradShaders[WinogradF23KernelTemplate.Output],
            new[] { (uint)n, (uint)k, (uint)outH, (uint)outW, (uint)tilesH, (uint)tilesW, (uint)epilogue },
            checked((int)((long)k * n * tilesH * tilesW)), m, y, bias ?? y);

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