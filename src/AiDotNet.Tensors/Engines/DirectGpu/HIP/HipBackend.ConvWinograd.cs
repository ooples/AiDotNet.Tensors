// Copyright (c) AiDotNet. All rights reserved.
// #1368: Winograd F(2x2, 3x3) forward and input gradient and F(3x3, 2x2) weight gradient on HIP, through the shared
// WinogradF23Pipeline: the tile transforms below (emitted from WinogradF23KernelTemplate) around BatchedGemm. HIP's
// backward convolution gets tuned slots here for the first time, with the direct kernels it always ran as the
// references, so Winograd is served only on the shape classes where it measures faster.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IWinogradF23Transforms, IWinogradFusedFilterGradient
{
    private IntPtr _winogradModule, _winogradFusedModule;
    private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardDataSlot, _conv2DBackwardFilterSlot;

    private bool WinogradKernelsReady => HasTunedKernel(WinogradF23KernelTemplate.Filter);

    /// <summary>
    /// Whether the fused weight gradient's own module built. It is compiled apart from the transforms because its
    /// kernels are the only ones here that declare shared memory and a fixed 256-wide group: a device compiler that
    /// rejects them then drops only the fused candidate, and the transforms keep serving the other three routes.
    /// </summary>
    private bool FusedFilterGradientReady =>
        HasTunedKernel(WinogradF23KernelTemplate.FusedFilterGradient)
        && HasTunedKernel(WinogradF23KernelTemplate.FusedFilterGradientDirect)
        && HasTunedKernel(WinogradF23KernelTemplate.FusedFilterGradientFinish);

    private void CompileWinogradKernels()
    {
        // Strict math: the transforms' coefficients are 0, +-1 and +-1/2, and contracting or reassociating them would
        // move the result away from the other backends' instances of the same template.
        CompileWinogradModule(WinogradKernelGroup.Base, "winograd", ref _winogradModule, "Winograd kernels");
        CompileWinogradModule(WinogradKernelGroup.Fused, "winograd_fused", ref _winogradFusedModule,
            "Winograd fused weight-gradient kernels");
    }

    private void CompileWinogradModule(WinogradKernelGroup group, string moduleName, ref IntPtr module, string label)
    {
        var failure = CompileKernelModule(WinogradF23KernelTemplate.GetModuleSource(WinogradKernelDialect.Hip, group),
            moduleName, ref module, WinogradF23KernelTemplate.GetKernelNames(group), useFastMath: false);
        if (failure is not null)
            System.Diagnostics.Trace.TraceWarning($"HIP {label} unavailable: {failure.Message}");
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
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("hip.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_backward_input"), (in TunedConv2DBackwardArgs a) => LaunchDxDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("hip.conv2d_dx.winograd_f23", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinograd(a)) LaunchDxDirect(a); }),
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
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("hip.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_backward_kernel"), (in TunedConv2DBackwardArgs a) => LaunchDwDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("hip.conv2d_dw.winograd_f32", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinograd(a)) LaunchDwDirect(a); }),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("hip.conv2d_dw.winograd_fused", TunedKernelOrigin.Builtin,
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
            a => a.Output.Handle == a.First.Handle || a.Output.Handle == a.Second.Handle);

    // Per-route entry points, for the slot candidates above and for the tests, which run one route regardless of what
    // the slot would pick. False when the route declines the shape (or its module did not compile); nothing is
    // launched then.
    internal bool TryConv2DForwardWinograd(in TunedConv2DArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryForward(this, this, a);

    internal bool TryConv2DBackwardDataWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardData(this, this, a);

    internal bool TryConv2DBackwardFilterWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardFilter(this, this, a);

    private unsafe void LaunchBackwardDirect(string name, in TunedConv2DBackwardArgs a, bool weightGradient)
    {
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException("HIP kernel not found: " + name);
        IntPtr first = a.First.Handle, second = a.Second.Handle, output = a.Output.Handle;
        int batch = a.Batch, inC = a.InChannels, inH = a.InHeight, inW = a.InWidth, outC = a.OutChannels;
        int outH = a.OutHeight, outW = a.OutWidth, kh = a.KernelH, kw = a.KernelW, sh = a.StrideH, sw = a.StrideW;
        int ph = a.PadH, pw = a.PadW, dh = a.DilationH, dw = a.DilationW;
        void** args = stackalloc void*[18];
        args[0] = &first; args[1] = &second; args[2] = &output; args[3] = &batch; args[4] = &inC; args[5] = &inH;
        args[6] = &inW; args[7] = &outC; args[8] = &outH; args[9] = &outW; args[10] = &kh; args[11] = &kw;
        args[12] = &sh; args[13] = &sw; args[14] = &ph; args[15] = &pw; args[16] = &dh; args[17] = &dw;
        if (weightGradient)
        {
            // One thread per gradKernel element (see the kernel): a flat 1D grid has no 65,535 limit.
            long total = (long)outC * inC * kh * kw;
            LaunchKernel(kernel, (uint)((total + 255) / 256), 256, args);
            return;
        }
        LaunchKernel3D(kernel, (uint)((inW + 15) / 16), (uint)((inH + 15) / 16), (uint)(batch * inC), 16, 16, 1, args);
    }

    private void LaunchDxDirect(in TunedConv2DBackwardArgs a) => LaunchBackwardDirect("conv2d_backward_input", a, false);

    private void LaunchDwDirect(in TunedConv2DBackwardArgs a) => LaunchBackwardDirect("conv2d_backward_kernel", a, true);

    // The fused weight gradient: one launch of 256-wide blocks (plus the finish when the tiles split).
    internal bool TryConv2DBackwardFilterWinogradFused(in TunedConv2DBackwardArgs a, int splits = 0)
        => FusedFilterGradientReady && WinogradF23Pipeline.TryBackwardFilterFused(this, this, a, splits);

    int IWinogradFusedFilterGradient.FusedFilterGradientGroupTarget => ComputeUnits;

    unsafe void IWinogradFusedFilterGradient.FusedFilterGradient(bool direct, IGpuBuffer x, IGpuBuffer res, IGpuBuffer dy,
        int groups, int[] ints)
    {
        IntPtr xPtr = x.Handle, resPtr = res.Handle, dyPtr = dy.Handle;
        int* fusedInts = stackalloc int[ints.Length];
        void** fusedArgs = stackalloc void*[3 + ints.Length];
        fusedArgs[0] = &xPtr; fusedArgs[1] = &resPtr; fusedArgs[2] = &dyPtr;
        for (int i = 0; i < ints.Length; i++) { fusedInts[i] = ints[i]; fusedArgs[3 + i] = &fusedInts[i]; }
        LaunchKernel(TunedKernel(direct ? WinogradF23KernelTemplate.FusedFilterGradientDirect
            : WinogradF23KernelTemplate.FusedFilterGradient), (uint)groups, (uint)WinogradF23KernelTemplate.FusedGroupSize, fusedArgs);
    }

    unsafe void IWinogradFusedFilterGradient.FusedFilterGradientFinish(IGpuBuffer partials, IGpuBuffer dw, int kc, int splits)
    {
        int* finishInts = stackalloc int[] { kc, splits };
        LaunchWinograd(WinogradF23KernelTemplate.FusedFilterGradientFinish, 9L * kc, partials, dw, finishInts, 2);
    }

    private unsafe void LaunchWinograd(string name, long threads, IGpuBuffer read, IGpuBuffer write, int* ints, int count)
    {
        IntPtr readPtr = read.Handle, writePtr = write.Handle;
        void** args = stackalloc void*[2 + count];
        args[0] = &readPtr; args[1] = &writePtr;
        for (int i = 0; i < count; i++) args[2 + i] = &ints[i];
        LaunchKernel(TunedKernel(name), (uint)((threads + 255) / 256), 256, args);
    }

    unsafe void IWinogradF23Transforms.Filter(IGpuBuffer w, IGpuBuffer u, int k, int c, bool flip)
    {
        int* ints = stackalloc int[] { k, c };
        LaunchWinograd(flip ? WinogradF23KernelTemplate.FilterFlip : WinogradF23KernelTemplate.Filter, (long)k * c, w, u, ints, 2);
    }

    unsafe void IWinogradF23Transforms.Input(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH,
        int tilesW)
    {
        int* ints = stackalloc int[] { n, c, h, w, pad, tilesH, tilesW };
        LaunchWinograd(WinogradF23KernelTemplate.Input, (long)c * n * tilesH * tilesW, x, v, ints, 7);
    }

    unsafe void IWinogradF23Transforms.Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH,
        int tilesW, IGpuBuffer? bias, int epilogue)
    {
        IntPtr readPtr = m.Handle, writePtr = y.Handle, biasPtr = (bias ?? y).Handle;
        int* ints = stackalloc int[] { n, k, outH, outW, tilesH, tilesW, epilogue };
        void** args = stackalloc void*[10];
        args[0] = &readPtr; args[1] = &writePtr; args[2] = &biasPtr;
        for (int i = 0; i < 7; i++) args[3 + i] = &ints[i];
        long threads = (long)k * n * tilesH * tilesW;
        LaunchKernel(TunedKernel(WinogradF23KernelTemplate.Output), (uint)((threads + 255) / 256), 256, args);
    }

    unsafe void IWinogradF23Transforms.GradOutputTiles(IGpuBuffer dy, IGpuBuffer u, int n, int k, int outH, int outW,
        int tilesH, int tilesW)
    {
        int* ints = stackalloc int[] { n, k, outH, outW, tilesH, tilesW };
        LaunchWinograd(WinogradF23KernelTemplate.GradOutputTiles, (long)k * n * tilesH * tilesW, dy, u, ints, 6);
    }

    unsafe void IWinogradF23Transforms.InputTilesTileMajor(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w,
        int pad, int tilesH, int tilesW)
    {
        int* ints = stackalloc int[] { n, c, h, w, pad, tilesH, tilesW };
        LaunchWinograd(WinogradF23KernelTemplate.InputTilesTileMajor, (long)c * n * tilesH * tilesW, x, v, ints, 7);
    }

    unsafe void IWinogradF23Transforms.FilterGradient(IGpuBuffer m, IGpuBuffer dw, int k, int c, int splits)
    {
        int* ints = stackalloc int[] { k, c, splits };
        LaunchWinograd(WinogradF23KernelTemplate.FilterGradient, (long)k * c, m, dw, ints, 3);
    }
}