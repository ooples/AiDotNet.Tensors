// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #1479: the stride-2 input gradients (3 x 3 four-phase, 1 x 1, and the compiled plan's projection-shortcut pair) on HIP,
// through PortableConvDgradS2 and ConvDgradS2KernelTemplate's kernels, compiled on first need.
public sealed partial class HipBackend : IConvShortcutPairBackend
{
    private IntPtr _dgradS2Module;
    private bool? _dgradS2Ready;

    // The module, built once; a compile the device rejects leaves the other input-gradient routes serving these shapes,
    // and says why.
    private bool DgradS2Ready
    {
        get
        {
            if (_dgradS2Ready is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dgradS2Ready is null)
                {
                    var failure = CompileKernelModule(ConvDgradS2KernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                        "conv_dgrad_s2", ref _dgradS2Module, ConvDgradS2KernelTemplate.GetKernelNames(), useFastMath: false);
                    if (failure is not null)
                        System.Diagnostics.Trace.TraceWarning($"HIP stride-2 input-gradient kernels unavailable: {failure.Message}");
                    bool ok = failure is null;
                    foreach (var name in ConvDgradS2KernelTemplate.GetKernelNames()) ok &= HasTunedKernel(name);
                    _dgradS2Ready = ok;
                }
                return _dgradS2Ready.Value;
            }
        }
    }

    private unsafe void LaunchDgradS2(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        int count = buffers.Length + ints.Length;
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[count];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel(TunedKernel(kernelName), (uint)groups, (uint)ConvDgradS2KernelTemplate.GroupSize, args);
    }

    private int DgradS2GroupTarget => ((IWinogradFusedFilterGradient)this).FusedFilterGradientGroupTarget;

    // Per-route entry points, for the slot candidates and the tests. False: nothing launched.
    internal bool TryConv2DBackwardInputS2K3(in TunedConv2DBackwardArgs a, int splits = 0)
        => PortableConvDgradS2.AdmitsK3(a) && DgradS2Ready
           && PortableConvDgradS2.TryK3(this, LaunchDgradS2, a, DgradS2GroupTarget, splits);

    internal bool TryConv2DBackwardInputS2K1(in TunedConv2DBackwardArgs a, int splits = 0)
        => PortableConvDgradS2.AdmitsK1(a) && DgradS2Ready
           && PortableConvDgradS2.TryK1(this, LaunchDgradS2, a, DgradS2GroupTarget, splits);

    internal bool TryConv2DBackwardInputS2K3Pair(in TunedConv2DBackwardArgs a, IGpuBuffer dyShortcut, IGpuBuffer wShortcut,
        int shortcutChannels, int splits = 0)
        => PortableConvDgradS2.AdmitsK3(a) && DgradS2Ready
           && PortableConvDgradS2.TryPair(this, LaunchDgradS2, a, dyShortcut, wShortcut, shortcutChannels,
               DgradS2GroupTarget, splits);

    bool IConvShortcutPairBackend.SupportsConv2DBackwardInputShortcutPair(int batch, int inChannels, int inHeight,
        int inWidth, int outChannels, int shortcutChannels, int outHeight, int outWidth)
        => PortableConvDgradS2.AdmitsPair(batch, inChannels, inHeight, inWidth, outChannels, shortcutChannels, outHeight,
            outWidth) && DgradS2Ready;

    bool IConvShortcutPairBackend.TryConv2DBackwardInputShortcutPair(IGpuBuffer gradOutput, IGpuBuffer kernel,
        IGpuBuffer gradShortcut, IGpuBuffer kernelShortcut, IGpuBuffer gradInput, int batch, int inChannels, int inHeight,
        int inWidth, int outChannels, int shortcutChannels, int outHeight, int outWidth)
        => TryConv2DBackwardInputS2K3Pair(new TunedConv2DBackwardArgs(gradOutput, kernel, gradInput, false, batch,
            inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, 3, 3, 2, 2, 1, 1, 1, 1), gradShortcut,
            kernelShortcut, shortcutChannels);
}
