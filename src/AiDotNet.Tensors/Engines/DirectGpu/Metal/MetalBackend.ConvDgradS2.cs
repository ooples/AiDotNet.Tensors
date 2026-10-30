// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL instance, whose ConvDgradS2PortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #1480: the stride-2 input gradients (3 x 3 four-phase, 1 x 1, and the compiled plan's projection-shortcut pair) on
// Metal, through PortableConvDgradS2 and ConvDgradS2KernelTemplate's kernels, compiled as their own library on first need.
public sealed partial class MetalBackend : IConvShortcutPairBackend
{
    private const string DgradS2LibraryName = "ConvDgradS2";
    private IntPtr _dgradS2Library;
    private bool? _dgradS2Ready;

    // The library, compiled once, and every kernel able to run its 256 threads in one threadgroup (the compiler can cap a
    // pipeline below that under register pressure); otherwise the other input-gradient routes serve these shapes, and
    // the trace says why.
    private bool DgradS2Ready
    {
        get
        {
            if (_dgradS2Ready is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dgradS2Ready is null)
                {
                    bool ok;
                    try
                    {
                        _dgradS2Library = _shaderLibrary.CompileLibrary(DgradS2LibraryName,
                            ConvDgradS2KernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                        ok = _dgradS2Library != IntPtr.Zero;
                        if (ok)
                            foreach (var name in ConvDgradS2KernelTemplate.GetKernelNames())
                                ok &= GetPipeline(DgradS2LibraryName, _dgradS2Library, name).MaxTotalThreadsPerThreadgroup
                                      >= ConvDgradS2KernelTemplate.GroupSize;
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal stride-2 input-gradient kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _dgradS2Ready = ok;
                }
                return _dgradS2Ready.Value;
            }
        }
    }

    private void LaunchDgradS2(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var pipeline = GetPipeline(DgradS2LibraryName, _dgradS2Library, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)ConvDgradS2KernelTemplate.GroupSize, 1, 1));
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
