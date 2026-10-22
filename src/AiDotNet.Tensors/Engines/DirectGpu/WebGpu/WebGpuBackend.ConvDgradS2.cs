#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1483: the stride-2 input gradients (3 x 3 four-phase, 1 x 1, and the compiled plan's projection-shortcut pair) on
// WebGPU, through PortableConvDgradS2 and the WGSL restatement of ConvDgradS2KernelTemplate (WebGpuConvDgradS2Kernels),
// built on first need.
public sealed partial class WebGpuBackend : IConvShortcutPairBackend
{
    private const string DgradS2ModulePrefix = "ConvDgradS2_";
    private bool? _dgradS2Ready;

    // Every pipeline, built once; a module the implementation rejects leaves the other input-gradient routes serving
    // these shapes, and says why.
    private bool DgradS2Ready
    {
        get
        {
            if (_dgradS2Ready is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dgradS2Ready is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in ConvDgradS2KernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(DgradS2ModulePrefix + name, WebGpuConvDgradS2Kernels.SourceFor(name),
                                    name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU stride-2 input-gradient kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _dgradS2Ready = ok;
                }
                return _dgradS2Ready.Value;
            }
        }
    }

    private void LaunchDgradS2(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds the ints (bit-cast) and its padding words.
        var uniforms = new float[WebGpuConvDgradS2Kernels.UniformWords(kernelName)];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(DgradS2ModulePrefix + kernelName, WebGpuConvDgradS2Kernels.SourceFor(kernelName), kernelName,
            buffers, uniforms, groups * ConvDgradS2KernelTemplate.GroupSize).GetAwaiter().GetResult();
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
#endif

