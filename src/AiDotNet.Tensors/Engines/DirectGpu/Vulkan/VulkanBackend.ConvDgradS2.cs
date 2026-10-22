// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1482: the stride-2 input gradients (3 x 3 four-phase, 1 x 1, and the compiled plan's projection-shortcut pair) on
// Vulkan, through PortableConvDgradS2 and ConvDgradS2KernelTemplate's kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend : IConvShortcutPairBackend
{
    private static readonly Dictionary<string, string> s_dgradS2Shaders = BuildDgradS2Shaders();

    private static Dictionary<string, string> BuildDgradS2Shaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in ConvDgradS2KernelTemplate.GetKernelNames())
            shaders[name] = ConvDgradS2KernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _dgradS2Ready;

    // Every pipeline, built once; a shader the driver rejects leaves the other input-gradient routes serving these
    // shapes, and says why.
    private bool DgradS2Ready
    {
        get
        {
            if (_dgradS2Ready is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dgradS2Ready is null)
                {
                    bool ok = true;
                    try
                    {
                        foreach (var spec in ConvDgradS2KernelTemplate.Kernels)
                            if (GetOrCreateGlslPipeline(s_dgradS2Shaders[spec.Name], spec.Buffers.Length,
                                    (uint)(spec.Ints.Length * sizeof(uint))) is null)
                            {
                                System.Diagnostics.Trace.TraceWarning(
                                    $"Vulkan stride-2 input-gradient kernels unavailable: '{spec.Name}' did not build.");
                                ok = false;
                                break;
                            }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan stride-2 input-gradient kernels unavailable: {ex.Message}");
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
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_dgradS2Shaders[kernelName], push, groups * ConvDgradS2KernelTemplate.GroupSize, buffers);
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
