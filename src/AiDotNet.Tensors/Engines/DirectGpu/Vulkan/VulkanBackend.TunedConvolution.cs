// Copyright (c) AiDotNet. All rights reserved.
// Convolution forward through the tuned-kernel registry on Vulkan (#1115).

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>2-D convolution arguments (NCHW input, [Cout, Cin, kH, kW] filter).</summary>
internal readonly struct VulkanConv2DArgs
{
    internal VulkanConv2DArgs(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        Input = input; Kernel = kernel; Output = output;
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth;
        OutChannels = outChannels; OutHeight = outHeight; OutWidth = outWidth;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW;
        PadH = padH; PadW = padW; DilationH = dilationH; DilationW = dilationW;
    }

    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Kernel { get; }
    internal IGpuBuffer Output { get; }
    internal int Batch { get; }
    internal int InChannels { get; }
    internal int InHeight { get; }
    internal int InWidth { get; }
    internal int OutChannels { get; }
    internal int OutHeight { get; }
    internal int OutWidth { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int StrideH { get; }
    internal int StrideW { get; }
    internal int PadH { get; }
    internal int PadW { get; }
    internal int DilationH { get; }
    internal int DilationW { get; }

    internal int OutputCount => Batch * OutChannels * OutHeight * OutWidth;

    /// <summary>The kernel's push-constant block, in declaration order.</summary>
    internal uint[] PushConstants() => new[]
    {
        (uint)Batch, (uint)InChannels, (uint)InHeight, (uint)InWidth,
        (uint)OutChannels, (uint)OutHeight, (uint)OutWidth, (uint)KernelH, (uint)KernelW,
        (uint)StrideH, (uint)StrideW, (uint)PadH, (uint)PadW, (uint)DilationH, (uint)DilationW,
    };

    /// <summary>Shape class: every geometric parameter (output extent follows from them).</summary>
    internal unsafe TunedShape Shape()
    {
        int* d = stackalloc int[15];
        d[0] = Batch; d[1] = InChannels; d[2] = InHeight; d[3] = InWidth; d[4] = OutChannels;
        d[5] = KernelH; d[6] = KernelW; d[7] = StrideH; d[8] = StrideW; d[9] = PadH; d[10] = PadW;
        d[11] = DilationH; d[12] = DilationW; d[13] = OutHeight; d[14] = OutWidth;
        return TunedShape.Create(TunedKernelDType.Float32, new ReadOnlySpan<int>(d, 15));
    }
}

/// <summary>
/// Convolution forward through the tuned-kernel registry. The reference is the established kernel (one invocation
/// per output element); the candidates compute several output channels of an output pixel per invocation, each
/// selected per shape class only after the correctness and paired-timing gate.
/// </summary>
public sealed unsafe partial class VulkanBackend
{
    // The variants keep the reference's summation order; the tolerance only covers a driver contracting the
    // multiply-add differently between the two shaders.
    private const double ConvolutionTolerance = 1e-4;
    private TunedKernelSlot<VulkanConv2DArgs>? _conv2DForwardSlot;

    internal TunedKernelSlot<VulkanConv2DArgs> Conv2DForwardSlot
    {
        get
        {
            if (_conv2DForwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_conv2DForwardSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<VulkanConv2DArgs>>
                    {
                        new VulkanTunedKernelCandidate<VulkanConv2DArgs>("glsl.conv2d.invocation_per_output",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in VulkanConv2DArgs a) => LaunchConv2DReference(a)),
                    };
                    foreach (int tile in VulkanTunedRowKernels.ConvChannelTiles)
                    {
                        int t = tile;
                        string source = VulkanTunedRowKernels.Conv2DChannelTile(t);
                        candidates.Add(new VulkanTunedKernelCandidate<VulkanConv2DArgs>("generated.conv2d.channels" + t,
                            TunedKernelOrigin.Generated, true,
                            // shape[4] is the output channel count: a tile wider than it has nothing to share.
                            shape => shape[4] >= t && HasTunedPipeline(source, 3, 15 * sizeof(uint)),
                            (in VulkanConv2DArgs a) => TryDispatchConvPoolGlsl(source, a.PushConstants(),
                                a.Batch * ((a.OutChannels + t - 1) / t) * a.OutHeight * a.OutWidth,
                                a.Input, a.Kernel, a.Output)));
                    }
                    _conv2DForwardSlot = new TunedKernelSlot<VulkanConv2DArgs>(TunedKernelOp.Conv2DForward,
                        TunedDeviceKey,
                        new VulkanTunedKernelHarness<VulkanConv2DArgs>(this, ConvolutionTolerance,
                            a => new[] { (a.Output, a.OutputCount) },
                            a => Aliases(a.Output, a.Input) || Aliases(a.Output, a.Kernel)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _conv2DForwardSlot;
            }
        }
    }

    private void LaunchConv2DReference(in VulkanConv2DArgs a) =>
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv2D, a.PushConstants(), a.OutputCount,
            a.Input, a.Kernel, a.Output);
}
