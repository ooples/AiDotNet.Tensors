// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The transposed-convolution forward as one GEMM per batch chunk and an overlap-add on Vulkan, through
// PortableConvTransposeGemm and ConvTransposeGemmKernelTemplate's kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend
{
    private static readonly Dictionary<string, string> s_convTransposeGemmShaders = BuildConvTransposeGemmShaders();

    private static Dictionary<string, string> BuildConvTransposeGemmShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in ConvTransposeGemmKernelTemplate.GetKernelNames())
            shaders[name] = ConvTransposeGemmKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _convTransposeGemmReady;

    // Both pipelines, built once; a shader the driver rejects leaves the conv_transpose2d shader serving every shape,
    // and says why.
    private bool ConvTransposeGemmReady
    {
        get
        {
            if (_convTransposeGemmReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_convTransposeGemmReady is null)
                {
                    bool ok = true;
                    try
                    {
                        foreach (var spec in ConvTransposeGemmKernelTemplate.Kernels)
                            if (GetOrCreateGlslPipeline(s_convTransposeGemmShaders[spec.Name], spec.Buffers.Length,
                                    (uint)(spec.Ints.Length * sizeof(uint))) is null)
                            {
                                System.Diagnostics.Trace.TraceWarning(
                                    $"Vulkan transposed-convolution GEMM kernels unavailable: '{spec.Name}' did not build.");
                                ok = false;
                                break;
                            }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan transposed-convolution GEMM kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _convTransposeGemmReady = ok;
                }
                return _convTransposeGemmReady.Value;
            }
        }
    }

    private void LaunchConvTransposeGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_convTransposeGemmShaders[kernelName], push,
            groups * ConvTransposeGemmKernelTemplate.GroupSize, buffers);
    }

    /// <summary>The transposed forward through one GEMM per batch chunk. False: nothing launched.</summary>
    internal bool TryConvTranspose2DGemm(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output, int batch,
        int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH,
        int kernelW, int strideH, int strideW, int padH, int padW)
        => PortableConvTransposeGemm.Enabled && ConvTransposeGemmReady
           && PortableConvTransposeGemm.TryForward(this, LaunchConvTransposeGemm, input, kernel, output, batch,
               inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW,
               padH, padW);
}
