// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The jagged-rows grouped GEMM (IGroupedGemmBackend) on Vulkan, through PortableGroupedGemm and GroupedGemmKernelTemplate's
// kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend : IGroupedGemmBackend
{
    private static readonly Dictionary<string, string> s_groupedGemmShaders = BuildGroupedGemmShaders();

    private static Dictionary<string, string> BuildGroupedGemmShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in GroupedGemmKernelTemplate.GetKernelNames())
            shaders[name] = GroupedGemmKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _groupedGemmReady;

    // Every pipeline, built once; a shader the driver rejects leaves the host op, and says why.
    private bool GroupedGemmReady
    {
        get
        {
            if (_groupedGemmReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_groupedGemmReady is null)
                {
                    bool ok = true;
                    try
                    {
                        foreach (var spec in GroupedGemmKernelTemplate.Kernels)
                            if (GetOrCreateGlslPipeline(s_groupedGemmShaders[spec.Name], spec.Buffers.Length,
                                    (uint)(spec.Ints.Length * sizeof(uint))) is null)
                            {
                                System.Diagnostics.Trace.TraceWarning($"Vulkan grouped GEMM kernels unavailable: '{spec.Name}' did not build.");
                                ok = false;
                                break;
                            }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan grouped GEMM kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _groupedGemmReady = ok;
                }
                return _groupedGemmReady.Value;
            }
        }
    }

    private void LaunchGroupedGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_groupedGemmShaders[kernelName], push, groups * GroupedGemmKernelTemplate.GroupSize, buffers);
    }

    bool IGroupedGemmBackend.TryGroupedRowsGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer y, int[] ends, int m, int n, int k,
        bool transposeB, IGpuBuffer? bias, bool biasPerGroup)
        => GroupedGemmReady
           && PortableGroupedGemm.TryForward(this, LaunchGroupedGemm, a, b, y, ends, m, n, k, transposeB, bias, biasPerGroup);

    bool IGroupedGemmBackend.TryGroupedRowsWeightGrad(IGpuBuffer a, IGpuBuffer c, IGpuBuffer w, int[] ends, int m, int k, int n)
        => GroupedGemmReady && PortableGroupedGemm.TryWeightGrad(this, LaunchGroupedGemm, a, c, w, ends, m, k, n);
}
