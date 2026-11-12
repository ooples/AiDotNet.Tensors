// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Linq;

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

    // 0 not yet asked, 1 ready, 2 declined.
    private int _groupedGemmReady;

    // Every pipeline, built once; a shader the driver rejects leaves the host op, and says why.
    private bool GroupedGemmReady
    {
        get
        {
            int cached = System.Threading.Volatile.Read(ref _groupedGemmReady);
            if (cached != 0) return cached == 1;
            lock (_tunedSlotsLock)
            {
                if (_groupedGemmReady == 0)
                {
                    bool ok;
                    try
                    {
                        ok = GroupedGemmKernelTemplate.Kernels.All(spec =>
                            GetOrCreateGlslPipeline(s_groupedGemmShaders[spec.Name], spec.Buffers.Length,
                                (uint)(spec.Ints.Length * sizeof(uint))) is not null);
                        if (!ok) System.Diagnostics.Trace.TraceWarning("Vulkan grouped GEMM kernels unavailable: a pipeline did not build.");
                    }
                    catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan grouped GEMM kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    System.Threading.Volatile.Write(ref _groupedGemmReady, ok ? 1 : 2);
                }
                return _groupedGemmReady == 1;
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
