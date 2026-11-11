// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The fused MoE kernels (IMoeKernelBackend): MoeKernelTemplate's kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend : IMoeKernelBackend
{
    private static readonly Dictionary<string, string> s_moeShaders = BuildMoeShaders();

    private static Dictionary<string, string> BuildMoeShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in MoeKernelTemplate.GetKernelNames()) shaders[name] = MoeKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    // 0 not yet asked, 1 ready, 2 declined.
    private int _moeReady;

    bool IMoeKernelBackend.MoeKernelsReady
    {
        get
        {
            int cached = System.Threading.Volatile.Read(ref _moeReady);
            if (cached != 0) return cached == 1;
            lock (_tunedSlotsLock)
            {
                if (_moeReady == 0)
                {
                    bool ok;
                    try
                    {
                        ok = MoeKernelTemplate.Kernels.All(spec =>
                            GetOrCreateGlslPipeline(s_moeShaders[spec.Name], spec.Buffers.Length,
                                (uint)(spec.Ints.Length * sizeof(uint))) is not null);
                        if (!ok) System.Diagnostics.Trace.TraceWarning("Vulkan fused MoE kernels unavailable: a pipeline did not build.");
                    }
                    catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan fused MoE kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    System.Threading.Volatile.Write(ref _moeReady, ok ? 1 : 2);
                }
                return _moeReady == 1;
            }
        }
    }

    void IMoeKernelBackend.LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        if (!TryDispatchConvPoolGlsl(s_moeShaders[kernelName], push, groups * MoeKernelTemplate.GroupSize, buffers))
            throw new InvalidOperationException($"Vulkan could not dispatch the fused MoE kernel '{kernelName}'.");
    }
}
