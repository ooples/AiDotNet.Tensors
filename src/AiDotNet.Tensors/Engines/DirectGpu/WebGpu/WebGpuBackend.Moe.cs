#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The fused MoE kernels (IMoeKernelBackend) on WebGPU: the WGSL restatement of MoeKernelTemplate (WebGpuMoeKernels),
// every pipeline built on first need.
public sealed partial class WebGpuBackend : IMoeKernelBackend
{
    private const string MoeModulePrefix = "Moe_";
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
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in MoeKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(MoeModulePrefix + name, WebGpuMoeKernels.SourceFor(name), name)
                                    .GetAwaiter().GetResult();
                        }
                        catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU fused MoE kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    System.Threading.Volatile.Write(ref _moeReady, ok ? 1 : 2);
                }
                return _moeReady == 1;
            }
        }
    }

    void IMoeKernelBackend.LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var uniforms = new float[WebGpuMoeKernels.UniformWords];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(MoeModulePrefix + kernelName, WebGpuMoeKernels.SourceFor(kernelName), kernelName,
            buffers, uniforms, groups * MoeKernelTemplate.GroupSize).GetAwaiter().GetResult();
    }
}
#endif
