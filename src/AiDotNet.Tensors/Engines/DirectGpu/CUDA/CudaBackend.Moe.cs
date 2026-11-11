// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The fused MoE kernels (IMoeKernelBackend): MoeKernelTemplate's HIP C++ dialect compiled by NVRTC on first need.
public sealed partial class CudaBackend : IMoeKernelBackend
{
    private IntPtr _moeModule;
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
                    bool ok = false;
                    try
                    {
                        using var ctx = PushContext();
                        _moeModule = CompileKernelModule(_cuDevice, MoeKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                            "moe_kernels", MoeKernelTemplate.GetKernelNames(), useFastMath: false);
                        ok = _moeModule != IntPtr.Zero;
                        foreach (var name in MoeKernelTemplate.GetKernelNames()) ok &= _kernelCache.ContainsKey(name);
                    }
                    catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"CUDA fused MoE kernels unavailable: {ex.Message}");
                    }
                    System.Threading.Volatile.Write(ref _moeReady, ok ? 1 : 2);
                }
                return _moeReady == 1;
            }
        }
    }

    unsafe void IMoeKernelBackend.LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var kernel = _kernelCache[kernelName];
        using var ctx = PushContext();
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[buffers.Length + ints.Length];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel3D(kernel, (uint)groups, 1, 1, (uint)MoeKernelTemplate.GroupSize, 1, 1, args, 0);
    }
}
