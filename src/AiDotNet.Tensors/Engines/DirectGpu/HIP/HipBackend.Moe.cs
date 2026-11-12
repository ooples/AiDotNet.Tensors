// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The fused MoE kernels (IMoeKernelBackend): MoeKernelTemplate's HIP dialect compiled on first need.
public sealed partial class HipBackend : IMoeKernelBackend
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
                    var failure = CompileKernelModule(MoeKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                        "moe_kernels", ref _moeModule, MoeKernelTemplate.GetKernelNames(), useFastMath: false);
                    if (failure is not null)
                        System.Diagnostics.Trace.TraceWarning($"HIP fused MoE kernels unavailable: {failure.Message}");
                    bool ok = failure is null;
                    foreach (var name in MoeKernelTemplate.GetKernelNames()) ok &= HasTunedKernel(name);
                    System.Threading.Volatile.Write(ref _moeReady, ok ? 1 : 2);
                }
                return _moeReady == 1;
            }
        }
    }

    unsafe void IMoeKernelBackend.LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[buffers.Length + ints.Length];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel(TunedKernel(kernelName), (uint)groups, (uint)MoeKernelTemplate.GroupSize, args);
    }
}
