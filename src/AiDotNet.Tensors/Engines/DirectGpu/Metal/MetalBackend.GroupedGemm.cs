// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL and Vulkan instances, whose GroupedGemmPortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The jagged-rows grouped GEMM (IGroupedGemmBackend) on Metal, through PortableGroupedGemm and GroupedGemmKernelTemplate's
// kernels, compiled as their own library on first need.
public sealed partial class MetalBackend : IGroupedGemmBackend
{
    private const string GroupedGemmLibraryName = "GroupedGemm";
    private IntPtr _groupedGemmLibrary;
    // 0 not yet asked, 1 ready, 2 declined.
    private int _groupedGemmReady;

    // The library, compiled once, and both kernels able to run their 256 threads in one threadgroup; otherwise the host op
    // runs, and the trace says why.
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
                        _groupedGemmLibrary = _shaderLibrary.CompileLibrary(GroupedGemmLibraryName,
                            GroupedGemmKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                        ok = _groupedGemmLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in GroupedGemmKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(GroupedGemmLibraryName, _groupedGemmLibrary, name).MaxTotalThreadsPerThreadgroup
                                      >= GroupedGemmKernelTemplate.GroupSize;
                    }
                    catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal grouped GEMM kernels unavailable: {ex.Message}");
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
        var pipeline = GetPipeline(GroupedGemmLibraryName, _groupedGemmLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)GroupedGemmKernelTemplate.GroupSize, 1, 1));
    }

    bool IGroupedGemmBackend.TryGroupedRowsGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer y, int[] ends, int m, int n, int k,
        bool transposeB, IGpuBuffer? bias, bool biasPerGroup)
        => GroupedGemmReady
           && PortableGroupedGemm.TryForward(this, LaunchGroupedGemm, a, b, y, ends, m, n, k, transposeB, bias, biasPerGroup);

    bool IGroupedGemmBackend.TryGroupedRowsWeightGrad(IGpuBuffer a, IGpuBuffer c, IGpuBuffer w, int[] ends, int m, int k, int n)
        => GroupedGemmReady && PortableGroupedGemm.TryWeightGrad(this, LaunchGroupedGemm, a, c, w, ends, m, k, n);
}
