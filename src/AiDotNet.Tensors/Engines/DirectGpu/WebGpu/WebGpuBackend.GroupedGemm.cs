#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The jagged-rows grouped GEMM (IGroupedGemmBackend) on WebGPU, through PortableGroupedGemm and the WGSL restatement of
// GroupedGemmKernelTemplate (WebGpuGroupedGemmKernels), built on first need.
public sealed partial class WebGpuBackend : IGroupedGemmBackend
{
    private const string GroupedGemmModulePrefix = "GroupedGemm_";
    // 0 not yet asked, 1 ready, 2 declined.
    private int _groupedGemmReady;

    // Every pipeline, built once; a module the implementation rejects leaves the host op, and says why.
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
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in GroupedGemmKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(GroupedGemmModulePrefix + name, WebGpuGroupedGemmKernels.SourceFor(name),
                                    name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU grouped GEMM kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    System.Threading.Volatile.Write(ref _groupedGemmReady, ok ? 1 : 2);
                }
                return _groupedGemmReady == 1;
            }
        }
    }

    private void LaunchGroupedGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds eight ints (bit-cast); the weight gradient's seven leave the last word as padding.
        var uniforms = new float[WebGpuGroupedGemmKernels.UniformWords];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(GroupedGemmModulePrefix + kernelName, WebGpuGroupedGemmKernels.SourceFor(kernelName), kernelName,
            buffers, uniforms, groups * GroupedGemmKernelTemplate.GroupSize).GetAwaiter().GetResult();
    }

    bool IGroupedGemmBackend.TryGroupedRowsGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer y, int[] ends, int m, int n, int k,
        bool transposeB, IGpuBuffer? bias, bool biasPerGroup)
        => GroupedGemmReady
           && PortableGroupedGemm.TryForward(this, LaunchGroupedGemm, a, b, y, ends, m, n, k, transposeB, bias, biasPerGroup);

    bool IGroupedGemmBackend.TryGroupedRowsWeightGrad(IGpuBuffer a, IGpuBuffer c, IGpuBuffer w, int[] ends, int m, int k, int n)
        => GroupedGemmReady && PortableGroupedGemm.TryWeightGrad(this, LaunchGroupedGemm, a, c, w, ends, m, k, n);
}
#endif
