#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The jagged-rows grouped GEMM (IGroupedGemmBackend) on WebGPU, through PortableGroupedGemm and the WGSL restatement of
// GroupedGemmKernelTemplate (WebGpuGroupedGemmKernels), built on first need.
public sealed partial class WebGpuBackend : IGroupedGemmBackend
{
    private const string GroupedGemmModulePrefix = "GroupedGemm_";
    private bool? _groupedGemmReady;

    // Every pipeline, built once; a module the implementation rejects leaves the host op, and says why.
    private bool GroupedGemmReady
    {
        get
        {
            if (_groupedGemmReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_groupedGemmReady is null)
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
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU grouped GEMM kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _groupedGemmReady = ok;
                }
                return _groupedGemmReady.Value;
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
