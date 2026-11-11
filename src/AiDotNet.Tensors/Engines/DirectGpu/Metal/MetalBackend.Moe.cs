// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL and Vulkan instances, whose MoeFusedPortTests run on a real device; MetalShaderCompilationTests compiles it.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The fused MoE kernels (IMoeKernelBackend): MoeKernelTemplate's Metal library, compiled on first need.
public sealed partial class MetalBackend : IMoeKernelBackend
{
    private const string MoeLibraryName = "Moe";
    private IntPtr _moeLibrary;
    private bool? _moeReady;

    bool IMoeKernelBackend.MoeKernelsReady
    {
        get
        {
            if (_moeReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_moeReady is null)
                {
                    bool ok;
                    try
                    {
                        _moeLibrary = _shaderLibrary.CompileLibrary(MoeLibraryName,
                            MoeKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                        ok = _moeLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in MoeKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(MoeLibraryName, _moeLibrary, name).MaxTotalThreadsPerThreadgroup
                                      >= MoeKernelTemplate.GroupSize;
                    }
                    catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal fused MoE kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _moeReady = ok;
                }
                return _moeReady.Value;
            }
        }
    }

    void IMoeKernelBackend.LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var pipeline = GetPipeline(MoeLibraryName, _moeLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1), new MTLSize((ulong)MoeKernelTemplate.GroupSize, 1, 1));
    }
}
