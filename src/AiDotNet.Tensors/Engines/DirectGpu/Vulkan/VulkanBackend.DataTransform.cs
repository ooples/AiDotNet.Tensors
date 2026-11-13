// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Linq;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The sweep data transforms (IDataTransformBackend) on Vulkan, through PortableDataTransform and
// DataTransformKernelTemplate's kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend : IDataTransformBackend
{
    private static readonly Dictionary<string, string> s_dataTransformShaders = BuildDataTransformShaders();

    private static Dictionary<string, string> BuildDataTransformShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in DataTransformKernelTemplate.GetKernelNames())
            shaders[name] = DataTransformKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _dataTransformReady;

    // Every pipeline, built once; a shader the driver rejects leaves the host ops, and says why.
    private bool DataTransformReady
    {
        get
        {
            if (_dataTransformReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dataTransformReady is null)
                {
                    bool ok = true;
                    try
                    {
                        var failed = DataTransformKernelTemplate.Kernels.FirstOrDefault(spec =>
                            GetOrCreateGlslPipeline(s_dataTransformShaders[spec.Name], spec.Buffers.Length,
                                (uint)(spec.Ints.Length * sizeof(uint))) is null);
                        if (failed is not null)
                        {
                            System.Diagnostics.Trace.TraceWarning(
                                $"Vulkan data transform kernels unavailable: '{failed.Name}' did not build.");
                            ok = false;
                        }
                    }
                    catch (Exception ex) when (PortableDataTransform.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan data transform kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _dataTransformReady = ok;
                }
                return _dataTransformReady.Value;
            }
        }
    }

    private void LaunchDataTransform(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // Explicit push words, one per int, in the shader's push-constant order (no 64-bit pairs to align).
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_dataTransformShaders[kernelName], push, groups * DataTransformKernelTemplate.GroupSize, buffers);
    }

    bool IDataTransformBackend.TryBoxMuller(IGpuBuffer uniforms, IGpuBuffer output, int count, BoxMullerBranch branch,
        float mean, float std, bool clamp, float low, float high)
        => DataTransformReady
           && PortableDataTransform.TryBoxMuller(LaunchDataTransform, uniforms, output, count, branch, mean, std, clamp, low, high);

    bool IDataTransformBackend.TryUniformThreshold(IGpuBuffer uniforms, IGpuBuffer? thresholds, IGpuBuffer output, int count,
        bool above, float threshold, float onSuccess, float otherwise)
        => DataTransformReady
           && PortableDataTransform.TryUniformThreshold(LaunchDataTransform, uniforms, thresholds, output, count, above, threshold,
               onSuccess, otherwise);

    bool IDataTransformBackend.TryLinearFilter(IGpuBuffer x, IGpuBuffer b, IGpuBuffer a, IGpuBuffer? zi, IGpuBuffer y,
        IGpuBuffer workspace, int signals, int length, int nb, int na, bool guard, float maxAbs)
        => DataTransformReady
           && PortableDataTransform.TryLinearFilter(LaunchDataTransform, x, b, a, zi, y, workspace, signals, length, nb, na, guard,
               maxAbs);

    bool IDataTransformBackend.TryGatherRows(IGpuBuffer table, IGpuBuffer ids, IGpuBuffer output, int rows, int rowSize)
        => DataTransformReady && PortableDataTransform.TryGatherRows(LaunchDataTransform, table, ids, output, rows, rowSize);

    bool IDataTransformBackend.TryScatterRows(IGpuBuffer source, IGpuBuffer ids, IGpuBuffer table, int rows, int rowSize)
        => DataTransformReady && PortableDataTransform.TryScatterRows(LaunchDataTransform, source, ids, table, rows, rowSize);

    bool IDataTransformBackend.TryRoundIntegral(IGpuBuffer input, IGpuBuffer output, int count, RoundingMode mode, float low,
        float high)
        => DataTransformReady && PortableDataTransform.TryRoundIntegral(LaunchDataTransform, input, output, count, mode, low, high);
}
