// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Raw bytes and numeric text into float32 (IDataDecodeBackend) on Vulkan, through PortableDataDecode and
// DataDecodeKernelTemplate's kernels as GLSL compute shaders, built on first need. Every binding is a uint / int array,
// so no value is loaded as a float (a float load flushes subnormals on this driver family).
public sealed unsafe partial class VulkanBackend : IDataDecodeBackend
{
    private static readonly Dictionary<string, string> s_dataDecodeShaders = BuildDataDecodeShaders();

    private static Dictionary<string, string> BuildDataDecodeShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in DataDecodeKernelTemplate.GetKernelNames())
            shaders[name] = DataDecodeKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _dataDecodeReady;

    // Every pipeline, built once; a shader the driver rejects leaves the host op, and says why.
    private bool DataDecodeReady
    {
        get
        {
            if (_dataDecodeReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dataDecodeReady is null)
                {
                    bool ok;
                    try
                    {
                        ok = DataDecodeKernelTemplate.Kernels.All(spec =>
                            GetOrCreateGlslPipeline(s_dataDecodeShaders[spec.Name], spec.Buffers.Length,
                                (uint)(spec.Ints.Length * sizeof(uint))) is not null);
                        if (!ok) System.Diagnostics.Trace.TraceWarning("Vulkan data decode kernels unavailable: a pipeline did not build.");
                    }
                    catch (Exception ex) when (PortableDataDecode.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan data decode kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _dataDecodeReady = ok;
                }
                return _dataDecodeReady.Value;
            }
        }
    }

    private void LaunchDataDecode(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The whole push block, every int in order (the size-only overloads push just the first word).
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_dataDecodeShaders[kernelName], push, groups * DataDecodeKernelTemplate.GroupSize, buffers);
    }

    bool IDataDecodeBackend.TryDecodeBytes(IGpuBuffer words, IGpuBuffer output, int count, NumericStorageType sourceType,
        bool littleEndian)
        => DataDecodeReady && PortableDataDecode.TryDecode(LaunchDataDecode, words, output, count, sourceType, littleEndian);

    bool IDataDecodeBackend.TryCountSeparators(IGpuBuffer words, IGpuBuffer segmentCounts, int length, byte delimiter)
        => DataDecodeReady && PortableDataDecode.TryCountSeparators(LaunchDataDecode, words, segmentCounts, length, delimiter);

    bool IDataDecodeBackend.TryExclusiveScan(IGpuBuffer values, IGpuBuffer total, int count)
        => DataDecodeReady && PortableDataDecode.TryScan(LaunchDataDecode, values, total, count);

    bool IDataDecodeBackend.TryScatterFieldBounds(IGpuBuffer words, IGpuBuffer segmentOffsets, IGpuBuffer bounds, int length,
        byte delimiter, bool trailingNewline)
        => DataDecodeReady && PortableDataDecode.TryScatterBounds(LaunchDataDecode, words, segmentOffsets, bounds, length,
            delimiter, trailingNewline);

    bool IDataDecodeBackend.TryParseFields(IGpuBuffer words, IGpuBuffer bounds, IGpuBuffer output, IGpuBuffer status,
        int fields, int styleMask, float missingValue)
        => DataDecodeReady && PortableDataDecode.TryParse(LaunchDataDecode, words, bounds, output, status, fields, styleMask,
            missingValue);

    bool IDataDecodeBackend.TryEncodeBytes(IGpuBuffer input, IGpuBuffer words, int count, NumericStorageType targetType,
        bool littleEndian)
        => DataDecodeReady && PortableDataDecode.TryEncode(LaunchDataDecode, input, words, count, targetType, littleEndian);
}
