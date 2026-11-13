#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Raw bytes and numeric text into float32 (IDataDecodeBackend) on WebGPU, through PortableDataDecode and the WGSL
// restatement of DataDecodeKernelTemplate (WebGpuDataDecodeKernels), built on first need.
public sealed partial class WebGpuBackend : IDataDecodeBackend
{
    private const string DataDecodeModulePrefix = "DataDecode_";
    private bool? _dataDecodeReady;

    // Every pipeline, built once; a module the implementation rejects leaves the host op, and says why.
    private bool DataDecodeReady
    {
        get
        {
            if (_dataDecodeReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dataDecodeReady is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in DataDecodeKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(DataDecodeModulePrefix + name, WebGpuDataDecodeKernels.SourceFor(name),
                                    name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex) when (PortableDataDecode.IsBuildFailure(ex))
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU data decode kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _dataDecodeReady = ok;
                }
                return _dataDecodeReady.Value;
            }
        }
    }

    private void LaunchDataDecode(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds eight ints (bit-cast); a kernel with fewer leaves the rest as padding.
        var uniforms = new float[WebGpuDataDecodeKernels.UniformWords];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(DataDecodeModulePrefix + kernelName, WebGpuDataDecodeKernels.SourceFor(kernelName), kernelName,
            buffers, uniforms, groups * DataDecodeKernelTemplate.GroupSize).GetAwaiter().GetResult();
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
#endif
