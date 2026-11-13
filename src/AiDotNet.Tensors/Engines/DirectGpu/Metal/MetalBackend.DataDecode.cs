// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL, HIP and Vulkan instances, whose DataDecodePortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Raw bytes and numeric text into float32 (IDataDecodeBackend) on Metal, through PortableDataDecode and
// DataDecodeKernelTemplate's kernels, compiled as their own library on first need.
public sealed partial class MetalBackend : IDataDecodeBackend
{
    private const string DataDecodeLibraryName = "DataDecode";
    private IntPtr _dataDecodeLibrary;
    private bool? _dataDecodeReady;

    // The library, compiled once, and every kernel able to run its 256 threads in one threadgroup; otherwise the host op
    // runs, and the trace says why.
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
                        _dataDecodeLibrary = _shaderLibrary.CompileLibrary(DataDecodeLibraryName,
                            DataDecodeKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                        ok = _dataDecodeLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in DataDecodeKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(DataDecodeLibraryName, _dataDecodeLibrary, name).MaxTotalThreadsPerThreadgroup
                                      >= DataDecodeKernelTemplate.GroupSize;
                    }
                    catch (Exception ex) when (PortableDataDecode.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal data decode kernels unavailable: {ex.Message}");
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
        var pipeline = GetPipeline(DataDecodeLibraryName, _dataDecodeLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)DataDecodeKernelTemplate.GroupSize, 1, 1));
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
}
