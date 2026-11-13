// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Raw bytes and numeric text into float32 (IDataDecodeBackend) on CUDA, through PortableDataDecode and
// DataDecodeKernelTemplate's kernels (its HIP form is valid CUDA C++), compiled on first need in a module of their own.
public sealed partial class CudaBackend : IDataDecodeBackend
{
    private bool? _dataDecodeReady;

    // The module, built once; a compile NVRTC rejects leaves the host op, and says why.
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
                        using var _ = PushContext();
                        CompileKernelModule(_cuDevice, DataDecodeKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                            "data_decode", DataDecodeKernelTemplate.GetKernelNames(), useFastMath: false);
                        ok = true;
                        foreach (var name in DataDecodeKernelTemplate.GetKernelNames()) ok &= HasTunedKernel(name);
                    }
                    catch (Exception ex) when (PortableDataDecode.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"CUDA data decode kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _dataDecodeReady = ok;
                }
                return _dataDecodeReady.Value;
            }
        }
    }

    private unsafe void LaunchDataDecode(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        int count = buffers.Length + ints.Length;
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[count];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        using var _ = PushContext();
        LaunchKernel(_kernelCache[kernelName], (uint)groups, (uint)DataDecodeKernelTemplate.GroupSize, args);
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
