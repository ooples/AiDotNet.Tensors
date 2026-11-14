// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The sweep data transforms (IDataTransformBackend) on HIP, through PortableDataTransform and
// DataTransformKernelTemplate's kernels, compiled on first need.
public sealed partial class HipBackend : IDataTransformBackend
{
    private IntPtr _dataTransformModule;
    private bool? _dataTransformReady;

    // The module, built once with IEEE math; a compile the device rejects leaves the host ops, and says why.
    private bool DataTransformReady
    {
        get
        {
            if (_dataTransformReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dataTransformReady is null)
                {
                    bool ok;
                    try
                    {
                        var failure = CompileKernelModule(DataTransformKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                            "data_transform", ref _dataTransformModule, DataTransformKernelTemplate.GetKernelNames(),
                            useFastMath: false);
                        if (failure is not null)
                            System.Diagnostics.Trace.TraceWarning($"HIP data transform kernels unavailable: {failure.Message}");
                        ok = failure is null;
                        foreach (var name in DataTransformKernelTemplate.GetKernelNames()) ok &= HasTunedKernel(name);
                    }
                    catch (Exception ex) when (PortableDataTransform.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"HIP data transform kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _dataTransformReady = ok;
                }
                return _dataTransformReady.Value;
            }
        }
    }

    private unsafe void LaunchDataTransform(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        int count = buffers.Length + ints.Length;
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[count];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel(TunedKernel(kernelName), (uint)groups, (uint)DataTransformKernelTemplate.GroupSize, args);
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
