// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The sweep data transforms (IDataTransformBackend) on CUDA: DataTransformKernelTemplate's HIP dialect is
// CUDA C++ as well, compiled by NVRTC on first need and launched through PortableDataTransform.
public sealed partial class CudaBackend : IDataTransformBackend
{
    private readonly object _dataTransformLock = new();
    private bool _dataTransformTried;
    private bool _dataTransformReady;

    // The module, built once with IEEE math (the NaN guards, the exact threshold comparisons and the rounding need it);
    // otherwise the host ops run, and the trace says why.
    private bool DataTransformReady
    {
        get
        {
            lock (_dataTransformLock)
            {
                if (_dataTransformTried) return _dataTransformReady;
                _dataTransformTried = true;
                try
                {
                    using var _ = PushContext();
                    if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                    CompileKernelModule(device, DataTransformKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                        "data_transform_kernels", DataTransformKernelTemplate.GetKernelNames(), useFastMath: false);
                    bool ok = true;
                    foreach (var name in DataTransformKernelTemplate.GetKernelNames()) ok &= _kernelCache.ContainsKey(name);
                    _dataTransformReady = ok;
                }
                catch (Exception ex) when (PortableDataTransform.IsBuildFailure(ex))
                {
                    System.Diagnostics.Trace.TraceWarning($"CUDA data transform kernels unavailable: {ex.GetType().Name}: {ex.Message}");
                }
                return _dataTransformReady;
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
        using var _ = PushContext();
        LaunchKernel(_kernelCache[kernelName], (uint)groups, (uint)DataTransformKernelTemplate.GroupSize, args);
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
