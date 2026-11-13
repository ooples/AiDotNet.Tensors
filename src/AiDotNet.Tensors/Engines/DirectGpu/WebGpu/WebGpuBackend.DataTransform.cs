#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The sweep data transforms (IDataTransformBackend) on WebGPU, through PortableDataTransform and the WGSL
// restatement of DataTransformKernelTemplate (WebGpuDataTransformKernels), built on first need.
public sealed partial class WebGpuBackend : IDataTransformBackend
{
    private const string DataTransformModulePrefix = "DataTransform_";
    private bool? _dataTransformReady;

    // Every pipeline, built once; a module the implementation rejects leaves the host ops, and says why.
    private bool DataTransformReady
    {
        get
        {
            if (_dataTransformReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_dataTransformReady is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in DataTransformKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(DataTransformModulePrefix + name,
                                    WebGpuDataTransformKernels.SourceFor(name), name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex) when (PortableDataTransform.IsBuildFailure(ex))
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU data transform kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _dataTransformReady = ok;
                }
                return _dataTransformReady.Value;
            }
        }
    }

    private void LaunchDataTransform(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds eight ints (bit-cast); the unused words are padding.
        var uniforms = new float[WebGpuDataTransformKernels.UniformWords];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(DataTransformModulePrefix + kernelName, WebGpuDataTransformKernels.SourceFor(kernelName),
            kernelName, buffers, uniforms, groups * DataTransformKernelTemplate.GroupSize).GetAwaiter().GetResult();
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
#endif
