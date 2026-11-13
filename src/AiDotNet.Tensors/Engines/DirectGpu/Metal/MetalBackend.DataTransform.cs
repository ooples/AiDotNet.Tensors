// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL, HIP and Vulkan instances, whose DataTransformPortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The sweep data transforms (IDataTransformBackend) on Metal, through PortableDataTransform and
// DataTransformKernelTemplate's kernels, compiled as their own strict-math library on first need.
public sealed partial class MetalBackend : IDataTransformBackend
{
    private const string DataTransformLibraryName = "DataTransform";
    private IntPtr _dataTransformLibrary;
    private bool? _dataTransformReady;

    // The library, compiled once with fast math off (the NaN guards, the exact threshold comparisons and the rounding need
    // IEEE semantics), and every kernel able to run 256 threads in one threadgroup; otherwise the host ops run.
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
                        _dataTransformLibrary = _shaderLibrary.CompileLibrary(DataTransformLibraryName,
                            DataTransformKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal), requireStrictFloatingPoint: true);
                        ok = _dataTransformLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in DataTransformKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(DataTransformLibraryName, _dataTransformLibrary, name).MaxTotalThreadsPerThreadgroup
                                      >= DataTransformKernelTemplate.GroupSize;
                    }
                    catch (Exception ex) when (PortableDataTransform.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal data transform kernels unavailable: {ex.Message}");
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
        var pipeline = GetPipeline(DataTransformLibraryName, _dataTransformLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)DataTransformKernelTemplate.GroupSize, 1, 1));
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
