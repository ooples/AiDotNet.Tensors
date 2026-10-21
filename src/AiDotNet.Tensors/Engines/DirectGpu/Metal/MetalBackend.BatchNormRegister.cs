// Copyright (c) AiDotNet. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL instance, whose BatchNormRegisterPortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #1489: the register-resident training BatchNorm (IBatchNormVarianceBackend) on Metal, through PortableBatchNorm and
// BatchNormRegisterKernelTemplate's kernels, compiled as their own library on first need.
public sealed partial class MetalBackend : IBatchNormVarianceBackend
{
    private const string BatchNormRegisterLibraryName = "BatchNormRegister";
    private IntPtr _batchNormRegisterLibrary;
    private bool? _batchNormRegisterReady;

    // The library, compiled once, and every kernel able to run its 256 threads in one threadgroup (the compiler can cap
    // a pipeline below that under register pressure); otherwise the composed BatchNorm path runs, and says why.
    private bool BatchNormRegisterReady
    {
        get
        {
            if (_batchNormRegisterReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_batchNormRegisterReady is null)
                {
                    bool ok;
                    try
                    {
                        _batchNormRegisterLibrary = _shaderLibrary.CompileLibrary(BatchNormRegisterLibraryName,
                            BatchNormRegisterKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                        ok = _batchNormRegisterLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in BatchNormRegisterKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(BatchNormRegisterLibraryName, _batchNormRegisterLibrary, name)
                                    .MaxTotalThreadsPerThreadgroup >= BatchNormRegisterKernelTemplate.GroupSize;
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal register-resident BatchNorm kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _batchNormRegisterReady = ok;
                }
                return _batchNormRegisterReady.Value;
            }
        }
    }

    private void LaunchBatchNormRegister(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var pipeline = GetPipeline(BatchNormRegisterLibraryName, _batchNormRegisterLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)BatchNormRegisterKernelTemplate.GroupSize, 1, 1));
    }

    bool IBatchNormVarianceBackend.TryBatchNormTrainingWithVariance(IGpuBuffer input, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer saveMean, IGpuBuffer saveVariance,
        int batch, int channels, int spatial, float epsilon, bool relu, IGpuBuffer? residual)
        => BatchNormRegisterReady && PortableBatchNorm.TryForward(this, LaunchBatchNormRegister, input, output, gamma, beta,
            saveMean, saveVariance, batch, channels, spatial, epsilon, relu, residual);

    bool IBatchNormVarianceBackend.TryBatchNormBackwardFromVariance(IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma,
        IGpuBuffer gradBeta, int batch, int channels, int spatial, float epsilon, IGpuBuffer? reluOutput,
        IGpuBuffer? gradResidual, IGpuBuffer? beta, IGpuBuffer? gradOutput2)
        => BatchNormRegisterReady && PortableBatchNorm.TryBackward(this, LaunchBatchNormRegister, gradOutput, input, gamma,
            saveMean, saveVariance, gradInput, gradGamma, gradBeta, batch, channels, spatial, epsilon, reluOutput,
            gradResidual, beta, gradOutput2);
}
