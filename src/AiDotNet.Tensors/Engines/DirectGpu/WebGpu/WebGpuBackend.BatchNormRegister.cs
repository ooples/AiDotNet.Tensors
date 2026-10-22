#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1489: the register-resident training BatchNorm (IBatchNormVarianceBackend) on WebGPU, through PortableBatchNorm and
// the WGSL restatement of BatchNormRegisterKernelTemplate (WebGpuBatchNormRegisterKernels), built on first need.
public sealed partial class WebGpuBackend : IBatchNormVarianceBackend
{
    private const string BatchNormRegisterModulePrefix = "BatchNormRegister_";
    private bool? _batchNormRegisterReady;

    // Every pipeline, built once; a module the implementation rejects leaves the composed BatchNorm path, and says why.
    private bool BatchNormRegisterReady
    {
        get
        {
            if (_batchNormRegisterReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_batchNormRegisterReady is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in BatchNormRegisterKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(BatchNormRegisterModulePrefix + name,
                                    WebGpuBatchNormRegisterKernels.SourceFor(name), name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU register-resident BatchNorm kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _batchNormRegisterReady = ok;
                }
                return _batchNormRegisterReady.Value;
            }
        }
    }

    private void LaunchBatchNormRegister(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds the six ints (bit-cast) and two words of padding.
        var uniforms = new float[8];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(BatchNormRegisterModulePrefix + kernelName, WebGpuBatchNormRegisterKernels.SourceFor(kernelName),
            kernelName, buffers, uniforms, groups * BatchNormRegisterKernelTemplate.GroupSize).GetAwaiter().GetResult();
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
            gradResidual, beta, gradOutput2, canRebuildMask: false);   // WGSL's fma need not round once: read the mask
}
#endif
