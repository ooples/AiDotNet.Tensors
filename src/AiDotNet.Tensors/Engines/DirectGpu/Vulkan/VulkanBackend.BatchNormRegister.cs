// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1489: the register-resident training BatchNorm (IBatchNormVarianceBackend) on Vulkan, through PortableBatchNorm and
// BatchNormRegisterKernelTemplate's kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend : IBatchNormVarianceBackend
{
    private static readonly Dictionary<string, string> s_batchNormRegisterShaders = BuildBatchNormRegisterShaders();

    private static Dictionary<string, string> BuildBatchNormRegisterShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in BatchNormRegisterKernelTemplate.GetKernelNames())
            shaders[name] = BatchNormRegisterKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _batchNormRegisterReady;

    // Every pipeline, built once; a shader the driver rejects leaves the composed BatchNorm path, and says why.
    private bool BatchNormRegisterReady
    {
        get
        {
            if (_batchNormRegisterReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_batchNormRegisterReady is null)
                {
                    bool ok = true;
                    try
                    {
                        foreach (var spec in BatchNormRegisterKernelTemplate.Kernels)
                            if (GetOrCreateGlslPipeline(s_batchNormRegisterShaders[spec.Name], spec.Buffers.Length,
                                    (uint)(spec.Ints.Length * sizeof(uint))) is null)
                            {
                                System.Diagnostics.Trace.TraceWarning(
                                    $"Vulkan register-resident BatchNorm kernels unavailable: '{spec.Name}' did not build.");
                                ok = false;
                                break;
                            }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan register-resident BatchNorm kernels unavailable: {ex.Message}");
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
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_batchNormRegisterShaders[kernelName], push,
            groups * BatchNormRegisterKernelTemplate.GroupSize, buffers);
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
