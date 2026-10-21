// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #1489: the register-resident training BatchNorm (IBatchNormVarianceBackend) on HIP, through PortableBatchNorm and
// BatchNormRegisterKernelTemplate's kernels, compiled at initialization beside the Winograd modules.
public sealed partial class HipBackend : IBatchNormVarianceBackend
{
    private IntPtr _batchNormRegisterModule;

    private bool BatchNormRegisterReady
    {
        get
        {
            foreach (var name in BatchNormRegisterKernelTemplate.GetKernelNames())
                if (!HasTunedKernel(name)) return false;
            return true;
        }
    }

    // Strict math, as the Winograd modules: the mask the backward rebuilds must round as the forward's value did.
    private void CompileBatchNormRegisterKernels()
    {
        var failure = CompileKernelModule(BatchNormRegisterKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
            "bn_register", ref _batchNormRegisterModule, BatchNormRegisterKernelTemplate.GetKernelNames(), useFastMath: false);
        if (failure is not null)
            System.Diagnostics.Trace.TraceWarning($"HIP register-resident BatchNorm kernels unavailable: {failure.Message}");
    }

    private unsafe void LaunchBatchNormRegister(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        int count = buffers.Length + ints.Length;
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[count];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel(TunedKernel(kernelName), (uint)groups, (uint)BatchNormRegisterKernelTemplate.GroupSize, args);
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
