// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The compiled plan's fused mean-squared-error node (IFusedMseLossBackend, Kernels.CudaMseLossKernels).
public sealed partial class CudaBackend : IFusedMseLossBackend
{
    // One block reduces the whole loss; past this many elements a single block is slower than the composed ops' grid.
    private const int MseLossMaxLength = 1 << 22;

    private readonly object _mseLossModuleLock = new();
    private bool _mseLossModuleTried;
    private bool _mseLossModuleReady;

    private bool EnsureMseLossModule()
    {
        lock (_mseLossModuleLock)
        {
            if (_mseLossModuleTried) return _mseLossModuleReady;
            _mseLossModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaMseLossKernels.GetSource(), "mse_loss_kernels",
                    Kernels.CudaMseLossKernels.GetKernelNames());
                foreach (var name in Kernels.CudaMseLossKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _mseLossModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Fused MSE loss kernels unavailable: {ex.GetType().Name}: {ex.Message}");
            }
            return _mseLossModuleReady;
        }
    }

    unsafe bool IFusedMseLossBackend.TryMseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int length)
    {
        if (length <= 0 || length > MseLossMaxLength || predictions.Size < length || targets.Size < length || loss.Size < 1
            || !EnsureMseLossModule() || !_kernelCache.TryGetValue("mse_loss_mean", out var meanKernel))
            return false;
        using var _ = PushContext();
        IntPtr p = predictions.Handle, y = targets.Handle, l = loss.Handle;
        int n = length;
        void** args = stackalloc void*[4];
        args[0] = &p; args[1] = &y; args[2] = &l; args[3] = &n;
        LaunchKernel(meanKernel, 1, 1024, args);
        return true;
    }

    unsafe bool IFusedMseLossBackend.TryMseLossBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradLoss,
        IGpuBuffer gradPredictions, IGpuBuffer? gradTargets, int length)
    {
        if (length <= 0 || predictions.Size < length || targets.Size < length || gradLoss.Size < 1
            || gradPredictions.Size < length || (gradTargets is not null && gradTargets.Size < length)
            || !EnsureMseLossModule() || !_kernelCache.TryGetValue("mse_loss_backward", out var backwardKernel))
            return false;
        using var _ = PushContext();
        IntPtr p = predictions.Handle, y = targets.Handle, g = gradLoss.Handle, dp = gradPredictions.Handle;
        IntPtr dy = gradTargets?.Handle ?? IntPtr.Zero;
        int n = length;
        void** args = stackalloc void*[6];
        args[0] = &p; args[1] = &y; args[2] = &g; args[3] = &dp; args[4] = &dy; args[5] = &n;
        LaunchKernel(backwardKernel, (uint)Math.Min(4096, (length + 255) / 256), 256, args);
        return true;
    }
}
