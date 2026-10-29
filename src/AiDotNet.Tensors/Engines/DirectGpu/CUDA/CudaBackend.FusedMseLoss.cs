// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The compiled plan's fused mean-squared-error node (IFusedMseLossBackend, Kernels.CudaMseLossKernels) and its fused
// linear head + loss node (IFusedLinearMseLossBackend).
public sealed partial class CudaBackend : IFusedMseLossBackend, IFusedLinearMseLossBackend
{
    // The linear head + loss kernels (one block of 512 threads, CudaMseLossKernels): at most 16 outputs (w's rows are
    // padded to 20 floats and read as float4 chunks), with x, w, y, the partial predictions and the backward's dp staged
    // in at most 48 KB of shared memory. A parity LSTM's head is [32, 64] x [64, 10] (22 KB).
    private const int LinearMseMaxOutputs = 16, LinearMseThreads = 512, LinearMseRowFloats = 20, LinearMseKParts = 4;
    // 48 KB (the default per-block limit) less the kernels' static shared memory (the forward's 16 warp sums).
    private const int LinearMseMaxSharedFloats = 12 * 1024 - 16;

    private static long Round4(long n) => (n + 3) & ~3L;

    // Floats of dynamic shared memory, in the kernels' layout: x [m][k + 1], w [k][20], y, partials [4][m][20], dp [m][20],
    // the bias [16].
    private static long LinearMseSharedFloats(int m, int k, int n)
        => Round4((long)m * (k + 1)) + (long)k * LinearMseRowFloats + Round4((long)m * n)
           + (LinearMseKParts + 1L) * m * LinearMseRowFloats + LinearMseMaxOutputs;

    private static bool LinearMseShapeFits(int m, int k, int n)
        => m > 0 && k > 0 && n > 0 && n <= LinearMseMaxOutputs && LinearMseSharedFloats(m, k, n) <= LinearMseMaxSharedFloats;

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

    bool IFusedLinearMseLossBackend.SupportsLinearMseLoss(int m, int k, int n)
        => LinearMseShapeFits(m, k, n) && EnsureMseLossModule();

    unsafe bool IFusedLinearMseLossBackend.TryLinearMseLoss(IGpuBuffer x, IGpuBuffer w, IGpuBuffer? b, IGpuBuffer y,
        IGpuBuffer loss, int m, int k, int n)
    {
        if (!LinearMseShapeFits(m, k, n) || x.Size < m * k || w.Size < k * n || (b is not null && b.Size < n)
            || y.Size < m * n || loss.Size < 1
            || !EnsureMseLossModule() || !_kernelCache.TryGetValue("linear_mse_loss", out var headKernel))
            return false;
        using var _ = PushContext();
        IntPtr px = x.Handle, pw = w.Handle, pb = b?.Handle ?? IntPtr.Zero, py = y.Handle, pl = loss.Handle;
        int mm = m, kk = k, nn = n;
        void** args = stackalloc void*[8];
        args[0] = &px; args[1] = &pw; args[2] = &pb; args[3] = &py; args[4] = &pl;
        args[5] = &mm; args[6] = &kk; args[7] = &nn;
        LaunchKernelWithSharedMem(headKernel, 1, LinearMseThreads, (uint)(LinearMseSharedFloats(m, k, n) * sizeof(float)), args);
        return true;
    }

    unsafe bool IFusedLinearMseLossBackend.TryLinearMseLossBackward(IGpuBuffer x, IGpuBuffer w, IGpuBuffer? b,
        IGpuBuffer y, IGpuBuffer gradLoss, IGpuBuffer dx, IGpuBuffer dw, IGpuBuffer? db, int m, int k, int n)
    {
        if (!LinearMseShapeFits(m, k, n) || x.Size < m * k || w.Size < k * n || (b is not null && b.Size < n)
            || y.Size < m * n || gradLoss.Size < 1 || dx.Size < m * k || dw.Size < k * n || (db is not null && db.Size < n)
            || !EnsureMseLossModule() || !_kernelCache.TryGetValue("linear_mse_loss_backward", out var headBackwardKernel))
            return false;
        using var _ = PushContext();
        IntPtr px = x.Handle, pw = w.Handle, pb = b?.Handle ?? IntPtr.Zero, py = y.Handle, pg = gradLoss.Handle;
        IntPtr pdx = dx.Handle, pdw = dw.Handle, pdb = db?.Handle ?? IntPtr.Zero;
        int mm = m, kk = k, nn = n;
        void** args = stackalloc void*[11];
        args[0] = &px; args[1] = &pw; args[2] = &pb; args[3] = &py; args[4] = &pg;
        args[5] = &pdx; args[6] = &pdw; args[7] = &pdb; args[8] = &mm; args[9] = &kk; args[10] = &nn;
        LaunchKernelWithSharedMem(headBackwardKernel, 1, LinearMseThreads, (uint)(LinearMseSharedFloats(m, k, n) * sizeof(float)), args);
        return true;
    }
}
