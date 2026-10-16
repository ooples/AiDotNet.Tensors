using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Fused Winograd F(2x2, 3x3) (<see cref="Kernels.CudaConvWinogradFusedKernels"/>): a filter transform, then one kernel
/// that transforms its input patches in shared memory, runs the sixteen tile GEMMs and applies the output transform.
/// Same admission and the same fp32 arithmetic as the unfused F(2x2) path. The module compiles on first use.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly object _winogradFusedModuleLock = new();
    private bool _winogradFusedModuleTried, _winogradFusedModuleReady;
    private IGpuBuffer? _winogradFusedU, _winogradFusedPartials;

    internal bool EnsureWinogradFusedModule()
    {
        lock (_winogradFusedModuleLock)
        {
            if (_winogradFusedModuleTried) return _winogradFusedModuleReady;
            _winogradFusedModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaConvWinogradFusedKernels.GetSource(), "conv_winograd_fused_kernels",
                    Kernels.CudaConvWinogradFusedKernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvWinogradFusedKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _winogradFusedModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Fused Winograd convolution kernels unavailable: {ex.Message}");
            }
            return _winogradFusedModuleReady;
        }
    }

    /// <summary>Forward y = conv(x, w) through the fused F(2x2) kernel. Overwrites y. False (nothing launched) when the
    /// shape is not admitted or the module is unavailable.</summary>
    internal bool TryConv2DForwardWinogradF23Fused(in CudaConv2DArgs a)
    {
        if (!WinogradF23Admits(a) || !EnsureWinogradFusedModule()) return false;
        return RunWinogradF23Fused(a.Input, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.PadH, "wfz_filter", a.Kernel,
            a.OutChannels, a.InChannels, a.Output, a.OutChannels, a.OutHeight, a.OutWidth);
    }

    /// <summary>
    /// Input gradient dX = conv(dY, w') (w' the 180-degree-rotated, channel-transposed filter, padding 2 - pad) through
    /// the fused F(2x2) kernel. Overwrites dX. Same admission as <see cref="TryConv2DBackwardDataWinogradF23"/>.
    /// </summary>
    internal bool TryConv2DBackwardDataWinogradF23Fused(in CudaConv2DBackwardArgs a)
    {
        if (!WinogradF23AdmitsData(a) || !EnsureWinogradFusedModule()) return false;
        return RunWinogradF23Fused(a.First, a.Batch, a.OutChannels, a.OutHeight, a.OutWidth, 2 - a.PadH, "wfz_filter_flip",
            a.Second, a.OutChannels, a.InChannels, a.Output, a.InChannels, a.InHeight, a.InWidth);
    }

    // y[n][outC][outH][outW] = conv3x3_s1(x[n][inC][h][w], padding pad): the named filter transform of the forward-layout
    // weights [fwdK][fwdC][3][3] into U[16][inC][outC], then the fused kernel (512 threads, one block per SM) over 32
    // output channels x 32 tiles per block. On deep layers the tiles alone give few blocks (512@4x4 at batch 8: 16
    // blocks of 64 eight-channel steps each), so the channel reduction is split across blockIdx.z: the split count
    // minimizes waves x steps per block (the critical path), smallest count on ties; the partial outputs are summed in
    // split order (conv2d_wgrad_splitk_reduce), so the result stays deterministic.
    private unsafe bool RunWinogradF23Fused(IGpuBuffer x, int n, int inC, int h, int w, int pad, string filterKernel,
        IGpuBuffer weights, int fwdK, int fwdC, IGpuBuffer y, int outC, int outH, int outW)
    {
        int tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (p > int.MaxValue || 16L * inC * outC > int.MaxValue || (long)n * inC * h * w > int.MaxValue) return false;
        var u = TryConvScratch(ref _winogradFusedU, 16L * inC * outC);
        if (u is null) return false;

        using var _ = PushContext();
        IntPtr wPtr = weights.Handle, uPtr = u.Handle, xPtr = x.Handle;
        uint block = (uint)DefaultBlockSize;

        void** fa = stackalloc void*[4];
        fa[0] = &wPtr; fa[1] = &uPtr; fa[2] = &fwdK; fa[3] = &fwdC;
        LaunchKernel(_kernelCache[filterKernel], (uint)(((long)fwdK * fwdC + block - 1) / block), block, fa);

        long blocks = ((p + 31) / 32) * ((outC + 31) / 32);
        int sms = Math.Max(1, _multiProcessorCount), totalSteps = (inC + 7) / 8;
        int splits = 1;
        long bestCost = long.MaxValue;
        for (int s = 1; s <= Math.Min(16, totalSteps); s++)
        {
            // In half-step units: each wave costs its blocks' steps plus ~3 steps of per-block prologue and epilogue
            // (measured: 16 splits of 128@16 ran 15 one-step waves in 95 us), and each extra split one half step of
            // partial traffic and reduce.
            long cost = (blocks * s + sms - 1) / sms * 2 * ((totalSteps + s - 1) / s + 3) + (s - 1);
            if (cost < bestCost) { bestCost = cost; splits = s; }
        }
        int rPerSplit = (totalSteps + splits - 1) / splits * 8;
        splits = (inC + rPerSplit - 1) / rPerSplit;
        long outputs = (long)n * outC * outH * outW;
        IntPtr reduce = IntPtr.Zero;
        IGpuBuffer target = y;
        if (splits > 1)
        {
            if (!_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce) || outputs * splits > int.MaxValue) return false;
            var partials = TryConvScratch(ref _winogradFusedPartials, splits * outputs);
            if (partials is null) return false;
            target = partials;
        }
        IntPtr tPtr = target.Handle;

        void** ca = stackalloc void*[14];
        ca[0] = &xPtr; ca[1] = &uPtr; ca[2] = &tPtr; ca[3] = &n; ca[4] = &inC; ca[5] = &h; ca[6] = &w; ca[7] = &outC;
        ca[8] = &outH; ca[9] = &outW; ca[10] = &pad; ca[11] = &tilesH; ca[12] = &tilesW; ca[13] = &rPerSplit;
        LaunchKernel3D(_kernelCache["wfz_conv"], (uint)((p + 31) / 32), (uint)((outC + 31) / 32), (uint)splits, 512, 1, 1, ca, 0);
        if (splits > 1) LaunchSplitKReduce(reduce, target, y, (int)outputs, splits);
        return true;
    }
}
