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
    private IGpuBuffer? _winogradFusedU;

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
    internal bool TryConv2DForwardWinogradF23Fused(in CudaConv2DArgs a, bool oneBlockPerSm = false)
    {
        if (!WinogradF23Admits(a) || !EnsureWinogradFusedModule()) return false;
        return RunWinogradF23Fused(a.Input, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.PadH, "wfz_filter", a.Kernel,
            a.OutChannels, a.InChannels, a.Output, a.OutChannels, a.OutHeight, a.OutWidth, oneBlockPerSm);
    }

    /// <summary>
    /// Input gradient dX = conv(dY, w') (w' the 180-degree-rotated, channel-transposed filter, padding 2 - pad) through
    /// the fused F(2x2) kernel. Overwrites dX. Same admission as <see cref="TryConv2DBackwardDataWinogradF23"/>.
    /// </summary>
    internal bool TryConv2DBackwardDataWinogradF23Fused(in CudaConv2DBackwardArgs a, bool oneBlockPerSm = false)
    {
        if (!WinogradF23AdmitsData(a) || !EnsureWinogradFusedModule()) return false;
        return RunWinogradF23Fused(a.First, a.Batch, a.OutChannels, a.OutHeight, a.OutWidth, 2 - a.PadH, "wfz_filter_flip",
            a.Second, a.OutChannels, a.InChannels, a.Output, a.InChannels, a.InHeight, a.InWidth, oneBlockPerSm);
    }

    // y[n][outC][outH][outW] = conv3x3_s1(x[n][inC][h][w], padding pad): the named filter transform of the forward-layout
    // weights [fwdK][fwdC][3][3] into U[16][inC][outC], then the fused kernel over 32 output channels x 32 tiles per block.
    private unsafe bool RunWinogradF23Fused(IGpuBuffer x, int n, int inC, int h, int w, int pad, string filterKernel,
        IGpuBuffer weights, int fwdK, int fwdC, IGpuBuffer y, int outC, int outH, int outW, bool oneBlockPerSm)
    {
        int tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (p > int.MaxValue || 16L * inC * outC > int.MaxValue || (long)n * inC * h * w > int.MaxValue) return false;
        var u = TryConvScratch(ref _winogradFusedU, 16L * inC * outC);
        if (u is null) return false;

        using var _ = PushContext();
        IntPtr wPtr = weights.Handle, uPtr = u.Handle, xPtr = x.Handle, yPtr = y.Handle;
        uint block = (uint)DefaultBlockSize;

        void** fa = stackalloc void*[4];
        fa[0] = &wPtr; fa[1] = &uPtr; fa[2] = &fwdK; fa[3] = &fwdC;
        LaunchKernel(_kernelCache[filterKernel], (uint)(((long)fwdK * fwdC + block - 1) / block), block, fa);

        void** ca = stackalloc void*[14];
        ca[0] = &xPtr; ca[1] = &uPtr; ca[2] = &yPtr; ca[3] = &n; ca[4] = &inC; ca[5] = &h; ca[6] = &w; ca[7] = &outC;
        ca[8] = &outH; ca[9] = &outW; ca[10] = &pad; ca[11] = &tilesH; ca[12] = &tilesW;
        LaunchKernel3D(_kernelCache[oneBlockPerSm ? "wfz_conv1" : "wfz_conv"], (uint)((p + 31) / 32), (uint)((outC + 31) / 32), 1,
            256, 1, 1, ca, 0);
        return true;
    }
}
