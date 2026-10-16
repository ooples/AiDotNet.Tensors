using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Winograd F(2x2, 3x3) forward convolution (<see cref="Kernels.CudaConvWinogradKernels"/>) for 3x3, stride 1,
/// dilation 1: filter transform, input transform, sixteen GEMMs on the #1325 engine main loop, output transform.
/// The module compiles on first use.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly object _winogradModuleLock = new();
    private bool _winogradModuleTried, _winogradModuleReady;
    private IGpuBuffer? _winogradU, _winogradV, _winogradM;

    internal bool EnsureWinogradModule()
    {
        lock (_winogradModuleLock)
        {
            if (_winogradModuleTried) return _winogradModuleReady;
            _winogradModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaConvWinogradKernels.GetSource(), "conv_winograd_kernels",
                    Kernels.CudaConvWinogradKernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvWinogradKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _winogradModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Winograd convolution kernels unavailable: {ex.Message}");
            }
            return _winogradModuleReady;
        }
    }

    /// <summary>True for the shapes this path computes: 3x3, stride 1, dilation 1, equal padding.</summary>
    internal static bool WinogradF23Admits(in CudaConv2DArgs a)
        => a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 1 && a.StrideW == 1 && a.DilationH == 1 && a.DilationW == 1
           && a.PadH == a.PadW && a.OutHeight > 0 && a.OutWidth > 0;

    internal bool TryConv2DForwardWinogradF23(in CudaConv2DArgs a)
    {
        if (!WinogradF23Admits(a) || !EnsureWinogradModule()) return false;
        // y[n][K] = conv(x[n][C], w[K][C]): U[x][K][C] from w, GEMM rows = K, reduction = C.
        return RunWinogradF23(a.Input, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.PadH,
            "wino_f23_filter", a.Kernel, a.OutChannels, a.InChannels, a.Output, a.OutChannels, a.OutHeight, a.OutWidth);
    }

    /// <summary>True for the input-gradient shapes this path computes (the forward conv's 3x3, stride 1, dilation 1).</summary>
    internal static bool WinogradF23AdmitsData(in CudaConv2DBackwardArgs a)
        => !a.WeightGradient && a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 1 && a.StrideW == 1
           && a.DilationH == 1 && a.DilationW == 1 && a.PadH == a.PadW && a.PadH <= 2 && a.OutHeight > 0 && a.OutWidth > 0;

    /// <summary>
    /// Input gradient dX = conv(dY, w') with w' the 180-degree-rotated, channel-transposed filter and padding 2 - pad:
    /// the forward Winograd pipeline over dY (K channels, outH x outW) producing dX (C channels, H x W).
    /// </summary>
    internal bool TryConv2DBackwardDataWinogradF23(in CudaConv2DBackwardArgs a)
    {
        if (!WinogradF23AdmitsData(a) || !EnsureWinogradModule()) return false;
        return RunWinogradF23(a.First, a.Batch, a.OutChannels, a.OutHeight, a.OutWidth, 2 - a.PadH,
            "wino_f23_filter_flip", a.Second, a.OutChannels, a.InChannels, a.Output, a.InChannels, a.InHeight, a.InWidth);
    }

    // The pipeline for y[n][outC][outH][outW] = conv3x3_s1(x[n][inC][h][w], padding pad): the named filter transform of
    // w (forward layout [fwdK][fwdC][3][3]) into U[16][outC][inC], the input transform, sixteen GEMMs, the output transform.
    private unsafe bool RunWinogradF23(IGpuBuffer x, int n, int inC, int h, int w, int pad,
        string filterKernel, IGpuBuffer weights, int fwdK, int fwdC, IGpuBuffer y, int outC, int outH, int outW)
    {
        int tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (16L * inC * p > int.MaxValue || 16L * outC * p > int.MaxValue || 16L * outC * inC > int.MaxValue) return false;
        int pi = (int)p;
        var u = TryConvScratch(ref _winogradU, 16L * outC * inC);
        var v = TryConvScratch(ref _winogradV, 16L * inC * p);
        var m = TryConvScratch(ref _winogradM, 16L * outC * p);
        if (u is null || v is null || m is null) return false;

        using var _ = PushContext();
        IntPtr wPtr = weights.Handle, uPtr = u.Handle, xPtr = x.Handle, vPtr = v.Handle, mPtr = m.Handle, yPtr = y.Handle;
        uint block = (uint)DefaultBlockSize;

        void** fa = stackalloc void*[4];
        fa[0] = &wPtr; fa[1] = &uPtr; fa[2] = &fwdK; fa[3] = &fwdC;
        LaunchKernel(_kernelCache[filterKernel], (uint)(((long)fwdK * fwdC + block - 1) / block), block, fa);

        void** ia = stackalloc void*[9];
        ia[0] = &xPtr; ia[1] = &vPtr; ia[2] = &n; ia[3] = &inC; ia[4] = &h; ia[5] = &w; ia[6] = &pad; ia[7] = &tilesH; ia[8] = &tilesW;
        LaunchKernel(_kernelCache["wino_f23_input"], (uint)((inC * p + block - 1) / block), block, ia);

        // 128x128 tiles only while they still fill the SMs (the sixteen GEMMs share the grid through z).
        int tile = (long)((pi + 127) / 128) * ((outC + 127) / 128) * 16 >= Math.Max(1, _multiProcessorCount) ? 128 : 64;
        int gx = (pi + tile - 1) / tile, gy = (outC + tile - 1) / tile;
        void** ga = stackalloc void*[6];
        ga[0] = &uPtr; ga[1] = &vPtr; ga[2] = &mPtr; ga[3] = &outC; ga[4] = &pi; ga[5] = &inC;
        LaunchKernel3D(_kernelCache[tile == 128 ? "wino_gemm128" : "wino_gemm64"], (uint)gx, (uint)gy, 16, block, 1, 1, ga, 0);

        void** oa = stackalloc void*[8];
        oa[0] = &mPtr; oa[1] = &yPtr; oa[2] = &n; oa[3] = &outC; oa[4] = &outH; oa[5] = &outW; oa[6] = &tilesH; oa[7] = &tilesW;
        LaunchKernel(_kernelCache["wino_f23_output"], (uint)((outC * p + block - 1) / block), block, oa);
        return true;
    }
}
