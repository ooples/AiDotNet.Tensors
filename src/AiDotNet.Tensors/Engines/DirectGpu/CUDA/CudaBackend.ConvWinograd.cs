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

    // Batched-GEMM tile edges: the wino_gemm128 and wino_gemm64 instantiations.
    private const int WinogradLargeTile = 128, WinogradSmallTile = 64;
    // Weight-gradient split-K: split the P reduction until the grid holds this many blocks per SM, never leaving a
    // split with fewer than WinogradMinTilesPerSplit tiles of P.
    private const long WinogradSplitOccupancy = 2;
    private const int WinogradMinTilesPerSplit = 64;

    /// <summary>
    /// The batched-GEMM tile edge: 128x128 tiles only while they still fill the SMs (the sixteen GEMMs share the grid
    /// through z), 64x64 otherwise.
    /// </summary>
    private int ChooseWinogradTile(long rows, long cols)
        => (rows + WinogradLargeTile - 1) / WinogradLargeTile * ((cols + WinogradLargeTile - 1) / WinogradLargeTile) * 16
           >= Math.Max(1, _multiProcessorCount) ? WinogradLargeTile : WinogradSmallTile;

    internal bool EnsureWinogradModule()
    {
        lock (_winogradModuleLock)
        {
            if (_winogradModuleTried) return _winogradModuleReady;
            _winogradModuleTried = true;
            try
            {
                using var _ = PushContext();
                // Every refusal is logged: the result is cached for the backend's lifetime, so a silent one would
                // leave every 3x3 conv skipping the Winograd candidate with nothing to say why.
                var deviceResult = CudaNativeBindings.cuCtxGetDevice(out int device);
                if (deviceResult != CudaResult.Success)
                {
                    System.Diagnostics.Trace.TraceWarning($"Winograd convolution kernels unavailable: cuCtxGetDevice returned {deviceResult}.");
                    return false;
                }
                CompileKernelModule(device, Kernels.CudaConvWinogradKernels.GetSource(), "conv_winograd_kernels",
                    Kernels.CudaConvWinogradKernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvWinogradKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Winograd convolution kernels unavailable: '{name}' missing after compile.");
                        return false;
                    }
                _winogradModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Winograd convolution kernels unavailable: {ex.Message}");
            }
            return _winogradModuleReady;
        }
    }

    /// <summary>
    /// The same admission over a tuned-slot shape key (both conv slots: [5] kH, [6] kW, [7] sH, [8] sW, [9] pH, [10] pW,
    /// [11] dH, [12] dW, [13] outH, [14] outW).
    /// </summary>
    internal static bool WinogradF23AdmitsShape(AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedShape s)
        => s[5] == 3 && s[6] == 3 && s[7] == 1 && s[8] == 1 && s[11] == 1 && s[12] == 1 && s[9] == s[10] && s[9] <= 2
           && s[13] > 0 && s[14] > 0;

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
    internal static bool WinogradF23AdmitsData(in TunedConv2DBackwardArgs a)
        => !a.WeightGradient && a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 1 && a.StrideW == 1
           && a.DilationH == 1 && a.DilationW == 1 && a.PadH == a.PadW && a.PadH <= 2 && a.OutHeight > 0 && a.OutWidth > 0;

    /// <summary>
    /// Input gradient dX = conv(dY, w') with w' the 180-degree-rotated, channel-transposed filter and padding 2 - pad:
    /// the forward Winograd pipeline over dY (K channels, outH x outW) producing dX (C channels, H x W).
    /// </summary>
    internal bool TryConv2DBackwardDataWinogradF23(in TunedConv2DBackwardArgs a)
    {
        if (!WinogradF23AdmitsData(a) || !EnsureWinogradModule()) return false;
        return RunWinogradF23(a.First, a.Batch, a.OutChannels, a.OutHeight, a.OutWidth, 2 - a.PadH,
            "wino_f23_filter_flip", a.Second, a.OutChannels, a.InChannels, a.Output, a.InChannels, a.InHeight, a.InWidth);
    }

    private IGpuBuffer? _winogradWu, _winogradWv, _winogradWm;

    /// <summary>True for the weight-gradient shapes this path computes (the forward conv's 3x3, stride 1, dilation 1).</summary>
    internal static bool WinogradF32AdmitsFilter(in TunedConv2DBackwardArgs a)
        => a.WeightGradient && a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 1 && a.StrideW == 1
           && a.DilationH == 1 && a.DilationW == 1 && a.PadH == a.PadW && a.PadH <= 2 && a.OutHeight > 0 && a.OutWidth > 0;

    /// <summary>
    /// Weight gradient through Winograd F(3x3, 2x2): dY transform U[16][K][P], x transform V[16][P][C], sixteen GEMMs
    /// over the P tiles (split along P when the K x C grid alone would leave SMs idle), and the output transform that sums
    /// the splits into dW[K][C][3][3]. Overwrites the gradient.
    /// </summary>
    internal unsafe bool TryConv2DBackwardFilterWinogradF32(in TunedConv2DBackwardArgs a)
    {
        if (!WinogradF32AdmitsFilter(a) || !EnsureWinogradModule()) return false;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, h = a.InHeight, w = a.InWidth, pad = a.PadH;
        int outH = a.OutHeight, outW = a.OutWidth, tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (16L * k * p > int.MaxValue || 16L * c * p > int.MaxValue) return false;
        int pi = (int)p;
        int tile = ChooseWinogradTile(c, k);
        int gx = (c + tile - 1) / tile, gy = (k + tile - 1) / tile;
        int sms = Math.Max(1, _multiProcessorCount);
        long blocks = (long)gx * gy * 16;
        long targetBlocks = WinogradSplitOccupancy * sms;
        int splits = 1;
        if (blocks < targetBlocks)
            splits = (int)Math.Max(1, Math.Min((targetBlocks + blocks - 1) / blocks, pi / WinogradMinTilesPerSplit));
        // Each split's slice of P is a whole number of the GEMM main loop's K tiles (CE_BK).
        int kPerSplit = ((pi + splits - 1) / splits + ConvEngineTileK - 1) / ConvEngineTileK * ConvEngineTileK;
        splits = (pi + kPerSplit - 1) / kPerSplit;
        if (16L * splits * k * c > int.MaxValue) return false;
        var u = TryConvScratch(ref _winogradWu, 16L * k * p);
        var v = TryConvScratch(ref _winogradWv, 16L * p * c);
        var m = TryConvScratch(ref _winogradWm, 16L * splits * k * c);
        if (u is null || v is null || m is null) return false;

        using var _ = PushContext();
        IntPtr dyPtr = a.Second.Handle, xPtr = a.First.Handle, uPtr = u.Handle, vPtr = v.Handle, mPtr = m.Handle, dwPtr = a.Output.Handle;
        uint block = (uint)DefaultBlockSize;

        void** da = stackalloc void*[8];
        da[0] = &dyPtr; da[1] = &uPtr; da[2] = &n; da[3] = &k; da[4] = &outH; da[5] = &outW; da[6] = &tilesH; da[7] = &tilesW;
        LaunchKernel(_kernelCache["wino_f32w_dy"], (uint)((k * p + block - 1) / block), block, da);

        void** ia = stackalloc void*[9];
        ia[0] = &xPtr; ia[1] = &vPtr; ia[2] = &n; ia[3] = &c; ia[4] = &h; ia[5] = &w; ia[6] = &pad; ia[7] = &tilesH; ia[8] = &tilesW;
        LaunchKernel(_kernelCache["wino_f32w_input"], (uint)((c * p + block - 1) / block), block, ia);

        // M[x][s] (K x C) = U[x] (K x P) * V[x] (P x C) over the P slice s.
        void** ga = stackalloc void*[8];
        ga[0] = &uPtr; ga[1] = &vPtr; ga[2] = &mPtr; ga[3] = &k; ga[4] = &c; ga[5] = &pi; ga[6] = &splits; ga[7] = &kPerSplit;
        LaunchKernel3D(_kernelCache[tile == WinogradLargeTile ? "wino_gemm128" : "wino_gemm64"], (uint)gx, (uint)gy, (uint)(16 * splits), block, 1, 1, ga, 0);

        void** oa = stackalloc void*[5];
        oa[0] = &mPtr; oa[1] = &dwPtr; oa[2] = &k; oa[3] = &c; oa[4] = &splits;
        LaunchKernel(_kernelCache["wino_f32w_output"], (uint)((k * c + block - 1) / block), block, oa);
        return true;
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

        int tile = ChooseWinogradTile(pi, outC);
        int gx = (pi + tile - 1) / tile, gy = (outC + tile - 1) / tile;
        int oneSplit = 1, wholeK = inC;
        void** ga = stackalloc void*[8];
        ga[0] = &uPtr; ga[1] = &vPtr; ga[2] = &mPtr; ga[3] = &outC; ga[4] = &pi; ga[5] = &inC; ga[6] = &oneSplit; ga[7] = &wholeK;
        LaunchKernel3D(_kernelCache[tile == WinogradLargeTile ? "wino_gemm128" : "wino_gemm64"], (uint)gx, (uint)gy, 16, block, 1, 1, ga, 0);

        void** oa = stackalloc void*[8];
        oa[0] = &mPtr; oa[1] = &yPtr; oa[2] = &n; oa[3] = &outC; oa[4] = &outH; oa[5] = &outW; oa[6] = &tilesH; oa[7] = &tilesW;
        LaunchKernel(_kernelCache["wino_f23_output"], (uint)((outC * p + block - 1) / block), block, oa);
        return true;
    }
}
