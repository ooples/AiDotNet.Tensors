using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Winograd F(4x4, 3x3) (<see cref="Kernels.CudaConvWinogradF43Kernels"/>): the reduced-precision sibling of the F(2x2, 3x3)
/// path, 3x3 stride 1 dilation 1 only. It runs only under the explicit opt-in
/// <see cref="CudaDispatchPolicy.ConvWinogradF4"/>: about 2e-6 of sum|terms| fp32 error against F(2x2)'s 1e-7 (owner
/// decision, #1335: never a default tuned-slot candidate). The module compiles on first use.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly object _winogradF43ModuleLock = new();
    private bool _winogradF43ModuleTried, _winogradF43ModuleReady;
    private IGpuBuffer? _winograd4U, _winograd4V, _winograd4M;

    internal bool EnsureWinogradF43Module()
    {
        lock (_winogradF43ModuleLock)
        {
            if (_winogradF43ModuleTried) return _winogradF43ModuleReady;
            _winogradF43ModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaConvWinogradF43Kernels.GetSource(), "conv_winograd_f43_kernels",
                    Kernels.CudaConvWinogradF43Kernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvWinogradF43Kernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _winogradF43ModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Winograd F(4x4) convolution kernels unavailable: {ex.Message}");
            }
            return _winogradF43ModuleReady;
        }
    }

    /// <summary>
    /// Input gradient dX = conv(dY, w') through F(4x4, 3x3), w' the 180-degree-rotated, channel-transposed filter, padding
    /// 2 - pad (the same admission as the F(2x2) input gradient). Overwrites dX. False (nothing launched) when the shape
    /// is not admitted, the module is unavailable or scratch cannot be sized.
    /// </summary>
    internal bool TryConv2DBackwardDataWinogradF43(in TunedConv2DBackwardArgs a)
    {
        if (!WinogradF23AdmitsData(a) || !EnsureWinogradF43Module()) return false;
        return RunWinogradF43(a.First, a.Batch, a.OutChannels, a.OutHeight, a.OutWidth, 2 - a.PadH,
            "wino_f43_filter_flip", a.Second, a.OutChannels, a.InChannels, a.Output, a.InChannels, a.InHeight, a.InWidth);
    }

    private IGpuBuffer? _winograd4Wu, _winograd4Wv, _winograd4Wm;

    /// <summary>
    /// Weight gradient through Winograd F(3x3, 4x4) (opt-in, as above): dY transform U[36][K][P], x transform V[36][P][C],
    /// thirty-six GEMMs over the P tiles (split along P when the K x C grid alone would leave SMs idle), and the output
    /// transform that sums the splits into dW[K][C][3][3]. Overwrites the gradient. Same admission as the F(3,2) path.
    /// </summary>
    internal unsafe bool TryConv2DBackwardFilterWinogradF34(in TunedConv2DBackwardArgs a)
    {
        if (!WinogradF32AdmitsFilter(a) || !EnsureWinogradF43Module()) return false;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, h = a.InHeight, w = a.InWidth, pad = a.PadH;
        int outH = a.OutHeight, outW = a.OutWidth, tilesH = (outH + 3) / 4, tilesW = (outW + 3) / 4;
        long p = (long)n * tilesH * tilesW;
        if (36L * k * p > int.MaxValue || 36L * c * p > int.MaxValue) return false;
        int pi = (int)p;
        int tile = (long)((c + 127) / 128) * ((k + 127) / 128) * 36 >= Math.Max(1, _multiProcessorCount) ? 128 : 64;
        int gx = (c + tile - 1) / tile, gy = (k + tile - 1) / tile;
        int sms = Math.Max(1, _multiProcessorCount);
        long blocks = (long)gx * gy * 36;
        int splits = 1;
        if (blocks < 2L * sms) splits = (int)Math.Max(1, Math.Min((2L * sms + blocks - 1) / blocks, pi / 64));
        int kPerSplit = ((pi + splits - 1) / splits + 7) / 8 * 8;
        splits = (pi + kPerSplit - 1) / kPerSplit;
        if (36L * splits * k * c > int.MaxValue) return false;
        var u = TryConvScratch(ref _winograd4Wu, 36L * k * p);
        var v = TryConvScratch(ref _winograd4Wv, 36L * p * c);
        var m = TryConvScratch(ref _winograd4Wm, 36L * splits * k * c);
        if (u is null || v is null || m is null) return false;

        using var _ = PushContext();
        IntPtr dyPtr = a.Second.Handle, xPtr = a.First.Handle, uPtr = u.Handle, vPtr = v.Handle, mPtr = m.Handle, dwPtr = a.Output.Handle;
        uint block = (uint)DefaultBlockSize;

        void** da = stackalloc void*[8];
        da[0] = &dyPtr; da[1] = &uPtr; da[2] = &n; da[3] = &k; da[4] = &outH; da[5] = &outW; da[6] = &tilesH; da[7] = &tilesW;
        LaunchKernel(_kernelCache["wino_f34w_dy"], (uint)((k * p + block - 1) / block), block, da);

        void** ia = stackalloc void*[9];
        ia[0] = &xPtr; ia[1] = &vPtr; ia[2] = &n; ia[3] = &c; ia[4] = &h; ia[5] = &w; ia[6] = &pad; ia[7] = &tilesH; ia[8] = &tilesW;
        LaunchKernel(_kernelCache["wino_f34w_input"], (uint)((c * p + block - 1) / block), block, ia);

        // M[x][s] (K x C) = U[x] (K x P) * V[x] (P x C) over the P slice s.
        void** ga = stackalloc void*[8];
        ga[0] = &uPtr; ga[1] = &vPtr; ga[2] = &mPtr; ga[3] = &k; ga[4] = &c; ga[5] = &pi; ga[6] = &splits; ga[7] = &kPerSplit;
        LaunchKernel3D(_kernelCache[tile == 128 ? "wino4_gemm128" : "wino4_gemm64"], (uint)gx, (uint)gy, (uint)(36 * splits), block, 1, 1, ga, 0);

        void** oa = stackalloc void*[5];
        oa[0] = &mPtr; oa[1] = &dwPtr; oa[2] = &k; oa[3] = &c; oa[4] = &splits;
        LaunchKernel(_kernelCache["wino_f34w_output"], (uint)((k * c + block - 1) / block), block, oa);
        return true;
    }

    // y[n][outC][outH][outW] = conv3x3_s1(x[n][inC][h][w], padding pad) through F(4x4, 3x3): the named filter transform
    // (forward-layout weights [fwdK][fwdC][3][3]) into U[36][outC][inC], the input transform V[36][inC][P], thirty-six
    // GEMMs M[x] = U[x] V[x], the output transform. P = n * ceil(outH / 4) * ceil(outW / 4).
    private unsafe bool RunWinogradF43(IGpuBuffer x, int n, int inC, int h, int w, int pad,
        string filterKernel, IGpuBuffer weights, int fwdK, int fwdC, IGpuBuffer y, int outC, int outH, int outW)
    {
        int tilesH = (outH + 3) / 4, tilesW = (outW + 3) / 4;
        long p = (long)n * tilesH * tilesW;
        if (36L * inC * p > int.MaxValue || 36L * outC * p > int.MaxValue || 36L * outC * inC > int.MaxValue) return false;
        int pi = (int)p;
        var u = TryConvScratch(ref _winograd4U, 36L * outC * inC);
        var v = TryConvScratch(ref _winograd4V, 36L * inC * p);
        var m = TryConvScratch(ref _winograd4M, 36L * outC * p);
        if (u is null || v is null || m is null) return false;

        using var _ = PushContext();
        IntPtr wPtr = weights.Handle, uPtr = u.Handle, xPtr = x.Handle, vPtr = v.Handle, mPtr = m.Handle, yPtr = y.Handle;
        uint block = (uint)DefaultBlockSize;

        void** fa = stackalloc void*[4];
        fa[0] = &wPtr; fa[1] = &uPtr; fa[2] = &fwdK; fa[3] = &fwdC;
        LaunchKernel(_kernelCache[filterKernel], (uint)(((long)fwdK * fwdC + block - 1) / block), block, fa);

        void** ia = stackalloc void*[9];
        ia[0] = &xPtr; ia[1] = &vPtr; ia[2] = &n; ia[3] = &inC; ia[4] = &h; ia[5] = &w; ia[6] = &pad; ia[7] = &tilesH; ia[8] = &tilesW;
        LaunchKernel(_kernelCache["wino_f43_input"], (uint)((inC * p + block - 1) / block), block, ia);

        // 128x128 tiles only while they still fill the SMs (the thirty-six GEMMs share the grid through z).
        int tile = (long)((pi + 127) / 128) * ((outC + 127) / 128) * 36 >= Math.Max(1, _multiProcessorCount) ? 128 : 64;
        int gx = (pi + tile - 1) / tile, gy = (outC + tile - 1) / tile;
        int oneSplit = 1, wholeK = inC;
        void** ga = stackalloc void*[8];
        ga[0] = &uPtr; ga[1] = &vPtr; ga[2] = &mPtr; ga[3] = &outC; ga[4] = &pi; ga[5] = &inC; ga[6] = &oneSplit; ga[7] = &wholeK;
        LaunchKernel3D(_kernelCache[tile == 128 ? "wino4_gemm128" : "wino4_gemm64"], (uint)gx, (uint)gy, 36, block, 1, 1, ga, 0);

        IntPtr noBias = IntPtr.Zero;
        int noEpilogue = 0;
        void** oa = stackalloc void*[10];
        oa[0] = &mPtr; oa[1] = &yPtr; oa[2] = &n; oa[3] = &outC; oa[4] = &outH; oa[5] = &outW; oa[6] = &tilesH; oa[7] = &tilesW;
        oa[8] = &noBias; oa[9] = &noEpilogue;
        LaunchKernel(_kernelCache["wino_f43_output"], (uint)((outC * p + block - 1) / block), block, oa);
        return true;
    }
}
