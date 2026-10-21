using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Winograd F(2x2, 3x3) forward convolution (<see cref="Kernels.CudaConvWinogradKernels"/>) for 3x3, stride 1,
/// dilation 1: filter transform, input transform, sixteen GEMMs on the #1325 engine main loop, output transform.
/// The module compiles on first use.
/// </summary>
public sealed partial class CudaBackend : IWinogradF23Transforms, IWinogradFusedFilterGradient
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
            flip: false, a.Kernel, a.OutChannels, a.InChannels, a.Output, a.OutChannels, a.OutHeight, a.OutWidth,
            a.Bias, a.EpilogueFlags);
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
            flip: true, a.Second, a.OutChannels, a.InChannels, a.Output, a.InChannels, a.InHeight, a.InWidth,
            bias: null, epilogue: 0);
    }

    private IGpuBuffer? _winogradWu, _winogradWv, _winogradWm, _winogradDwPartials;

    // The fused weight gradient's (wino_f32w_fused) tiles per chunk (WF_PC) and channel tile (WF_TK = WF_TC).
    private const int WinogradFusedChunk = 8, WinogradFusedTile = 32;

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
        IntPtr uPtr = u.Handle, vPtr = v.Handle, mPtr = m.Handle;
        uint block = (uint)DefaultBlockSize;
        IWinogradF23Transforms transforms = this;
        transforms.GradOutputTiles(a.Second, u, n, k, outH, outW, tilesH, tilesW);
        transforms.InputTilesTileMajor(a.First, v, n, c, h, w, pad, tilesH, tilesW);

        // M[x][s] (K x C) = U[x] (K x P) * V[x] (P x C) over the P slice s.
        void** ga = stackalloc void*[8];
        ga[0] = &uPtr; ga[1] = &vPtr; ga[2] = &mPtr; ga[3] = &k; ga[4] = &c; ga[5] = &pi; ga[6] = &splits; ga[7] = &kPerSplit;
        LaunchKernel3D(_kernelCache[tile == WinogradLargeTile ? "wino_gemm128" : "wino_gemm64"], (uint)gx, (uint)gy, (uint)(16 * splits), block, 1, 1, ga, 0);

        transforms.FilterGradient(m, a.Output, k, c, splits);
        return true;
    }

    /// <summary>
    /// Weight gradient through Winograd F(3x3, 2x2) in one fused launch (wino_f32w_fused): each block transforms its dY
    /// and x tiles in shared memory, accumulates all sixteen components in registers and output-transforms them, so no
    /// U, V or M planes go through memory. Unsplit, the kernel writes dW itself (wino_f32w_fused_direct, staged so its
    /// stores are contiguous rows); split, it writes component-major partials that conv2d_wgrad_winograd_finish sums. The
    /// tiles are split for about one block per SM (measured on an RTX 3080: more blocks per SM lost to the extra
    /// partials). ResNet-18, batch 8, standalone: 128@16 63 -> 35 us and 256@8 44 -> 36 (vs the four-launch pipeline),
    /// 64@32 51 -> 39 and 512@4 53 -> 41 (vs r8v). Overwrites the gradient.
    /// </summary>
    internal bool TryConv2DBackwardFilterWinogradFused(in TunedConv2DBackwardArgs a)
    {
        if (!WinogradF32AdmitsFilter(a) || !EnsureWinogradModule()) return false;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, h = a.InHeight, w = a.InWidth, pad = a.PadH;
        int outH = a.OutHeight, outW = a.OutWidth, tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (p > int.MaxValue || (long)n * c * h * w > int.MaxValue || (long)n * k * outH * outW > int.MaxValue) return false;
        int pi = (int)p;
        int gx = (c + WinogradFusedTile - 1) / WinogradFusedTile, gy = (k + WinogradFusedTile - 1) / WinogradFusedTile;
        int sms = Math.Max(1, _multiProcessorCount);
        int splits = (int)Math.Max(1, Math.Min(Math.Round((double)sms / ((long)gx * gy)), pi / WinogradFusedChunk));
        int tilesPerSplit = ((pi + splits - 1) / splits + WinogradFusedChunk - 1) / WinogradFusedChunk * WinogradFusedChunk;
        splits = (pi + tilesPerSplit - 1) / tilesPerSplit;
        long kc = (long)k * c;
        if (9L * splits * kc > int.MaxValue) return false;
        IGpuBuffer? partials = null;
        if (splits > 1 && (partials = TryConvScratch(ref _winogradDwPartials, 9L * splits * kc)) is null) return false;

        using var _ = PushContext();
        IWinogradFusedFilterGradient fused = this;
        fused.FusedFilterGradient(splits == 1, a.First, partials ?? a.Output, a.Second, gx * gy * splits,
            new[] { n, c, h, w, k, outH, outW, pad, tilesH, tilesW, tilesPerSplit });
        if (partials is not null) fused.FusedFilterGradientFinish(partials, a.Output, (int)kc, splits);
        return true;
    }

    int IWinogradFusedFilterGradient.FusedFilterGradientGroupTarget => _multiProcessorCount;

    // CUDA's own fused kernels (CudaConvWinogradKernels): a (ceil(C/32), ceil(K/32), splits) grid of 256-thread blocks,
    // arguments (x, dy, out, ints), where the template's kernels take a 1-D grid and (x, res, dy, ints).
    unsafe void IWinogradFusedFilterGradient.FusedFilterGradient(bool direct, IGpuBuffer x, IGpuBuffer res, IGpuBuffer dy,
        int groups, int[] ints)
    {
        int gx = (ints[1] + WinogradFusedTile - 1) / WinogradFusedTile, gy = (ints[4] + WinogradFusedTile - 1) / WinogradFusedTile;
        IntPtr xPtr = x.Handle, dyPtr = dy.Handle, resPtr = res.Handle;
        int* fusedInts = stackalloc int[ints.Length];
        void** fusedArgs = stackalloc void*[3 + ints.Length];
        fusedArgs[0] = &xPtr; fusedArgs[1] = &dyPtr; fusedArgs[2] = &resPtr;
        for (int i = 0; i < ints.Length; i++) { fusedInts[i] = ints[i]; fusedArgs[3 + i] = &fusedInts[i]; }
        LaunchKernel3D(_kernelCache[direct ? "wino_f32w_fused_direct" : "wino_f32w_fused"], (uint)gx, (uint)gy,
            (uint)(groups / (gx * gy)), (uint)DefaultBlockSize, 1, 1, fusedArgs, 0);
    }

    unsafe void IWinogradFusedFilterGradient.FusedFilterGradientFinish(IGpuBuffer partials, IGpuBuffer dw, int kc, int splits)
    {
        IntPtr partPtr = partials.Handle, dwPtr = dw.Handle;
        void** finishArgs = stackalloc void*[4];
        finishArgs[0] = &partPtr; finishArgs[1] = &dwPtr; finishArgs[2] = &kc; finishArgs[3] = &splits;
        LaunchKernel(_kernelCache["conv2d_wgrad_winograd_finish"], (uint)((9L * kc + DefaultBlockSize - 1) / DefaultBlockSize),
            (uint)DefaultBlockSize, finishArgs);
    }

    // The pipeline for y[n][outC][outH][outW] = conv3x3_s1(x[n][inC][h][w], padding pad): the filter transform of w
    // (forward layout [fwdK][fwdC][3][3], flipped for the input gradient) into U[16][outC][inC], the input transform,
    // sixteen GEMMs, the output transform. The tile transforms are this backend's IWinogradF23Transforms; the GEMM is
    // CUDA's own, on the engine main loop, where the other backends' shared pipeline uses BatchedGemm.
    private unsafe bool RunWinogradF23(IGpuBuffer x, int n, int inC, int h, int w, int pad,
        bool flip, IGpuBuffer weights, int fwdK, int fwdC, IGpuBuffer y, int outC, int outH, int outW, IGpuBuffer? bias,
        int epilogue)
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
        IntPtr uPtr = u.Handle, vPtr = v.Handle, mPtr = m.Handle;
        uint block = (uint)DefaultBlockSize;
        IWinogradF23Transforms transforms = this;
        transforms.Filter(weights, u, fwdK, fwdC, flip);
        transforms.Input(x, v, n, inC, h, w, pad, tilesH, tilesW);

        int tile = ChooseWinogradTile(pi, outC);
        int gx = (pi + tile - 1) / tile, gy = (outC + tile - 1) / tile;
        int oneSplit = 1, wholeK = inC;
        void** ga = stackalloc void*[8];
        ga[0] = &uPtr; ga[1] = &vPtr; ga[2] = &mPtr; ga[3] = &outC; ga[4] = &pi; ga[5] = &inC; ga[6] = &oneSplit; ga[7] = &wholeK;
        LaunchKernel3D(_kernelCache[tile == WinogradLargeTile ? "wino_gemm128" : "wino_gemm64"], (uint)gx, (uint)gy, 16, block, 1, 1, ga, 0);

        transforms.Output(m, y, n, outC, outH, outW, tilesH, tilesW, bias, epilogue);
        return true;
    }

    // IWinogradF23Transforms on CUDA: the module compiled from CudaConvWinogradKernels, whose kernels take the same
    // arguments as WinogradF23KernelTemplate's, one thread per transformed element in DefaultBlockSize blocks.
    private unsafe void LaunchWinograd(string name, long threads, IGpuBuffer read, IGpuBuffer write, int* ints, int count)
    {
        using var _ = PushContext();
        IntPtr readPtr = read.Handle, writePtr = write.Handle;
        void** args = stackalloc void*[2 + count];
        args[0] = &readPtr; args[1] = &writePtr;
        for (int i = 0; i < count; i++) args[2 + i] = &ints[i];
        uint block = (uint)DefaultBlockSize;
        LaunchKernel(_kernelCache[name], (uint)((threads + block - 1) / block), block, args);
    }

    unsafe void IWinogradF23Transforms.Filter(IGpuBuffer w, IGpuBuffer u, int k, int c, bool flip)
    {
        int* ints = stackalloc int[] { k, c };
        LaunchWinograd(flip ? "wino_f23_filter_flip" : "wino_f23_filter", (long)k * c, w, u, ints, 2);
    }

    unsafe void IWinogradF23Transforms.Input(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH,
        int tilesW)
    {
        int* ints = stackalloc int[] { n, c, h, w, pad, tilesH, tilesW };
        LaunchWinograd("wino_f23_input", (long)c * n * tilesH * tilesW, x, v, ints, 7);
    }

    unsafe void IWinogradF23Transforms.Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH,
        int tilesW, IGpuBuffer? bias, int epilogue)
    {
        using var _ = PushContext();
        IntPtr readPtr = m.Handle, writePtr = y.Handle, biasPtr = (bias ?? y).Handle;
        int* ints = stackalloc int[] { n, k, outH, outW, tilesH, tilesW, epilogue };
        void** args = stackalloc void*[10];
        args[0] = &readPtr; args[1] = &writePtr; args[2] = &biasPtr;
        for (int i = 0; i < 7; i++) args[3 + i] = &ints[i];
        long threads = (long)k * n * tilesH * tilesW;
        uint block = (uint)DefaultBlockSize;
        LaunchKernel(_kernelCache["wino_f23_output"], (uint)((threads + block - 1) / block), block, args);
    }

    unsafe void IWinogradF23Transforms.GradOutputTiles(IGpuBuffer dy, IGpuBuffer u, int n, int k, int outH, int outW,
        int tilesH, int tilesW)
    {
        int* ints = stackalloc int[] { n, k, outH, outW, tilesH, tilesW };
        LaunchWinograd("wino_f32w_dy", (long)k * n * tilesH * tilesW, dy, u, ints, 6);
    }

    unsafe void IWinogradF23Transforms.InputTilesTileMajor(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w,
        int pad, int tilesH, int tilesW)
    {
        int* ints = stackalloc int[] { n, c, h, w, pad, tilesH, tilesW };
        LaunchWinograd("wino_f32w_input", (long)c * n * tilesH * tilesW, x, v, ints, 7);
    }

    unsafe void IWinogradF23Transforms.FilterGradient(IGpuBuffer m, IGpuBuffer dw, int k, int c, int splits)
    {
        int* ints = stackalloc int[] { k, c, splits };
        LaunchWinograd("wino_f32w_output", (long)k * c, m, dw, ints, 3);
    }
}
