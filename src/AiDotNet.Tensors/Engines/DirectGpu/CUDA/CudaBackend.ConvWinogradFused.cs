using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Fused Winograd F(2x2, 3x3) (<see cref="Kernels.CudaConvWinogradFusedKernels"/>): one kernel per convolution that
/// transforms its filters and its input patches in shared memory, runs the sixteen tile GEMMs and applies the output
/// transform. Same admission and the same fp32 arithmetic as the unfused F(2x2) path. The module compiles on first use.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly object _winogradFusedModuleLock = new();
    private bool _winogradFusedModuleTried, _winogradFusedModuleReady;
    private IGpuBuffer? _winogradFusedPartials;

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
    /// shape is not admitted or the module is unavailable. <paramref name="forcedSplits"/> &gt; 0 replaces the chosen
    /// split count of the channel reduction (capped at one split per eight-channel step), so a test can run the split
    /// reducer on any device; 0 keeps the SM-count heuristic.</summary>
    internal bool TryConv2DForwardWinogradF23Fused(in CudaConv2DArgs a, int forcedSplits = 0)
    {
        if (!WinogradF23Admits(a) || !EnsureWinogradFusedModule()) return false;
        // Reduction over the input channels, outputs the output channels, filters w[outC][inC][3][3].
        return RunWinogradF23Fused(a.Input, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.PadH, a.Kernel, flipped: false,
            a.Output, a.OutChannels, a.OutHeight, a.OutWidth, forcedSplits);
    }

    /// <summary>
    /// Input gradient dX = conv(dY, w') (w' the 180-degree-rotated, channel-transposed filter, padding 2 - pad) through
    /// the fused F(2x2) kernel. Overwrites dX. Same admission as <see cref="TryConv2DBackwardDataWinogradF23"/>.
    /// </summary>
    internal bool TryConv2DBackwardDataWinogradF23Fused(in TunedConv2DBackwardArgs a)
    {
        if (!WinogradF23AdmitsData(a) || !EnsureWinogradFusedModule()) return false;
        // Reduction over dY's channels (the forward's outC), outputs dX's channels, filters read rotated.
        return RunWinogradF23Fused(a.First, a.Batch, a.OutChannels, a.OutHeight, a.OutWidth, 2 - a.PadH, a.Second, flipped: true,
            a.Output, a.InChannels, a.InHeight, a.InWidth);
    }

    // y[n][outC][outH][outW] = conv3x3_s1(x[n][inC][h][w], padding pad) through the fused kernel (512 threads, one block
    // per SM) over 32 output channels x 32 tiles per block, with weights w[outC][inC][3][3] (flipped: false) or the
    // input gradient's w[inC][outC][3][3] read rotated (flipped: true). On deep layers the tiles alone give few blocks
    // (512@4x4 at batch 8: 16 blocks of 64 eight-channel steps each), so the channel reduction is split across
    // blockIdx.z; each split applies the (linear) output transform to its partial products and writes its own slice,
    // summed in split order (conv2d_wgrad_splitk_reduce), so the result stays deterministic.
    private unsafe bool RunWinogradF23Fused(IGpuBuffer x, int n, int inC, int h, int w, int pad, IGpuBuffer weights,
        bool flipped, IGpuBuffer y, int outC, int outH, int outW, int forcedSplits = 0)
    {
        int tilesH = (outH + 1) / 2, tilesW = (outW + 1) / 2;
        long p = (long)n * tilesH * tilesW;
        if (p > int.MaxValue || 9L * inC * outC > int.MaxValue || (long)n * inC * h * w > int.MaxValue) return false;

        long blocks = ((p + 31) / 32) * ((outC + 31) / 32);
        int splits = forcedSplits > 0
            ? SplitWinogradF23FusedReduction(inC, Math.Min(forcedSplits, (inC + 7) / 8), out int rPerSplit)
            : SplitWinogradF23FusedReduction(inC, ChooseWinogradF23FusedSplits(blocks, inC, Math.Max(1, _multiProcessorCount)), out rPerSplit);
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

        using var _ = PushContext();
        IntPtr xPtr = x.Handle, wPtr = weights.Handle, tPtr = target.Handle;
        void** ca = stackalloc void*[14];
        ca[0] = &xPtr; ca[1] = &wPtr; ca[2] = &tPtr; ca[3] = &n; ca[4] = &inC; ca[5] = &h; ca[6] = &w; ca[7] = &outC;
        ca[8] = &outH; ca[9] = &outW; ca[10] = &pad; ca[11] = &tilesH; ca[12] = &tilesW; ca[13] = &rPerSplit;
        // Even tile rows: the pair loader (a 4 x 6 window per two neighbouring tiles), bit-identical to the one-tile loader.
        bool pair = tilesW % 2 == 0;
        LaunchKernel3D(_kernelCache[(flipped ? "wfz_conv_wf" : "wfz_conv_w") + (pair ? "_pair" : "")], (uint)((p + 31) / 32), (uint)((outC + 31) / 32),
            (uint)splits, 512, 1, 1, ca, 0);
        if (splits > 1) LaunchSplitKReduce(reduce, target, y, (int)outputs, splits);
        return true;
    }

    /// <summary>How many splits the fused kernel cuts a reduction over <paramref name="inC"/> channels into, for
    /// <paramref name="blocks"/> tile blocks on a device with <paramref name="sms"/> SMs (before rounding to whole
    /// eight-channel steps, <see cref="SplitWinogradF23FusedReduction"/>).</summary>
    internal static int ChooseWinogradF23FusedSplits(long blocks, int inC, int sms)
    {
        // Measured per-block prologue + epilogue cost, in eight-channel steps (see the cost comment below).
        const int PrologueEpilogueSteps = 3;
        // Most splits the heuristic considers (the bound the measured cost model was taken under).
        const int MaxSplits = 16;
        int totalSteps = (inC + 7) / 8;
        int splits = 1;
        long bestCost = long.MaxValue;
        for (int s = 1; s <= Math.Min(MaxSplits, totalSteps); s++)
        {
            // In half-step units: each wave costs its blocks' steps plus ~3 steps of per-block prologue and epilogue
            // (measured: 16 splits of 128@16 ran 15 one-step waves in 95 us), and each extra split one half step of
            // partial traffic and reduce. Smallest count on ties.
            long cost = (blocks * s + sms - 1) / sms * 2 * ((totalSteps + s - 1) / s + PrologueEpilogueSteps) + (s - 1);
            if (cost < bestCost) { bestCost = cost; splits = s; }
        }
        return splits;
    }

    /// <summary>Rounds a split count to whole eight-channel steps: each split reduces <paramref name="rPerSplit"/>
    /// channels; returns the number of splits that leaves non-empty.</summary>
    internal static int SplitWinogradF23FusedReduction(int inC, int splits, out int rPerSplit)
    {
        int totalSteps = (inC + 7) / 8;
        rPerSplit = (totalSteps + splits - 1) / splits * 8;
        return (inC + rPerSplit - 1) / rPerSplit;
    }
}
