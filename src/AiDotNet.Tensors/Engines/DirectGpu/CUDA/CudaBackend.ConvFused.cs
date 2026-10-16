using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Launchers for the fused ResNet basic-block training-forward kernels (<see cref="Kernels.CudaConvFusedKernels"/>).
/// The module compiles on first use, so a backend that never runs a fused block pays nothing at start-up. Spike.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly object _fusedBlockModuleLock = new();
    private bool _fusedBlockModuleTried;
    private bool _fusedBlockModuleReady;

    internal bool EnsureFusedBlockModule()
    {
        lock (_fusedBlockModuleLock)
        {
            if (_fusedBlockModuleTried) return _fusedBlockModuleReady;
            _fusedBlockModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaConvFusedKernels.GetSource(), "conv_fused_block_kernels",
                    Kernels.CudaConvFusedKernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvFusedKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _fusedBlockModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Fused basic-block kernels unavailable: {ex.Message}");
            }
            return _fusedBlockModuleReady;
        }
    }

    /// <summary>
    /// y = conv(x [through BN+ReLU when <paramref name="proScale"/> is given], w), accumulating each output channel's
    /// sum and sum of squares into <paramref name="sum"/>/<paramref name="sumSq"/> (doubles, zeroed by the previous
    /// finalize). One K split, so the epilogue sees final values.
    /// </summary>
    internal unsafe void FusedConvBatchStats(in CudaConv2DArgs a, IGpuBuffer sum, IGpuBuffer sumSq,
        IGpuBuffer? proScale = null, IGpuBuffer? proShift = null)
    {
        if (!EnsureFusedBlockModule()) throw new GpuCapabilityUnavailableException("fused basic-block kernels unavailable");
        long columns = (long)a.Batch * a.OutHeight * a.OutWidth;
        long kTotal = (long)a.InChannels * a.KernelH * a.KernelW;
        // The engine's own tiling plan: a grid too small for the SMs splits K.
        ConvEnginePlan(columns, a.OutChannels, kTotal, out int tile, out int tilesN, out int tilesM,
            out int splits, out int kPerSplit);
        bool tap = a.InChannels % ConvEngineTileK == 0;
        if (splits > 1)
        {
            FusedConvSplitBatchStats(a, sum, sumSq, proScale, proShift, tile, tilesN, tilesM, splits, kPerSplit, tap);
            return;
        }
        string name = (proScale is null ? "fb_conv_stats" : "fb_bnrelu_conv_stats") + tile + (tap ? "_tap" : "_gen");
        var kernel = _kernelCache[name];

        using var _ = PushContext();
        IntPtr xPtr = a.Input.Handle, wPtr = a.Kernel.Handle, yPtr = a.Output.Handle;
        IntPtr sPtr = sum.Handle, qPtr = sumSq.Handle;
        IntPtr scPtr = proScale?.Handle ?? IntPtr.Zero, shPtr = proShift?.Handle ?? IntPtr.Zero;
        int batch = a.Batch, inC = a.InChannels, inH = a.InHeight, inW = a.InWidth, outC = a.OutChannels;
        int outH = a.OutHeight, outW = a.OutWidth, kH = a.KernelH, kW = a.KernelW, sH = a.StrideH, sW = a.StrideW;
        int pH = a.PadH, pW = a.PadW, dH = a.DilationH, dW = a.DilationW;
        void** args = stackalloc void*[22];
        args[0] = &xPtr; args[1] = &wPtr; args[2] = &yPtr;
        args[3] = &batch; args[4] = &inC; args[5] = &inH; args[6] = &inW;
        args[7] = &outC; args[8] = &outH; args[9] = &outW;
        args[10] = &kH; args[11] = &kW; args[12] = &sH; args[13] = &sW;
        args[14] = &pH; args[15] = &pW; args[16] = &dH; args[17] = &dW;
        args[18] = &sPtr; args[19] = &qPtr; args[20] = &scPtr; args[21] = &shPtr;
        LaunchKernel3D(kernel, (uint)tilesN, (uint)tilesM, 1, (uint)DefaultBlockSize, 1, 1, args, 0);
    }

    private IGpuBuffer? _fusedBlockPartials;

    // Split-K form: the convolution (with the BN+ReLU prologue when given; the engine's own kernel otherwise) writes the
    // split partials, and fb_splitk_reduce_stats sums them and reduces the batch statistics in one pass.
    private unsafe void FusedConvSplitBatchStats(in CudaConv2DArgs a, IGpuBuffer sum, IGpuBuffer sumSq,
        IGpuBuffer? proScale, IGpuBuffer? proShift, int tile, int tilesN, int tilesM, int splits, int kPerSplit, bool tap)
    {
        long outputs = (long)a.Batch * a.OutChannels * a.OutHeight * a.OutWidth;
        var partials = TryConvScratch(ref _fusedBlockPartials, splits * outputs)
            ?? throw new GpuCapabilityUnavailableException("no scratch for split-K partials");
        string convName = proScale is null
            ? (tile == 128 ? "conv2d_fprop_ig128_" : "conv2d_fprop_ig64_") + (tap ? "tap" : "gen")
            : "fb_bnrelu_conv_split" + tile + (tap ? "_tap" : "_gen");
        var conv = _kernelCache[convName];
        var reduce = _kernelCache["fb_splitk_reduce_stats"];

        using var _ = PushContext();
        IntPtr xPtr = a.Input.Handle, wPtr = a.Kernel.Handle, pPtr = partials.Handle;
        IntPtr scPtr = proScale?.Handle ?? IntPtr.Zero, shPtr = proShift?.Handle ?? IntPtr.Zero;
        int batch = a.Batch, inC = a.InChannels, inH = a.InHeight, inW = a.InWidth, outC = a.OutChannels;
        int outH = a.OutHeight, outW = a.OutWidth, kH = a.KernelH, kW = a.KernelW, sH = a.StrideH, sW = a.StrideW;
        int pH = a.PadH, pW = a.PadW, dH = a.DilationH, dW = a.DilationW;
        void** args = stackalloc void*[21];
        args[0] = &xPtr; args[1] = &wPtr; args[2] = &pPtr;
        args[3] = &batch; args[4] = &inC; args[5] = &inH; args[6] = &inW;
        args[7] = &outC; args[8] = &outH; args[9] = &outW;
        args[10] = &kH; args[11] = &kW; args[12] = &sH; args[13] = &sW;
        args[14] = &pH; args[15] = &pW; args[16] = &dH; args[17] = &dW; args[18] = &kPerSplit;
        args[19] = &scPtr; args[20] = &shPtr;
        LaunchKernel3D(conv, (uint)tilesN, (uint)tilesM, (uint)splits, (uint)DefaultBlockSize, 1, 1, args, 0);

        IntPtr yPtr = a.Output.Handle, sPtr = sum.Handle, qPtr = sumSq.Handle;
        int count = (int)outputs, spatial = outH * outW;
        void** rargs = stackalloc void*[8];
        rargs[0] = &pPtr; rargs[1] = &yPtr; rargs[2] = &count; rargs[3] = &splits;
        rargs[4] = &outC; rargs[5] = &spatial; rargs[6] = &sPtr; rargs[7] = &qPtr;
        LaunchKernel3D(reduce, (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize), 1, 1, (uint)DefaultBlockSize, 1, 1,
            rargs, (uint)(2 * outC * sizeof(float)));
    }

    /// <summary>Batch statistics to mean, inverse std and the folded scale/shift; re-zeroes the sums.</summary>
    internal unsafe void FusedBatchNormFinalize(IGpuBuffer sum, IGpuBuffer sumSq, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer mean, IGpuBuffer invStd, IGpuBuffer scale, IGpuBuffer shift, int channels, long count, float epsilon)
    {
        if (!EnsureFusedBlockModule()) throw new GpuCapabilityUnavailableException("fused basic-block kernels unavailable");
        var kernel = _kernelCache["fb_bn_finalize"];
        using var _ = PushContext();
        IntPtr s = sum.Handle, q = sumSq.Handle, g = gamma.Handle, b = beta.Handle;
        IntPtr m = mean.Handle, i = invStd.Handle, sc = scale.Handle, sh = shift.Handle;
        double n = count;
        void** args = stackalloc void*[11];
        args[0] = &s; args[1] = &q; args[2] = &g; args[3] = &b; args[4] = &m; args[5] = &i; args[6] = &sc; args[7] = &sh;
        args[8] = &channels; args[9] = &n; args[10] = &epsilon;
        LaunchKernel(kernel, (uint)((channels + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    /// <summary>out = relu(y * scale + shift + residual) over NCHW with per-channel scale/shift.</summary>
    internal unsafe void FusedBatchNormAddRelu(IGpuBuffer y, IGpuBuffer scale, IGpuBuffer shift, IGpuBuffer residual,
        IGpuBuffer output, int total, int channels, int spatial)
    {
        if (!EnsureFusedBlockModule()) throw new GpuCapabilityUnavailableException("fused basic-block kernels unavailable");
        var kernel = _kernelCache["fb_bn_add_relu"];
        using var _ = PushContext();
        IntPtr yp = y.Handle, sc = scale.Handle, sh = shift.Handle, r = residual.Handle, o = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &yp; args[1] = &sc; args[2] = &sh; args[3] = &r; args[4] = &o;
        args[5] = &total; args[6] = &channels; args[7] = &spatial;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }
}
