// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Channels-last activations for a compiled plan's convolution stacks (IChannelsLastBackend): cuDNN's passes with a
// format per tensor (CuDnnConvolution.ConvolutionChannelsLast), batchnorm_nhwc_* of CudaNormalizationKernels, and
// layout_transpose_tiles at the stack's edges.
public sealed partial class CudaBackend : IChannelsLastBackend
{
    // The statistics and backward reductions split the rows so the launch fills the device: about
    // BatchNormNhwcTargetBlocks blocks of (128 channels x a row range), no split under BatchNormNhwcMinRowsPerSplit
    // rows (4 per thread row), at most BatchNormNhwcMaxSplits partials per channel for the finalize to merge.
    private const int BatchNormNhwcMaxSplits = 1024;
    private const int BatchNormNhwcMinRowsPerSplit = 32;
    private const int BatchNormNhwcTargetBlocks = 4 * 84;

    /// <summary>Channels-last convolution passes and BatchNorms launched (tests).</summary>
    internal static long ChannelsLastConvolutions, ChannelsLastBatchNorms;

    /// <summary>
    /// Whether FP32 plans keep conv stacks channels-last (tests). Off: measured on a DCGAN training step (RTX 3090,
    /// nsys, 180 steady steps) channels-last cost 10.59 ms of kernel time against NCHW's 10.24 ms even with every pass
    /// routed by measurement -- cuDNN's TF32 engines prepare their operands in either layout (NCHW: transposes; NHWC:
    /// convertTensor, as PyTorch's FP32 channels_last trace shows too), so the layout saves no conversion in FP32 and
    /// adds the stack-edge changes. Half-precision activations, which cuDNN consumes channels-last natively, are where
    /// the layout is meant to pay.
    /// </summary>
    internal static bool s_channelsLastAtFp32;

    // cuDNN's tensor-core convolutions are native to channels-last; without cuDNN (or tensor cores) the backend's own
    // convolutions are NCHW, and the layout would only add passes. Every channels-last plan today is FP32.
    bool IChannelsLastBackend.PrefersChannelsLast
    {
        get
        {
            if (!s_channelsLastAtFp32) return false;
            if (!IsAvailable || !ConvTensorCoresAvailable || !CudaDispatchPolicy.ConvReducedPrecisionTF32) return false;
            return CudnnConvLoaded();
        }
    }

    bool IChannelsLastBackend.AcceptsChannelsLast(int channels, int outChannels)
        => IsAvailable && channels > 0 && outChannels > 0
           && _kernelCache.ContainsKey("batchnorm_nhwc_stats") && _kernelCache.ContainsKey("layout_transpose_tiles");

    private IGpuBuffer? _channelsLastScratchA;   // a mixed-layout pass's converted activation
    private IGpuBuffer? _channelsLastScratchB;   // a channels-last filter gradient before its change to [k, c, r, s]

    private IGpuBuffer? _channelsLastScratchC;   // the NCHW route's input-side activation
    private IGpuBuffer? _channelsLastScratchD;   // the NCHW route's output-side activation

    private TunedKernelSlot<ChannelsLastConvArgs>? _channelsLastConvSlot;

    /// <summary>Channels-last convolution passes the NCHW route ran (tests, diagnostics).</summary>
    internal static long ChannelsLastNchwRoutes;

    bool IChannelsLastBackend.TryConvolutionChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch,
        int channels, int height, int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth,
        int padH, int padW, int strideH, int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc)
    {
        if (!IsAvailable || pass is < 0 or > 2) return false;
        using var _ = PushContext();
        var args = new ChannelsLastConvArgs(pass, a, b, c, batch, channels, height, width, outChannels, kernelH, kernelW,
            outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, inputNhwc, outputNhwc);
        // cuDNN's TF32 engines prepare their operands in either layout, so its channels-last descriptors are not always
        // the faster route: the tuned slot measures them against the NCHW convolutions between layout changes per pass
        // and geometry (its evidence gate, persisted, determinism-aware) and keeps the winner. A capture replays the
        // cached decision; an unmeasured pass met inside one takes the NCHW reference.
        var chosen = ChannelsLastConvSlot.Resolve(args.Shape(), args);
        if (chosen is null || chosen.Id == ChannelsLastNchwId) return RunChannelsLastNchw(args);
        return TryCudnnChannelsLastRoute(args) || RunChannelsLastNchw(args);
    }

    private const string ChannelsLastNchwId = "channels_last.nchw_route";

    private TunedKernelSlot<ChannelsLastConvArgs> ChannelsLastConvSlot
    {
        get
        {
            if (_channelsLastConvSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_channelsLastConvSlot is null)
                {
                    var candidates = new ITunedKernelCandidate<ChannelsLastConvArgs>[]
                    {
                        new CudaTunedKernelCandidate<ChannelsLastConvArgs>(ChannelsLastNchwId, TunedKernelOrigin.Builtin,
                            true, _ => true, (in ChannelsLastConvArgs x) => RunChannelsLastNchw(x)),
                        // cuDNN with channels-last descriptors; where it takes no algorithm for the formats the NCHW
                        // route runs instead, so the candidate always writes the result (and cannot win the gate).
                        new CudaTunedKernelCandidate<ChannelsLastConvArgs>("channels_last.cudnn", TunedKernelOrigin.Vendor,
                            true, _ => CudnnConvLoaded(),
                            (in ChannelsLastConvArgs x) => { if (!TryCudnnChannelsLastRoute(x)) RunChannelsLastNchw(x); }),
                    };
                    _channelsLastConvSlot = new TunedKernelSlot<ChannelsLastConvArgs>(TunedKernelOp.ChannelsLastConvolution,
                        TunedDeviceKey,
                        new CudaTunedKernelHarness<ChannelsLastConvArgs>(this, ConvolutionTolerance,
                            x => new[] { (x.C, x.OutputCount) },
                            x => x.C.Handle == x.A.Handle || x.C.Handle == x.B.Handle),
                        () => GpuDeterminism.IsActive, candidates);
                }
                return Created(_channelsLastConvSlot);
            }
        }
    }

    private bool RunChannelsLastNchw(in ChannelsLastConvArgs x)
        => TryNchwChannelsLastRoute(x.Pass, x.A, x.B, x.C, x.Batch, x.Channels, x.Height, x.Width, x.OutChannels,
            x.KernelH, x.KernelW, x.OutHeight, x.OutWidth, x.PadH, x.PadW, x.StrideH, x.StrideW, x.DilationH, x.DilationW,
            x.InputNhwc, x.OutputNhwc);

    private bool TryCudnnChannelsLastRoute(in ChannelsLastConvArgs x)
        => TryCudnnChannelsLastRoute(x.Pass, x.A, x.B, x.C, x.Batch, x.Channels, x.Height, x.Width, x.OutChannels,
            x.KernelH, x.KernelW, x.OutHeight, x.OutWidth, x.PadH, x.PadW, x.StrideH, x.StrideW, x.DilationH, x.DilationW,
            x.InputNhwc, x.OutputNhwc);

    // The pass as NCHW convolutions: channels-last operands change to NCHW in scratch, a channels-last result is
    // computed NCHW in scratch and changed after.
    private bool TryNchwChannelsLastRoute(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int channels,
        int height, int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth, int padH, int padW,
        int strideH, int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc)
    {
        int inSpatial = height * width, outSpatial = outHeight * outWidth;
        long inputCount = (long)batch * channels * inSpatial, outputCount = (long)batch * outChannels * outSpatial;
        IChannelsLastBackend self = this;
        IGpuBuffer? inputSide = inputNhwc ? TryConvScratch(ref _channelsLastScratchC, inputCount) : null;
        IGpuBuffer? outputSide = outputNhwc ? TryConvScratch(ref _channelsLastScratchD, outputCount) : null;
        if ((inputNhwc && inputSide is null) || (outputNhwc && outputSide is null)) return false;
        switch (pass)
        {
            case 0:
            {
                if (inputNhwc && !self.TryChangeLayout(a, inputSide!, batch, channels, inSpatial, toChannelsLast: false))
                    return false;
                Conv2D(inputNhwc ? inputSide! : a, b, outputNhwc ? outputSide! : c, batch, channels, height, width,
                    outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
                if (outputNhwc && !self.TryChangeLayout(outputSide!, c, batch, outChannels, outSpatial, toChannelsLast: true))
                    return false;
                break;
            }
            case 1:
            {
                if (outputNhwc && !self.TryChangeLayout(b, outputSide!, batch, outChannels, outSpatial, toChannelsLast: false))
                    return false;
                Conv2DBackwardInput(outputNhwc ? outputSide! : b, a, inputNhwc ? inputSide! : c, batch, channels, height,
                    width, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH,
                    dilationW);
                if (inputNhwc && !self.TryChangeLayout(inputSide!, c, batch, channels, inSpatial, toChannelsLast: true))
                    return false;
                break;
            }
            default:
            {
                if (inputNhwc && !self.TryChangeLayout(a, inputSide!, batch, channels, inSpatial, toChannelsLast: false))
                    return false;
                if (outputNhwc && !self.TryChangeLayout(b, outputSide!, batch, outChannels, outSpatial, toChannelsLast: false))
                    return false;
                Conv2DBackwardKernel(inputNhwc ? inputSide! : a, outputNhwc ? outputSide! : b, c, batch, channels, height,
                    width, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH,
                    dilationW);
                break;
            }
        }
        System.Threading.Interlocked.Increment(ref ChannelsLastConvolutions);
        System.Threading.Interlocked.Increment(ref ChannelsLastNchwRoutes);
        return true;
    }

    // The pass on cuDNN with channels-last descriptors (a mixed-layout pass through a converted copy).
    private bool TryCudnnChannelsLastRoute(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int channels,
        int height, int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth, int padH, int padW,
        int strideH, int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc)
    {
        if (!CudnnConvLoaded() || _cudnnConv is null || _cudnnContext is null) return false;
        if (RunCudnnChannelsLast(pass, a, b, c, batch, channels, height, width, outChannels, kernelH, kernelW, outHeight,
                outWidth, padH, padW, strideH, strideW, dilationH, dilationW, inputNhwc, outputNhwc))
            return true;
        if (inputNhwc == outputNhwc) return false;

        // cuDNN takes few passes whose two activations differ in layout (the backward passes of a stack's first layer,
        // its input NCHW): the smaller activation changes to the other's layout in scratch, and the pass runs in one.
        long inputCount = (long)batch * channels * height * width, outputCount = (long)batch * outChannels * outHeight * outWidth;
        int inSpatial = height * width, outSpatial = outHeight * outWidth;
        bool convertInputSide = inputCount <= outputCount;   // the x / dx side takes the output side's layout
        bool layout = convertInputSide ? outputNhwc : inputNhwc;
        var scratch = TryConvScratch(ref _channelsLastScratchA, convertInputSide ? inputCount : outputCount);
        if (scratch is null) return false;
        IChannelsLastBackend self = this;
        switch (pass)
        {
            case 0:   // a = x, c = y
                if (convertInputSide)
                    return self.TryChangeLayout(a, scratch, batch, channels, inSpatial, layout)
                           && RunCudnnChannelsLast(0, scratch, b, c, batch, channels, height, width, outChannels, kernelH,
                               kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, layout, layout);
                return RunCudnnChannelsLast(0, a, b, scratch, batch, channels, height, width, outChannels, kernelH, kernelW,
                           outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, layout, layout)
                       && self.TryChangeLayout(scratch, c, batch, outChannels, outSpatial, outputNhwc);
            case 1:   // b = dy, c = dx
                if (convertInputSide)
                    return RunCudnnChannelsLast(1, a, b, scratch, batch, channels, height, width, outChannels, kernelH,
                               kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, layout, layout)
                           && self.TryChangeLayout(scratch, c, batch, channels, inSpatial, inputNhwc);
                return self.TryChangeLayout(b, scratch, batch, outChannels, outSpatial, layout)
                       && RunCudnnChannelsLast(1, a, scratch, c, batch, channels, height, width, outChannels, kernelH,
                           kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, layout, layout);
            default:  // a = x, b = dy
                if (convertInputSide)
                    return self.TryChangeLayout(a, scratch, batch, channels, inSpatial, layout)
                           && RunCudnnChannelsLast(2, scratch, b, c, batch, channels, height, width, outChannels, kernelH,
                               kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, layout, layout);
                return self.TryChangeLayout(b, scratch, batch, outChannels, outSpatial, layout)
                       && RunCudnnChannelsLast(2, a, scratch, c, batch, channels, height, width, outChannels, kernelH,
                           kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, layout, layout);
        }
    }

    private bool RunCudnnChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int channels,
        int height, int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth, int padH, int padW,
        int strideH, int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc)
    {
        bool allNhwc = inputNhwc && outputNhwc;
        int filterSpatial = kernelH * kernelW;
        IChannelsLastBackend self = this;
        bool launched;
        if (pass == 2 && allNhwc)
        {
            // The filter gradient of channels-last activations: cuDNN computes it only into a channels-last filter
            // [k, r, s, c], so it goes to scratch and changes to [k, c, r, s] after.
            var scratch = TryConvScratch(ref _channelsLastScratchB, (long)outChannels * channels * filterSpatial);
            launched = scratch is not null
                       && LaunchCudnnChannelsLast(pass, a, b, scratch, batch, channels, height, width, outChannels, kernelH,
                           kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, inputNhwc,
                           outputNhwc, filterNhwc: true)
                       && self.TryChangeLayout(scratch, c, outChannels, channels, filterSpatial, toChannelsLast: false);
        }
        else
        {
            launched = LaunchCudnnChannelsLast(pass, a, b, c, batch, channels, height, width, outChannels, kernelH, kernelW,
                outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW, inputNhwc, outputNhwc,
                filterNhwc: false);
            if (!launched && allNhwc)
            {
                // Some channels-last forward and data-gradient shapes (few input channels) run only with the filter
                // channels-last too: it changes to [k, r, s, c] in scratch first.
                var filter = pass == 0 ? b : a;
                var scratch = TryConvScratch(ref _channelsLastScratchB, (long)outChannels * channels * filterSpatial);
                launched = scratch is not null
                           && self.TryChangeLayout(filter, scratch, outChannels, channels, filterSpatial, toChannelsLast: true)
                           && LaunchCudnnChannelsLast(pass, pass == 0 ? a : scratch, pass == 0 ? scratch : b, c, batch,
                               channels, height, width, outChannels, kernelH, kernelW, outHeight, outWidth, padH, padW,
                               strideH, strideW, dilationH, dilationW, inputNhwc, outputNhwc, filterNhwc: true);
            }
        }
        if (launched) System.Threading.Interlocked.Increment(ref ChannelsLastConvolutions);
        return launched;
    }

    private bool LaunchCudnnChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int channels,
        int height, int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth, int padH, int padW,
        int strideH, int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc, bool filterNhwc)
    {
        var context = _cudnnContext!;
        lock (context.StreamBindingLock)
        {
            context.BindStream(_stream);
            return _cudnnConv!.ConvolutionChannelsLast(pass, a.Handle, b.Handle, c.Handle, batch, channels, height, width,
                outChannels, kernelH, kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH, dilationW,
                inputNhwc, outputNhwc, CudnnConvMath, GpuDeterminism.IsActive, IsStreamCapturing(), filterNhwc);
        }
    }

    // Whether cuDNN's convolutions loaded (EnsureCudnnConv throws where the library is absent).
    private bool CudnnConvLoaded()
    {
        try
        {
            EnsureCudnnConv();
        }
        catch (InvalidOperationException)
        {
            return false;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        return _cudnnConv is not null;
    }

    private static int BatchNormNhwcSplits(int rows, int channels)
    {
        int channelBlocks = (channels / 4 + 31) / 32;
        int wanted = (BatchNormNhwcTargetBlocks + channelBlocks - 1) / channelBlocks;
        int rowLimit = Math.Max(1, rows / BatchNormNhwcMinRowsPerSplit);
        return Math.Max(1, Math.Min(Math.Min(wanted, rowLimit), BatchNormNhwcMaxSplits));
    }

    // The finalize's block: a warp per 32 splits, up to 256 threads.
    private static uint BatchNormNhwcFinalizeThreads(int splits) => (uint)Math.Min(256, Math.Max(32, (splits + 31) / 32 * 32));

    private static bool BatchNormNhwcShape(int rows, int channels)
        => rows > 0 && channels > 0 && (channels & 3) == 0 && (long)rows * channels <= int.MaxValue;

    private uint ElementwiseBlocks(long quads)
        => (uint)Math.Max(1, Math.Min((quads + DefaultBlockSize - 1) / DefaultBlockSize, 65535L * 8));

    unsafe bool IChannelsLastBackend.TryBatchNormChannelsLast(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma,
        IGpuBuffer beta, IGpuBuffer saveMean, IGpuBuffer saveVariance, int rows, int channels, float epsilon, int act,
        float slope)
    {
        long n = (long)rows * channels;
        if (!IsAvailable || !BatchNormNhwcShape(rows, channels) || act is < 0 or > 2
            || (act == 2 && !ChannelEpilogueActivation.IsSlope(slope))
            || input.Size < n || output.Size < n || gamma.Size < channels || beta.Size < channels
            || saveMean.Size < channels || saveVariance.Size < channels
            || !_kernelCache.TryGetValue("batchnorm_nhwc_stats", out var stats)
            || !_kernelCache.TryGetValue("batchnorm_nhwc_finalize", out var finalize)
            || !_kernelCache.TryGetValue("batchnorm_nhwc_apply", out var apply))
            return false;
        int splits = BatchNormNhwcSplits(rows, channels);
        lock (_batchNormSplitLock)
        {
            if (EnsureBatchNormSplitPartials(channels * splits * 3) is not { } partials) return false;
            using var _ = PushContext();
            IntPtr pi = input.Handle, po = output.Handle, pg = gamma.Handle, pb = beta.Handle, pm = saveMean.Handle,
                pv = saveVariance.Handle, pp = partials.Handle;
            int r = rows, c = channels, sp = splits, a = act;
            float eps = epsilon, sl = act == 2 ? slope : 0f;
            long quads = n / 4;
            void** statArgs = stackalloc void*[5];
            statArgs[0] = &pi; statArgs[1] = &pp; statArgs[2] = &r; statArgs[3] = &c; statArgs[4] = &sp;
            LaunchKernel2D(stats, (uint)((channels / 4 + 31) / 32), (uint)splits, 1, 32, 8, statArgs);
            void** finArgs = stackalloc void*[6];
            finArgs[0] = &pi; finArgs[1] = &pp; finArgs[2] = &pm; finArgs[3] = &pv; finArgs[4] = &c; finArgs[5] = &sp;
            LaunchKernel(finalize, (uint)channels, BatchNormNhwcFinalizeThreads(splits), finArgs);
            void** args = stackalloc void*[11];
            args[0] = &pi; args[1] = &po; args[2] = &pg; args[3] = &pb; args[4] = &pm; args[5] = &pv; args[6] = &quads;
            args[7] = &c; args[8] = &eps; args[9] = &a; args[10] = &sl;
            LaunchKernel(apply, ElementwiseBlocks(quads), DefaultBlockSize, args);
        }
        System.Threading.Interlocked.Increment(ref ChannelsLastBatchNorms);
        return true;
    }

    unsafe bool IChannelsLastBackend.TryBatchNormChannelsLastBackward(IGpuBuffer gradOutput, IGpuBuffer? maskSource,
        IGpuBuffer? gradOutput2, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int channels, float epsilon,
        int maskMode, int act, float slope, bool accumulate)
    {
        long n = (long)rows * channels;
        if (!IsAvailable || !BatchNormNhwcShape(rows, channels) || maskMode is < 0 or > 2 || act is < 0 or > 2
            || (maskMode != 0) != (act != 0) || (act == 2 && !ChannelEpilogueActivation.IsSlope(slope))
            || (maskMode != 0 && maskSource is null)
            || (maskMode == 1 && maskSource!.Size < n) || (maskMode == 2 && maskSource!.Size < channels)
            || gradOutput.Size < n || input.Size < n || gradInput.Size < n || (gradOutput2 is not null && gradOutput2.Size < n)
            || gamma.Size < channels || saveMean.Size < channels || saveVariance.Size < channels
            || gradGamma.Size < channels || gradBeta.Size < channels
            || !_kernelCache.TryGetValue("batchnorm_nhwc_bwd_reduce", out var reduce)
            || !_kernelCache.TryGetValue("batchnorm_nhwc_bwd_finalize", out var finalize)
            || !_kernelCache.TryGetValue("batchnorm_nhwc_bwd_apply", out var apply))
            return false;
        int splits = BatchNormNhwcSplits(rows, channels);
        lock (_batchNormSplitLock)
        {
            if (EnsureBatchNormSplitPartials(channels * splits * 2) is not { } partials) return false;
            using var _ = PushContext();
            IntPtr pdy = gradOutput.Handle, pms = maskSource?.Handle ?? IntPtr.Zero, pdy2 = gradOutput2?.Handle ?? IntPtr.Zero,
                px = input.Handle, pg = gamma.Handle, pm = saveMean.Handle, pv = saveVariance.Handle, pdx = gradInput.Handle,
                pdg = gradGamma.Handle, pdb = gradBeta.Handle, pp = partials.Handle;
            int r = rows, c = channels, sp = splits, mode = maskMode, a = act, acc = accumulate ? 1 : 0;
            float eps = epsilon, sl = act == 2 ? slope : 0f, invN = 1.0f / rows;
            long quads = n / 4;
            void** reduceArgs = stackalloc void*[15];
            reduceArgs[0] = &pdy; reduceArgs[1] = &pms; reduceArgs[2] = &pdy2; reduceArgs[3] = &px; reduceArgs[4] = &pg;
            reduceArgs[5] = &pm; reduceArgs[6] = &pv; reduceArgs[7] = &pp; reduceArgs[8] = &r; reduceArgs[9] = &c;
            reduceArgs[10] = &eps; reduceArgs[11] = &sp; reduceArgs[12] = &mode; reduceArgs[13] = &a; reduceArgs[14] = &sl;
            LaunchKernel2D(reduce, (uint)((channels / 4 + 31) / 32), (uint)splits, 1, 32, 8, reduceArgs);
            void** finArgs = stackalloc void*[5];
            finArgs[0] = &pp; finArgs[1] = &pdg; finArgs[2] = &pdb; finArgs[3] = &c; finArgs[4] = &sp;
            LaunchKernel(finalize, (uint)channels, BatchNormNhwcFinalizeThreads(splits), finArgs);
            void** args = stackalloc void*[18];
            args[0] = &pdy; args[1] = &pms; args[2] = &pdy2; args[3] = &px; args[4] = &pg; args[5] = &pm; args[6] = &pv;
            args[7] = &pdg; args[8] = &pdb; args[9] = &pdx; args[10] = &quads; args[11] = &c; args[12] = &eps;
            args[13] = &mode; args[14] = &a; args[15] = &sl; args[16] = &acc; args[17] = &invN;
            LaunchKernel(apply, ElementwiseBlocks(quads), DefaultBlockSize, args);
        }
        System.Threading.Interlocked.Increment(ref ChannelsLastBatchNorms);
        return true;
    }

    unsafe bool IChannelsLastBackend.TryChangeLayout(IGpuBuffer input, IGpuBuffer output, int batch, int channels,
        int spatial, bool toChannelsLast)
    {
        long n = (long)batch * channels * spatial;
        if (!IsAvailable || batch <= 0 || channels <= 0 || spatial <= 0 || batch > MaxGridDimY || n > int.MaxValue
            || input.Size < n || output.Size < n || !_kernelCache.TryGetValue("layout_transpose_tiles", out var kernel))
            return false;
        int rows = toChannelsLast ? channels : spatial, cols = toChannelsLast ? spatial : channels;
        if ((rows + 31) / 32 > MaxGridDimY) return false;
        using var _ = PushContext();
        IntPtr pi = input.Handle, po = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pi; args[1] = &po; args[2] = &rows; args[3] = &cols;
        LaunchKernel2D(kernel, (uint)((cols + 31) / 32), (uint)((rows + 31) / 32), (uint)batch, 32, 8, args);
        return true;
    }
}

/// <summary>One channels-last convolution pass's arguments (<see cref="IChannelsLastBackend.TryConvolutionChannelsLast"/>).</summary>
internal readonly struct ChannelsLastConvArgs
{
    internal ChannelsLastConvArgs(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int channels, int height,
        int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth, int padH, int padW, int strideH,
        int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc)
    {
        Pass = pass; A = a; B = b; C = c; Batch = batch; Channels = channels; Height = height; Width = width;
        OutChannels = outChannels; KernelH = kernelH; KernelW = kernelW; OutHeight = outHeight; OutWidth = outWidth;
        PadH = padH; PadW = padW; StrideH = strideH; StrideW = strideW; DilationH = dilationH; DilationW = dilationW;
        InputNhwc = inputNhwc; OutputNhwc = outputNhwc;
    }

    internal int Pass { get; }
    internal IGpuBuffer A { get; }
    internal IGpuBuffer B { get; }
    internal IGpuBuffer C { get; }
    internal int Batch { get; }
    internal int Channels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int OutChannels { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int OutHeight { get; }
    internal int OutWidth { get; }
    internal int PadH { get; }
    internal int PadW { get; }
    internal int StrideH { get; }
    internal int StrideW { get; }
    internal int DilationH { get; }
    internal int DilationW { get; }
    internal bool InputNhwc { get; }
    internal bool OutputNhwc { get; }

    /// <summary>The pass's output element count (y, dx or dw).</summary>
    internal int OutputCount => Pass switch
    {
        0 => Batch * OutChannels * OutHeight * OutWidth,
        1 => Batch * Channels * Height * Width,
        _ => OutChannels * Channels * KernelH * KernelW,
    };

    internal unsafe TunedShape Shape()
    {
        int* d = stackalloc int[17];
        d[0] = Batch; d[1] = Channels; d[2] = Height; d[3] = Width; d[4] = OutChannels;
        d[5] = KernelH; d[6] = KernelW; d[7] = StrideH; d[8] = StrideW; d[9] = PadH; d[10] = PadW;
        d[11] = DilationH; d[12] = DilationW; d[13] = OutHeight; d[14] = OutWidth; d[15] = Pass;
        d[16] = (InputNhwc ? 1 : 0) | (OutputNhwc ? 2 : 0);
        return TunedShape.Create(TunedKernelDType.Float32, new ReadOnlySpan<int>(d, 17));
    }
}
