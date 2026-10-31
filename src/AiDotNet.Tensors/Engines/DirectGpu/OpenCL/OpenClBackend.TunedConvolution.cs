using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Convolution forward through the tuned-kernel registry on OpenCL (#1114), as CudaBackend.TunedConvolution: the
    // reference is the established generic dispatch (Winograd F(2x2,3x3) for 3x3 stride 1, else tiled up to 7x7, else
    // direct); the candidates are the explicit tiled and direct kernels and im2col + GEMM.
    public sealed partial class OpenClBackend : IConvBiasActivationBackend
    {
        // Convolution reassociates long K = Cin*kH*kW dot products; Winograd and GEMM differ by a few ulps of the scale.
        private const double ConvolutionTolerance = 1e-4;
        private TunedKernelSlot<TunedConv2DArgs>? _conv2DForwardSlot;

        internal TunedKernelSlot<TunedConv2DArgs> Conv2DForwardSlot
        {
            get
            {
                if (_conv2DForwardSlot is { } s) return s;
                lock (_tunedSlotsLock)
                {
                    if (_conv2DForwardSlot is null)
                    {
                        // The first applicable candidate is the slot's reference: every tuning run executes it once for the
                        // correctness check and times it in every pair. im2col + GEMM comes first so that is a GEMM wherever
                        // its scratch fits: the generic dispatch is a direct convolution there, up to 1.4 s a call on a batch-64
                        // ResNet-50 layer, so measuring one shape spent the whole 2 s tuning budget and every other shape of a
                        // process ran on its untuned default. The generic dispatch stays the reference where the scratch does not fit.
                        var candidates = new ITunedKernelCandidate<TunedConv2DArgs>[]
                        {
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.im2col_gemm", TunedKernelOrigin.Generated, true,
                                shape => TunedConv2DArgs.GemmScratchFits(shape) && HasTunedKernel(Kernels.OpenClTunedRowKernels.Im2ColName),
                                (in TunedConv2DArgs a) => { if (!Conv2DIm2ColGemm(a)) ConvEpiloguePasses(a); }),
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.generic", TunedKernelOrigin.Builtin, true,
                                _ => true, (in TunedConv2DArgs a) => { Conv2DGeneric(a); ConvEpiloguePasses(a); }),
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.tiled", TunedKernelOrigin.Builtin, true,
                                shape => shape[5] <= 7 && shape[6] <= 7 && HasTunedKernel("conv2d_tiled"),
                                (in TunedConv2DArgs a) => Conv2DTiled(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.direct", TunedKernelOrigin.Builtin, true,
                                _ => HasTunedKernel("conv2d_direct"), (in TunedConv2DArgs a) => Conv2DDirect(a)),
                            // #1370: Winograd as transforms around one batched GEMM, where the generic route's Winograd
                            // kernel computes each output tile's whole channel sum in one work item.
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.winograd_gemm", TunedKernelOrigin.Builtin, true,
                                shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                                (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinograd(a)) { Conv2DGeneric(a); ConvEpiloguePasses(a); } }),
                            // F(4x4, 3x3): 36 per-point GEMMs for 4x fewer multiplies than direct, against F(2x2, 3x3)'s 2.25x.
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.winograd_f43_gemm", TunedKernelOrigin.Builtin, true,
                                shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradF43KernelsReady,
                                (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinogradF43(a)) { Conv2DGeneric(a); ConvEpiloguePasses(a); } }),
                            // Fused F(2x2, 3x3): transforms, per-point products and output transform in one launch, so
                            // the 16-plane V and M never reach global memory.
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.winograd_fused", TunedKernelOrigin.Builtin, true,
                                shape => WinogradF23Pipeline.AdmitsShape(shape)
                                    && HasTunedKernel(Kernels.WinogradFusedKernels.Forward, WinogradFusedGroupSize),
                                (in TunedConv2DArgs a) => Conv2DWinogradFused(a)),
                            // im2col (none for a 1x1 stride-1 unpadded batch-1 conv) + the register-blocked split-K GEMM,
                            // which keeps every CU busy on the small late-layer products CLBlast leaves underfilled.
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.im2col_rb_gemm", TunedKernelOrigin.Builtin, true,
                                shape => TunedConv2DArgs.GemmScratchFits(shape) && shape[0] == 1
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.Im2ColWide)
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.Rb64SplitK, Rb64GroupSide * Rb64GroupSide),
                                (in TunedConv2DArgs a) => Conv2DIm2ColRbGemm(a)),
                            // A 1x1, stride-1, unpadded convolution over a batch: one strided-batched GEMM per image with
                            // the filter shared (A stride 0), written straight into the NCHW output. im2col + GEMM copies the
                            // whole input into a column buffer and permutes the whole output back, two passes over each,
                            // and DenseNet-121 runs 58 such convolutions on its growing concatenations.
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.pointwise_batched", TunedKernelOrigin.Builtin, true,
                                shape => shape[0] > 1 && shape[5] == 1 && shape[6] == 1 && shape[7] == 1 && shape[8] == 1
                                    && shape[9] == 0 && shape[10] == 0 && ClBlastNative.IsAvailable,
                                (in TunedConv2DArgs a) =>
                                {
                                    if (!Conv2DPointwiseBatched(a)) Conv2DGeneric(a);
                                    ConvEpiloguePasses(a);
                                }),
                            // The same 1x1 convolution through conv1x1_rb128v: no vendor call (CLBlast's strided-batched
                            // and plain SGEMM cost 20-170 us of host time a call, so DenseNet-121's 58 of them held the
                            // host behind the device), the bias and ReLU in the output kernel, and K split where the
                            // tile grid underfills the device (the 4 x 4 maps).
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.pointwise_rb", TunedKernelOrigin.Builtin, true,
                                shape => shape[5] == 1 && shape[6] == 1 && shape[7] == 1 && shape[8] == 1
                                    && shape[9] == 0 && shape[10] == 0 && (long)shape[2] * shape[3] % 4 == 0
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.Conv1x1Rb128V, Rb128GroupSide * Rb128GroupSide)
                                    && HasTunedKernel(Kernels.WinogradFusedKernels.SplitReduce),
                                (in TunedConv2DArgs a) => Conv2DPointwiseRb(a)),
                            // A stem over at most four input channels: each work item reads a pixel's inputs once for 16
                            // output channels.
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.small_channel", TunedKernelOrigin.Builtin, true,
                                shape => shape[1] <= SmallChannelConvMaxChannels
                                    && shape[1] * shape[5] * shape[6] <= Kernels.WinogradFusedKernels.SmallChannelMaxTaps
                                    && HasTunedKernel(Kernels.WinogradFusedKernels.SmallChannel),
                                (in TunedConv2DArgs a) => Conv2DSmallChannel(a)),
                        };
                        _conv2DForwardSlot = new TunedKernelSlot<TunedConv2DArgs>(TunedKernelOp.Conv2DForward, TunedDeviceKey,
                            new BackendTunedKernelHarness<TunedConv2DArgs>(this, ConvolutionTolerance, () => IsAvailable,
                                MeasureTunedKernelMilliseconds, a => new[] { (a.Output, a.OutputCount) },
                                a => a.Output.Handle == a.Input.Handle || a.Output.Handle == a.Kernel.Handle),
                            () => GpuDeterminism.IsActive, candidates);
                        // A shape the process has no tuning budget left for: the generic reference is a direct convolution,
                        // tens of times slower than GEMM on large batches (ResNet-50 at batch 64: 42 s an epoch against 13.5 s
                        // tuned). Each candidate's own applicability narrows this list: fused Winograd admits 3x3 stride-1
                        // shapes, the register-blocked split-K GEMM batch 1, im2col + GEMM whatever its scratch fits.
                        _conv2DForwardSlot.UntunedPreference = static _ => s_conv2DForwardUntunedPreference;
                    }
                    return _conv2DForwardSlot;
                }
            }
        }

        private static readonly string[] s_conv2DForwardUntunedPreference =
            { "opencl.conv2d.winograd_fused", "opencl.conv2d.im2col_rb_gemm", "opencl.conv2d.pointwise_rb", "opencl.conv2d.im2col_gemm" };

        private static void ConvArgs(DirectOpenClKernel k, in TunedConv2DArgs a, bool withKernelGeometry)
        {
            uint arg = 0;
            k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(a.Kernel)); k.SetArg(arg++, BufferHandle(a.Output));
            k.SetArg(arg++, a.Batch); k.SetArg(arg++, a.InChannels); k.SetArg(arg++, a.InHeight); k.SetArg(arg++, a.InWidth);
            k.SetArg(arg++, a.OutChannels); k.SetArg(arg++, a.OutHeight); k.SetArg(arg++, a.OutWidth);
            if (!withKernelGeometry) return;
            k.SetArg(arg++, a.KernelH); k.SetArg(arg++, a.KernelW); k.SetArg(arg++, a.StrideH); k.SetArg(arg++, a.StrideW);
            k.SetArg(arg++, a.PadH); k.SetArg(arg++, a.PadW); k.SetArg(arg++, a.DilationH); k.SetArg(arg++, a.DilationW);
        }

        // The epilogue flags the output kernels take (see conv_epilogue): bit 0 bias, bit 1 ReLU. The routes that write
        // the output themselves (fused Winograd, F(4x4) output transform, split-K GEMM) apply these; the others run
        // ConvEpiloguePasses after the convolution.
        private static int ConvEpilogueFlags(in TunedConv2DArgs a) => ConvEpilogue.Flags(a);

        // A bias buffer for the epilogue argument; the output stands in when there is none (the kernel never reads it).
        private static IntPtr ConvEpilogueBias(in TunedConv2DArgs a) => BufferHandle(a.Bias ?? a.Output);

        private void ConvEpiloguePasses(in TunedConv2DArgs a)
        {
            if (!TryConvNchwEpilogue(a)) ConvEpilogue.RunPasses(this, a);
        }

        // out[b] = W[Cout, Cin] * x[b][Cin, H * W] for every image b, the filter shared across the batch. False when CLBlast
        // declined (the caller then runs the generic route).
        internal bool Conv2DPointwiseBatched(in TunedConv2DArgs a)
            => TryExecuteClBlastBatchedGemm(a.Kernel, a.Input, a.Output, a.OutChannels, a.OutHeight * a.OutWidth,
                a.InChannels, a.Batch, 1f, 0f, strideA: 0);

        // y[n] = W * x[n] for every image n by conv1x1_rb128v, K split into partial planes summed by wino_f23_split_reduce
        // (which applies the epilogue) when the 128 x 128 tile grid alone covers fewer than about four groups per compute
        // unit; each slice keeps at least 64 channels.
        private void Conv2DPointwiseRb(in TunedConv2DArgs a)
        {
            int hw = a.OutHeight * a.OutWidth, m = a.OutChannels, k = a.InChannels;
            int n = checked(a.Batch * hw);
            int tiles = ((m + Rb128Tile - 1) / Rb128Tile) * ((n + Rb128Tile - 1) / Rb128Tile);
            int splits = SplitKFactor(tiles, k, Rb128MinSlice, Rb128GroupsPerComputeUnit, SplitComputeUnits);
            int kChunk = RoundUpToMultiple((k + splits - 1) / splits, Rb128ChunkAlign);
            splits = (k + kChunk - 1) / kChunk;
            int count = checked(m * n);
            using IGpuBuffer? partial = splits > 1 ? AllocateBuffer(checked(splits * count)) : null;
            var gemm = _kernelCache[Kernels.BatchedGemmKernels.Conv1x1Rb128V];
            gemm.SetArg(0, BufferHandle(a.Kernel)); gemm.SetArg(1, BufferHandle(a.Input));
            gemm.SetArg(2, BufferHandle(partial ?? a.Output));
            gemm.SetArg(3, m); gemm.SetArg(4, n); gemm.SetArg(5, k); gemm.SetArg(6, hw); gemm.SetArg(7, kChunk);
            gemm.SetArg(8, ConvEpilogueBias(a)); gemm.SetArg(9, partial is null ? ConvEpilogueFlags(a) : 0);
            gemm.Execute3D(RoundUpToMultiple(n, Rb128Tile) / Rb128OutputsPerItem,
                RoundUpToMultiple(m, Rb128Tile) / Rb128OutputsPerItem, splits, Rb128GroupSide, Rb128GroupSide, 1);
            if (partial is null) return;
            var reduce = _kernelCache[Kernels.WinogradFusedKernels.SplitReduce];
            reduce.SetArg(0, BufferHandle(partial)); reduce.SetArg(1, BufferHandle(a.Output));
            reduce.SetArg(2, count); reduce.SetArg(3, splits);
            reduce.SetArg(4, ConvEpilogueBias(a)); reduce.SetArg(5, ConvEpilogueFlags(a));
            reduce.SetArg(6, hw); reduce.SetArg(7, m);
            reduce.Execute1D(RoundUpToMultiple(count, ReduceGroupSize), ReduceGroupSize);
        }

        // Split-K sizing, shared by every split-K route: K is cut into slices, each at least minSlice deep, until the tile
        // grid times the slices covers about groupsPerComputeUnit work groups per compute unit. At least one slice. Each
        // caller then aligns the slice depth to its kernel's channel step.
        internal static int SplitKFactor(int tiles, int depth, int minSlice, int groupsPerComputeUnit, int computeUnits)
        {
            int target = groupsPerComputeUnit * computeUnits;
            return Math.Max(1, Math.Min((target + tiles - 1) / tiles, depth / minSlice));
        }

        // The compute units split-K sizes its grid for: ComputeUnits (detected, or AIDOTNET_GPU_COMPUTE_UNITS), or
        // DefaultComputeUnits when the device reported none.
        private int SplitComputeUnits => ComputeUnits > 0 ? ComputeUnits : DefaultComputeUnits;

        private const int DefaultComputeUnits = 8;

        // The work-group size of the split-K reduce kernels (wino_f23_split_reduce, gemm_splitk_reduce).
        private const int ReduceGroupSize = 256;

        // conv1x1_rb128v: a 128 x 128 output tile per 16 x 16 group, 8 x 8 outputs per work item, K in 8-channel steps.
        // Split while the grid covers fewer than about four groups per compute unit, each slice at least 64 channels.
        private const int Rb128Tile = 128;
        private const int Rb128OutputsPerItem = 8;
        private const int Rb128GroupSide = Rb128Tile / Rb128OutputsPerItem;
        private const int Rb128ChunkAlign = 8;
        private const int Rb128MinSlice = 64;
        private const int Rb128GroupsPerComputeUnit = 4;

        // gemm_rb64_splitk: a 64 x 64 tile per 16 x 16 group, 4 x 4 outputs per work item, K in 16-deep steps. Split
        // while the grid covers fewer than about four groups per compute unit, each slice at least 64 deep.
        private const int Rb64Tile = 64;
        private const int Rb64OutputsPerItem = 4;
        private const int Rb64GroupSide = Rb64Tile / Rb64OutputsPerItem;
        private const int Rb64ChunkAlign = 16;
        private const int Rb64MinSlice = 64;
        private const int Rb64GroupsPerComputeUnit = 4;

        // wino_f23_fused_forward: 32 output channels x 32 tiles per 256-item group, channels in 8-channel steps (WF_BO,
        // WF_BP and WF_CK in its source). Split while the grid covers fewer than about two groups per compute unit, each
        // layer at least 32 channels.
        private const int WinogradFusedBlock = 32;
        private const int WinogradFusedGroupSize = 256;
        private const int WinogradFusedChunkAlign = 8;
        private const int WinogradFusedMinSlice = 32;
        private const int WinogradFusedGroupsPerComputeUnit = 2;

        private const int SmallChannelConvMaxChannels = 4;

        private void Conv2DSmallChannel(in TunedConv2DArgs a)
        {
            DirectOpenClKernel k;
            if (a.KernelH == 3 && a.KernelW == 3 && _kernelCache.TryGetValue(Kernels.WinogradFusedKernels.SmallChannel3x3, out var k3))
            {
                k = k3;
                ConvArgs(k, a, withKernelGeometry: false);
                k.SetArg(10, a.StrideH); k.SetArg(11, a.StrideW); k.SetArg(12, a.PadH); k.SetArg(13, a.PadW);
                k.SetArg(14, a.DilationH); k.SetArg(15, a.DilationW);
                k.SetArg(16, ConvEpilogueBias(a)); k.SetArg(17, ConvEpilogueFlags(a));
            }
            else
            {
                k = _kernelCache[Kernels.WinogradFusedKernels.SmallChannel];
                ConvArgs(k, a, withKernelGeometry: true);
                k.SetArg(18, ConvEpilogueBias(a)); k.SetArg(19, ConvEpilogueFlags(a));
            }
            const int block = Kernels.WinogradFusedKernels.SmallChannelBlock;
            k.Execute2D(RoundUpToMultiple(checked(a.Batch * a.OutHeight * a.OutWidth), 64),
                (a.OutChannels + block - 1) / block, 64, 1);
        }

        private void Conv2DGeneric(in TunedConv2DArgs a)
            => Conv2DGenericCore(a.Input, a.Kernel, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels,
                a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW);

        private void Conv2DDirect(in TunedConv2DArgs a)
        {
            var k = _kernelCache["conv2d_direct"];
            ConvArgs(k, a, withKernelGeometry: true);
            k.SetArg(18, ConvEpilogueBias(a)); k.SetArg(19, ConvEpilogueFlags(a));
            k.ExecutePlanes(a.OutWidth, a.OutHeight, a.Batch * a.OutChannels);
        }

        private void Conv2DTiled(in TunedConv2DArgs a)
        {
            const int TileOut = 16;
            var k = _kernelCache["conv2d_tiled"];
            ConvArgs(k, a, withKernelGeometry: true);
            int tileInH = TileOut * a.StrideH + (a.KernelH - 1) * a.DilationH;
            int tileInW = TileOut * a.StrideW + (a.KernelW - 1) * a.DilationW;
            k.SetLocalArg(18, tileInH * tileInW * sizeof(float));
            k.SetArg(19, ConvEpilogueBias(a)); k.SetArg(20, ConvEpilogueFlags(a));
            int tilesX = (a.OutWidth + TileOut - 1) / TileOut, tilesY = (a.OutHeight + TileOut - 1) / TileOut;
            k.Execute3D(tilesX * TileOut, tilesY * TileOut, a.Batch * a.OutChannels, TileOut, TileOut, 1);
        }

        private void Conv2DWinogradFused(in TunedConv2DArgs a)
        {
            int tilesH = (a.OutHeight + 1) / 2, tilesW = (a.OutWidth + 1) / 2;
            int p = a.Batch * tilesH * tilesW;
            int tileGroups = (p + WinogradFusedBlock - 1) / WinogradFusedBlock;
            int channelGroups = (a.OutChannels + WinogradFusedBlock - 1) / WinogradFusedBlock;
            int rChunk = WinogradFusedChannelChunk(tileGroups * channelGroups, a.InChannels, SplitComputeUnits);
            int splits = (a.InChannels + rChunk - 1) / rChunk;
            int count = a.Batch * a.OutChannels * a.OutHeight * a.OutWidth;
            using IGpuBuffer? partial = splits > 1 ? AllocateBuffer(checked(splits * count)) : null;
            var k = _kernelCache[Kernels.WinogradFusedKernels.Forward];
            uint arg = 0;
            k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(a.Kernel));
            k.SetArg(arg++, BufferHandle(partial ?? a.Output));
            k.SetArg(arg++, a.Batch); k.SetArg(arg++, a.InChannels); k.SetArg(arg++, a.InHeight); k.SetArg(arg++, a.InWidth);
            k.SetArg(arg++, a.OutChannels); k.SetArg(arg++, a.OutHeight); k.SetArg(arg++, a.OutWidth); k.SetArg(arg++, a.PadH);
            k.SetArg(arg++, tilesH); k.SetArg(arg++, tilesW);
            k.SetArg(arg++, ConvEpilogueBias(a)); k.SetArg(arg++, partial is null ? ConvEpilogueFlags(a) : 0);
            k.SetArg(arg, rChunk);
            k.Execute3D(tileGroups * WinogradFusedGroupSize, channelGroups, splits, WinogradFusedGroupSize, 1, 1);
            if (partial is null) return;
            var reduce = _kernelCache[Kernels.WinogradFusedKernels.SplitReduce];
            reduce.SetArg(0, BufferHandle(partial)); reduce.SetArg(1, BufferHandle(a.Output));
            reduce.SetArg(2, count); reduce.SetArg(3, splits);
            reduce.SetArg(4, ConvEpilogueBias(a)); reduce.SetArg(5, ConvEpilogueFlags(a));
            reduce.SetArg(6, a.OutHeight * a.OutWidth); reduce.SetArg(7, a.OutChannels);
            reduce.Execute1D(RoundUpToMultiple(count, ReduceGroupSize), ReduceGroupSize);
        }

        // The channels each work-group layer of the fused Winograd reduces: all of them, unless the (tile, output channel)
        // grid leaves the device underfilled, as VGG-11's 2 x 2 and 4 x 4 maps (16 and 64 groups at batch 32) and
        // DenseNet-121's growth convolutions (4 and 16) did. Then the channels are split, at least 32 per layer, until
        // the grid covers about two groups per compute unit. A multiple of the kernel's 8-channel step. Measured at batch
        // 32 on gfx1012 (11 reported units): 512 channels on a 2 x 2 map 246 us whole, 204 us in two layers, 274 us in
        // four; 128 channels into 32 on 4 x 4 maps 48 us whole, 30-32 us in two to four; on the 64-group 4 x 4 maps every
        // split was slower (587 us whole, 621 us in two).
        internal static int WinogradFusedChannelChunk(int groups, int channels, int computeUnits)
        {
            int splits = SplitKFactor(groups, channels, WinogradFusedMinSlice, WinogradFusedGroupsPerComputeUnit, computeUnits);
            return RoundUpToMultiple((channels + splits - 1) / splits, WinogradFusedChunkAlign);
        }

        // Batch 1 only (the slot admits it only there): y[Cout, L] = W[Cout, P] * col[P, L], col the input itself for a
        // 1x1 stride-1 unpadded convolution.
        private void Conv2DIm2ColRbGemm(in TunedConv2DArgs a)
        {
            int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
            if (a.KernelH == 1 && a.KernelW == 1 && a.StrideH == 1 && a.StrideW == 1 && a.PadH == 0 && a.PadW == 0)
            {
                RbSplitKGemm(a.Kernel, a.Input, a.Output, a.OutChannels, l, p, a.Bias, ConvEpilogueFlags(a));
                return;
            }
            long colCount = (long)p * l;
            using var col = AllocateBuffer(checked((int)colCount));
            var k = _kernelCache[Kernels.BatchedGemmKernels.Im2ColWide];
            uint arg = 0;
            k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(col));
            k.SetArg(arg++, a.InChannels); k.SetArg(arg++, a.InHeight); k.SetArg(arg++, a.InWidth);
            k.SetArg(arg++, a.KernelH); k.SetArg(arg++, a.KernelW); k.SetArg(arg++, a.StrideH); k.SetArg(arg++, a.StrideW);
            k.SetArg(arg++, a.PadH); k.SetArg(arg++, a.PadW); k.SetArg(arg++, a.DilationH); k.SetArg(arg++, a.DilationW);
            k.SetArg(arg++, a.OutHeight); k.SetArg(arg++, a.OutWidth);
            k.Execute1D(RoundUpToMultiple(checked((int)colCount), 256), 256);
            RbSplitKGemm(a.Kernel, col, a.Output, a.OutChannels, l, p, a.Bias, ConvEpilogueFlags(a));
        }

        // C[M x N] = A[M x K] * B[K x N] (row-major, contiguous) through gemm_rb64_splitk: K is split into slices of at
        // least 64 until the 64 x 64 tile grid times the slices covers about four work groups per compute unit, and the
        // slices' partial planes are summed by gemm_splitk_reduce. The epilogue (see ConvEpilogueFlags; bias per row of C)
        // runs in whichever kernel writes C.
        internal void RbSplitKGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k, IGpuBuffer? bias = null,
            int epilogue = 0)
        {
            int tiles = ((m + Rb64Tile - 1) / Rb64Tile) * ((n + Rb64Tile - 1) / Rb64Tile);
            int splits = SplitKFactor(tiles, k, Rb64MinSlice, Rb64GroupsPerComputeUnit, SplitComputeUnits);
            int kChunk = RoundUpToMultiple((k + splits - 1) / splits, Rb64ChunkAlign);
            splits = (k + kChunk - 1) / kChunk;
            var gemm = _kernelCache[Kernels.BatchedGemmKernels.Rb64SplitK];
            using IGpuBuffer? partial = splits > 1 ? AllocateBuffer(checked(splits * m * n)) : null;
            gemm.SetArg(0, BufferHandle(a)); gemm.SetArg(1, BufferHandle(b)); gemm.SetArg(2, BufferHandle(partial ?? c));
            gemm.SetArg(3, m); gemm.SetArg(4, n); gemm.SetArg(5, k); gemm.SetArg(6, k); gemm.SetArg(7, n);
            gemm.SetArg(8, kChunk);
            gemm.SetArg(9, BufferHandle(bias ?? c)); gemm.SetArg(10, partial is null ? epilogue : 0);
            gemm.Execute3D(RoundUpToMultiple(n, Rb64Tile) / Rb64OutputsPerItem, RoundUpToMultiple(m, Rb64Tile) / Rb64OutputsPerItem,
                splits, Rb64GroupSide, Rb64GroupSide, 1);
            if (partial is null) return;
            var reduce = _kernelCache[Kernels.BatchedGemmKernels.SplitKReduce];
            reduce.SetArg(0, BufferHandle(partial)); reduce.SetArg(1, BufferHandle(c));
            reduce.SetArg(2, m * n); reduce.SetArg(3, splits);
            reduce.SetArg(4, BufferHandle(bias ?? c)); reduce.SetArg(5, epilogue); reduce.SetArg(6, n);
            reduce.Execute1D(RoundUpToMultiple(m * n, ReduceGroupSize), ReduceGroupSize);
        }

        // tmp[Cout, B * L] = W[Cout, P] * col[P, B * L] as ONE GEMM, then a permute to the output's [B, Cout, L] (none for
        // B = 1).
        // True when the bias and ReLU were applied too (conv_gemm_epilogue); false when the caller still runs them.
        private bool Conv2DIm2ColGemm(in TunedConv2DArgs a)
        {
            int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
            // A 1x1, stride-1, unpadded convolution's im2col is the input itself ([Cin, H*W] per batch item): multiply it
            // directly instead of copying it to a column buffer first.
            if (a.KernelH == 1 && a.KernelW == 1 && a.StrideH == 1 && a.StrideW == 1 && a.PadH == 0 && a.PadW == 0
                && a.Batch == 1)
            {
                Gemm(a.Kernel, a.Input, a.Output, a.OutChannels, l, p);
                return ConvEpilogueFlags(a) == 0 || TryConvGemmEpilogue(a.Output, a);
            }
            using var col = AllocateBuffer(checked(a.Batch * p * l));
            var k = _kernelCache[Kernels.OpenClTunedRowKernels.Im2ColName];
            uint arg = 0;
            k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(col));
            k.SetArg(arg++, a.Batch); k.SetArg(arg++, a.InChannels); k.SetArg(arg++, a.InHeight); k.SetArg(arg++, a.InWidth);
            k.SetArg(arg++, a.KernelH); k.SetArg(arg++, a.KernelW); k.SetArg(arg++, a.StrideH); k.SetArg(arg++, a.StrideW);
            k.SetArg(arg++, a.PadH); k.SetArg(arg++, a.PadW); k.SetArg(arg++, a.DilationH); k.SetArg(arg++, a.DilationW);
            k.SetArg(arg++, a.OutHeight); k.SetArg(arg++, a.OutWidth);
            k.Execute1D(RoundUpToMultiple(a.Batch * l * p, 64), 64);   // one work item per col element
            if (a.Batch == 1)
            {
                Gemm(a.Kernel, col, a.Output, a.OutChannels, l, p);
                return ConvEpilogueFlags(a) == 0 || TryConvGemmEpilogue(a.Output, a);
            }
            using var tmp = AllocateBuffer(checked(a.OutChannels * a.Batch * l));
            Gemm(a.Kernel, col, tmp, a.OutChannels, a.Batch * l, p);
            // The permute back to NCHW and the epilogue in one pass.
            if (TryConvGemmEpilogue(tmp, a)) return true;
            Permute(tmp, a.Output, new[] { a.OutChannels, a.Batch, l }, new[] { 1, 0, 2 });
            return false;
        }
    }
}
