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
                        var candidates = new ITunedKernelCandidate<TunedConv2DArgs>[]
                        {
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.generic", TunedKernelOrigin.Builtin, true,
                                _ => true, (in TunedConv2DArgs a) => { Conv2DGeneric(a); ConvEpiloguePasses(a); }),
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.im2col_gemm", TunedKernelOrigin.Generated, true,
                                shape => TunedConv2DArgs.GemmScratchFits(shape) && HasTunedKernel(Kernels.OpenClTunedRowKernels.Im2ColName),
                                (in TunedConv2DArgs a) => { Conv2DIm2ColGemm(a); ConvEpiloguePasses(a); }),
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
                                shape => WinogradF23Pipeline.AdmitsShape(shape) && HasTunedKernel(Kernels.WinogradFusedKernels.Forward),
                                (in TunedConv2DArgs a) => Conv2DWinogradFused(a)),
                            // im2col (none for a 1x1 stride-1 unpadded batch-1 conv) + the register-blocked split-K GEMM,
                            // which keeps every CU busy on the small late-layer products CLBlast leaves underfilled.
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.im2col_rb_gemm", TunedKernelOrigin.Builtin, true,
                                shape => TunedConv2DArgs.GemmScratchFits(shape) && shape[0] == 1
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.Im2ColWide)
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.Rb64SplitK),
                                (in TunedConv2DArgs a) => Conv2DIm2ColRbGemm(a)),
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
            { "opencl.conv2d.winograd_fused", "opencl.conv2d.im2col_rb_gemm", "opencl.conv2d.im2col_gemm" };

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

        private void ConvEpiloguePasses(in TunedConv2DArgs a) => ConvEpilogue.RunPasses(this, a);

        private void Conv2DGeneric(in TunedConv2DArgs a)
            => Conv2DGenericCore(a.Input, a.Kernel, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels,
                a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW);

        private void Conv2DDirect(in TunedConv2DArgs a)
        {
            var k = _kernelCache["conv2d_direct"];
            ConvArgs(k, a, withKernelGeometry: true);
            k.SetArg(18, ConvEpilogueBias(a)); k.SetArg(19, ConvEpilogueFlags(a));
            k.Execute3D(a.OutWidth, a.OutHeight, a.Batch * a.OutChannels, 8, 8, 1);
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
            var k = _kernelCache[Kernels.WinogradFusedKernels.Forward];
            uint arg = 0;
            k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(a.Kernel)); k.SetArg(arg++, BufferHandle(a.Output));
            k.SetArg(arg++, a.Batch); k.SetArg(arg++, a.InChannels); k.SetArg(arg++, a.InHeight); k.SetArg(arg++, a.InWidth);
            k.SetArg(arg++, a.OutChannels); k.SetArg(arg++, a.OutHeight); k.SetArg(arg++, a.OutWidth); k.SetArg(arg++, a.PadH);
            k.SetArg(arg++, tilesH); k.SetArg(arg++, tilesW);
            k.SetArg(arg++, ConvEpilogueBias(a)); k.SetArg(arg++, ConvEpilogueFlags(a));
            k.Execute2D(RoundUpToMultiple(p, 32) / 32 * 256, (a.OutChannels + 31) / 32, 256, 1);
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
            int tiles = ((m + 63) / 64) * ((n + 63) / 64);
            int target = 4 * (int)Math.Max(1u, _context?.MaxComputeUnits ?? 8u);
            int splits = Math.Max(1, Math.Min((target + tiles - 1) / tiles, k / 64));
            int kChunk = RoundUpToMultiple((k + splits - 1) / splits, 16);
            splits = (k + kChunk - 1) / kChunk;
            var gemm = _kernelCache[Kernels.BatchedGemmKernels.Rb64SplitK];
            IGpuBuffer? partial = splits > 1 ? AllocateBuffer(checked(splits * m * n)) : null;
            try
            {
                gemm.SetArg(0, BufferHandle(a)); gemm.SetArg(1, BufferHandle(b)); gemm.SetArg(2, BufferHandle(partial ?? c));
                gemm.SetArg(3, m); gemm.SetArg(4, n); gemm.SetArg(5, k); gemm.SetArg(6, k); gemm.SetArg(7, n);
                gemm.SetArg(8, kChunk);
                gemm.SetArg(9, BufferHandle(bias ?? c)); gemm.SetArg(10, partial is null ? epilogue : 0);
                gemm.Execute3D(RoundUpToMultiple(n, 64) / 4, RoundUpToMultiple(m, 64) / 4, splits, 16, 16, 1);
                if (partial is null) return;
                var reduce = _kernelCache[Kernels.BatchedGemmKernels.SplitKReduce];
                reduce.SetArg(0, BufferHandle(partial)); reduce.SetArg(1, BufferHandle(c));
                reduce.SetArg(2, m * n); reduce.SetArg(3, splits);
                reduce.SetArg(4, BufferHandle(bias ?? c)); reduce.SetArg(5, epilogue); reduce.SetArg(6, n);
                reduce.Execute1D(RoundUpToMultiple(m * n, 256), 256);
            }
            finally
            {
                partial?.Dispose();
            }
        }

        // tmp[Cout, B * L] = W[Cout, P] * col[P, B * L] as ONE GEMM, then a permute to the output's [B, Cout, L] (none for
        // B = 1).
        private void Conv2DIm2ColGemm(in TunedConv2DArgs a)
        {
            int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
            // A 1x1, stride-1, unpadded convolution's im2col is the input itself ([Cin, H*W] per batch item): multiply it
            // directly instead of copying it to a column buffer first.
            if (a.KernelH == 1 && a.KernelW == 1 && a.StrideH == 1 && a.StrideW == 1 && a.PadH == 0 && a.PadW == 0
                && a.Batch == 1)
            {
                Gemm(a.Kernel, a.Input, a.Output, a.OutChannels, l, p);
                return;
            }
            using var col = AllocateBuffer(checked(a.Batch * p * l));
            var k = _kernelCache[Kernels.OpenClTunedRowKernels.Im2ColName];
            uint arg = 0;
            k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(col));
            k.SetArg(arg++, a.Batch); k.SetArg(arg++, a.InChannels); k.SetArg(arg++, a.InHeight); k.SetArg(arg++, a.InWidth);
            k.SetArg(arg++, a.KernelH); k.SetArg(arg++, a.KernelW); k.SetArg(arg++, a.StrideH); k.SetArg(arg++, a.StrideW);
            k.SetArg(arg++, a.PadH); k.SetArg(arg++, a.PadW); k.SetArg(arg++, a.DilationH); k.SetArg(arg++, a.DilationW);
            k.SetArg(arg++, a.OutHeight); k.SetArg(arg++, a.OutWidth);
            k.Execute1D(RoundUpToMultiple(a.Batch * l, 64), 64);
            if (a.Batch == 1)
            {
                Gemm(a.Kernel, col, a.Output, a.OutChannels, l, p);
                return;
            }
            using var tmp = AllocateBuffer(checked(a.OutChannels * a.Batch * l));
            Gemm(a.Kernel, col, tmp, a.OutChannels, a.Batch * l, p);
            Permute(tmp, a.Output, new[] { a.OutChannels, a.Batch, l }, new[] { 1, 0, 2 });
        }
    }
}
