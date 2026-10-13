using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Convolution forward through the tuned-kernel registry on OpenCL (#1114), as CudaBackend.TunedConvolution: the
    // reference is the established generic dispatch (Winograd F(2x2,3x3) for 3x3 stride 1, else tiled up to 7x7, else
    // direct); the candidates are the explicit tiled and direct kernels and im2col + GEMM.
    public sealed partial class OpenClBackend
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
                                _ => true, (in TunedConv2DArgs a) => Conv2DGeneric(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.im2col_gemm", TunedKernelOrigin.Generated, true,
                                shape => TunedConv2DArgs.GemmScratchFits(shape) && HasTunedKernel(Kernels.OpenClTunedRowKernels.Im2ColName),
                                (in TunedConv2DArgs a) => Conv2DIm2ColGemm(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.tiled", TunedKernelOrigin.Builtin, true,
                                shape => shape[5] <= 7 && shape[6] <= 7 && HasTunedKernel("conv2d_tiled"),
                                (in TunedConv2DArgs a) => Conv2DTiled(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DArgs>("opencl.conv2d.direct", TunedKernelOrigin.Builtin, true,
                                _ => HasTunedKernel("conv2d_direct"), (in TunedConv2DArgs a) => Conv2DDirect(a)),
                        };
                        _conv2DForwardSlot = new TunedKernelSlot<TunedConv2DArgs>(TunedKernelOp.Conv2DForward, TunedDeviceKey,
                            new BackendTunedKernelHarness<TunedConv2DArgs>(this, ConvolutionTolerance, () => IsAvailable,
                                MeasureTunedKernelMilliseconds, a => new[] { (a.Output, a.OutputCount) },
                                a => a.Output.Handle == a.Input.Handle || a.Output.Handle == a.Kernel.Handle),
                            () => GpuDeterminism.IsActive, candidates);
                    }
                    return _conv2DForwardSlot;
                }
            }
        }

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

        private void Conv2DGeneric(in TunedConv2DArgs a)
            => Conv2DGenericCore(a.Input, a.Kernel, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels,
                a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW);

        private void Conv2DDirect(in TunedConv2DArgs a)
        {
            var k = _kernelCache["conv2d_direct"];
            ConvArgs(k, a, withKernelGeometry: true);
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
            int tilesX = (a.OutWidth + TileOut - 1) / TileOut, tilesY = (a.OutHeight + TileOut - 1) / TileOut;
            k.Execute3D(tilesX * TileOut, tilesY * TileOut, a.Batch * a.OutChannels, TileOut, TileOut, 1);
        }

        // tmp[Cout, B * L] = W[Cout, P] * col[P, B * L] as ONE GEMM, then a permute to the output's [B, Cout, L] (none for
        // B = 1).
        private void Conv2DIm2ColGemm(in TunedConv2DArgs a)
        {
            int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
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
