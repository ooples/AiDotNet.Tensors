// Copyright (c) AiDotNet. All rights reserved.
// Convolution forward through the tuned-kernel registry on Metal (#1116).
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// As CudaBackend / OpenClBackend.TunedConvolution: the reference is Metal's established direct convolution (a thread
// per output element); the candidate is im2col + ONE GEMM over the whole batch.
public sealed partial class MetalBackend
{
    // Convolution reassociates long K = Cin*kH*kW dot products; GEMM and the direct loop differ by a few ulps of the scale.
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
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("metal.conv2d.direct", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedConv2DArgs a) => Conv2DDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("metal.conv2d.im2col_gemm", TunedKernelOrigin.Generated, true,
                            shape => TunedConv2DArgs.GemmScratchFits(shape) && HasTunedKernel(TunedRowKernelTemplate.Im2ColName, 1),
                            (in TunedConv2DArgs a) => Conv2DIm2ColGemm(a)),
                        // #1369: Winograd F(2x2,3x3) as transforms around one batched GEMM.
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("metal.conv2d.winograd_gemm", TunedKernelOrigin.Builtin, true,
                            shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinograd(a)) Conv2DDirect(a); }),
                    };
                    _conv2DForwardSlot = new TunedKernelSlot<TunedConv2DArgs>(TunedKernelOp.Conv2DForward, TunedDeviceKey,
                        new BackendTunedKernelHarness<TunedConv2DArgs>(this, ConvolutionTolerance, () => IsAvailable,
                            MeasureTunedKernelMilliseconds, a => new[] { (a.Output, a.OutputCount) },
                            a => SameBuffer(a.Output, a.Input) || SameBuffer(a.Output, a.Kernel)),
                        () => GpuDeterminism.IsActive, candidates);
                }
                return _conv2DForwardSlot;
            }
        }
    }

    private void Conv2DDirect(in TunedConv2DArgs a)
        => TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv2d_direct",
            a.Batch * a.OutChannels * a.OutHeight * a.OutWidth, new[] { a.Input, a.Kernel, a.Output },
            new[] { a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels, a.OutHeight, a.OutWidth,
                a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW });

    // tmp[Cout, B * L] = W[Cout, P] * col[P, B * L] as ONE GEMM, then a permute to the output's [B, Cout, L] (none for
    // B = 1).
    private void Conv2DIm2ColGemm(in TunedConv2DArgs a)
    {
        int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
        using var col = AllocateBuffer(checked(a.Batch * p * l));
        var pipeline = GetPipeline(TunedRowLibraryName, _tunedRowLibrary, TunedRowKernelTemplate.Im2ColName);
        var (groups, threads) = pipeline.Calculate1DDispatch(a.Batch * l);
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(AsMetal(a.Input), 0);
            encoder.SetBuffer(AsMetal(col), 1);
            int[] geometry =
            {
                a.Batch, a.InChannels, a.InHeight, a.InWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW,
                a.PadH, a.PadW, a.DilationH, a.DilationW, a.OutHeight, a.OutWidth,
            };
            for (int i = 0; i < geometry.Length; i++) encoder.SetBytes(geometry[i], (ulong)(i + 2));
            encoder.DispatchThreadgroups(groups, threads);
        }
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
