using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Convolution forward through the tuned-kernel registry on Vulkan (#1115): the established direct shader as the
// reference, im2col + GEMM as the candidate (see CudaBackend.TunedConvolution).
public sealed unsafe partial class VulkanBackend
{
    private const double ConvolutionTolerance = 1e-4;
    private TunedKernelSlot<TunedConv2DArgs>? _conv2DForwardSlot;

    // FP32 im2col to [P, B * L] (P = Cin*kH*kW, L = outH*outW): one invocation per (batch, output pixel). The batch
    // sits inside each row so ONE GEMM covers every batch item.
    private const string Im2ColShader = @"#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) writeonly buffer C { float col[]; };
layout(push_constant) uniform P { int batch; int channels; int height; int width; int kernelH; int kernelW; int strideH;
    int strideW; int padH; int padW; int dilationH; int dilationW; int outH; int outW; } pc;
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    int l = pc.outH * pc.outW;
    if (idx >= pc.batch * l) return;
    int b = idx / l, s = idx % l, oh = s / pc.outW, ow = s % pc.outW;
    int patchSize = pc.channels * pc.kernelH * pc.kernelW;
    for (int c = 0; c < pc.channels; c++)
        for (int kh = 0; kh < pc.kernelH; kh++)
            for (int kw = 0; kw < pc.kernelW; kw++)
            {
                int ih = oh * pc.strideH - pc.padH + kh * pc.dilationH;
                int iw = ow * pc.strideW - pc.padW + kw * pc.dilationW;
                float v = (ih >= 0 && ih < pc.height && iw >= 0 && iw < pc.width) ? x[((b * pc.channels + c) * pc.height + ih) * pc.width + iw] : 0.0;
                col[((c * pc.kernelH + kh) * pc.kernelW + kw) * (pc.batch * l) + b * l + s] = v;
            }
}";

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
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("vulkan.conv2d.direct", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedConv2DArgs a) => Conv2DDirectReference(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("vulkan.conv2d.im2col_gemm", TunedKernelOrigin.Generated, true,
                            shape => TunedConv2DArgs.GemmScratchFits(shape) && TunedVariantsAvailable,
                            (in TunedConv2DArgs a) => Conv2DIm2ColGemm(a)),
                        // #1371: Winograd F(2x2,3x3) as transforms around one batched GEMM.
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("vulkan.conv2d.winograd_gemm", TunedKernelOrigin.Builtin, true,
                            shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinograd(a)) Conv2DDirectReference(a); }),
                    };
                    _conv2DForwardSlot = new TunedKernelSlot<TunedConv2DArgs>(TunedKernelOp.Conv2DForward, TunedDeviceKey,
                        new BackendTunedKernelHarness<TunedConv2DArgs>(this, ConvolutionTolerance, () => CanMeasureTunedKernels,
                            MeasureTunedKernelMilliseconds, a => new[] { (a.Output, a.OutputCount) },
                            a => ReferenceEquals(a.Output, a.Input) || ReferenceEquals(a.Output, a.Kernel)),
                        () => GpuDeterminism.IsActive, candidates);
                }
                return _conv2DForwardSlot;
            }
        }
    }

    private static uint[] ConvPush(in TunedConv2DArgs a) => new uint[]
    {
        (uint)a.Batch, (uint)a.InChannels, (uint)a.InHeight, (uint)a.InWidth, (uint)a.OutChannels, (uint)a.OutHeight,
        (uint)a.OutWidth, (uint)a.KernelH, (uint)a.KernelW, (uint)a.StrideH, (uint)a.StrideW, (uint)a.PadH, (uint)a.PadW,
        (uint)a.DilationH, (uint)a.DilationW,
    };

    private void Conv2DDirectReference(in TunedConv2DArgs a)
    {
        if (!TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv2D, ConvPush(a), a.OutputCount, a.Input, a.Kernel, a.Output))
            throw new GpuCapabilityUnavailableException("The Vulkan Conv2D shader is unavailable.");
    }

    // tmp[Cout, B * L] = W[Cout, P] * col[P, B * L] as ONE GEMM, then a permute to the output's [B, Cout, L] (none for B = 1).
    private void Conv2DIm2ColGemm(in TunedConv2DArgs a)
    {
        int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
        using var col = AllocateBuffer(checked(a.Batch * p * l));
        GlslNaryOp(Im2ColShader, new[] { a.Input, col }, a.Batch * l, new uint[]
        {
            (uint)a.Batch, (uint)a.InChannels, (uint)a.InHeight, (uint)a.InWidth, (uint)a.KernelH, (uint)a.KernelW, (uint)a.StrideH,
            (uint)a.StrideW, (uint)a.PadH, (uint)a.PadW, (uint)a.DilationH, (uint)a.DilationW, (uint)a.OutHeight, (uint)a.OutWidth,
        });
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