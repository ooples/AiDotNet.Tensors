#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Convolution forward through the tuned-kernel registry on WebGPU (#1117): the established direct kernel as the
// reference, im2col + GEMM as the candidate (see VulkanBackend.TunedConvolution).
public sealed partial class WebGpuBackend
{
    private const double ConvolutionTolerance = 1e-4;
    private TunedKernelSlot<TunedConv2DArgs>? _conv2DForwardSlot;

    // FP32 im2col to [P, B * L] (P = Cin*kH*kW, L = outH*outW): one invocation per (batch, output pixel). The batch
    // sits inside each row so ONE GEMM covers every batch item.
    private const string Im2ColSource = @"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> col: array<f32>;
struct ConvParams {
    batch: i32, channels: i32, height: i32, width: i32,
    kernel_h: i32, kernel_w: i32, stride_h: i32, stride_w: i32,
    pad_h: i32, pad_w: i32, dilation_h: i32, dilation_w: i32,
    out_h: i32, out_w: i32, pad0: i32, pad1: i32,
}
@group(0) @binding(2) var<uniform> p: ConvParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    let l = p.out_h * p.out_w;
    if (idx >= p.batch * l) { return; }
    let b = idx / l;
    let s = idx % l;
    let oh = s / p.out_w;
    let ow = s % p.out_w;
    for (var c = 0; c < p.channels; c = c + 1) {
        for (var kh = 0; kh < p.kernel_h; kh = kh + 1) {
            for (var kw = 0; kw < p.kernel_w; kw = kw + 1) {
                let ih = oh * p.stride_h - p.pad_h + kh * p.dilation_h;
                let iw = ow * p.stride_w - p.pad_w + kw * p.dilation_w;
                var v: f32 = 0.0;
                if (ih >= 0 && ih < p.height && iw >= 0 && iw < p.width) {
                    v = x[((b * p.channels + c) * p.height + ih) * p.width + iw];
                }
                col[((c * p.kernel_h + kh) * p.kernel_w + kw) * (p.batch * l) + b * l + s] = v;
            }
        }
    }
}
";

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
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("webgpu.conv2d.direct", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedConv2DArgs a) => Conv2DDirectReference(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("webgpu.conv2d.im2col_gemm", TunedKernelOrigin.Generated, true,
                            shape => IsAvailable && TunedConv2DArgs.GemmScratchFits(shape),
                            (in TunedConv2DArgs a) => Conv2DIm2ColGemm(a)),
                        // #1372: Winograd F(2x2,3x3) as transforms around one batched GEMM.
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("webgpu.conv2d.winograd_gemm", TunedKernelOrigin.Builtin, true,
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

    private void Conv2DDirectReference(in TunedConv2DArgs a)
    {
        var uniforms = MakeConvUniforms(a.Batch, a.InChannels, a.OutChannels, a.InHeight, a.InWidth, a.OutHeight, a.OutWidth,
            a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(a.OutWidth, a.OutHeight, a.Batch * a.OutChannels);
        WithDistinctBindings(new[] { a.Input, a.Kernel, a.Output }, b =>
            Dispatch3Buffer3DAsync("Convolution", WebGpuKernels.ConvolutionSource, "conv2d",
                b[0], b[1], b[2], uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult());
    }

    // tmp[Cout, B * L] = W[Cout, P] * col[P, B * L] as ONE GEMM, then a permute to the output's [B, Cout, L] (none for B = 1).
    private void Conv2DIm2ColGemm(in TunedConv2DArgs a)
    {
        int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
        long workgroups = ((long)a.Batch * l + 255) / 256;
        if (workgroups > WebGpuTunedRowKernels.MaxWorkgroups)
            throw new GpuCapabilityUnavailableException("The im2col dispatch exceeds WebGPU's workgroup-count limit.");
        using var col = AllocateBuffer(checked(a.Batch * p * l));
        var uniforms = new int[]
        {
            a.Batch, a.InChannels, a.InHeight, a.InWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW,
            a.PadH, a.PadW, a.DilationH, a.DilationW, a.OutHeight, a.OutWidth, 0, 0,
        };
        var bits = new float[uniforms.Length];
        for (int i = 0; i < uniforms.Length; i++) bits[i] = BitConverter.Int32BitsToSingle(uniforms[i]);
        TunedDispatch("conv2d_im2col", Im2ColSource, new[] { a.Input, col }, bits, workgroups);
        if (a.Batch == 1)
        {
            // The GEMM writes alpha * acc + beta * C; zero the output so a stale NaN cannot survive beta = 0.
            Fill(a.Output, 0f, checked(a.OutChannels * l));
            Gemm(a.Kernel, col, a.Output, a.OutChannels, l, p);
            return;
        }
        using var tmp = AllocateBuffer(checked(a.OutChannels * a.Batch * l));
        Gemm(a.Kernel, col, tmp, a.OutChannels, a.Batch * l, p);
        Permute(tmp, a.Output, new[] { a.OutChannels, a.Batch, l }, new[] { 1, 0, 2 });
    }
}
#endif
