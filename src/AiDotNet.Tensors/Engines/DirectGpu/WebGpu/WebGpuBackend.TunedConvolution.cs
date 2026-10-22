#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Convolution forward through the tuned-kernel registry on WebGPU (#1117): the established direct kernel as the
// reference, im2col + GEMM as the candidate (see VulkanBackend.TunedConvolution).
public sealed partial class WebGpuBackend : IConvBiasActivationBackend
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
                            (in TunedConv2DArgs a) => { Conv2DIm2ColGemm(a); ConvEpilogue.RunPasses(this, a); }),
                        // #1372: Winograd F(2x2,3x3) as transforms around one batched GEMM.
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("webgpu.conv2d.winograd_gemm", TunedKernelOrigin.Builtin, true,
                            shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinograd(a)) Conv2DDirectReference(a); }),
                        // #1432: F(4x4, 3x3), 36 per-point GEMMs for 4x fewer multiplies than direct (F(2x2): 2.25x).
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("webgpu.conv2d.winograd_f43_gemm", TunedKernelOrigin.Builtin, true,
                            shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradF43KernelsReady,
                            (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinogradF43(a)) Conv2DDirectReference(a); }),
                        // #1432: fused F(2x2, 3x3): transforms, per-point products and output transform in one launch.
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("webgpu.conv2d.winograd_fused", TunedKernelOrigin.Builtin, true,
                            shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradFusedForwardReady,
                            (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinogradFused(a)) Conv2DDirectReference(a); }),
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

    // The direct kernel with the args' epilogue (ConvEpilogue): its own module, since the bias is a fourth binding (input,
    // kernel and bias read, then the output, as WithDistinctBindings orders them). Without an epilogue the established
    // conv2d kernel runs.
    private void Conv2DDirectReference(in TunedConv2DArgs a)
    {
        var uniforms = MakeConvUniforms(a.Batch, a.InChannels, a.OutChannels, a.InHeight, a.InWidth, a.OutHeight, a.OutWidth,
            a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(a.OutWidth, a.OutHeight, a.Batch * a.OutChannels);
        int flags = ConvEpilogue.Flags(a);
        if (flags == 0)
        {
            WithDistinctBindings(new[] { a.Input, a.Kernel, a.Output }, b =>
                Dispatch3Buffer3DAsync("Convolution", WebGpuKernels.ConvolutionSource, "conv2d",
                    b[0], b[1], b[2], uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult());
            return;
        }
        // The uniform struct gains the flags in its first padding slot (see ConvEpilogueSource).
        var withFlags = new float[Math.Max(uniforms.Length, 16)];
        Array.Copy(uniforms, withFlags, uniforms.Length);
        withFlags[15] = BitConverter.Int32BitsToSingle(flags);
        using var placeholder = a.Bias is null ? AllocateBuffer(1) : null;
        var bias = a.Bias ?? placeholder ?? a.Output;
        WithDistinctBindings(new[] { a.Input, a.Kernel, bias, a.Output }, b =>
            Dispatch4Buffer3DAsync("ConvolutionEpilogue", ConvEpilogueSource, "conv2d_epilogue",
                b[0], b[1], b[2], b[3], withFlags, wgX, wgY, wgZ).GetAwaiter().GetResult());
    }

    // WebGpuKernels.ConvolutionSource's conv2d with the epilogue: bias at binding 2, the output at 3, the uniform at 4.
    // epi bit 0 adds bias[oc], bit 1 applies ReLU (r < 0 is false for NaN, which passes).
    private const string ConvEpilogueSource = @"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
struct ConvParams {
    batch_size: u32, in_channels: u32, out_channels: u32, in_height: u32, in_width: u32, out_height: u32, out_width: u32,
    kernel_height: u32, kernel_width: u32, stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32, dilation_h: u32,
    dilation_w: u32, epi: u32,
}
@group(0) @binding(4) var<uniform> params: ConvParams;

@compute @workgroup_size(8, 8, 1)
fn conv2d_epilogue(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    let out_c = gid.z % params.out_channels;
    let batch = gid.z / params.out_channels;
    if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch_size) {
        return;
    }
    var acc: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var ky: u32 = 0u; ky < params.kernel_height; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < params.kernel_width; kx = kx + 1u) {
                let in_y = i32(out_y * params.stride_h + ky * params.dilation_h) - i32(params.pad_h);
                let in_x = i32(out_x * params.stride_w + kx * params.dilation_w) - i32(params.pad_w);
                if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
                    let in_idx = batch * params.in_channels * params.in_height * params.in_width
                               + ic * params.in_height * params.in_width
                               + u32(in_y) * params.in_width + u32(in_x);
                    let k_idx = out_c * params.in_channels * params.kernel_height * params.kernel_width
                              + ic * params.kernel_height * params.kernel_width
                              + ky * params.kernel_width + kx;
                    acc = acc + input[in_idx] * kernel[k_idx];
                }
            }
        }
    }
    var r = acc;
    if ((params.epi & 1u) != 0u) { r = r + bias[out_c]; }
    if ((params.epi & 2u) != 0u && r < 0.0) { r = 0.0; }
    let out_idx = batch * params.out_channels * params.out_height * params.out_width
                + out_c * params.out_height * params.out_width
                + out_y * params.out_width + out_x;
    output[out_idx] = r;
}
";

    // tmp[Cout, B * L] = W[Cout, P] * col[P, B * L] as ONE GEMM, then a permute to the output's [B, Cout, L] (none for B = 1).
    private void Conv2DIm2ColGemm(in TunedConv2DArgs a)
    {
        int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
        long workgroups = ((long)a.Batch * l + 255) / 256;   // past 65535 the dispatch folds (WebGpuDispatchFold)
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

    /// <inheritdoc/>
    public bool TryConv2DBiasActivation(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, FusedActivationType activation)
    {
        if (!ConvEpilogue.Fuses(activation) || batch <= 0 || outHeight <= 0 || outWidth <= 0) return false;
        var args = new TunedConv2DArgs(input, kernel, output, batch, inChannels, inHeight, inWidth, outChannels, outHeight,
            outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, bias, activation);
        if (!Conv2DForwardSlot.TryExecute(args.Shape(), args)) Conv2DDirectReference(args);
        return true;
    }
}
#endif
