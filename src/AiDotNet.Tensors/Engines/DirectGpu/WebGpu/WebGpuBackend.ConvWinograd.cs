#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1372: Winograd F(2x2, 3x3) forward and input gradient and F(3x3, 2x2) weight gradient on WebGPU, through the shared
// WinogradF23Pipeline: the WGSL tile transforms of WebGpuWinogradKernels around BatchedGemm. Backward convolution gets
// tuned slots here for the first time, with the direct kernels it always ran as the references, so Winograd is served
// only on the shape classes where it measures faster.
public sealed partial class WebGpuBackend : IWinogradF23Transforms
{
    private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardDataSlot, _conv2DBackwardFilterSlot;
    private bool? _winogradReady;

    private const string WinogradModulePrefix = "Winograd_";

    // Every Winograd pipeline, built once. A module the implementation rejects would otherwise throw from inside a
    // tuned slot on the first 3x3 convolution; this turns it into a candidate that is not offered, and says why.
    private bool WinogradKernelsReady
    {
        get
        {
            if (_winogradReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_winogradReady is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in WinogradF23KernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(WinogradModulePrefix + name, WebGpuWinogradKernels.SourceFor(name), name)
                                    .GetAwaiter().GetResult();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU Winograd kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _winogradReady = ok;
                }
                return _winogradReady.Value;
            }
        }
    }

    internal TunedKernelSlot<TunedConv2DBackwardArgs> Conv2DBackwardDataSlot
    {
        get
        {
            if (_conv2DBackwardDataSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _conv2DBackwardDataSlot ??= new TunedKernelSlot<TunedConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardData,
                    TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, new ITunedKernelCandidate<TunedConv2DBackwardArgs>[]
                    {
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("webgpu.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                            true, _ => true, (in TunedConv2DBackwardArgs a) => LaunchDxDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("webgpu.conv2d_dx.winograd_f23", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinograd(a)) LaunchDxDirect(a); }),
                    });
                return _conv2DBackwardDataSlot;
            }
        }
    }

    internal TunedKernelSlot<TunedConv2DBackwardArgs> Conv2DBackwardFilterSlot
    {
        get
        {
            if (_conv2DBackwardFilterSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _conv2DBackwardFilterSlot ??= new TunedKernelSlot<TunedConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardFilter,
                    TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, new ITunedKernelCandidate<TunedConv2DBackwardArgs>[]
                    {
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("webgpu.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                            true, _ => true, (in TunedConv2DBackwardArgs a) => LaunchDwDirect(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("webgpu.conv2d_dw.winograd_f32", TunedKernelOrigin.Builtin,
                            true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinograd(a)) LaunchDwDirect(a); }),
                    });
                return _conv2DBackwardFilterSlot;
            }
        }
    }

    private BackendTunedKernelHarness<TunedConv2DBackwardArgs> BackwardHarness() =>
        new(this, ConvolutionTolerance, () => CanMeasureTunedKernels, MeasureTunedKernelMilliseconds,
            a => new[] { (a.Output, a.OutputCount) },
            a => ReferenceEquals(a.Output, a.First) || ReferenceEquals(a.Output, a.Second));

    // Per-route entry points, for the slot candidates above and for the tests, which run one route regardless of what
    // the slot would pick. False when the route declines the shape (or its modules did not build); nothing is launched
    // then.
    internal bool TryConv2DForwardWinograd(in TunedConv2DArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryForward(this, this, a);

    internal bool TryConv2DBackwardDataWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardData(this, this, a);

    internal bool TryConv2DBackwardFilterWinograd(in TunedConv2DBackwardArgs a)
        => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardFilter(this, this, a);

    private static float[] BackwardUniforms(in TunedConv2DBackwardArgs a)
        => MakeConvUniforms(a.Batch, a.InChannels, a.OutChannels, a.InHeight, a.InWidth, a.OutHeight, a.OutWidth,
            a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW);

    // The direct kernels: one invocation per input element (8x8 groups over W x H x N*C) or per weight.
    private void LaunchDxDirect(in TunedConv2DBackwardArgs a)
    {
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(a.InWidth, a.InHeight, a.Batch * a.InChannels);
        Dispatch3Buffer3DAsync("Conv2DBackward", WebGpuKernels.Conv2DBackwardSource, "conv2d_backward_input",
            a.First, a.Second, a.Output, BackwardUniforms(a), wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    private void LaunchDwDirect(in TunedConv2DBackwardArgs a)
        => Dispatch3BufferAsync("Conv2DBackwardKernel", WebGpuKernels.Conv2DBackwardKernelSource, "conv2d_backward_kernel",
            a.First, a.Second, a.Output, BackwardUniforms(a), a.OutChannels * a.InChannels * a.KernelH * a.KernelW)
            .GetAwaiter().GetResult();

    private void RunWinograd(string name, long threads, IGpuBuffer read, IGpuBuffer write, params int[] ints)
    {
        // A uniform struct is a 16-byte multiple: the ints, bit-cast, then zero padding.
        var uniforms = new float[(ints.Length + 3) / 4 * 4];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        Dispatch2BufferAsync(WinogradModulePrefix + name, WebGpuWinogradKernels.SourceFor(name), name, read, write, uniforms,
            checked((int)threads)).GetAwaiter().GetResult();
    }

    void IWinogradF23Transforms.Filter(IGpuBuffer w, IGpuBuffer u, int k, int c, bool flip)
        => RunWinograd(flip ? WinogradF23KernelTemplate.FilterFlip : WinogradF23KernelTemplate.Filter, (long)k * c, w, u, k, c);

    void IWinogradF23Transforms.Input(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH, int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.Input, (long)c * n * tilesH * tilesW, x, v, n, c, h, w, pad, tilesH, tilesW);

    void IWinogradF23Transforms.Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH, int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.Output, (long)k * n * tilesH * tilesW, m, y, n, k, outH, outW, tilesH, tilesW);

    void IWinogradF23Transforms.GradOutputTiles(IGpuBuffer dy, IGpuBuffer u, int n, int k, int outH, int outW, int tilesH,
        int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.GradOutputTiles, (long)k * n * tilesH * tilesW, dy, u, n, k, outH, outW,
            tilesH, tilesW);

    void IWinogradF23Transforms.InputTilesTileMajor(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad,
        int tilesH, int tilesW)
        => RunWinograd(WinogradF23KernelTemplate.InputTilesTileMajor, (long)c * n * tilesH * tilesW, x, v, n, c, h, w, pad,
            tilesH, tilesW);

    void IWinogradF23Transforms.FilterGradient(IGpuBuffer m, IGpuBuffer dw, int k, int c, int splits)
        => RunWinograd(WinogradF23KernelTemplate.FilterGradient, (long)k * c, m, dw, k, c, splits);
}
#endif
