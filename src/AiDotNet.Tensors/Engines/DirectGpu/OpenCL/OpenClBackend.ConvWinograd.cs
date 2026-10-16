using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1370: Winograd F(2x2, 3x3) forward and input gradient and F(3x3, 2x2) weight gradient on OpenCL, through the
    // shared WinogradF23Pipeline: the transforms below (emitted from WinogradF23KernelTemplate) around BatchedGemm.
    // Backward convolution gets a tuned slot here for the first time, with the direct kernels it always ran as the
    // reference, so Winograd is served only on the shape classes where it measures faster.
    public sealed partial class OpenClBackend : IWinogradF23Transforms
    {
        private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardDataSlot, _conv2DBackwardFilterSlot;

        private bool WinogradKernelsReady => HasTunedKernel(WinogradF23KernelTemplate.Filter);

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
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                                true, _ => HasTunedKernel("conv2d_backward_input"), (in TunedConv2DBackwardArgs a) => LaunchDxDirect(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.winograd_f23", TunedKernelOrigin.Builtin,
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
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                                true, _ => HasTunedKernel("conv2d_backward_weights"), (in TunedConv2DBackwardArgs a) => LaunchDwDirect(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dw.winograd_f32", TunedKernelOrigin.Builtin,
                                true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                                (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinograd(a)) LaunchDwDirect(a); }),
                        });
                    return _conv2DBackwardFilterSlot;
                }
            }
        }

        private BackendTunedKernelHarness<TunedConv2DBackwardArgs> BackwardHarness() =>
            new(this, ConvolutionTolerance, () => IsAvailable, MeasureTunedKernelMilliseconds,
                a => new[] { (a.Output, a.OutputCount) },
                a => a.Output.Handle == a.First.Handle || a.Output.Handle == a.Second.Handle);

        // Per-route entry points, for the slot candidates above and for the tests, which run one route regardless of
        // what the slot would pick. False when the route declines the shape (or its module did not compile); nothing is
        // launched then.
        internal bool TryConv2DForwardWinograd(in TunedConv2DArgs a)
            => WinogradKernelsReady && WinogradF23Pipeline.TryForward(this, this, a);

        internal bool TryConv2DBackwardDataWinograd(in TunedConv2DBackwardArgs a)
            => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardData(this, this, a);

        internal bool TryConv2DBackwardFilterWinograd(in TunedConv2DBackwardArgs a)
            => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardFilter(this, this, a);

        private DirectOpenClKernel WinogradKernel(string name, IGpuBuffer read, IGpuBuffer write)
        {
            var k = _kernelCache[name];
            k.SetArg(0, BufferHandle(read));
            k.SetArg(1, BufferHandle(write));
            return k;
        }

        private static void RunWinograd(DirectOpenClKernel k, long threads)
            => k.Execute1D(RoundUpToMultiple(checked((int)threads), 256), 256);

        void IWinogradF23Transforms.Filter(IGpuBuffer w, IGpuBuffer u, int k, int c, bool flip)
        {
            var kernel = WinogradKernel(flip ? WinogradF23KernelTemplate.FilterFlip : WinogradF23KernelTemplate.Filter, w, u);
            kernel.SetArg(2, k); kernel.SetArg(3, c);
            RunWinograd(kernel, (long)k * c);
        }

        void IWinogradF23Transforms.Input(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH, int tilesW)
            => RunInputTransform(WinogradF23KernelTemplate.Input, x, v, n, c, h, w, pad, tilesH, tilesW);

        void IWinogradF23Transforms.InputTilesTileMajor(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad,
            int tilesH, int tilesW)
            => RunInputTransform(WinogradF23KernelTemplate.InputTilesTileMajor, x, v, n, c, h, w, pad, tilesH, tilesW);

        private void RunInputTransform(string name, IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad,
            int tilesH, int tilesW)
        {
            var kernel = WinogradKernel(name, x, v);
            uint arg = 2;
            kernel.SetArg(arg++, n); kernel.SetArg(arg++, c); kernel.SetArg(arg++, h); kernel.SetArg(arg++, w);
            kernel.SetArg(arg++, pad); kernel.SetArg(arg++, tilesH); kernel.SetArg(arg++, tilesW);
            RunWinograd(kernel, (long)c * n * tilesH * tilesW);
        }

        void IWinogradF23Transforms.Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH, int tilesW)
            => RunTileTransform(WinogradF23KernelTemplate.Output, m, y, n, k, outH, outW, tilesH, tilesW);

        void IWinogradF23Transforms.GradOutputTiles(IGpuBuffer dy, IGpuBuffer u, int n, int k, int outH, int outW,
            int tilesH, int tilesW)
            => RunTileTransform(WinogradF23KernelTemplate.GradOutputTiles, dy, u, n, k, outH, outW, tilesH, tilesW);

        private void RunTileTransform(string name, IGpuBuffer read, IGpuBuffer write, int n, int k, int outH, int outW,
            int tilesH, int tilesW)
        {
            var kernel = WinogradKernel(name, read, write);
            uint arg = 2;
            kernel.SetArg(arg++, n); kernel.SetArg(arg++, k); kernel.SetArg(arg++, outH); kernel.SetArg(arg++, outW);
            kernel.SetArg(arg++, tilesH); kernel.SetArg(arg++, tilesW);
            RunWinograd(kernel, (long)k * n * tilesH * tilesW);
        }

        void IWinogradF23Transforms.FilterGradient(IGpuBuffer m, IGpuBuffer dw, int k, int c)
        {
            var kernel = WinogradKernel(WinogradF23KernelTemplate.FilterGradient, m, dw);
            kernel.SetArg(2, k); kernel.SetArg(3, c);
            RunWinograd(kernel, (long)k * c);
        }
    }
}