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

        private bool WinogradF43KernelsReady => HasTunedKernel(Kernels.WinogradF43Kernels.Filter);

        private const int WinogradF43Points = 36;

        // y = conv3x3_s1(x) through F(4x4, 3x3): U[36][K][C] = G g G^T, V[36][C][P] = B^T d B, the 36 per-point GEMMs
        // M[x] (K x P) = U[x] (K x C) * V[x] (C x P), and Y = A^T m A. False, with nothing launched, when the shape is
        // not admitted or a plane would not fit 32-bit indexing.
        internal bool TryConv2DForwardWinogradF43(in TunedConv2DArgs a)
        {
            if (!WinogradF43KernelsReady || !WinogradF23Pipeline.Admits(a)) return false;
            int n = a.Batch, inC = a.InChannels, outC = a.OutChannels, outH = a.OutHeight, outW = a.OutWidth;
            int tilesH = (outH + 3) / 4, tilesW = (outW + 3) / 4;
            long p = (long)n * tilesH * tilesW;
            long uCount = WinogradF43Points * (long)outC * inC, vCount = WinogradF43Points * inC * p;
            long mCount = WinogradF43Points * outC * p;
            if (uCount > int.MaxValue || vCount > int.MaxValue || mCount > int.MaxValue) return false;
            int pi = (int)p;
            IGpuBuffer? owned = null;
            IGpuBuffer u = GetCachedF43Filter(outC, inC) ?? (owned = AllocateBuffer((int)uCount));
            using var v = AllocateBuffer((int)vCount);
            using var m = AllocateBuffer((int)mCount);
            try
            {
                if (owned is not null)
                {
                    var filter = WinogradKernel(Kernels.WinogradF43Kernels.Filter, a.Kernel, owned);
                    filter.SetArg(2, outC); filter.SetArg(3, inC);
                    RunWinograd(filter, (long)outC * inC);
                    if (StoreF43Filter(owned, outC, inC)) owned = null;
                }

                var input = WinogradKernel(Kernels.WinogradF43Kernels.Input, a.Input, v);
                uint arg = 2;
                input.SetArg(arg++, n); input.SetArg(arg++, inC); input.SetArg(arg++, a.InHeight); input.SetArg(arg++, a.InWidth);
                input.SetArg(arg++, a.PadH); input.SetArg(arg++, tilesH); input.SetArg(arg++, tilesW);
                RunWinograd(input, (long)inC * p);

                BatchedGemm(u, v, m, outC, pi, inC, WinogradF43Points);

                var output = WinogradKernel(Kernels.WinogradF43Kernels.Output, m, a.Output);
                arg = 2;
                output.SetArg(arg++, n); output.SetArg(arg++, outC); output.SetArg(arg++, outH); output.SetArg(arg++, outW);
                output.SetArg(arg++, tilesH); output.SetArg(arg++, tilesW);
                RunWinograd(output, (long)outC * p);
                return true;
            }
            finally
            {
                owned?.Dispose();
            }
        }

        // Transformed filters by weight identity (ConvWeightIdentity): inference reuses them, where every call
        // re-transformed the weights (a 512x512 filter writes 37.7 MB of U: 0.3 ms of a 1.9 ms convolution). Bounded,
        // since U is four times the weights; past the budget a convolution transforms into scratch as before.
        // An eighth of device memory, at most 768 MiB: the buffer pool takes a quarter, and this cache is not drained on
        // out-of-memory (an entry goes only with its weights).
        private long F43FilterCacheBudgetBytes => Math.Min(768L << 20, GlobalMemoryBytes / 8);
        private readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, F43FilterEntry> _f43Filters = new();
        private readonly object _f43FilterLock = new();
        private long _f43FilterBytes;
        // Buffers of entries whose weights were collected: freed by the next cache access, never on the finalizer thread.
        private readonly System.Collections.Concurrent.ConcurrentQueue<IGpuBuffer> _f43Retired = new();
        private readonly System.Collections.Generic.List<WeakReference<F43FilterEntry>> _f43Live = new();

        private sealed class F43FilterEntry
        {
            internal F43FilterEntry(OpenClBackend owner, IGpuBuffer u, int version, int outC, int inC, long bytes)
            {
                Owner = owner; U = u; Version = version; OutC = outC; InC = inC; Bytes = bytes;
            }

            internal OpenClBackend Owner { get; }
            internal IGpuBuffer U { get; }
            internal int Version { get; }
            internal int OutC { get; }
            internal int InC { get; }
            internal long Bytes { get; }

            // The key is a host weight array: when it is collected the entry goes, and its device buffer is retired.
            ~F43FilterEntry()
            {
                System.Threading.Interlocked.Add(ref Owner._f43FilterBytes, -Bytes);
                Owner._f43Retired.Enqueue(U);
            }
        }

        private IGpuBuffer? GetCachedF43Filter(int outC, int inC)
        {
            if (!ConvWeightIdentity.TryGet(out var key, out int version)) return null;
            lock (_f43FilterLock)
            {
                DrainRetiredF43Filters();
                if (_f43Filters.TryGetValue(key, out var entry) && entry.Version == version && entry.OutC == outC
                    && entry.InC == inC)
                    return entry.U;
            }
            return null;
        }

        private void DrainRetiredF43Filters()
        {
            while (_f43Retired.TryDequeue(out var retired)) retired.Dispose();
            _f43Live.RemoveAll(w => !w.TryGetTarget(out _));
        }

        // Frees every cached transform; called when the backend is disposed.
        private void ReleaseF43FilterCache()
        {
            lock (_f43FilterLock)
            {
                foreach (var weak in _f43Live)
                {
                    if (!weak.TryGetTarget(out var entry)) continue;
                    GC.SuppressFinalize(entry);
                    entry.U.Dispose();
                }
                _f43Live.Clear();
                while (_f43Retired.TryDequeue(out var retired)) retired.Dispose();
                _f43FilterBytes = 0;
            }
        }

        // True when the cache took ownership of u.
        private bool StoreF43Filter(IGpuBuffer u, int outC, int inC)
        {
            if (!ConvWeightIdentity.TryGet(out var key, out int version)) return false;
            long bytes = (long)WinogradF43Points * outC * inC * sizeof(float);
            lock (_f43FilterLock)
            {
                DrainRetiredF43Filters();
                if (_f43Filters.TryGetValue(key, out var stale))
                {
                    _f43Filters.Remove(key);
                    System.Threading.Interlocked.Add(ref _f43FilterBytes, -stale.Bytes);
                    GC.SuppressFinalize(stale);
                    stale.U.Dispose();
                }
                if (System.Threading.Interlocked.Read(ref _f43FilterBytes) + bytes > F43FilterCacheBudgetBytes) return false;
                var added = new F43FilterEntry(this, u, version, outC, inC, bytes);
                _f43Filters.Add(key, added);
                _f43Live.Add(new WeakReference<F43FilterEntry>(added));
                System.Threading.Interlocked.Add(ref _f43FilterBytes, bytes);
                return true;
            }
        }

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

        void IWinogradF23Transforms.FilterGradient(IGpuBuffer m, IGpuBuffer dw, int k, int c, int splits)
        {
            var kernel = WinogradKernel(WinogradF23KernelTemplate.FilterGradient, m, dw);
            kernel.SetArg(2, k); kernel.SetArg(3, c); kernel.SetArg(4, splits);
            RunWinograd(kernel, (long)k * c);
        }
    }
}