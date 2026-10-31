using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1370: Winograd F(2x2, 3x3) forward and input gradient and F(3x3, 2x2) weight gradient on OpenCL, through the
    // shared WinogradF23Pipeline: the transforms below (emitted from WinogradF23KernelTemplate) around BatchedGemm.
    // Backward convolution gets a tuned slot here for the first time, with the direct kernels it always ran as the
    // reference, so Winograd is served only on the shape classes where it measures faster.
    public sealed partial class OpenClBackend : IWinogradF23Transforms, IWinogradFusedFilterGradient, IWinogradForwardFilterCache
    {
        private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardDataSlot, _conv2DBackwardFilterSlot;

        private bool WinogradKernelsReady => HasTunedKernel(WinogradF23KernelTemplate.Filter);

        // 0 not yet asked, 1 ready, 2 declined.
        private int _fusedFilterGradientReady;

        /// <summary>
        /// Whether the fused weight gradient can run here. Its own module has to have built, and both accumulation
        /// kernels have to accept the 256-wide group the launch always asks for: OpenCL reports
        /// CL_KERNEL_WORK_GROUP_SIZE per kernel, and a kernel below 256 would fail the launch with
        /// CL_INVALID_WORK_GROUP_SIZE rather than fall through to the direct route.
        /// </summary>
        private bool FusedFilterGradientReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _fusedFilterGradientReady);
                if (cached != 0) return cached == 1;
                bool ready = FusedGroupFits(WinogradF23KernelTemplate.FusedFilterGradient)
                    && FusedGroupFits(WinogradF23KernelTemplate.FusedFilterGradientDirect)
                    && HasTunedKernel(WinogradF23KernelTemplate.FusedFilterGradientFinish);
                if (!ready)
                    System.Diagnostics.Trace.TraceWarning(
                        "OpenCL fused Winograd weight gradient unavailable: a kernel is missing or cannot run "
                        + $"{WinogradF23KernelTemplate.FusedGroupSize} work-items.");
                System.Threading.Volatile.Write(ref _fusedFilterGradientReady, ready ? 1 : 2);
                return ready;
            }
        }

        // A zero means the driver did not report a per-kernel limit; the device limit is then all there is to go on,
        // as in ClampLocalSizeForKernel.
        private bool FusedGroupFits(string name)
        {
            if (!HasTunedKernel(name)) return false;
            int limit = KernelWorkGroupSize(name);
            return limit == 0 || limit >= WinogradF23KernelTemplate.FusedGroupSize;
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
                            // The column GEMM first, as the reference wherever it applies: the direct kernel took up to 160 ms
                            // a call on ResNet-50 shapes and was timed in every tuning pair (see Conv2DForwardSlot).
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.column_gemm", TunedKernelOrigin.Builtin,
                                true, AdmitsColumnGemm, (in TunedConv2DBackwardArgs a) => Conv2DBackwardInputColumnGemm(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                                true, _ => HasTunedKernel("conv2d_backward_input"), (in TunedConv2DBackwardArgs a) => LaunchDxDirect(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.winograd_f23", TunedKernelOrigin.Builtin,
                                true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                                (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinograd(a)) LaunchDxDirect(a); }),
                            // The stride-2 input gradients of ConvDgradS2KernelTemplate (#1476 ported): the 3 x 3 four-phase kernel and
                            // the 1 x 1 kernel written straight into dX with its zero phases.
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.s2k3", TunedKernelOrigin.Builtin,
                                true, shape => PortableConvDgradS2.K3Enabled && PortableConvDgradS2.AdmitsK3Shape(shape) && DgradS2Ready,
                                (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardInputS2K3(a)) LaunchDxDirect(a); }),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.s2k1", TunedKernelOrigin.Builtin,
                                true, shape => PortableConvDgradS2.K1Enabled && PortableConvDgradS2.AdmitsK1Shape(shape) && DgradS2Ready,
                                (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardInputS2K1(a)) LaunchDxDirect(a); }),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dx.pointwise_gemm", TunedKernelOrigin.Builtin,
                                true, AdmitsPointwiseGemm, (in TunedConv2DBackwardArgs a) => Conv2DBackwardInputPointwiseGemm(a)),
                        });
                    _conv2DBackwardDataSlot.UntunedPreference ??= static _ => s_conv2DBackwardDataUntunedPreference;
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
                            // The column GEMM first, as the reference wherever it applies: the direct kernel took up to 160 ms
                            // a call on ResNet-50 shapes and was timed in every tuning pair (see Conv2DForwardSlot).
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dw.column_gemm", TunedKernelOrigin.Builtin,
                                true, AdmitsColumnGemm, (in TunedConv2DBackwardArgs a) => Conv2DBackwardFilterColumnGemm(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                                true, _ => HasTunedKernel("conv2d_backward_weights"), (in TunedConv2DBackwardArgs a) => LaunchDwDirect(a)),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dw.winograd_f32", TunedKernelOrigin.Builtin,
                                true, shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                                (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinograd(a)) LaunchDwDirect(a); }),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dw.winograd_fused", TunedKernelOrigin.Builtin,
                                true, shape => WinogradF23Pipeline.AdmitsShape(shape) && FusedFilterGradientReady,
                                (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradFused(a)) LaunchDwDirect(a); }),
                            new DelegateTunedKernelCandidate<TunedConv2DBackwardArgs>("opencl.conv2d_dw.pointwise_gemm", TunedKernelOrigin.Builtin,
                                true, AdmitsPointwiseGemm, (in TunedConv2DBackwardArgs a) => Conv2DBackwardFilterPointwiseGemm(a)),
                        });
                    _conv2DBackwardFilterSlot.UntunedPreference ??= static _ => s_conv2DBackwardFilterUntunedPreference;
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
            IGpuBuffer u = GetCachedWinogradFilter(WinogradF43Points, outC, inC) ?? (owned = AllocateBuffer((int)uCount));
            using var v = AllocateBuffer((int)vCount);
            using var m = AllocateBuffer((int)mCount);
            try
            {
                if (owned is not null)
                {
                    var filter = WinogradKernel(Kernels.WinogradF43Kernels.Filter, a.Kernel, owned);
                    filter.SetArg(2, outC); filter.SetArg(3, inC);
                    RunWinograd(filter, (long)outC * inC);
                    if (StoreWinogradFilter(owned, WinogradF43Points, outC, inC)) owned = null;
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
                output.SetArg(arg++, ConvEpilogueBias(a)); output.SetArg(arg++, ConvEpilogueFlags(a));
                RunWinograd(output, (long)outC * p);
                return true;
            }
            finally
            {
                owned?.Dispose();
            }
        }

        // Transformed filters by weight identity (ConvWeightIdentity), F(4x4, 3x3) and F(2x2, 3x3) forward each in a table of
        // its own: inference reuses them, where every call re-transformed the weights (a 512x512 filter writes 37.7 MB of
        // U: 0.3 ms of a 1.9 ms convolution). Bounded, since U is up to four times the weights; past the budget a
        // convolution transforms into scratch as before.
        // An eighth of device memory, at most 768 MiB: the buffer pool takes half (PooledMemoryDivisor), and this cache
        // is not drained on out-of-memory (an entry goes only with its weights). One cache serves F(2x2) and F(4x4).
        private const long WinogradFilterCacheMaxBytes = 768L << 20;
        private const int WinogradFilterCacheMemoryDivisor = 8;
        private long WinogradFilterCacheBudgetBytes =>
            Math.Min(WinogradFilterCacheMaxBytes, GlobalMemoryBytes / WinogradFilterCacheMemoryDivisor);
        private readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, WinogradFilterEntry> _f43Filters = new();
        private readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, WinogradFilterEntry> _f23Filters = new();

        // The small-map GEMM's gathered weights ((C x H x W) x (O x H x W), keyed as one "point"), in the same budget.
        private readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, WinogradFilterEntry> _smallMapWeights = new();

        // The point count the small-map GEMM files its gathered weights under (outC = its columns, inC = its rows).
        internal const int SmallMapWeightPoints = 1;

        private System.Runtime.CompilerServices.ConditionalWeakTable<object, WinogradFilterEntry> WinogradFilterTable(int points)
            => points switch
            {
                WinogradF43Points => _f43Filters,
                SmallMapWeightPoints => _smallMapWeights,
                _ => _f23Filters,
            };

        IGpuBuffer? IWinogradForwardFilterCache.CachedForwardFilter(int points, int k, int c)
            => GetCachedWinogradFilter(points, k, c);

        bool IWinogradForwardFilterCache.StoreForwardFilter(IGpuBuffer u, int points, int k, int c)
            => StoreWinogradFilter(u, points, k, c);
        private readonly object _winogradFilterLock = new();
        private long _winogradFilterBytes;
        // Buffers of entries whose weights were collected: freed by the next cache access, never on the finalizer thread.
        private readonly System.Collections.Concurrent.ConcurrentQueue<IGpuBuffer> _winogradFilterRetired = new();
        private readonly System.Collections.Generic.List<WeakReference<WinogradFilterEntry>> _winogradFilterLive = new();

        private sealed class WinogradFilterEntry
        {
            internal WinogradFilterEntry(OpenClBackend owner, IGpuBuffer u, int version, int outC, int inC, long bytes)
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
            ~WinogradFilterEntry()
            {
                System.Threading.Interlocked.Add(ref Owner._winogradFilterBytes, -Bytes);
                Owner._winogradFilterRetired.Enqueue(U);
            }
        }

        private IGpuBuffer? GetCachedWinogradFilter(int points, int outC, int inC)
        {
            if (!ConvWeightIdentity.TryGet(out var key, out int version)) return null;
            lock (_winogradFilterLock)
            {
                DrainRetiredWinogradFilters();
                if (WinogradFilterTable(points).TryGetValue(key, out var entry) && entry.Version == version && entry.OutC == outC
                    && entry.InC == inC)
                    return entry.U;
            }
            return null;
        }

        private void DrainRetiredWinogradFilters()
        {
            while (_winogradFilterRetired.TryDequeue(out var retired)) retired.Dispose();
            _winogradFilterLive.RemoveAll(w => !w.TryGetTarget(out _));
        }

        // Frees every cached transform; called when the backend is disposed.
        private void ReleaseWinogradFilterCache()
        {
            lock (_winogradFilterLock)
            {
                foreach (var weak in _winogradFilterLive)
                {
                    if (!weak.TryGetTarget(out var entry)) continue;
                    GC.SuppressFinalize(entry);
                    entry.U.Dispose();
                }
                _winogradFilterLive.Clear();
                while (_winogradFilterRetired.TryDequeue(out var retired)) retired.Dispose();
                _winogradFilterBytes = 0;
            }
        }

        // True when the cache took ownership of u.
        private bool StoreWinogradFilter(IGpuBuffer u, int points, int outC, int inC)
        {
            if (!ConvWeightIdentity.TryGet(out var key, out int version)) return false;
            long bytes = (long)points * outC * inC * sizeof(float);
            var table = WinogradFilterTable(points);
            lock (_winogradFilterLock)
            {
                DrainRetiredWinogradFilters();
                if (table.TryGetValue(key, out var stale))
                {
                    table.Remove(key);
                    System.Threading.Interlocked.Add(ref _winogradFilterBytes, -stale.Bytes);
                    GC.SuppressFinalize(stale);
                    stale.U.Dispose();
                }
                if (System.Threading.Interlocked.Read(ref _winogradFilterBytes) + bytes > WinogradFilterCacheBudgetBytes) return false;
                var added = new WinogradFilterEntry(this, u, version, outC, inC, bytes);
                table.Add(key, added);
                _winogradFilterLive.Add(new WeakReference<WinogradFilterEntry>(added));
                System.Threading.Interlocked.Add(ref _winogradFilterBytes, bytes);
                return true;
            }
        }

        internal bool TryConv2DBackwardDataWinograd(in TunedConv2DBackwardArgs a)
            => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardData(this, this, a);

        internal bool TryConv2DBackwardFilterWinograd(in TunedConv2DBackwardArgs a)
            => WinogradKernelsReady && WinogradF23Pipeline.TryBackwardFilter(this, this, a);

        // The fused weight gradient: one launch of 256-wide groups (plus the finish when the tiles split).
        internal bool TryConv2DBackwardFilterWinogradFused(in TunedConv2DBackwardArgs a, int splits = 0)
            => FusedFilterGradientReady && WinogradF23Pipeline.TryBackwardFilterFused(this, this, a, splits);

        int IWinogradFusedFilterGradient.FusedFilterGradientGroupTarget => ComputeUnits;

        void IWinogradFusedFilterGradient.FusedFilterGradient(bool direct, IGpuBuffer x, IGpuBuffer res, IGpuBuffer dy,
            int groups, int[] ints)
        {
            var fused = WinogradKernel(direct ? WinogradF23KernelTemplate.FusedFilterGradientDirect
                : WinogradF23KernelTemplate.FusedFilterGradient, x, res);
            fused.SetArg(2, BufferHandle(dy));
            for (int i = 0; i < ints.Length; i++) fused.SetArg((uint)(3 + i), ints[i]);
            fused.Execute1D(groups * WinogradF23KernelTemplate.FusedGroupSize, WinogradF23KernelTemplate.FusedGroupSize);
        }

        void IWinogradFusedFilterGradient.FusedFilterGradientFinish(IGpuBuffer partials, IGpuBuffer dw, int kc, int splits)
        {
            var finish = WinogradKernel(WinogradF23KernelTemplate.FusedFilterGradientFinish, partials, dw);
            finish.SetArg(2, kc); finish.SetArg(3, splits);
            RunWinograd(finish, 9L * kc);
        }

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

        void IWinogradF23Transforms.Output(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH, int tilesW,
            IGpuBuffer? bias, int epilogue)
        {
            var kernel = WinogradKernel(WinogradF23KernelTemplate.Output, m, y);
            uint arg = 2;
            kernel.SetArg(arg++, BufferHandle(bias ?? y));
            kernel.SetArg(arg++, n); kernel.SetArg(arg++, k); kernel.SetArg(arg++, outH); kernel.SetArg(arg++, outW);
            kernel.SetArg(arg++, tilesH); kernel.SetArg(arg++, tilesW); kernel.SetArg(arg++, epilogue);
            RunWinograd(kernel, (long)k * n * tilesH * tilesW);
        }

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