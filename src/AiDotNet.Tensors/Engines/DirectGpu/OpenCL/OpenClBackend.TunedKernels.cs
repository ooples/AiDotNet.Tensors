using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Tuned-kernel registry dispatch on OpenCL (#1114): softmax and its backward, LayerNorm forward / input gradient /
    // affine gradients and RMSNorm's gamma gradient each run through a TunedKernelSlot whose reference candidate is the
    // established kernel, with the generated OpenClTunedRowKernels variants as the alternatives. The slot measures and
    // picks per shape class (or takes a shipped / persisted profile entry), exactly as CudaBackend.TunedKernels.
    public sealed partial class OpenClBackend
    {
        private const double TunedRowTolerance = 1e-4;

        private readonly object _tunedSlotsLock = new object();
        private readonly object _tuneMeasureLock = new object();
        private TunedKernelSlot<TunedSoftmaxArgs>? _softmaxSlot;
        private TunedKernelSlot<TunedSoftmaxBackwardArgs>? _softmaxBackwardSlot;
        private TunedKernelSlot<TunedLayerNormArgs>? _layerNormSlot;
        private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormBackwardSlot;
        private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormGradParamsSlot;
        private TunedKernelSlot<TunedNormBackwardArgs>? _rmsNormGradGammaSlot;
        private string? _tunedDeviceKey;

        /// <summary>The registry device key, e.g. <c>opencl:Advanced Micro Devices, Inc.:gfx1012</c>.</summary>
        internal string TunedDeviceKey => _tunedDeviceKey ??= $"opencl:{DeviceVendor}:{DeviceName}";

        private bool HasTunedKernel(string name) => _kernelCache.ContainsKey(name);

        private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

        // The main queue has no profiling enabled (only a separate per-thread profiling queue does, and the candidates
        // enqueue on the main one), so a measurement drains the queue, times `repetitions` runs on the host and drains
        // again: device time plus launch overhead, the same for every candidate it compares.
        internal double MeasureTunedKernelMilliseconds(Action run, int repetitions)
        {
            lock (_tuneMeasureLock)
            {
                Synchronize();
                var watch = Stopwatch.StartNew();
                for (int i = 0; i < repetitions; i++) run();
                Synchronize();
                return watch.Elapsed.TotalMilliseconds;
            }
        }

        private BackendTunedKernelHarness<TArgs> Harness<TArgs>(Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs,
            Func<TArgs, bool> aliased) where TArgs : struct
            => new BackendTunedKernelHarness<TArgs>(this, TunedRowTolerance, () => IsAvailable, MeasureTunedKernelMilliseconds,
                outputs, aliased);

        private static int RowGlobalSize(int rows, int lanes)
            => checked((int)(((long)rows * lanes + OpenClTunedRowKernels.GroupSize - 1) / OpenClTunedRowKernels.GroupSize
                * OpenClTunedRowKernels.GroupSize));

        // ---------------------------------------------------------------- softmax

        internal TunedKernelSlot<TunedSoftmaxArgs> SoftmaxSlot
        {
            get
            {
                if (_softmaxSlot is { } s) return s;
                lock (_tunedSlotsLock)
                {
                    if (_softmaxSlot is null)
                    {
                        var candidates = new List<ITunedKernelCandidate<TunedSoftmaxArgs>>
                        {
                            new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("opencl.softmax.group", TunedKernelOrigin.Builtin, true,
                                _ => true, (in TunedSoftmaxArgs a) => SoftmaxReference(a.Input, a.Output, a.Rows, a.N)),
                        };
                        foreach (int lanes in OpenClTunedRowKernels.RowLanes)
                        {
                            int l = lanes;
                            string name = OpenClTunedRowKernels.SoftmaxName(l);
                            candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("generated.softmax.lanes" + l,
                                TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name),
                                (in TunedSoftmaxArgs a) => LaunchTunedRow2(name, l, a.Input, a.Output, a.Rows, a.N)));
                        }
                        _softmaxSlot = new TunedKernelSlot<TunedSoftmaxArgs>(TunedKernelOp.Softmax, TunedDeviceKey,
                            Harness<TunedSoftmaxArgs>(a => new[] { (a.Output, a.Rows * a.N) }, a => a.Input.Handle == a.Output.Handle),
                            () => GpuDeterminism.IsActive, candidates.ToArray());
                    }
                    return _softmaxSlot;
                }
            }
        }

        private void LaunchTunedRow2(string kernelName, int lanes, IGpuBuffer input, IGpuBuffer output, int rows, int n)
        {
            var k = _kernelCache[kernelName];
            k.SetArg(0, BufferHandle(input)); k.SetArg(1, BufferHandle(output)); k.SetArg(2, rows); k.SetArg(3, n);
            k.Execute1D(RowGlobalSize(rows, lanes), OpenClTunedRowKernels.GroupSize);
        }

        internal TunedKernelSlot<TunedSoftmaxBackwardArgs> SoftmaxBackwardSlot
        {
            get
            {
                if (_softmaxBackwardSlot is { } s) return s;
                lock (_tunedSlotsLock)
                {
                    if (_softmaxBackwardSlot is null)
                    {
                        var candidates = new List<ITunedKernelCandidate<TunedSoftmaxBackwardArgs>>
                        {
                            new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("opencl.softmax_backward.group",
                                TunedKernelOrigin.Builtin, true, _ => true,
                                (in TunedSoftmaxBackwardArgs a) => SoftmaxBackwardReference(a.GradOutput, a.Output, a.GradInput, a.Rows, a.N)),
                        };
                        foreach (int lanes in OpenClTunedRowKernels.RowLanes)
                        {
                            int l = lanes;
                            string name = OpenClTunedRowKernels.SoftmaxBackwardName(l);
                            candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("generated.softmax_backward.lanes" + l,
                                TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name),
                                (in TunedSoftmaxBackwardArgs a) =>
                                {
                                    var k = _kernelCache[name];
                                    k.SetArg(0, BufferHandle(a.GradOutput)); k.SetArg(1, BufferHandle(a.Output));
                                    k.SetArg(2, BufferHandle(a.GradInput)); k.SetArg(3, a.Rows); k.SetArg(4, a.N);
                                    k.Execute1D(RowGlobalSize(a.Rows, l), OpenClTunedRowKernels.GroupSize);
                                }));
                        }
                        _softmaxBackwardSlot = new TunedKernelSlot<TunedSoftmaxBackwardArgs>(TunedKernelOp.SoftmaxBackward, TunedDeviceKey,
                            Harness<TunedSoftmaxBackwardArgs>(a => new[] { (a.GradInput, a.Rows * a.N) },
                                a => a.GradInput.Handle == a.GradOutput.Handle || a.GradInput.Handle == a.Output.Handle),
                            () => GpuDeterminism.IsActive, candidates.ToArray());
                    }
                    return _softmaxBackwardSlot;
                }
            }
        }

        // ---------------------------------------------------------------- LayerNorm

        internal TunedKernelSlot<TunedLayerNormArgs> LayerNormSlot
        {
            get
            {
                if (_layerNormSlot is { } s) return s;
                lock (_tunedSlotsLock)
                {
                    if (_layerNormSlot is null)
                    {
                        var candidates = new List<ITunedKernelCandidate<TunedLayerNormArgs>>
                        {
                            new DelegateTunedKernelCandidate<TunedLayerNormArgs>("opencl.layernorm_forward.group",
                                TunedKernelOrigin.Builtin, true, _ => true, (in TunedLayerNormArgs a) => LayerNormReference(a)),
                        };
                        foreach (int lanes in OpenClTunedRowKernels.RowLanes)
                        {
                            int l = lanes;
                            string name = OpenClTunedRowKernels.LayerNormName(l);
                            candidates.Add(new DelegateTunedKernelCandidate<TunedLayerNormArgs>("generated.layernorm_forward.lanes" + l,
                                TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name),
                                (in TunedLayerNormArgs a) =>
                                {
                                    var k = _kernelCache[name];
                                    k.SetArg(0, BufferHandle(a.Input)); k.SetArg(1, BufferHandle(a.Output)); k.SetArg(2, BufferHandle(a.Gamma));
                                    k.SetArg(3, BufferHandle(a.Beta)); k.SetArg(4, BufferHandle(a.SaveMean)); k.SetArg(5, BufferHandle(a.SaveInvVar));
                                    k.SetArg(6, a.Rows); k.SetArg(7, a.N); k.SetArg(8, a.Epsilon);
                                    k.Execute1D(RowGlobalSize(a.Rows, l), OpenClTunedRowKernels.GroupSize);
                                }));
                        }
                        _layerNormSlot = new TunedKernelSlot<TunedLayerNormArgs>(TunedKernelOp.LayerNorm, TunedDeviceKey,
                            Harness<TunedLayerNormArgs>(
                                a => new[] { (a.Output, a.Rows * a.N), (a.SaveMean, a.Rows), (a.SaveInvVar, a.Rows) },
                                a => a.Output.Handle == a.Input.Handle),
                            () => GpuDeterminism.IsActive, candidates.ToArray());
                    }
                    return _layerNormSlot;
                }
            }
        }

        internal TunedKernelSlot<TunedNormBackwardArgs> LayerNormBackwardSlot
        {
            get
            {
                if (_layerNormBackwardSlot is { } s) return s;
                lock (_tunedSlotsLock)
                {
                    if (_layerNormBackwardSlot is null)
                    {
                        var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
                        {
                            new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("opencl.layernorm_backward.group",
                                TunedKernelOrigin.Builtin, true, _ => true, (in TunedNormBackwardArgs a) => LayerNormBackwardInputReference(a)),
                        };
                        foreach (int lanes in OpenClTunedRowKernels.RowLanes)
                        {
                            int l = lanes;
                            string name = OpenClTunedRowKernels.LayerNormBackwardName(l);
                            candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("generated.layernorm_backward.lanes" + l,
                                TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name),
                                (in TunedNormBackwardArgs a) =>
                                {
                                    var k = _kernelCache[name];
                                    k.SetArg(0, BufferHandle(a.GradOutput)); k.SetArg(1, BufferHandle(a.Input));
                                    k.SetArg(2, BufferHandle(Required(a.Gamma))); k.SetArg(3, BufferHandle(a.Stat0));
                                    k.SetArg(4, BufferHandle(Required(a.Stat1))); k.SetArg(5, BufferHandle(a.Out0));
                                    k.SetArg(6, a.Rows); k.SetArg(7, a.N);
                                    k.Execute1D(RowGlobalSize(a.Rows, l), OpenClTunedRowKernels.GroupSize);
                                }));
                        }
                        _layerNormBackwardSlot = new TunedKernelSlot<TunedNormBackwardArgs>(TunedKernelOp.LayerNormBackward, TunedDeviceKey,
                            Harness<TunedNormBackwardArgs>(a => new[] { (a.Out0, a.Rows * a.N) },
                                a => a.Out0.Handle == a.GradOutput.Handle || a.Out0.Handle == a.Input.Handle),
                            () => GpuDeterminism.IsActive, candidates.ToArray());
                    }
                    return _layerNormBackwardSlot;
                }
            }
        }

        internal TunedKernelSlot<TunedNormBackwardArgs> LayerNormGradParametersSlot
        {
            get
            {
                if (_layerNormGradParamsSlot is { } s) return s;
                lock (_tunedSlotsLock)
                    return _layerNormGradParamsSlot ??= CreateColumnGradSlot(TunedKernelOp.LayerNormGradParameters,
                        "opencl.layernorm_grad_params.thread_per_column", rms: false);
            }
        }

        internal TunedKernelSlot<TunedNormBackwardArgs> RmsNormGradGammaSlot
        {
            get
            {
                if (_rmsNormGradGammaSlot is { } s) return s;
                lock (_tunedSlotsLock)
                    return _rmsNormGradGammaSlot ??= CreateColumnGradSlot(TunedKernelOp.RmsNormGradGamma,
                        "opencl.rmsnorm_grad_gamma.thread_per_column", rms: true);
            }
        }

        private TunedKernelSlot<TunedNormBackwardArgs> CreateColumnGradSlot(TunedKernelOp op, string referenceId, bool rms)
        {
            var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
            {
                new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(referenceId, TunedKernelOrigin.Builtin, true, _ => true,
                    (in TunedNormBackwardArgs a) => ColumnGradReference(rms, a)),
            };
            foreach (int rowLanes in OpenClTunedRowKernels.ColumnRowLanes)
            {
                int r = rowLanes;
                string name = rms ? OpenClTunedRowKernels.RmsNormGradGammaName(r) : OpenClTunedRowKernels.LayerNormGradParamsName(r);
                candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(
                    (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + r,
                    TunedKernelOrigin.Generated, true, _ => HasTunedKernel(name),
                    (in TunedNormBackwardArgs a) =>
                    {
                        var k = _kernelCache[name];
                        uint arg = 0;
                        k.SetArg(arg++, BufferHandle(a.GradOutput)); k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(a.Stat0));
                        if (!rms) k.SetArg(arg++, BufferHandle(Required(a.Stat1)));
                        k.SetArg(arg++, BufferHandle(a.Out0));
                        if (!rms) k.SetArg(arg++, BufferHandle(Required(a.Out1)));
                        k.SetArg(arg++, a.Rows); k.SetArg(arg++, a.N);
                        k.Execute2D(((a.N + 31) / 32) * 32, r, 32, r);
                    }));
            }
            return new TunedKernelSlot<TunedNormBackwardArgs>(op, TunedDeviceKey,
                Harness<TunedNormBackwardArgs>(
                    a => rms ? new[] { (a.Out0, a.N) } : new[] { (a.Out0, a.N), (Required(a.Out1), a.N) },
                    a => a.Out0.Handle == a.GradOutput.Handle || a.Out0.Handle == a.Input.Handle ||
                         (a.Out1 is { } o1 && (o1.Handle == a.GradOutput.Handle || o1.Handle == a.Input.Handle))),
                () => GpuDeterminism.IsActive, candidates.ToArray());
        }

        private static IGpuBuffer Required(IGpuBuffer? buffer)
            => buffer ?? throw new InvalidOperationException("A required tuned-kernel buffer argument is missing.");

        private static IntPtr BufferHandle(IGpuBuffer buffer) => ((DirectOpenClGpuBuffer)buffer).Buffer.Handle;

        // ---------------------------------------------------------------- reference launches (the established kernels)

        private static int PowerOfTwoLocal(int n)
        {
            int local = Math.Min(256, n);
            local = (int)Math.Pow(2, Math.Floor(Math.Log(Math.Max(1, local), 2)));
            return Math.Max(1, local);
        }

        private void SoftmaxReference(IGpuBuffer input, IGpuBuffer output, int rows, int n)
        {
            var kernel = _kernelCache["softmax"];
            kernel.SetArg(0, BufferHandle(input)); kernel.SetArg(1, BufferHandle(output)); kernel.SetArg(2, rows); kernel.SetArg(3, n);
            int local = PowerOfTwoLocal(n);
            kernel.SetLocalArg(4, local * sizeof(float));
            kernel.Execute1D(rows * local, local);
        }

        private void SoftmaxBackwardReference(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
        {
            var k = _kernelCache["softmax_backward"];
            k.SetArg(0, BufferHandle(gradOutput)); k.SetArg(1, BufferHandle(output)); k.SetArg(2, BufferHandle(gradInput));
            k.SetArg(3, rows); k.SetArg(4, n);
            int local = PowerOfTwoLocal(n);
            k.SetLocalArg(5, local * sizeof(float));
            k.Execute1D(rows * local, local);
        }

        private void LayerNormReference(in TunedLayerNormArgs a)
        {
            var k = _kernelCache["layernorm_forward"];
            k.SetArg(0, BufferHandle(a.Input)); k.SetArg(1, BufferHandle(a.Output)); k.SetArg(2, BufferHandle(a.Gamma));
            k.SetArg(3, BufferHandle(a.Beta)); k.SetArg(4, BufferHandle(a.SaveMean)); k.SetArg(5, BufferHandle(a.SaveInvVar));
            k.SetArg(6, a.Rows); k.SetArg(7, a.N); k.SetArg(8, a.Epsilon);
            int local = PowerOfTwoLocal(a.N);
            k.SetLocalArg(9, local * sizeof(float));
            k.Execute1D(a.Rows * local, local);
        }

        // layernorm_backward takes gradGamma / gradBeta arguments but writes only the input gradient (the affine
        // gradients are layernorm_grad_params' job), so the input-gradient reference binds Out0 to those slots too.
        private void LayerNormBackwardInputReference(in TunedNormBackwardArgs a)
        {
            var k = _kernelCache["layernorm_backward"];
            k.SetArg(0, BufferHandle(a.GradOutput)); k.SetArg(1, BufferHandle(a.Input)); k.SetArg(2, BufferHandle(Required(a.Gamma)));
            k.SetArg(3, BufferHandle(a.Stat0)); k.SetArg(4, BufferHandle(Required(a.Stat1))); k.SetArg(5, BufferHandle(a.Out0));
            k.SetArg(6, BufferHandle(a.Out0)); k.SetArg(7, BufferHandle(a.Out0));
            k.SetArg(8, a.Rows); k.SetArg(9, a.N); k.SetArg(10, 0f);
            int local = PowerOfTwoLocal(a.N);
            k.SetLocalArg(11, local * sizeof(float));
            k.SetLocalArg(12, local * sizeof(float));
            k.Execute1D(a.Rows * local, local);
        }

        private void ColumnGradReference(bool rms, in TunedNormBackwardArgs a)
        {
            var k = _kernelCache[rms ? "rmsnorm_grad_gamma" : "layernorm_grad_params"];
            uint arg = 0;
            k.SetArg(arg++, BufferHandle(a.GradOutput)); k.SetArg(arg++, BufferHandle(a.Input)); k.SetArg(arg++, BufferHandle(a.Stat0));
            if (!rms) k.SetArg(arg++, BufferHandle(Required(a.Stat1)));
            k.SetArg(arg++, BufferHandle(a.Out0));
            if (!rms) k.SetArg(arg++, BufferHandle(Required(a.Out1)));
            k.SetArg(arg++, a.Rows); k.SetArg(arg++, a.N);
            k.Execute1D(a.N, Math.Min(rms ? 256 : 64, a.N));
        }
    }
}