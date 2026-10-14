// Copyright (c) AiDotNet. All rights reserved.
// #1113: HIP serves softmax, softmax backward, LayerNorm forward / input gradient / affine gradients and the RMSNorm
// gamma gradient through the tuned-kernel registry, with its established kernels as each slot's reference.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    private const double TunedRowTolerance = 1e-4;

    private IntPtr _tunedRowModule;
    private readonly object _tunedSlotsLock = new object();
    private readonly object _tuneMeasureLock = new object();
    private TunedKernelSlot<TunedSoftmaxArgs>? _softmaxSlot;
    private TunedKernelSlot<TunedSoftmaxBackwardArgs>? _softmaxBackwardSlot;
    private TunedKernelSlot<TunedLayerNormArgs>? _layerNormSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormBackwardSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormGradParamsSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _rmsNormGradGammaSlot;
    private string? _tunedDeviceKey;

    /// <summary>The device a profile entry names: <c>hip:&lt;gfx target&gt;:&lt;device name&gt;</c>.</summary>
    internal string TunedDeviceKey => _tunedDeviceKey ??= $"hip:{_architectureTarget}:{DeviceName}";

    private void CompileTunedRowKernels()
    {
        // A variant that fails to compile is simply not offered (HasTunedKernel gates every generated candidate);
        // the established kernels keep serving every shape.
        CompileKernelModule(HipTunedRowKernels.GetSource(), "tuned_rows", ref _tunedRowModule, HipTunedRowKernels.GetKernelNames());
    }

    private bool HasTunedKernel(string name) => _kernelCache.ContainsKey(name);

    private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

    private static uint RowGrid(int rows, int lanes) => (uint)(((long)rows * lanes + 255) / 256);

    /// <summary>
    /// Device time of <paramref name="repetitions"/> runs, between two events on the backend's stream (the issue's
    /// "device events" harness): host launch overhead outside the stream does not count.
    /// </summary>
    internal double MeasureTunedKernelMilliseconds(Action run, int repetitions)
    {
        lock (_tuneMeasureLock)
        {
            IntPtr start = IntPtr.Zero, stop = IntPtr.Zero;
            try
            {
                HipNativeBindings.CheckError(HipNativeBindings.hipEventCreate(ref start), "hipEventCreate");
                HipNativeBindings.CheckError(HipNativeBindings.hipEventCreate(ref stop), "hipEventCreate");
                Synchronize();
                HipNativeBindings.CheckError(HipNativeBindings.hipEventRecord(start, _stream), "hipEventRecord");
                for (int i = 0; i < repetitions; i++) run();
                HipNativeBindings.CheckError(HipNativeBindings.hipEventRecord(stop, _stream), "hipEventRecord");
                HipNativeBindings.CheckError(HipNativeBindings.hipEventSynchronize(stop), "hipEventSynchronize");
                float ms = 0f;
                HipNativeBindings.CheckError(HipNativeBindings.hipEventElapsedTime(ref ms, start, stop), "hipEventElapsedTime");
                return ms;
            }
            finally
            {
                if (start != IntPtr.Zero) HipNativeBindings.hipEventDestroy(start);
                if (stop != IntPtr.Zero) HipNativeBindings.hipEventDestroy(stop);
            }
        }
    }

    private BackendTunedKernelHarness<TArgs> Harness<TArgs>(Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs,
        Func<TArgs, bool> aliased) where TArgs : struct
        => new BackendTunedKernelHarness<TArgs>(this, TunedRowTolerance, () => IsAvailable, MeasureTunedKernelMilliseconds,
            outputs, aliased);

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
                        new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("hip.softmax.block", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedSoftmaxArgs a) => SoftmaxReference(a.Input, a.Output, a.Rows, a.N)),
                    };
                    foreach (int lanes in HipTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = HipTunedRowKernels.SoftmaxName(l);
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
                        new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("hip.softmax_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in TunedSoftmaxBackwardArgs a) => SoftmaxBackwardReference(a.GradOutput, a.Output, a.GradInput, a.Rows, a.N)),
                    };
                    foreach (int lanes in HipTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = HipTunedRowKernels.SoftmaxBackwardName(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("generated.softmax_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name),
                            (in TunedSoftmaxBackwardArgs a) => LaunchTunedSoftmaxBackward(name, l, a)));
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
                        new DelegateTunedKernelCandidate<TunedLayerNormArgs>("hip.layernorm_forward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedLayerNormArgs a) => LayerNormReference(a)),
                    };
                    foreach (int lanes in HipTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = HipTunedRowKernels.LayerNormName(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedLayerNormArgs>("generated.layernorm_forward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name),
                            (in TunedLayerNormArgs a) => LaunchTunedLayerNorm(name, l, a)));
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
                        new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("hip.layernorm_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedNormBackwardArgs a) => LayerNormBackwardInputReference(a)),
                    };
                    foreach (int lanes in HipTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string name = HipTunedRowKernels.LayerNormBackwardName(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("generated.layernorm_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name),
                            (in TunedNormBackwardArgs a) => LaunchTunedLayerNormBackward(name, l, a)));
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
                    "hip.layernorm_grad_params.thread_per_column", rms: false);
        }
    }

    internal TunedKernelSlot<TunedNormBackwardArgs> RmsNormGradGammaSlot
    {
        get
        {
            if (_rmsNormGradGammaSlot is { } s) return s;
            lock (_tunedSlotsLock)
                return _rmsNormGradGammaSlot ??= CreateColumnGradSlot(TunedKernelOp.RmsNormGradGamma,
                    "hip.rmsnorm_grad_gamma.thread_per_column", rms: true);
        }
    }

    private TunedKernelSlot<TunedNormBackwardArgs> CreateColumnGradSlot(TunedKernelOp op, string referenceId, bool rms)
    {
        var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
        {
            new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(referenceId, TunedKernelOrigin.Builtin, true, _ => true,
                (in TunedNormBackwardArgs a) => ColumnGradReference(rms, a)),
        };
        foreach (int rowLanes in HipTunedRowKernels.ColumnRowLanes)
        {
            int r = rowLanes;
            string name = rms ? HipTunedRowKernels.RmsNormGradGammaName(r) : HipTunedRowKernels.LayerNormGradParamsName(r);
            candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(
                (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + r,
                TunedKernelOrigin.Generated, true, _ => HasTunedKernel(name),
                (in TunedNormBackwardArgs a) => LaunchTunedColumnGrad(name, r, rms, a)));
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

    private IntPtr TunedKernel(string name)
        => _kernelCache.TryGetValue(name, out var k) ? k : throw new GpuCapabilityUnavailableException($"HIP kernel not found: {name}");

    // ---------------------------------------------------------------- generated variants (L lanes per row, R row lanes per column tile)

    private unsafe void LaunchTunedRow2(string kernelName, int lanes, IGpuBuffer input, IGpuBuffer output, int rows, int n)
    {
        IntPtr kernel = TunedKernel(kernelName);
        IntPtr x = input.Handle, y = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &x; args[1] = &y; args[2] = &rows; args[3] = &n;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    private unsafe void LaunchTunedSoftmaxBackward(string kernelName, int lanes, in TunedSoftmaxBackwardArgs a)
    {
        IntPtr kernel = TunedKernel(kernelName);
        IntPtr g = a.GradOutput.Handle, y = a.Output.Handle, dx = a.GradInput.Handle;
        int rows = a.Rows, n = a.N;
        void** args = stackalloc void*[5];
        args[0] = &g; args[1] = &y; args[2] = &dx; args[3] = &rows; args[4] = &n;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    private unsafe void LaunchTunedLayerNorm(string kernelName, int lanes, in TunedLayerNormArgs a)
    {
        IntPtr kernel = TunedKernel(kernelName);
        IntPtr x = a.Input.Handle, y = a.Output.Handle, gm = a.Gamma.Handle, bt = a.Beta.Handle;
        IntPtr mean = a.SaveMean.Handle, inv = a.SaveInvVar.Handle;
        int rows = a.Rows, n = a.N;
        float eps = a.Epsilon;
        void** args = stackalloc void*[9];
        args[0] = &x; args[1] = &y; args[2] = &gm; args[3] = &bt; args[4] = &mean; args[5] = &inv;
        args[6] = &rows; args[7] = &n; args[8] = &eps;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    private unsafe void LaunchTunedLayerNormBackward(string kernelName, int lanes, in TunedNormBackwardArgs a)
    {
        IntPtr kernel = TunedKernel(kernelName);
        IntPtr g = a.GradOutput.Handle, x = a.Input.Handle, gm = Required(a.Gamma).Handle;
        IntPtr mean = a.Stat0.Handle, inv = Required(a.Stat1).Handle, dx = a.Out0.Handle;
        int rows = a.Rows, n = a.N;
        void** args = stackalloc void*[8];
        args[0] = &g; args[1] = &x; args[2] = &gm; args[3] = &mean; args[4] = &inv; args[5] = &dx;
        args[6] = &rows; args[7] = &n;
        LaunchKernel(kernel, RowGrid(rows, lanes), 256, args);
    }

    private unsafe void LaunchTunedColumnGrad(string kernelName, int rowLanes, bool rms, in TunedNormBackwardArgs a)
    {
        IntPtr kernel = TunedKernel(kernelName);
        IntPtr g = a.GradOutput.Handle, x = a.Input.Handle, s0 = a.Stat0.Handle, o0 = a.Out0.Handle;
        int rows = a.Rows, n = a.N;
        uint grid = (uint)((n + 31) / 32);
        if (rms)
        {
            void** args = stackalloc void*[6];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &o0; args[4] = &rows; args[5] = &n;
            LaunchKernel3D(kernel, grid, 1, 1, 32, (uint)rowLanes, 1, args);
        }
        else
        {
            IntPtr s1 = Required(a.Stat1).Handle, o1 = Required(a.Out1).Handle;
            void** args = stackalloc void*[8];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &s1; args[4] = &o0; args[5] = &o1;
            args[6] = &rows; args[7] = &n;
            LaunchKernel3D(kernel, grid, 1, 1, 32, (uint)rowLanes, 1, args);
        }
    }

    // ---------------------------------------------------------------- references (HIP's established kernels, as the ops launched them)

    private unsafe void SoftmaxReference(IGpuBuffer input, IGpuBuffer output, int rows, int n)
    {
        IntPtr kernel = TunedKernel("softmax");
        // 1 block per row, 256 threads cooperating on the reduction; shared memory holds one partial per wavefront.
        uint sharedBytes = (DefaultBlockSize / 64 + 1) * sizeof(float);
        IntPtr x = input.Handle, y = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &x; args[1] = &y; args[2] = &rows; args[3] = &n;
        LaunchKernel(kernel, (uint)rows, DefaultBlockSize, args, sharedBytes);
    }

    private unsafe void SoftmaxBackwardReference(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
    {
        IntPtr kernel = TunedKernel("softmax_backward");
        IntPtr g = gradOutput.Handle, y = output.Handle, dx = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &g; args[1] = &y; args[2] = &dx; args[3] = &rows; args[4] = &n;
        LaunchKernel(kernel, (uint)((rows + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    private unsafe void LayerNormReference(in TunedLayerNormArgs a)
    {
        IntPtr kernel = TunedKernel("layernorm_forward");
        IntPtr x = a.Input.Handle, y = a.Output.Handle, gm = a.Gamma.Handle, bt = a.Beta.Handle;
        IntPtr mean = a.SaveMean.Handle, inv = a.SaveInvVar.Handle;
        int rows = a.Rows, n = a.N;
        float eps = a.Epsilon;
        void** args = stackalloc void*[9];
        args[0] = &x; args[1] = &y; args[2] = &gm; args[3] = &bt; args[4] = &mean; args[5] = &inv;
        args[6] = &rows; args[7] = &n; args[8] = &eps;
        LaunchKernel(kernel, (uint)((rows + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    // layernorm_backward takes gradGamma / gradBeta arguments but writes only the input gradient (the affine gradients
    // are layernorm_grad_params' job), so the input-gradient reference binds Out0 to those slots too.
    private unsafe void LayerNormBackwardInputReference(in TunedNormBackwardArgs a)
    {
        IntPtr kernel = TunedKernel("layernorm_backward");
        IntPtr g = a.GradOutput.Handle, x = a.Input.Handle, gm = Required(a.Gamma).Handle;
        IntPtr mean = a.Stat0.Handle, inv = Required(a.Stat1).Handle, dx = a.Out0.Handle;
        IntPtr unusedGamma = dx, unusedBeta = dx;
        int rows = a.Rows, n = a.N;
        float eps = 0f;
        void** args = stackalloc void*[11];
        args[0] = &g; args[1] = &x; args[2] = &gm; args[3] = &mean; args[4] = &inv; args[5] = &dx;
        args[6] = &unusedGamma; args[7] = &unusedBeta; args[8] = &rows; args[9] = &n; args[10] = &eps;
        LaunchKernel(kernel, (uint)((rows + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    private unsafe void ColumnGradReference(bool rms, in TunedNormBackwardArgs a)
    {
        IntPtr kernel = TunedKernel(rms ? "rmsnorm_grad_gamma" : "layernorm_grad_params");
        IntPtr g = a.GradOutput.Handle, x = a.Input.Handle, s0 = a.Stat0.Handle, o0 = a.Out0.Handle;
        int rows = a.Rows, n = a.N;
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        if (rms)
        {
            void** args = stackalloc void*[6];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &o0; args[4] = &rows; args[5] = &n;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
        else
        {
            IntPtr s1 = Required(a.Stat1).Handle, o1 = Required(a.Out1).Handle;
            void** args = stackalloc void*[8];
            args[0] = &g; args[1] = &x; args[2] = &s0; args[3] = &s1; args[4] = &o0; args[5] = &o1;
            args[6] = &rows; args[7] = &n;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
    }
}