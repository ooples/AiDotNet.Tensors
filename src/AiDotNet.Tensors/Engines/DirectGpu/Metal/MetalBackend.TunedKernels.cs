// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Tuned-kernel registry dispatch on Metal (#1116).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Softmax and its backward, LayerNorm forward / input gradient / affine gradients and RMSNorm's gamma gradient each run
// through a TunedKernelSlot whose reference candidate is Metal's established kernel (or, for the backward halves the
// established kernel computes together on one thread, its per-row / per-column split), with the TunedRowKernelTemplate
// variants as the alternatives. The slot measures and picks per shape class, or takes a shipped / persisted profile
// entry, exactly as on CUDA, OpenCL and Vulkan.
public sealed partial class MetalBackend
{
    private const string TunedRowLibraryName = "TunedRow";
    private const double TunedRowTolerance = 1e-4;

    private IntPtr _tunedRowLibrary;
    private readonly object _tunedSlotsLock = new object();
    private readonly object _tuneMeasureLock = new object();
    private TunedKernelSlot<TunedSoftmaxArgs>? _softmaxSlot;
    private TunedKernelSlot<TunedSoftmaxBackwardArgs>? _softmaxBackwardSlot;
    private TunedKernelSlot<TunedLayerNormArgs>? _layerNormSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormBackwardSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormGradParamsSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _rmsNormGradGammaSlot;
    private string? _tunedDeviceKey;

    /// <summary>The registry device key, e.g. <c>metal:Apple M2 Pro</c>.</summary>
    internal string TunedDeviceKey => _tunedDeviceKey ??= "metal:" + DeviceName;

    /// <summary>True when the tuned-row library compiled, so the generated variants and split references exist.</summary>
    internal bool TunedRowKernelsAvailable => _tunedRowLibrary != IntPtr.Zero;

    /// <summary>True when the last measurement ran on the device clock rather than the host's.</summary>
    internal bool LastTunedMeasurementUsedDeviceClock { get; private set; }

    private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

    /// <summary>
    /// Time of <paramref name="repetitions"/> back-to-back runs: the span from the first command buffer's GPU start to
    /// the last one's GPU end (<see cref="MetalGpuTiming"/>), on the device clock. Every Metal op waits for its command
    /// buffer, so that span is what the runs cost a caller. A driver that reports no GPU interval is timed on the host.
    /// </summary>
    internal double MeasureTunedKernelMilliseconds(Action run, int repetitions)
    {
        lock (_tuneMeasureLock)
        {
            var watch = Stopwatch.StartNew();
            double? device;
            MetalGpuTiming.Begin();
            try
            {
                for (int i = 0; i < repetitions; i++) run();
            }
            finally
            {
                device = MetalGpuTiming.End();
            }
            double host = watch.Elapsed.TotalMilliseconds;
            LastTunedMeasurementUsedDeviceClock = device.HasValue;
            return device ?? host;
        }
    }

    private BackendTunedKernelHarness<TArgs> Harness<TArgs>(Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs,
        Func<TArgs, bool> aliased) where TArgs : struct
        => new BackendTunedKernelHarness<TArgs>(this, TunedRowTolerance, () => IsAvailable, MeasureTunedKernelMilliseconds,
            outputs, aliased);

    // A generated kernel applies when its pipeline builds and a GroupSize-thread group fits it.
    private bool HasTunedKernel(string name, int threadsPerGroup)
    {
        var pipeline = TryGetPipeline(TunedRowLibraryName, _tunedRowLibrary, name);
        return pipeline is not null && pipeline.MaxTotalThreadsPerThreadgroup >= threadsPerGroup;
    }

    private static IGpuBuffer Required(IGpuBuffer? buffer)
        => buffer ?? throw new InvalidOperationException("A required tuned-kernel buffer argument is missing.");

    // Binds buffers to [[buffer(0..)]] and the scalars after them, then dispatches `groups` threadgroups of `threads`.
    private void DispatchTunedRow(string kernel, MTLSize groups, MTLSize threads, IGpuBuffer[] buffers, int[] ints,
        float? trailingFloat = null)
    {
        var pipeline = GetPipeline(TunedRowLibraryName, _tunedRowLibrary, kernel);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        int index = 0;
        foreach (var b in buffers) encoder.SetBuffer(AsMetal(b), index++);
        foreach (int v in ints) encoder.SetBytes(v, (ulong)index++);
        if (trailingFloat is float f) encoder.SetBytes(f, (ulong)index);
        encoder.DispatchThreadgroups(groups, threads);
    }

    private static MTLSize RowGroups(int rows, int lanes)
        => new MTLSize((ulong)(((long)rows * lanes + TunedRowKernelTemplate.GroupSize - 1) / TunedRowKernelTemplate.GroupSize), 1, 1);

    private static readonly MTLSize RowThreads = new MTLSize((ulong)TunedRowKernelTemplate.GroupSize, 1, 1);

    // A thread per row or per column for the split references (uint scalars, as the resident kernels take them).
    private void DispatchTunedReference(string kernel, int threads, IGpuBuffer[] buffers, params uint[] values)
    {
        if (threads <= 0) return;
        var pipeline = GetPipeline(TunedRowLibraryName, _tunedRowLibrary, kernel);
        var (groups, perGroup) = pipeline.Calculate1DDispatch(threads);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer(AsMetal(buffers[i]), i);
        for (int i = 0; i < values.Length; i++) encoder.SetBytes(values[i], (ulong)(buffers.Length + i));
        encoder.DispatchThreadgroups(groups, perGroup);
    }

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
                        new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("metal.softmax.thread_per_row", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedSoftmaxArgs a) => SoftmaxReference(a.Input, a.Output, a.Rows, a.N)),
                    };
                    foreach (int lanes in TunedRowKernelTemplate.RowLanes)
                    {
                        int l = lanes;
                        string name = TunedRowKernelTemplate.SoftmaxName(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("generated.softmax.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name, TunedRowKernelTemplate.GroupSize),
                            (in TunedSoftmaxArgs a) => DispatchTunedRow(name, RowGroups(a.Rows, l), RowThreads,
                                new[] { a.Input, a.Output }, new[] { a.Rows, a.N })));
                    }
                    _softmaxSlot = new TunedKernelSlot<TunedSoftmaxArgs>(TunedKernelOp.Softmax, TunedDeviceKey,
                        Harness<TunedSoftmaxArgs>(a => new[] { (a.Output, a.Rows * a.N) }, a => SameBuffer(a.Input, a.Output)),
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
                        new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("metal.softmax_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in TunedSoftmaxBackwardArgs a) => SoftmaxBackwardReference(a.GradOutput, a.Output, a.GradInput, a.Rows, a.N)),
                    };
                    foreach (int lanes in TunedRowKernelTemplate.RowLanes)
                    {
                        int l = lanes;
                        string name = TunedRowKernelTemplate.SoftmaxBackwardName(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("generated.softmax_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name, TunedRowKernelTemplate.GroupSize),
                            (in TunedSoftmaxBackwardArgs a) => DispatchTunedRow(name, RowGroups(a.Rows, l), RowThreads,
                                new[] { a.GradOutput, a.Output, a.GradInput }, new[] { a.Rows, a.N })));
                    }
                    _softmaxBackwardSlot = new TunedKernelSlot<TunedSoftmaxBackwardArgs>(TunedKernelOp.SoftmaxBackward, TunedDeviceKey,
                        Harness<TunedSoftmaxBackwardArgs>(a => new[] { (a.GradInput, a.Rows * a.N) },
                            a => SameBuffer(a.GradInput, a.GradOutput) || SameBuffer(a.GradInput, a.Output)),
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
                        new DelegateTunedKernelCandidate<TunedLayerNormArgs>("metal.layernorm_forward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedLayerNormArgs a) => LayerNormReference(a)),
                    };
                    foreach (int lanes in TunedRowKernelTemplate.RowLanes)
                    {
                        int l = lanes;
                        string name = TunedRowKernelTemplate.LayerNormName(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedLayerNormArgs>("generated.layernorm_forward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name, TunedRowKernelTemplate.GroupSize),
                            (in TunedLayerNormArgs a) => DispatchTunedRow(name, RowGroups(a.Rows, l), RowThreads,
                                new[] { a.Input, a.Output, a.Gamma, a.Beta, a.SaveMean, a.SaveInvVar }, new[] { a.Rows, a.N }, a.Epsilon)));
                    }
                    _layerNormSlot = new TunedKernelSlot<TunedLayerNormArgs>(TunedKernelOp.LayerNorm, TunedDeviceKey,
                        Harness<TunedLayerNormArgs>(
                            a => new[] { (a.Output, a.Rows * a.N), (a.SaveMean, a.Rows), (a.SaveInvVar, a.Rows) },
                            a => SameBuffer(a.Output, a.Input)),
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
                        new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("metal.layernorm_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => TunedRowKernelsAvailable,
                            (in TunedNormBackwardArgs a) => LayerNormBackwardInputReference(a)),
                    };
                    foreach (int lanes in TunedRowKernelTemplate.RowLanes)
                    {
                        int l = lanes;
                        string name = TunedRowKernelTemplate.LayerNormBackwardName(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("generated.layernorm_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && HasTunedKernel(name, TunedRowKernelTemplate.GroupSize),
                            (in TunedNormBackwardArgs a) => DispatchTunedRow(name, RowGroups(a.Rows, l), RowThreads,
                                new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0 }, new[] { a.Rows, a.N })));
                    }
                    _layerNormBackwardSlot = new TunedKernelSlot<TunedNormBackwardArgs>(TunedKernelOp.LayerNormBackward, TunedDeviceKey,
                        Harness<TunedNormBackwardArgs>(a => new[] { (a.Out0, a.Rows * a.N) },
                            a => SameBuffer(a.Out0, a.GradOutput) || SameBuffer(a.Out0, a.Input)),
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
                    "metal.layernorm_grad_params.thread_per_column", rms: false);
        }
    }

    internal TunedKernelSlot<TunedNormBackwardArgs> RmsNormGradGammaSlot
    {
        get
        {
            if (_rmsNormGradGammaSlot is { } s) return s;
            lock (_tunedSlotsLock)
                return _rmsNormGradGammaSlot ??= CreateColumnGradSlot(TunedKernelOp.RmsNormGradGamma,
                    "metal.rmsnorm_grad_gamma.thread_per_column", rms: true);
        }
    }

    private TunedKernelSlot<TunedNormBackwardArgs> CreateColumnGradSlot(TunedKernelOp op, string referenceId, bool rms)
    {
        var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
        {
            new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(referenceId, TunedKernelOrigin.Builtin, true,
                _ => TunedRowKernelsAvailable, (in TunedNormBackwardArgs a) => ColumnGradReference(rms, a)),
        };
        foreach (int rowLanes in TunedRowKernelTemplate.ColumnRowLanes)
        {
            int r = rowLanes;
            string name = rms ? TunedRowKernelTemplate.RmsNormGradGammaName(r) : TunedRowKernelTemplate.LayerNormGradParamsName(r);
            candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(
                (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + r,
                TunedKernelOrigin.Generated, true, _ => HasTunedKernel(name, 32 * r),
                (in TunedNormBackwardArgs a) =>
                {
                    var groups = new MTLSize((ulong)((a.N + 31) / 32), 1, 1);
                    var threads = new MTLSize(32, (ulong)r, 1);
                    IGpuBuffer[] buffers = rms
                        ? new[] { a.GradOutput, a.Input, a.Stat0, a.Out0 }
                        : new[] { a.GradOutput, a.Input, a.Stat0, Required(a.Stat1), a.Out0, Required(a.Out1) };
                    DispatchTunedRow(name, groups, threads, buffers, new[] { a.Rows, a.N });
                }));
        }
        return new TunedKernelSlot<TunedNormBackwardArgs>(op, TunedDeviceKey,
            Harness<TunedNormBackwardArgs>(
                a => rms ? new[] { (a.Out0, a.N) } : new[] { (a.Out0, a.N), (Required(a.Out1), a.N) },
                a => SameBuffer(a.Out0, a.GradOutput) || SameBuffer(a.Out0, a.Input) ||
                     (a.Out1 is { } o1 && (SameBuffer(o1, a.GradOutput) || SameBuffer(o1, a.Input)))),
            () => GpuDeterminism.IsActive, candidates.ToArray());
    }

    private static bool SameBuffer(IGpuBuffer a, IGpuBuffer b) => ReferenceEquals(a, b) || a.Handle == b.Handle;

    // ---------------------------------------------------------------- reference launches (the established kernels)

    private void SoftmaxReference(IGpuBuffer input, IGpuBuffer output, int rows, int n)
    {
        var pipeline = GetPipeline("Normalization", _normalizationLibrary, "softmax_row");
        var (groups, threads) = pipeline.Calculate1DDispatch(rows);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(AsMetal(input), 0);
        encoder.SetBuffer(AsMetal(output), 1);
        encoder.SetBytes((uint)rows, 2);
        encoder.SetBytes((uint)n, 3);
        encoder.DispatchThreadgroups(groups, threads);
    }

    private void SoftmaxBackwardReference(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
    {
        var pipeline = GetPipeline("Normalization", _normalizationLibrary, "softmax_backward");
        var (groups, threads) = pipeline.Calculate1DDispatch(rows);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(AsMetal(gradOutput), 0);
        encoder.SetBuffer(AsMetal(output), 1);
        encoder.SetBuffer(AsMetal(gradInput), 2);
        encoder.SetBytes((uint)rows, 3);
        encoder.SetBytes((uint)n, 4);
        encoder.DispatchThreadgroups(groups, threads);
    }

    private void LayerNormReference(in TunedLayerNormArgs a)
        => DispatchResidentMetal("layer_norm_forward_serial_rows", a.Rows,
            new[] { a.Input, a.Output, a.Gamma, a.Beta, a.SaveMean, a.SaveInvVar },
            (uint)a.Rows, (uint)a.N, unchecked((uint)SingleToInt32BitsCompat(a.Epsilon)));

    private void LayerNormBackwardInputReference(in TunedNormBackwardArgs a)
        => DispatchTunedReference(MetalTunedRowKernels.LayerNormBackwardInputReference, a.Rows,
            new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0 }, (uint)a.Rows, (uint)a.N);

    private void ColumnGradReference(bool rms, in TunedNormBackwardArgs a)
    {
        if (rms)
            DispatchTunedReference(MetalTunedRowKernels.RmsNormGradGammaReference, a.N,
                new[] { a.GradOutput, a.Input, a.Stat0, a.Out0 }, (uint)a.Rows, (uint)a.N);
        else
            DispatchTunedReference(MetalTunedRowKernels.LayerNormGradParamsReference, a.N,
                new[] { a.GradOutput, a.Input, a.Stat0, Required(a.Stat1), a.Out0, Required(a.Out1) }, (uint)a.Rows, (uint)a.N);
    }

    private void RmsNormBackwardInputReference(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, int rows, int n)
        => DispatchTunedReference(MetalTunedRowKernels.RmsNormBackwardInputReference, rows,
            new[] { gradOutput, input, gamma, saveRms, gradInput }, (uint)rows, (uint)n);
}
