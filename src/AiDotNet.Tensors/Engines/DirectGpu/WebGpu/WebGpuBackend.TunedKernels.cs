#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// Tuned-kernel registry dispatch on WebGPU (#1117): softmax and its backward, LayerNorm forward / input gradient /
// affine gradients and RMSNorm's gamma gradient run through TunedKernelSlots whose reference is the established
// WGSL kernel, with the generated WebGpuTunedRowKernels variants as the alternatives (see VulkanBackend.TunedKernels).
public sealed partial class WebGpuBackend
{
    private const double TunedRowTolerance = 1e-4;

    private readonly object _tunedSlotsLock = new object();
    private TunedKernelSlot<TunedSoftmaxArgs>? _softmaxSlot;
    private TunedKernelSlot<TunedSoftmaxBackwardArgs>? _softmaxBackwardSlot;
    private TunedKernelSlot<TunedLayerNormArgs>? _layerNormSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormBackwardSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormGradParamsSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _rmsNormGradGammaSlot;
    private string? _tunedDeviceKey;

    /// <summary>The registry device key, e.g. <c>webgpu:AMD Radeon RX 5500 XT</c>.</summary>
    internal string TunedDeviceKey => _tunedDeviceKey ??= "webgpu:" + DeviceName;

    private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

    private readonly object _tuneTimingLock = new object();
    private int _tuneTimestampQuery = -2;   // -2 = not created yet, -1 = unavailable

    // Binds an unused storage slot of a reference kernel: WebGPU rejects a bind group that binds one buffer twice where
    // either binding is writable, so an unused output cannot reuse a real one.
    private IGpuBuffer? _tunedUnusedBinding;

    private IGpuBuffer UnusedTunedBinding()
    {
        lock (_tunedSlotsLock) return _tunedUnusedBinding ??= AllocateBuffer(4);
    }

    /// <summary>True when timing runs on the device's timestamp clock rather than the host's.</summary>
    internal bool TunedKernelTimingUsesDeviceClock
    {
        get { lock (_tuneTimingLock) return EnsureTuneTimestampQuery(); }
    }

    /// <summary>
    /// Time of <paramref name="repetitions"/> back-to-back runs. Every WebGPU dispatch is submitted and waited for, so
    /// the interval between a timestamp written before the first run and one written after the last is what the op
    /// costs a caller, on the device clock. A device without the <c>timestamp-query</c> feature is timed on the host.
    /// </summary>
    internal double MeasureTunedKernelMilliseconds(Action run, int repetitions)
    {
        // Its own lock, as on Vulkan: gates of different slots tuning concurrently would overwrite each other's query.
        lock (_tuneTimingLock)
        {
            if (EnsureTuneTimestampQuery())
            {
                WebGpuNativeBindings.WriteTimestamp(_tuneTimestampQuery, 0);
                for (int i = 0; i < repetitions; i++) run();
                WebGpuNativeBindings.WriteTimestamp(_tuneTimestampQuery, 1);
                double nanoseconds = WebGpuNativeBindings.ReadTimestampQueryAsync(_tuneTimestampQuery).GetAwaiter().GetResult();
                if (nanoseconds > 0) return nanoseconds / 1e6;
                // A failed read, or a clock that did not advance: stop using the device clock for this backend.
                ReleaseTuneTimestampQuery();
                _tuneTimestampQuery = -1;
            }

            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < repetitions; i++) run();
            return (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
        }
    }

    // Caller holds _tuneTimingLock.
    private bool EnsureTuneTimestampQuery()
    {
        if (_tuneTimestampQuery == -2) _tuneTimestampQuery = IsAvailable ? WebGpuNativeBindings.CreateTimestampQuery() : -1;
        return _tuneTimestampQuery >= 0;
    }

    // Caller holds _tuneTimingLock.
    private void ReleaseTuneTimestampQuery()
    {
        if (_tuneTimestampQuery >= 0) WebGpuNativeBindings.DestroyTimestampQuery(_tuneTimestampQuery);
    }

    private void DisposeTunedKernelResources()
    {
        lock (_tuneTimingLock)
        {
            ReleaseTuneTimestampQuery();
            _tuneTimestampQuery = -1;
        }
        lock (_tunedSlotsLock)
        {
            _tunedUnusedBinding?.Dispose();
            _tunedUnusedBinding = null;
            _tunedUnusedBinding2?.Dispose();
            _tunedUnusedBinding2 = null;
        }
    }

    private bool CanMeasureTunedKernels => IsAvailable;

    private BackendTunedKernelHarness<TArgs> Harness<TArgs>(Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs,
        Func<TArgs, bool> aliased) where TArgs : struct
        => new BackendTunedKernelHarness<TArgs>(this, TunedRowTolerance, () => CanMeasureTunedKernels, MeasureTunedKernelMilliseconds,
            outputs, aliased);

    private static float[] TunedParams(int rows, int n, float epsilon = 0f) =>
        new[] { BitConverter.Int32BitsToSingle(rows), BitConverter.Int32BitsToSingle(n), epsilon, 0f };

    // One generated variant: its own pipeline per source (keyed by the candidate id), the parameter block after the
    // storage buffers, workgroupsX workgroups.
    private void TunedDispatch(string key, string source, IGpuBuffer[] buffers, float[] uniforms, long workgroups)
        => WithDistinctBindings(buffers, bound =>
            DispatchWorkgroupsAsync("tuned." + key, source, "main", bound, uniforms, checked((int)workgroups)).GetAwaiter().GetResult());

    // The row kernels (WebGpuTunedRowKernels.RowPrologue) fold their workgroups over a 2-D grid.
    private void TunedRowDispatch(string key, string source, IGpuBuffer[] buffers, float[] uniforms, long workgroups)
        => WithDistinctBindings(buffers, bound =>
            DispatchWorkgroupsAsync("tuned." + key, source, "main", bound, uniforms, RowGridX(workgroups), RowGridY(workgroups))
                .GetAwaiter().GetResult());

    // A per-row dispatch of `rows` workgroups, folded over a 2-D grid (WebGPU allows 65535 workgroups per dimension by
    // default); the row kernels index rows as workgroup_id.x + workgroup_id.y * 65535.
    internal static int RowGridX(long workgroups) => (int)Math.Min(Math.Max(workgroups, 1), WebGpuTunedRowKernels.MaxWorkgroups);

    internal static int RowGridY(long workgroups) => (int)((Math.Max(workgroups, 1) + WebGpuTunedRowKernels.MaxWorkgroups - 1) / WebGpuTunedRowKernels.MaxWorkgroups);

    private async System.Threading.Tasks.Task DispatchWorkgroupsAsync(string moduleName, string source, string entryPoint,
        IGpuBuffer[] buffers, float[] uniformParams, int workgroupsX, int workgroupsY = 1)
    {
        ThrowIfNotInitialized();
        var wbufs = new WebGpuBuffer[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) wbufs[i] = (WebGpuBuffer)buffers[i];
        var pipelineId = await GetOrCreatePipelineAsync(moduleName, source, entryPoint);
        using var uniformBuffer = new WebGpuBuffer(uniformParams, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, wbufs);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroupsX, workgroupsY, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }
    private void RowVariant(string key, string source, IGpuBuffer[] buffers, int rows, int n, int lanes, float epsilon = 0f)
        => TunedRowDispatch(key, source, buffers, TunedParams(rows, n, epsilon), WebGpuTunedRowKernels.RowWorkgroups(rows, lanes));

    private Func<TunedShape, bool> RowFit(int lanes) =>
        shape => IsAvailable && WebGpuTunedRowKernels.RowLanesFit(shape[0], shape[1], lanes);

    /// <summary>
    /// WebGPU rejects a dispatch that binds one buffer both read-only and writable, so an op called in place (an input
    /// that is also its output) failed on every kernel. Runs <paramref name="run"/> with a binding list in which every
    /// buffer is distinct: an earlier occurrence of a repeated buffer is replaced by a scratch copy of it. Every binding
    /// list here puts the buffers a kernel reads before the ones it writes, so the copy is always the one read and the
    /// output lands in the caller's buffer.
    /// </summary>
    private void WithDistinctBindings(IGpuBuffer[] buffers, Action<IGpuBuffer[]> run)
    {
        IGpuBuffer[] bound = buffers;
        List<IGpuBuffer>? scratch = null;
        for (int i = 0; i < buffers.Length; i++)
        {
            for (int j = i + 1; j < buffers.Length; j++)
            {
                if (!ReferenceEquals(buffers[i], buffers[j])) continue;
                if (ReferenceEquals(bound, buffers)) bound = (IGpuBuffer[])buffers.Clone();
                var copy = AllocateBuffer(buffers[i].Size);
                Copy(buffers[i], copy, buffers[i].Size);
                (scratch ??= new List<IGpuBuffer>()).Add(copy);
                bound[i] = copy;
                break;
            }
        }
        try { run(bound); }
        finally
        {
            if (scratch is not null) foreach (var b in scratch) b.Dispose();
        }
    }
    // ---------------------------------------------------------------- references (the established kernels)

    private void SoftmaxReference(IGpuBuffer input, IGpuBuffer output, int rows, int n)
        => WithDistinctBindings(new[] { input, output }, b => SoftmaxAsync(b[0], b[1], rows, n).GetAwaiter().GetResult());

    // softmax_backward reduces one row per workgroup (workgroup_id.x is the row). It was dispatched as rows / 256
    // workgroups (one per 256 "elements"), so only the first row was ever computed and the rest kept their old contents.
    private void SoftmaxBackwardReference(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
        => WithDistinctBindings(new[] { gradOutput, output, gradInput }, b =>
            DispatchWorkgroupsAsync("SoftmaxBackward", WebGpuKernels.SoftmaxBackwardSource, "softmax_backward", b,
                new[] { BitConverter.Int32BitsToSingle(rows), BitConverter.Int32BitsToSingle(n), 0f, 0f }, RowGridX(rows), RowGridY(rows))
                .GetAwaiter().GetResult());

    // The established layer_norm kernel writes only the output; the row statistics the interface promises in
    // saveMean / saveInvVar (and the input-gradient candidates read) come from a statistics pass with the same 256-lane
    // reduction. They used to be filled with zeros.
    private void LayerNormReference(in TunedLayerNormArgs a)
    {
        var args = a;
        // The statistics come first: they read the input, which an in-place call is about to overwrite.
        TunedRowDispatch("layernorm_row_stats", WebGpuTunedRowKernels.LayerNormRowStats,
            new[] { a.Input, a.SaveMean, a.SaveInvVar }, TunedParams(a.Rows, a.N, a.Epsilon), a.Rows);
        WithDistinctBindings(new[] { args.Input, args.Gamma, args.Beta, args.Output }, b =>
            LayerNormAsync(b[0], b[1], b[2], b[3], args.Rows, args.N, args.Epsilon).GetAwaiter().GetResult());
    }

    private void LayerNormBackwardInputReference(in TunedNormBackwardArgs a)
    {
        var args = a;
        WithDistinctBindings(new[] { a.GradOutput, a.Input, a.Gamma ?? UnusedTunedBinding(), a.Out0 }, b =>
            Dispatch4BufferAsync("LayerNormBackwardData", WebGpuKernels.LayerNormBackwardDataSource, "layer_norm_backward_data",
                b[0], b[1], b[2], b[3], TunedParams(args.Rows, args.N, args.Epsilon), checked(args.Rows * args.N)).GetAwaiter().GetResult());
    }

    private void LayerNormGradParamsReference(in TunedNormBackwardArgs a)
    {
        var args = a;
        WithDistinctBindings(new[] { a.GradOutput, a.Input, a.Out0, a.Out1 ?? UnusedTunedBinding() }, b =>
            Dispatch4BufferAsync("LayerNormBackwardStats", WebGpuKernels.LayerNormBackwardStatsSource, "layer_norm_backward_stats",
                b[0], b[1], b[2], b[3], TunedParams(args.Rows, args.N, args.Epsilon), args.N).GetAwaiter().GetResult());
    }

    private void RmsNormGradGammaReference(in TunedNormBackwardArgs a)
    {
        var args = a;
        WithDistinctBindings(new[] { a.GradOutput, a.Input, a.Stat0, a.Out0 }, b =>
            Dispatch4BufferAsync("RmsNormGradGamma", WebGpuKernels.RmsNormGradGammaSource, "rmsnorm_grad_gamma",
                b[0], b[1], b[2], b[3], TunedParams(args.Rows, args.N), args.N).GetAwaiter().GetResult());
    }

    private IGpuBuffer? _tunedUnusedBinding2;

    private IGpuBuffer UnusedTunedBinding2()
    {
        lock (_tunedSlotsLock) return _tunedUnusedBinding2 ??= AllocateBuffer(4);
    }

    // ---------------------------------------------------------------- slots

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
                        new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("webgpu.softmax.workgroup_per_row", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedSoftmaxArgs a) => SoftmaxReference(a.Input, a.Output, a.Rows, a.N)),
                    };
                    foreach (int lanes in WebGpuTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string id = "generated.softmax.lanes" + l, source = WebGpuTunedRowKernels.Softmax(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxArgs>(id, TunedKernelOrigin.Generated, true, RowFit(l),
                            (in TunedSoftmaxArgs a) => RowVariant(id, source, new[] { a.Input, a.Output }, a.Rows, a.N, l)));
                    }
                    _softmaxSlot = new TunedKernelSlot<TunedSoftmaxArgs>(TunedKernelOp.Softmax, TunedDeviceKey,
                        Harness<TunedSoftmaxArgs>(a => new[] { (a.Output, a.Rows * a.N) }, a => ReferenceEquals(a.Input, a.Output)),
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
                        new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("webgpu.softmax_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in TunedSoftmaxBackwardArgs a) => SoftmaxBackwardReference(a.GradOutput, a.Output, a.GradInput, a.Rows, a.N)),
                    };
                    foreach (int lanes in WebGpuTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string id = "generated.softmax_backward.lanes" + l, source = WebGpuTunedRowKernels.SoftmaxBackward(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>(id, TunedKernelOrigin.Generated, true,
                            RowFit(l), (in TunedSoftmaxBackwardArgs a) =>
                                RowVariant(id, source, new[] { a.GradOutput, a.Output, a.GradInput }, a.Rows, a.N, l)));
                    }
                    _softmaxBackwardSlot = new TunedKernelSlot<TunedSoftmaxBackwardArgs>(TunedKernelOp.SoftmaxBackward, TunedDeviceKey,
                        Harness<TunedSoftmaxBackwardArgs>(a => new[] { (a.GradInput, a.Rows * a.N) },
                            a => ReferenceEquals(a.GradInput, a.GradOutput) || ReferenceEquals(a.GradInput, a.Output)),
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
                        new DelegateTunedKernelCandidate<TunedLayerNormArgs>("webgpu.layernorm_forward.workgroup_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedLayerNormArgs a) => LayerNormReference(a)),
                    };
                    foreach (int lanes in WebGpuTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string id = "generated.layernorm_forward.lanes" + l, source = WebGpuTunedRowKernels.LayerNorm(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedLayerNormArgs>(id, TunedKernelOrigin.Generated, true, RowFit(l),
                            (in TunedLayerNormArgs a) => RowVariant(id, source,
                                new[] { a.Input, a.Output, a.Gamma, a.Beta, a.SaveMean, a.SaveInvVar }, a.Rows, a.N, l, a.Epsilon)));
                    }
                    _layerNormSlot = new TunedKernelSlot<TunedLayerNormArgs>(TunedKernelOp.LayerNorm, TunedDeviceKey,
                        Harness<TunedLayerNormArgs>(a => new[] { (a.Output, a.Rows * a.N), (a.SaveMean, a.Rows), (a.SaveInvVar, a.Rows) },
                            a => ReferenceEquals(a.Input, a.Output)),
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
                        new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("webgpu.layernorm_backward_input.thread_per_element",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedNormBackwardArgs a) => LayerNormBackwardInputReference(a)),
                    };
                    foreach (int lanes in WebGpuTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string id = "generated.layernorm_backward.lanes" + l, source = WebGpuTunedRowKernels.LayerNormBackward(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(id, TunedKernelOrigin.Generated, true,
                            shape => RowFit(l)(shape), (in TunedNormBackwardArgs a) => RowVariant(id, source,
                                new[] { a.GradOutput, a.Input, a.Gamma ?? UnusedTunedBinding(), a.Stat0, a.Stat1 ?? UnusedTunedBinding(), a.Out0 },
                                a.Rows, a.N, l)));
                    }
                    _layerNormBackwardSlot = new TunedKernelSlot<TunedNormBackwardArgs>(TunedKernelOp.LayerNormBackward, TunedDeviceKey,
                        Harness<TunedNormBackwardArgs>(a => new[] { (a.Out0, a.Rows * a.N) },
                            a => ReferenceEquals(a.Out0, a.GradOutput) || ReferenceEquals(a.Out0, a.Input)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _layerNormBackwardSlot;
            }
        }
    }

    private bool ColumnVariantFits(int rowLanes)
    {
        int invocations = 32 * rowLanes;
        return invocations <= MaxWorkgroupSize && invocations <= (Limits?.MaxComputeWorkgroupSizeX ?? 256);
    }

    internal TunedKernelSlot<TunedNormBackwardArgs> LayerNormGradParametersSlot => ColumnSlot(ref _layerNormGradParamsSlot, rms: false);

    internal TunedKernelSlot<TunedNormBackwardArgs> RmsNormGradGammaSlot => ColumnSlot(ref _rmsNormGradGammaSlot, rms: true);

    private TunedKernelSlot<TunedNormBackwardArgs> ColumnSlot(ref TunedKernelSlot<TunedNormBackwardArgs>? field, bool rms)
    {
        if (field is { } s) return s;
        lock (_tunedSlotsLock)
        {
            if (field is null)
            {
                var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
                {
                    rms
                        ? new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("webgpu.rmsnorm_grad_gamma.thread_per_column",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedNormBackwardArgs a) => RmsNormGradGammaReference(a))
                        : new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("webgpu.layernorm_grad_params.thread_per_column",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedNormBackwardArgs a) => LayerNormGradParamsReference(a)),
                };
                foreach (int rowLanes in WebGpuTunedRowKernels.ColumnRowLanes)
                {
                    string id = (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + rowLanes;
                    string source = rms ? WebGpuTunedRowKernels.RmsNormGradGamma(rowLanes) : WebGpuTunedRowKernels.LayerNormGradParams(rowLanes);
                    candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(id, TunedKernelOrigin.Generated, true,
                        // 32 * R invocations per workgroup: the 32-row variant (1024) exceeds WebGPU's default limit of 256,
                        // so each variant is offered only where the device's limits admit it.
                        shape => IsAvailable && ColumnVariantFits(rowLanes) && WebGpuTunedRowKernels.ColumnFit(shape[1]),
                        (in TunedNormBackwardArgs a) => TunedDispatch(id, source,
                            rms ? new[] { a.GradOutput, a.Input, a.Stat0, a.Out0 }
                                : new[] { a.GradOutput, a.Input, a.Stat0, a.Stat1 ?? UnusedTunedBinding(), a.Out0, a.Out1 ?? UnusedTunedBinding2() },
                            TunedParams(a.Rows, a.N), (a.N + 31L) / 32)));
                }
                field = new TunedKernelSlot<TunedNormBackwardArgs>(
                    rms ? TunedKernelOp.RmsNormGradGamma : TunedKernelOp.LayerNormGradParameters, TunedDeviceKey,
                    Harness<TunedNormBackwardArgs>(
                        a => rms ? new[] { (a.Out0, a.N) } : new[] { (a.Out0, a.N), (a.Out1 ?? a.Out0, a.N) },
                        a => ReferenceEquals(a.Out0, a.GradOutput) || ReferenceEquals(a.Out0, a.Input)),
                    () => GpuDeterminism.IsActive, candidates.ToArray());
            }
            return field;
        }
    }
}
#endif
