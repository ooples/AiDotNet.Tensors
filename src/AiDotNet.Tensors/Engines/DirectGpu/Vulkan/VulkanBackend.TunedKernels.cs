using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Tuned-kernel registry dispatch on Vulkan (#1115): softmax and its backward, LayerNorm forward / input gradient /
// affine gradients and RMSNorm's gamma gradient run through TunedKernelSlots whose reference is the established
// shader, with the generated VulkanTunedRowKernels variants as the alternatives (see OpenClBackend.TunedKernels).
public sealed unsafe partial class VulkanBackend
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

    /// <summary>The registry device key, e.g. <c>vulkan:AMD Radeon RX 5500 XT</c>.</summary>
    internal string TunedDeviceKey => _tunedDeviceKey ??= "vulkan:" + _device.DeviceName;

    private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

    private readonly object _tuneTimingLock = new object();
    private IntPtr _tuneQueryPool;
    private bool _tuneQueryPoolUnavailable;

    /// <summary>True when timing runs on the device's timestamp clock rather than the host's.</summary>
    internal bool TunedKernelTimingUsesDeviceClock
    {
        get { lock (_tuneTimingLock) return EnsureTimestampPool(); }
    }

    /// <summary>
    /// Time of <paramref name="repetitions"/> back-to-back runs. Every dispatch is submitted and waited for, so the
    /// interval between a timestamp written before the first run and one written after the last is what the op
    /// costs a caller, on the device clock. A device without timestamp queries is timed on the host clock.
    /// </summary>
    internal double MeasureTunedKernelMilliseconds(Action run, int repetitions)
    {
        // Its own lock: gates of different slots tuning concurrently would otherwise overwrite each other's queries.
        // Nothing acquires a slot lock while holding it (run launches a candidate kernel), so the order
        // slot lock -> this lock is safe.
        lock (_tuneTimingLock)
        {
            if (EnsureTimestampPool())
            {
                if (SubmitsWithoutWaiting)
                {
                    // Both timestamps go into the command buffer the runs are recorded into, and the buffer is not
                    // submitted until all of them are in. Written in waited submissions of their own, the timestamps
                    // also counted the device's idle time between submissions (about 3.5 ms on a WDDM driver),
                    // which swamped a single run and skewed every comparison the tuner made.
                    _device.WaitForPendingSubmissions();
                    int limit = _device.CoalescedDispatchLimit;
                    _device.CoalescedDispatchLimit = int.MaxValue;
                    try
                    {
                        RecordTuneTimestamp(0, reset: true);
                        for (int i = 0; i < repetitions; i++) run();
                        RecordTuneTimestamp(1, reset: false);
                        _device.WaitForPendingSubmissions();
                    }
                    finally
                    {
                        _device.CoalescedDispatchLimit = limit;
                    }
                }
                else
                {
                    WriteTuneTimestamp(0, reset: true);
                    for (int i = 0; i < repetitions; i++) run();
                    WriteTuneTimestamp(1, reset: false);
                }
                ulong* ticks = stackalloc ulong[2];
                int result = VulkanTimestampBindings.vkGetQueryPoolResults(_device.Device, _tuneQueryPool, 0, 2,
                    (UIntPtr)(2 * sizeof(ulong)), ticks, sizeof(ulong),
                    VulkanTimestampBindings.VK_QUERY_RESULT_64_BIT | VulkanTimestampBindings.VK_QUERY_RESULT_WAIT_BIT);
                if (result == VulkanNativeBindings.VK_SUCCESS && ticks[1] >= ticks[0])
                    return (ticks[1] - ticks[0]) * (double)_device.Limits.timestampPeriod / 1e6;
                // A failed read or a wrapped counter: stop using the device clock for this process.
                _tuneQueryPoolUnavailable = true;
            }

            _device.WaitForPendingSubmissions();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < repetitions; i++) run();
            // The runs are submitted without waiting; the clock stops when they have executed.
            _device.WaitForPendingSubmissions();
            return (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
        }
    }

    // Caller holds _tuneTimingLock.
    private bool EnsureTimestampPool()
    {
        if (_tuneQueryPoolUnavailable) return false;
        if (_tuneQueryPool != IntPtr.Zero) return true;
        var limits = _device.Limits;
        if (limits.timestampComputeAndGraphics == 0 || !(limits.timestampPeriod > 0f))
        {
            _tuneQueryPoolUnavailable = true;
            return false;
        }
        var info = new VkQueryPoolCreateInfo
        {
            sType = VulkanTimestampBindings.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            pNext = null,
            flags = 0,
            queryType = VulkanTimestampBindings.VK_QUERY_TYPE_TIMESTAMP,
            queryCount = 2,
            pipelineStatistics = 0,
        };
        if (VulkanTimestampBindings.vkCreateQueryPool(_device.Device, &info, IntPtr.Zero, out _tuneQueryPool) !=
            VulkanNativeBindings.VK_SUCCESS || _tuneQueryPool == IntPtr.Zero)
        {
            _tuneQueryPool = IntPtr.Zero;
            _tuneQueryPoolUnavailable = true;
            return false;
        }
        return true;
    }

    // A timestamp recorded into the open (unsubmitted) command buffer, behind the dispatches recorded before it.
    private void RecordTuneTimestamp(uint query, bool reset)
    {
        lock (_computeLock)
        {
            var cmd = _device.OpenCommandBuffer();
            if (reset) VulkanTimestampBindings.vkCmdResetQueryPool(cmd, _tuneQueryPool, 0, 2);
            VulkanTimestampBindings.vkCmdWriteTimestamp(cmd, VulkanTimestampBindings.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                _tuneQueryPool, query);
        }
    }

    private void WriteTuneTimestamp(uint query, bool reset)
    {
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            var cmd = threadRes.CommandBuffer;
            VulkanNativeBindings.vkResetCommandBuffer(cmd, 0);
            var beginInfo = new VkCommandBufferBeginInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext = null,
                flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                pInheritanceInfo = null
            };
            VulkanNativeBindings.vkBeginCommandBuffer(cmd, &beginInfo);
            if (reset) VulkanTimestampBindings.vkCmdResetQueryPool(cmd, _tuneQueryPool, 0, 2);
            VulkanTimestampBindings.vkCmdWriteTimestamp(cmd, VulkanTimestampBindings.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                _tuneQueryPool, query);
            VulkanNativeBindings.vkEndCommandBuffer(cmd);
            _device.SubmitAndWait(cmd, threadRes.Fence);
        }
    }

    private void DisposeTunedKernelResources()
    {
        lock (_tuneTimingLock)
        {
            if (_tuneQueryPool != IntPtr.Zero)
            {
                try { VulkanTimestampBindings.vkDestroyQueryPool(_device.Device, _tuneQueryPool, IntPtr.Zero); } catch { }
                _tuneQueryPool = IntPtr.Zero;
            }
            _tuneQueryPoolUnavailable = true;
        }
    }

    private bool CanMeasureTunedKernels => IsAvailable && IsGlslCompilerAvailable && !_batchMode;

    private BackendTunedKernelHarness<TArgs> Harness<TArgs>(Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs,
        Func<TArgs, bool> aliased) where TArgs : struct
        => new BackendTunedKernelHarness<TArgs>(this, TunedRowTolerance, () => CanMeasureTunedKernels, MeasureTunedKernelMilliseconds,
            outputs, aliased);

    private static int RowThreads(int rows, int lanes) => checked(rows * lanes);

    private static uint[] Push(params int[] values)
    {
        var words = new uint[values.Length];
        for (int i = 0; i < values.Length; i++) words[i] = unchecked((uint)values[i]);
        return words;
    }

    // A generated variant applies only where the GLSL compiler can build it.
    private bool TunedVariantsAvailable => IsGlslCompilerAvailable;

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
                        new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("vulkan.softmax.thread_per_row", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedSoftmaxArgs a) => SoftmaxReference(a.Input, a.Output, a.Rows, a.N)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.Softmax(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("generated.softmax.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => VulkanTunedRowKernels.RowLanesFit((int)shape[1], l) && TunedVariantsAvailable,
                            (in TunedSoftmaxArgs a) => GlslNaryOp(source, new[] { a.Input, a.Output }, RowThreads(a.Rows, l), Push(a.Rows, a.N))));
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
                        new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("vulkan.softmax_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in TunedSoftmaxBackwardArgs a) => SoftmaxBackwardReference(a.GradOutput, a.Output, a.GradInput, a.Rows, a.N)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.SoftmaxBackward(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("generated.softmax_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => VulkanTunedRowKernels.RowLanesFit((int)shape[1], l) && TunedVariantsAvailable,
                            (in TunedSoftmaxBackwardArgs a) => GlslNaryOp(source, new[] { a.GradOutput, a.Output, a.GradInput },
                                RowThreads(a.Rows, l), Push(a.Rows, a.N))));
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
                        new DelegateTunedKernelCandidate<TunedLayerNormArgs>("vulkan.layernorm_forward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedLayerNormArgs a) => LayerNormReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.LayerNorm(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedLayerNormArgs>("generated.layernorm_forward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => VulkanTunedRowKernels.RowLanesFit((int)shape[1], l) && TunedVariantsAvailable,
                            (in TunedLayerNormArgs a) => GlslNaryOp(source,
                                new[] { a.Input, a.Output, a.Gamma, a.Beta, a.SaveMean, a.SaveInvVar }, RowThreads(a.Rows, l),
                                new[] { unchecked((uint)a.Rows), unchecked((uint)a.N), FloatBits(a.Epsilon) })));
                    }
                    _layerNormSlot = new TunedKernelSlot<TunedLayerNormArgs>(TunedKernelOp.LayerNorm, TunedDeviceKey,
                        Harness<TunedLayerNormArgs>(
                            a => new[] { (a.Output, a.Rows * a.N), (a.SaveMean, a.Rows), (a.SaveInvVar, a.Rows) },
                            a => ReferenceEquals(a.Output, a.Input)),
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
                        new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("vulkan.layernorm_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedNormBackwardArgs a) => LayerNormBackwardInputReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.LayerNormBackward(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("generated.layernorm_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => VulkanTunedRowKernels.RowLanesFit((int)shape[1], l) && TunedVariantsAvailable,
                            (in TunedNormBackwardArgs a) => GlslNaryOp(source,
                                new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0 },
                                RowThreads(a.Rows, l), Push(a.Rows, a.N))));
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

    internal TunedKernelSlot<TunedNormBackwardArgs> LayerNormGradParametersSlot
    {
        get
        {
            if (_layerNormGradParamsSlot is { } s) return s;
            lock (_tunedSlotsLock)
                return _layerNormGradParamsSlot ??= CreateColumnGradSlot(TunedKernelOp.LayerNormGradParameters,
                    "vulkan.layernorm_grad_params.thread_per_column", rms: false);
        }
    }

    internal TunedKernelSlot<TunedNormBackwardArgs> RmsNormGradGammaSlot
    {
        get
        {
            if (_rmsNormGradGammaSlot is { } s) return s;
            lock (_tunedSlotsLock)
                return _rmsNormGradGammaSlot ??= CreateColumnGradSlot(TunedKernelOp.RmsNormGradGamma,
                    "vulkan.rmsnorm_grad_gamma.thread_per_column", rms: true);
        }
    }

    private TunedKernelSlot<TunedNormBackwardArgs> CreateColumnGradSlot(TunedKernelOp op, string referenceId, bool rms)
    {
        var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
        {
            new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(referenceId, TunedKernelOrigin.Builtin, true, _ => true,
                (in TunedNormBackwardArgs a) => ColumnGradReference(rms, a)),
        };
        foreach (int rowLanes in VulkanTunedRowKernels.ColumnRowLanes)
        {
            int r = rowLanes;
            string source = rms ? VulkanTunedRowKernels.RmsNormGradGamma(r) : VulkanTunedRowKernels.LayerNormGradParams(r);
            candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(
                (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + r,
                TunedKernelOrigin.Generated, true, _ => TunedVariantsAvailable,
                (in TunedNormBackwardArgs a) =>
                {
                    // One workgroup of 32 * R invocations per 32-column tile.
                    int threads = checked(((a.N + 31) / 32) * 32 * r);
                    var buffers = rms
                        ? new[] { a.GradOutput, a.Input, a.Stat0, a.Out0 }
                        : new[] { a.GradOutput, a.Input, a.Stat0, Required(a.Stat1), a.Out0, Required(a.Out1) };
                    GlslNaryOp(source, buffers, threads, Push(a.Rows, a.N), 32 * r);
                }));
        }
        return new TunedKernelSlot<TunedNormBackwardArgs>(op, TunedDeviceKey,
            Harness<TunedNormBackwardArgs>(
                a => rms ? new[] { (a.Out0, a.N) } : new[] { (a.Out0, a.N), (Required(a.Out1), a.N) },
                a => ReferenceEquals(a.Out0, a.GradOutput) || ReferenceEquals(a.Out0, a.Input) ||
                     (a.Out1 is { } o1 && (ReferenceEquals(o1, a.GradOutput) || ReferenceEquals(o1, a.Input)))),
            () => GpuDeterminism.IsActive, candidates.ToArray());
    }

    private static IGpuBuffer Required(IGpuBuffer? buffer)
        => buffer ?? throw new InvalidOperationException("A required tuned-kernel buffer argument is missing.");

    // ---------------------------------------------------------------- reference launches (the established shaders)

    private void SoftmaxReference(IGpuBuffer input, IGpuBuffer output, int rows, int n)
        => GlslUnaryOp(VulkanGlslKernels.SoftmaxRows, input, output, rows, new uint[] { (uint)rows, (uint)n }, 2 * sizeof(uint));

    private void SoftmaxBackwardReference(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
        => GlslBinaryOp(VulkanGlslKernels.SoftmaxBackward, gradOutput, output, gradInput, rows,
            new uint[] { (uint)rows, (uint)n }, 2 * sizeof(uint));

    private void LayerNormReference(in TunedLayerNormArgs a)
    {
        DispatchNormalizationForward(
            new[] { a.Input, a.Output, a.Gamma, a.Beta, UnusedBinding, UnusedBinding, a.SaveMean, a.SaveInvVar }, a.Rows,
            new[] { 1u, (uint)a.Rows, (uint)a.N, 1u, 1u, FloatBits(a.Epsilon), 0u, 1u });
    }

    // NormalizationBackward (mode 1 LayerNorm, operation 0) writes only the input gradient; its gamma / beta bindings
    // are unused there and take the shared placeholder.
    private void LayerNormBackwardInputReference(in TunedNormBackwardArgs a)
    {
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward,
            new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0, UnusedBinding, UnusedBinding },
            a.Rows, new[] { 1u, 0u, (uint)a.Rows, (uint)a.N, 1u });
    }

    // NormalizationBackward operation 1: the affine gradients, one invocation per column. LayerNorm (mode 1) writes
    // gradGamma and gradBeta; RMSNorm (mode 3) writes gradGamma from the saved RMS in the scale binding.
    private void ColumnGradReference(bool rms, in TunedNormBackwardArgs a)
    {
        if (rms)
        {
            DispatchNormalizationBackward(
                new[] { a.GradOutput, a.Input, UnusedBinding, UnusedBinding, a.Stat0, UnusedBinding, a.Out0, UnusedBinding }, a.N,
                new[] { 3u, 1u, (uint)a.Rows, (uint)a.N, 1u });
            return;
        }
        DispatchNormalizationBackward(
            new[] { a.GradOutput, a.Input, UnusedBinding, a.Stat0, Required(a.Stat1), UnusedBinding, a.Out0, Required(a.Out1) }, a.N,
            new[] { 1u, 1u, (uint)a.Rows, (uint)a.N, 1u });
    }
}