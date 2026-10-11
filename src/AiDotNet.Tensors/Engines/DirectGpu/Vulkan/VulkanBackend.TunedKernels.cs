// Copyright (c) AiDotNet. All rights reserved.
// Tuned-kernel registry dispatch on Vulkan (#1115): slots, measurement harness and the device timestamp clock.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>Runs a candidate on arguments passed by reference.</summary>
internal delegate void VulkanTunedKernelInvoker<TArgs>(in TArgs args) where TArgs : struct;

/// <summary>A registry candidate backed by a delegate on the Vulkan backend.</summary>
internal sealed class VulkanTunedKernelCandidate<TArgs> : ITunedKernelCandidate<TArgs> where TArgs : struct
{
    private readonly Func<TunedShape, bool> _applicable;
    private readonly VulkanTunedKernelInvoker<TArgs> _execute;

    internal VulkanTunedKernelCandidate(string id, TunedKernelOrigin origin, bool deterministic,
        Func<TunedShape, bool> applicable, VulkanTunedKernelInvoker<TArgs> execute)
    {
        Id = id;
        Origin = origin;
        IsDeterministic = deterministic;
        _applicable = applicable;
        _execute = execute;
    }

    public string Id { get; }
    public TunedKernelOrigin Origin { get; }
    public bool IsDeterministic { get; }
    public bool IsApplicable(in TunedShape shape) => _applicable(shape);
    public void Execute(in TArgs args) => _execute(args);
}

/// <summary>Vulkan measurement services for one op family: host snapshots of outputs, device-timestamp timing.</summary>
internal sealed class VulkanTunedKernelHarness<TArgs> : ITunedKernelHarness<TArgs> where TArgs : struct
{
    private readonly VulkanBackend _backend;
    private readonly Func<TArgs, (IGpuBuffer Buffer, int Count)[]> _outputs;
    private readonly Func<TArgs, bool> _aliased;

    internal VulkanTunedKernelHarness(VulkanBackend backend, double relativeTolerance,
        Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs, Func<TArgs, bool> aliased)
    {
        _backend = backend;
        RelativeTolerance = relativeTolerance;
        _outputs = outputs;
        _aliased = aliased;
    }

    public double RelativeTolerance { get; }

    public bool CanMeasure(in TArgs args) => _backend.CanMeasureTunedKernels && !_aliased(args);

    public void PoisonOutput(in TArgs args)
    {
        foreach (var o in _outputs(args)) _backend.Fill(o.Buffer, float.NaN, o.Count);
    }

    public float[] SnapshotOutput(in TArgs args)
    {
        var outs = _outputs(args);
        int total = 0;
        foreach (var o in outs) total += o.Count;
        var result = new float[total];
        int offset = 0;
        foreach (var o in outs)
        {
            float[] host = _backend.DownloadBuffer(o.Buffer);
            Array.Copy(host, 0, result, offset, o.Count);
            offset += o.Count;
        }
        return result;
    }

    public double MeasureMilliseconds(ITunedKernelCandidate<TArgs> candidate, in TArgs args, int repetitions)
    {
        TArgs local = args;
        return _backend.MeasureTunedKernelMilliseconds(() => candidate.Execute(local), repetitions);
    }
}

/// <summary>Row softmax arguments.</summary>
internal readonly struct VulkanSoftmaxArgs
{
    internal VulkanSoftmaxArgs(IGpuBuffer input, IGpuBuffer output, int rows, int n)
    { Input = input; Output = output; Rows = rows; N = n; }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Output { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>Row softmax backward arguments.</summary>
internal readonly struct VulkanSoftmaxBackwardArgs
{
    internal VulkanSoftmaxBackwardArgs(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
    { GradOutput = gradOutput; Output = output; GradInput = gradInput; Rows = rows; N = n; }
    internal IGpuBuffer GradOutput { get; }
    internal IGpuBuffer Output { get; }
    internal IGpuBuffer GradInput { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>LayerNorm forward arguments.</summary>
internal readonly struct VulkanLayerNormArgs
{
    internal VulkanLayerNormArgs(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int rows, int n, float epsilon)
    {
        Input = input; Output = output; Gamma = gamma; Beta = beta; SaveMean = saveMean; SaveInvVar = saveInvVar;
        Rows = rows; N = n; Epsilon = epsilon;
    }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Output { get; }
    internal IGpuBuffer Gamma { get; }
    internal IGpuBuffer Beta { get; }
    internal IGpuBuffer SaveMean { get; }
    internal IGpuBuffer SaveInvVar { get; }
    internal int Rows { get; }
    internal int N { get; }
    internal float Epsilon { get; }
}

/// <summary>Normalization backward arguments (LayerNorm input gradient and the affine column reductions).</summary>
internal readonly struct VulkanNormBackwardArgs
{
    internal VulkanNormBackwardArgs(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer? gamma,
        IGpuBuffer stat0, IGpuBuffer? stat1, IGpuBuffer out0, IGpuBuffer? out1, int rows, int n)
    {
        GradOutput = gradOutput; Input = input; Gamma = gamma; Stat0 = stat0; Stat1 = stat1; Out0 = out0; Out1 = out1;
        Rows = rows; N = n;
    }
    internal IGpuBuffer GradOutput { get; }
    internal IGpuBuffer Input { get; }
    internal IGpuBuffer? Gamma { get; }
    /// <summary>Mean (LayerNorm) or RMS (RMSNorm) per row.</summary>
    internal IGpuBuffer Stat0 { get; }
    /// <summary>Inverse standard deviation per row (LayerNorm only).</summary>
    internal IGpuBuffer? Stat1 { get; }
    /// <summary>Input gradient, or gradGamma for a column reduction.</summary>
    internal IGpuBuffer Out0 { get; }
    /// <summary>gradBeta for the LayerNorm column reduction.</summary>
    internal IGpuBuffer? Out1 { get; }
    internal int Rows { get; }
    internal int N { get; }
}

/// <summary>
/// The Vulkan backend's tuned-kernel slots: every op family below dispatches through
/// <see cref="TunedKernelSlot{TArgs}"/>, with the established kernel as the reference and the generated variants of
/// <see cref="VulkanTunedRowKernels"/> as candidates.
/// </summary>
public sealed unsafe partial class VulkanBackend
{
    private const double RowOpTolerance = 1e-4;
    private readonly object _tunedSlotsLock = new();
    private readonly object _tuneTimingLock = new();
    private IntPtr _tuneQueryPool;
    private bool _tuneQueryPoolUnavailable;
    private string? _tunedDeviceKey;
    private TunedKernelSlot<VulkanSoftmaxArgs>? _softmaxSlot;
    private TunedKernelSlot<VulkanSoftmaxBackwardArgs>? _softmaxBackwardSlot;
    private TunedKernelSlot<VulkanLayerNormArgs>? _layerNormSlot;
    private TunedKernelSlot<VulkanNormBackwardArgs>? _layerNormBackwardSlot;
    private TunedKernelSlot<VulkanNormBackwardArgs>? _layerNormGradParamsSlot;
    private TunedKernelSlot<VulkanNormBackwardArgs>? _rmsNormGradGammaSlot;

    /// <summary>Device key used by the tuned-kernel registry: <c>vulkan:{device name}</c>.</summary>
    internal string TunedDeviceKey => _tunedDeviceKey ??= "vulkan:" + DeviceName;

    // A batch only records dispatches, so nothing run inside one can be snapshotted or timed.
    internal bool CanMeasureTunedKernels =>
        IsAvailable && !(_batchMode && Environment.CurrentManagedThreadId == _batchOwnerThreadId);

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
                WriteTuneTimestamp(0, reset: true);
                for (int i = 0; i < repetitions; i++) run();
                WriteTuneTimestamp(1, reset: false);
                ulong* ticks = stackalloc ulong[2];
                int result = VulkanTimestampBindings.vkGetQueryPoolResults(_device.Device, _tuneQueryPool, 0, 2,
                    (UIntPtr)(2 * sizeof(ulong)), ticks, sizeof(ulong),
                    VulkanTimestampBindings.VK_QUERY_RESULT_64_BIT | VulkanTimestampBindings.VK_QUERY_RESULT_WAIT_BIT);
                if (result == VulkanNativeBindings.VK_SUCCESS && ticks[1] >= ticks[0])
                    return (ticks[1] - ticks[0]) * (double)_device.Limits.timestampPeriod / 1e6;
                // A failed read or a wrapped counter: stop using the device clock for this process.
                _tuneQueryPoolUnavailable = true;
            }

            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < repetitions; i++) run();
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

    // A generated variant is a candidate only where its shader compiles and its pipeline builds on this device
    // (no libshaderc, or a driver that rejects it, leaves the reference alone in the slot). The answer is kept per
    // shader: applicability is asked again on every call that cannot be measured yet (inside a batch).
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> _tunedPipelineAvailable =
        new(StringComparer.Ordinal);

    private bool HasTunedPipeline(string source, int bindings, uint pushSize)
    {
        if (_tunedPipelineAvailable.TryGetValue(source, out bool known)) return known;
        bool available;
        try
        {
            available = GetOrCreateGlslPipeline(source, bindings, pushSize) is not null;
        }
        catch (OutOfMemoryException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"Vulkan tuned kernel variant unavailable: {ex.Message}");
            available = false;
        }
        _tunedPipelineAvailable[source] = available;
        return available;
    }

    private static bool Aliases(IGpuBuffer a, IGpuBuffer b) => ReferenceEquals(a, b) || a.Handle == b.Handle;

    private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

    private static IGpuBuffer Required(IGpuBuffer? buffer) =>
        buffer ?? throw new InvalidOperationException("A required tuned-kernel buffer argument is missing.");

    // ---------------------------------------------------------------- softmax

    internal TunedKernelSlot<VulkanSoftmaxArgs> SoftmaxSlot
    {
        get
        {
            if (_softmaxSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_softmaxSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<VulkanSoftmaxArgs>>
                    {
                        new VulkanTunedKernelCandidate<VulkanSoftmaxArgs>("glsl.softmax.invocation_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in VulkanSoftmaxArgs a) => LaunchSoftmaxReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.Softmax(l);
                        candidates.Add(new VulkanTunedKernelCandidate<VulkanSoftmaxArgs>("generated.softmax.lanes" + l,
                            TunedKernelOrigin.Generated, true,
                            shape => VulkanTunedRowKernels.RowLanesFit(l, shape[0], shape[1]) &&
                                     HasTunedPipeline(source, 2, 2 * sizeof(uint)),
                            (in VulkanSoftmaxArgs a) => GlslUnaryOp(source, a.Input, a.Output, a.Rows * l,
                                new[] { (uint)a.Rows, (uint)a.N }, 2 * sizeof(uint))));
                    }
                    _softmaxSlot = new TunedKernelSlot<VulkanSoftmaxArgs>(TunedKernelOp.Softmax, TunedDeviceKey,
                        new VulkanTunedKernelHarness<VulkanSoftmaxArgs>(this, RowOpTolerance,
                            a => new[] { (a.Output, a.Rows * a.N) },
                            a => Aliases(a.Input, a.Output)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _softmaxSlot;
            }
        }
    }

    private void LaunchSoftmaxReference(in VulkanSoftmaxArgs a) =>
        GlslUnaryOp(VulkanGlslKernels.SoftmaxRows, a.Input, a.Output, a.Rows,
            new[] { (uint)a.Rows, (uint)a.N }, 2 * sizeof(uint));

    // ---------------------------------------------------------------- softmax backward

    internal TunedKernelSlot<VulkanSoftmaxBackwardArgs> SoftmaxBackwardSlot
    {
        get
        {
            if (_softmaxBackwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_softmaxBackwardSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<VulkanSoftmaxBackwardArgs>>
                    {
                        new VulkanTunedKernelCandidate<VulkanSoftmaxBackwardArgs>("glsl.softmax_backward.invocation_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in VulkanSoftmaxBackwardArgs a) => LaunchSoftmaxBackwardReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.SoftmaxBackward(l);
                        candidates.Add(new VulkanTunedKernelCandidate<VulkanSoftmaxBackwardArgs>(
                            "generated.softmax_backward.lanes" + l, TunedKernelOrigin.Generated, true,
                            shape => VulkanTunedRowKernels.RowLanesFit(l, shape[0], shape[1]) &&
                                     HasTunedPipeline(source, 3, 2 * sizeof(uint)),
                            (in VulkanSoftmaxBackwardArgs a) => GlslBinaryOp(source, a.GradOutput, a.Output, a.GradInput,
                                a.Rows * l, new[] { (uint)a.Rows, (uint)a.N }, 2 * sizeof(uint))));
                    }
                    _softmaxBackwardSlot = new TunedKernelSlot<VulkanSoftmaxBackwardArgs>(TunedKernelOp.SoftmaxBackward,
                        TunedDeviceKey,
                        new VulkanTunedKernelHarness<VulkanSoftmaxBackwardArgs>(this, RowOpTolerance,
                            a => new[] { (a.GradInput, a.Rows * a.N) },
                            a => Aliases(a.GradInput, a.GradOutput) || Aliases(a.GradInput, a.Output)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _softmaxBackwardSlot;
            }
        }
    }

    private void LaunchSoftmaxBackwardReference(in VulkanSoftmaxBackwardArgs a) =>
        GlslBinaryOp(VulkanGlslKernels.SoftmaxBackward, a.GradOutput, a.Output, a.GradInput,
            a.Rows, new[] { (uint)a.Rows, (uint)a.N }, 2 * sizeof(uint));

    // ---------------------------------------------------------------- layer norm forward

    internal TunedKernelSlot<VulkanLayerNormArgs> LayerNormSlot
    {
        get
        {
            if (_layerNormSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_layerNormSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<VulkanLayerNormArgs>>
                    {
                        new VulkanTunedKernelCandidate<VulkanLayerNormArgs>("glsl.layernorm_forward.invocation_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in VulkanLayerNormArgs a) => LaunchLayerNormReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.LayerNormForward(l);
                        candidates.Add(new VulkanTunedKernelCandidate<VulkanLayerNormArgs>(
                            "generated.layernorm_forward.lanes" + l, TunedKernelOrigin.Generated, true,
                            shape => VulkanTunedRowKernels.RowLanesFit(l, shape[0], shape[1]) &&
                                     HasTunedPipeline(source, 6, 3 * sizeof(uint)),
                            (in VulkanLayerNormArgs a) => GlslNaryOp(source,
                                new[] { a.Input, a.Output, a.Gamma, a.Beta, a.SaveMean, a.SaveInvVar }, a.Rows * l,
                                new[] { (uint)a.Rows, (uint)a.N, FloatBits(a.Epsilon) })));
                    }
                    _layerNormSlot = new TunedKernelSlot<VulkanLayerNormArgs>(TunedKernelOp.LayerNorm, TunedDeviceKey,
                        new VulkanTunedKernelHarness<VulkanLayerNormArgs>(this, RowOpTolerance,
                            a => new[] { (a.Output, a.Rows * a.N), (a.SaveMean, a.Rows), (a.SaveInvVar, a.Rows) },
                            a => Aliases(a.Output, a.Input)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _layerNormSlot;
            }
        }
    }

    private void LaunchLayerNormReference(in VulkanLayerNormArgs a)
    {
        var unused = UnusedBinding;
        DispatchNormalizationForward(
            new[] { a.Input, a.Output, a.Gamma, a.Beta, unused, unused, a.SaveMean, a.SaveInvVar },
            a.Rows, new[] { 1u, (uint)a.Rows, (uint)a.N, 1u, 1u, FloatBits(a.Epsilon), 0u, 1u });
    }

    // ---------------------------------------------------------------- layer norm backward (input gradient)

    internal TunedKernelSlot<VulkanNormBackwardArgs> LayerNormBackwardSlot
    {
        get
        {
            if (_layerNormBackwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_layerNormBackwardSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<VulkanNormBackwardArgs>>
                    {
                        new VulkanTunedKernelCandidate<VulkanNormBackwardArgs>("glsl.layernorm_backward.invocation_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in VulkanNormBackwardArgs a) => LaunchLayerNormBackwardReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.LayerNormBackward(l);
                        candidates.Add(new VulkanTunedKernelCandidate<VulkanNormBackwardArgs>(
                            "generated.layernorm_backward.lanes" + l, TunedKernelOrigin.Generated, true,
                            shape => VulkanTunedRowKernels.RowLanesFit(l, shape[0], shape[1]) &&
                                     HasTunedPipeline(source, 6, 2 * sizeof(uint)),
                            (in VulkanNormBackwardArgs a) => GlslNaryOp(source,
                                new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0 },
                                a.Rows * l, new[] { (uint)a.Rows, (uint)a.N })));
                    }
                    _layerNormBackwardSlot = new TunedKernelSlot<VulkanNormBackwardArgs>(TunedKernelOp.LayerNormBackward,
                        TunedDeviceKey,
                        new VulkanTunedKernelHarness<VulkanNormBackwardArgs>(this, RowOpTolerance,
                            a => new[] { (a.Out0, a.Rows * a.N) },
                            a => Aliases(a.Out0, a.GradOutput) || Aliases(a.Out0, a.Input)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _layerNormBackwardSlot;
            }
        }
    }

    // The established kernel declares gradGamma/gradBeta bindings but its input-gradient operation never writes
    // them; the input-gradient buffer stands in for both.
    private void LaunchLayerNormBackwardReference(in VulkanNormBackwardArgs a) =>
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward,
            new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0, a.Out0, a.Out0 },
            a.Rows, new[] { 1u, 0u, (uint)a.Rows, (uint)a.N, 1u });

    // ---------------------------------------------------------------- column reductions (affine gradients)

    internal TunedKernelSlot<VulkanNormBackwardArgs> LayerNormGradParametersSlot
    {
        get
        {
            if (_layerNormGradParamsSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _layerNormGradParamsSlot ??= CreateColumnGradSlot(TunedKernelOp.LayerNormGradParameters,
                    "glsl.layernorm_grad_params.invocation_per_column", rms: false);
                return _layerNormGradParamsSlot;
            }
        }
    }

    internal TunedKernelSlot<VulkanNormBackwardArgs> RmsNormGradGammaSlot
    {
        get
        {
            if (_rmsNormGradGammaSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                _rmsNormGradGammaSlot ??= CreateColumnGradSlot(TunedKernelOp.RmsNormGradGamma,
                    "glsl.rmsnorm_grad_gamma.invocation_per_column", rms: true);
                return _rmsNormGradGammaSlot;
            }
        }
    }

    private TunedKernelSlot<VulkanNormBackwardArgs> CreateColumnGradSlot(TunedKernelOp op, string referenceId, bool rms)
    {
        var candidates = new List<ITunedKernelCandidate<VulkanNormBackwardArgs>>
        {
            new VulkanTunedKernelCandidate<VulkanNormBackwardArgs>(referenceId, TunedKernelOrigin.Builtin, true, _ => true,
                (in VulkanNormBackwardArgs a) => LaunchColumnGradReference(rms, a)),
        };
        foreach (int rowLanes in VulkanTunedRowKernels.ColumnRowLanes)
        {
            int r = rowLanes;
            string source = rms ? VulkanTunedRowKernels.RmsNormGradGamma(r) : VulkanTunedRowKernels.LayerNormGradParameters(r);
            int bindings = rms ? 4 : 6;
            candidates.Add(new VulkanTunedKernelCandidate<VulkanNormBackwardArgs>(
                (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + r,
                TunedKernelOrigin.Generated, true,
                shape => (long)shape[1] * r <= int.MaxValue && HasTunedPipeline(source, bindings, 2 * sizeof(uint)),
                (in VulkanNormBackwardArgs a) => GlslNaryOp(source,
                    rms
                        ? new[] { a.GradOutput, a.Input, a.Stat0, a.Out0 }
                        : new[] { a.GradOutput, a.Input, a.Stat0, Required(a.Stat1), a.Out0, Required(a.Out1) },
                    a.N * r, new[] { (uint)a.Rows, (uint)a.N })));
        }
        return new TunedKernelSlot<VulkanNormBackwardArgs>(op, TunedDeviceKey,
            new VulkanTunedKernelHarness<VulkanNormBackwardArgs>(this, RowOpTolerance,
                a => rms ? new[] { (a.Out0, a.N) } : new[] { (a.Out0, a.N), (Required(a.Out1), a.N) },
                a => Aliases(a.Out0, a.GradOutput) || Aliases(a.Out0, a.Input) ||
                     (a.Out1 is { } o1 && (Aliases(o1, a.GradOutput) || Aliases(o1, a.Input)))),
            () => GpuDeterminism.IsActive, candidates.ToArray());
    }

    // The established kernel's parameter-gradient operation reads gradOutput, input and the saved statistics and
    // writes gradGamma (and gradBeta for LayerNorm). Its other bindings are declared but untouched there, so buffers
    // already in the call stand in for them: the input for gamma, and gradGamma for the input gradient (and for
    // RMSNorm's absent gradBeta and saved mean).
    private void LaunchColumnGradReference(bool rms, in VulkanNormBackwardArgs a)
    {
        var buffers = rms
            ? new[] { a.GradOutput, a.Input, a.Input, a.Stat0, a.Stat0, a.Out0, a.Out0, a.Out0 }
            : new[] { a.GradOutput, a.Input, a.Input, a.Stat0, Required(a.Stat1), a.Out0, a.Out0, Required(a.Out1) };
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward, buffers, a.N,
            new[] { rms ? 3u : 1u, 1u, (uint)a.Rows, (uint)a.N, 1u });
    }
}
