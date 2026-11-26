// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

// Loss scaling (PyTorch's GradScaler) inside the plan: the loss-gradient seed holds the scale, so the backward is scaled
// with no extra op; the gradients are unscaled (and checked for NaN/Inf) right after the backward, before the L2 term,
// the clip and the optimizer, which then see exactly the gradients of an unscaled step; a non-finite step is discarded by
// the plan's existing finiteness guard. On CUDA the unscale, the check and the scale update are device kernels -- inside
// the captured step graph when there is one -- so a scaled step reads nothing back. Elsewhere they run on the host arrays.
internal sealed partial class CompiledTrainingPlan<T> : ICompiledTrainingPlanLossScaling
{
    private LossScalingOptions? _lossScaling;
    private LossScaleState _lossScaleHost;
    private Engines.DirectGpu.IDeviceGradScaler? _lossScaleDevice;
    // True while the device scaler holds the current state (the host copy is stale).
    private bool _lossScaleOnDevice;
    private readonly List<Engines.DirectGpu.IGpuBuffer> _lossScaleBuffers = new();
    private readonly List<int> _lossScaleSizes = new();

    /// <inheritdoc />
    public void ConfigureLossScaling(LossScalingOptions? options, LossScaleState? initialState = null)
    {
        if (initialState is { } given && (!(given.Scale > 0f) || float.IsInfinity(given.Scale)))
            throw new ArgumentOutOfRangeException(nameof(initialState),
                $"A loss scale state to resume from needs a finite positive scale, not {given.Scale}.");
        if (options is not null)
        {
            if (typeof(T) != typeof(float))
                throw new NotSupportedException($"Loss scaling needs a float plan, not {typeof(T).Name}.");
            options.Validate();
            if (initialState is { } resumed && (resumed.Scale < options.MinScale || resumed.Scale > options.MaxScale))
                throw new ArgumentOutOfRangeException(nameof(initialState),
                    $"The loss scale state to resume from has scale {resumed.Scale}, outside the options' "
                    + $"[MinScale, MaxScale] = [{options.MinScale}, {options.MaxScale}].");
            RequireDiscardingNonFinitePolicy(_nonFiniteGradientPolicy);
        }
        lock (_stepSync)
        {
            DisposeLossScaleDevice();
            _lossScaling = options is null ? null : new LossScalingOptions
            {
                InitialScale = options.InitialScale, GrowthFactor = options.GrowthFactor,
                BackoffFactor = options.BackoffFactor, GrowthInterval = options.GrowthInterval,
                MinScale = options.MinScale, MaxScale = options.MaxScale, Dynamic = options.Dynamic,
            };
            _lossScaleHost = initialState ?? new LossScaleState(options?.InitialScale ?? 1f, 0, 0, 0);
            // A step graph captured before this records the previous seed handling: capture again.
            if (_stepGraphExec != IntPtr.Zero) InvalidateCapturedStepGraph();
        }
    }

    /// <inheritdoc />
    public LossScaleState? ReadLossScaleState()
    {
        if (_lossScaling is null) return null;
        lock (_stepSync)
        {
            if (_lossScaleOnDevice && _lossScaleDevice is { } device)
            {
                using var __gpuOp = (_engine.As<Engines.DirectGpuTensorEngine>())?.EnterGpuOp() ?? default;
                return device.Read();
            }
            return _lossScaleHost;
        }
    }

    private static void RequireDiscardingNonFinitePolicy(Engines.Gpu.NonFiniteGradientPolicy policy)
    {
        if (policy is not (Engines.Gpu.NonFiniteGradientPolicy.LeaveUnscaled or Engines.Gpu.NonFiniteGradientPolicy.Propagate))
            throw new InvalidOperationException(
                $"Loss scaling skips a step whose gradients overflowed, which the non-finite gradient policy {policy} does not "
                + "do (it repairs or raises on them). Use LeaveUnscaled or Propagate.");
    }

    private void DisposeLossScaleDevice()
    {
        if (_lossScaleDevice is { } device)
        {
            if (_lossScaleOnDevice)
            {
                try { _lossScaleHost = device.Read(); }
                catch (Exception ex) when (ex is InvalidOperationException or ObjectDisposedException)
                {
                    // The device is gone (the backend was disposed first): the host copy keeps the last state read.
                    System.Diagnostics.Trace.TraceWarning(
                        $"The device loss scale could not be read back before its scaler was released ({ex.GetType().Name}); "
                        + $"the plan keeps scale {_lossScaleHost.Scale} from its last read.");
                }
            }
            device.Dispose();
        }
        _lossScaleDevice = null;
        _lossScaleOnDevice = false;
    }

    private void MoveLossScaleToHost()
    {
        if (!_lossScaleOnDevice || _lossScaleDevice is not { } device) return;
        _lossScaleHost = device.Read();
        _lossScaleOnDevice = false;
    }

    /// <summary>The scale the host-side seed is filled with (the eager step), brought back from the device first.</summary>
    private float HostLossScaleForSeed()
    {
        MoveLossScaleToHost();
        return _lossScaleHost.Scale;
    }

    /// <summary>
    /// Writes the loss-gradient seed on the device for an uncaptured resident step: 1, or the current scale. A device
    /// scaler that holds the state writes its own scale, so the step reads nothing back.
    /// </summary>
    private void FillLossGradientSeedOnDevice(Engines.DirectGpu.IDirectGpuBackend backend, Engines.DirectGpu.IGpuBuffer seed,
        int length)
    {
        if (_lossScaling is null) { backend.Fill(seed, 1f, length); return; }
        if (_lossScaleOnDevice && _lossScaleDevice is { } device) device.WriteSeed(seed, length);
        else backend.Fill(seed, _lossScaleHost.Scale, length);
    }

    // The distinct device buffers of the gradients the clip and the optimizer read, all current on one backend, or false.
    private bool TryCollectLossScaleBuffers(bool capturing, out Engines.DirectGpu.IDirectGpuBackend? backend)
    {
        backend = null;
        _lossScaleBuffers.Clear();
        _lossScaleSizes.Clear();
        if (typeof(T) != typeof(float)) return false;
        var seen = new HashSet<IntPtr>();
        foreach (var grad in ActiveOptimizerGradients())
        {
            if (grad is null || grad.Length == 0) continue;
            if (grad.TryGetGpuBuffer() is not { } buffer
                || !(capturing || grad._gpuBufferVersion == grad.GpuCacheVersion)
                || grad._gpuBackend is not Engines.DirectGpu.IDirectGpuBackend owner
                || !grad.IsContiguous)
                return false;
            if (backend is null) backend = owner;
            else if (!ReferenceEquals(backend, owner)) return false;
            if (seen.Add(buffer.Handle))
            {
                _lossScaleBuffers.Add(buffer);
                _lossScaleSizes.Add(grad.Length);
            }
        }
        return backend is not null;
    }

    private string DescribeUncollectableGradient()
    {
        var grads = ActiveOptimizerGradients();
        for (int i = 0; i < grads.Length; i++)
        {
            var grad = grads[i];
            if (grad is null || grad.Length == 0) continue;
            if (grad.TryGetGpuBuffer() is null) return $"gradient {i} [{string.Join(",", grad._shape)}] has no device buffer";
            if (grad._gpuBackend is not Engines.DirectGpu.IDirectGpuBackend) return $"gradient {i} is not on a direct GPU backend";
            if (!grad.IsContiguous) return $"gradient {i} is not contiguous";
        }
        return "the gradients are on more than one backend";
    }

    // The device scaler over the current gradient buffers, holding the current state; null when there is none.
    private Engines.DirectGpu.IDeviceGradScaler? EnsureLossScaleDevice(Engines.DirectGpu.IDeviceGradScalerFactory factory)
    {
        var options = _lossScaling ?? throw new InvalidOperationException("Loss scaling is off.");
        if (_lossScaleDevice is { } existing && existing.Covers(_lossScaleBuffers, _lossScaleSizes))
        {
            if (!_lossScaleOnDevice)
            {
                existing.Write(_lossScaleHost);
                _lossScaleOnDevice = true;
            }
            return existing;
        }
        MoveLossScaleToHost();
        _lossScaleDevice?.Dispose();
        _lossScaleDevice = factory.CreateGradScaler(_lossScaleBuffers, _lossScaleSizes,
            new Engines.DirectGpu.DeviceGradScalerSettings(options.GrowthFactor, options.BackoffFactor,
                options.GrowthInterval, options.MinScale, options.MaxScale, options.Dynamic),
            _lossScaleHost);
        _lossScaleOnDevice = _lossScaleDevice is not null;
        return _lossScaleDevice;
    }

    /// <summary>
    /// Before the step graph is captured (outside the capture): the device scaler over the gradients the graph writes,
    /// holding the current state, and the seed set to its scale. Throws when the graph cannot carry the scaling, so the
    /// capture is abandoned and the plan keeps the uncaptured steps, which scale on their own. With a gradient exchange
    /// the scaler may never enter a graph (an update that runs outside it unscales after the exchange, uncaptured), so a
    /// scaler that cannot be prepared there does not abandon the capture: the update graph's capture refuses it if it
    /// does record the scaler (<see cref="CaptureLossScaling"/>).
    /// </summary>
    private void PrepareLossScalingForCapture()
    {
        if (_lossScaling is null) return;
        if (!TryCollectLossScaleBuffers(capturing: false, out var backend)
            || backend is not Engines.DirectGpu.IDeviceGradScalerFactory factory
            || EnsureLossScaleDevice(factory) is not { } device
            || _lossGradSeed.TryGetGpuBuffer() is not { } seed)
        {
            if (_gradientExchange is not null) return;
            throw new InvalidOperationException(
                "Loss scaling needs every trained gradient and the loss-gradient seed on one CUDA device to run inside the "
                + "step graph.");
        }
        device.WriteSeed(seed, _lossGradSeed.Length);
        _lossScalePreparedHandles = DescribeGradientBuffers();
    }

    // The gradient buffers the scaler was prepared over, for the capture's mismatch message.
    private string[]? _lossScalePreparedHandles;

    private string[] DescribeGradientBuffers()
        => ActiveOptimizerGradients().Select(g => g is null || g.Length == 0 ? "-"
            : $"[{string.Join(",", g._shape)}]@{g.TryGetGpuBuffer()?.Handle.ToInt64():x}").ToArray();

    private string DescribeMovedGradientBuffers()
    {
        var now = DescribeGradientBuffers();
        var prepared = _lossScalePreparedHandles ?? Array.Empty<string>();
        var moved = new List<string>();
        for (int i = 0; i < now.Length; i++)
            if (i >= prepared.Length || prepared[i] != now[i])
                moved.Add($"gradient {i} {(i < prepared.Length ? prepared[i] : "?")} -> {now[i]}");
        return moved.Count == 0 ? "the same buffers, other sizes" : string.Join("; ", moved);
    }

    /// <summary>Records the unscale and the scale update into the step graph being captured, after its backward.</summary>
    private void CaptureLossScaling()
    {
        if (_lossScaling is null) return;
        string? unprepared =
            _lossScaleDevice is null ? "there is no device scaler"
            : !_lossScaleOnDevice ? "the device scaler does not hold the current scale"
            : !TryCollectLossScaleBuffers(capturing: true, out _) ? DescribeUncollectableGradient()
            : !_lossScaleDevice.Covers(_lossScaleBuffers, _lossScaleSizes) ? "the device scaler covers other gradient buffers ("
                + DescribeMovedGradientBuffers() + ")"
            : _lossGradSeed.TryGetGpuBuffer() is null ? "the loss-gradient seed is not on the device"
            : null;
        if (unprepared is not null)
            throw new InvalidOperationException($"The loss scaler was not prepared for the gradients this graph captures: {unprepared}.");
        var device = _lossScaleDevice!;
        var seed = _lossGradSeed.TryGetGpuBuffer()!;
        device.UnscaleAndUpdate(seed, _lossGradSeed.Length);
    }

    /// <summary>
    /// A captured step with a gradient exchange whose update runs outside the graph: the graph left the gradients scaled,
    /// so they are unscaled (and the scale moved) here, after the exchange, from the reduced gradient every rank holds
    /// alike. Without an exchange the graph already did it.
    /// </summary>
    private void UnscaleAfterExchange()
    {
        if (_gradientExchange is not null) UnscaleLossScaledGradients();
    }

    // Set when a step unscaled on a device without a device scaler: the plan's finiteness guard decides that step, and
    // CompleteLossScaledStep updates the scale from its verdict after the optimizer update.
    private bool _lossScaleVerdictPending;

    /// <summary>
    /// An uncaptured step's unscale, after its backward and before the L2 term, the clip and the optimizer: on the device
    /// when every gradient is current there (the resident step), else on the host arrays (the eager step).
    /// </summary>
    private void UnscaleLossScaledGradients()
    {
        if (_lossScaling is null) return;
        using var __gpuOp = (_engine.As<Engines.DirectGpuTensorEngine>())?.EnterGpuOp() ?? default;
        if (TryCollectLossScaleBuffers(capturing: false, out var backend) && backend is not null)
        {
            if (backend is Engines.DirectGpu.IDeviceGradScalerFactory factory
                && _lossGradSeed.TryGetGpuBuffer() is { } seed
                && EnsureLossScaleDevice(factory) is { } device)
            {
                device.UnscaleAndUpdate(seed, _lossGradSeed.Length);
                return;
            }
            // Every other GPU backend: unscale each gradient on the device. These backends read the finiteness guard on
            // the host as part of the update, so the scale follows that verdict (CompleteLossScaledStep).
            MoveLossScaleToHost();
            float inverse = 1f / _lossScaleHost.Scale;
            for (int i = 0; i < _lossScaleBuffers.Count; i++)
                backend.Scale(_lossScaleBuffers[i], _lossScaleBuffers[i], inverse, _lossScaleSizes[i]);
            _lossScaleVerdictPending = true;
            return;
        }
        UnscaleLossScaledGradientsOnHost();
    }

    /// <summary>After the optimizer update of a step <see cref="UnscaleLossScaledGradients"/> unscaled on a device with no
    /// device scaler: advance the scale from the finiteness verdict of the plan for that step.</summary>
    private void CompleteLossScaledStep()
    {
        if (!_lossScaleVerdictPending) return;
        _lossScaleVerdictPending = false;
        ResolveDeferredFiniteness();
        AdvanceHostLossScale(LastStepSkippedNonFiniteGradients);
    }

    private void AdvanceHostLossScale(bool nonFinite)
    {
        var options = _lossScaling ?? throw new InvalidOperationException("Loss scaling is off.");
        var state = _lossScaleHost;
        float scale = state.Scale;
        int tracker = state.GrowthTracker;
        long overflows = state.OverflowSteps;
        if (nonFinite)
        {
            overflows++;
            tracker = 0;
            if (options.Dynamic) scale = Math.Max(scale * options.BackoffFactor, options.MinScale);
        }
        else
        {
            tracker++;
            if (options.Dynamic && tracker >= options.GrowthInterval)
            {
                scale = Math.Min(scale * options.GrowthFactor, options.MaxScale);
                tracker = 0;
            }
        }
        _lossScaleHost = new LossScaleState(scale, tracker, overflows, state.Steps + 1);
    }

    private void UnscaleLossScaledGradientsOnHost()
    {
        MoveLossScaleToHost();
        float inverse = 1f / _lossScaleHost.Scale;
        bool nonFinite = false;
        // By the memory a gradient covers, as the device path dedupes by buffer: two tensors over exactly the same elements
        // (a tied weight's gradient, a view over a gradient's storage) are unscaled once. Optimizer gradients never
        // partially overlap (aliased views are of intermediate gradients, which this does not visit).
        var seen = new HashSet<(object Storage, int Offset, int Length)>(new StorageRangeComparer());
        foreach (var grad in ActiveOptimizerGradients())
        {
            if (grad is null || grad.Length == 0 || !seen.Add((grad.StorageIdentity, grad.StorageOffset, grad.Length))) continue;
            var typed = (Tensor<float>)(object)grad;
            using (var lease = typed.LeaseWritable())
            {
                var span = lease.Span;
                for (int i = 0; i < span.Length; i++)
                {
                    float v = span[i] * inverse;
                    span[i] = v;
                    nonFinite |= float.IsNaN(v) || float.IsInfinity(v);
                }
            }
            // The host write leaves a device copy stale; the optimizer must not read the scaled values there.
            if (grad._gpuBuffer is not null) grad.IncrementVersion();
        }
        AdvanceHostLossScale(nonFinite);
    }

    private sealed class StorageRangeComparer : IEqualityComparer<(object Storage, int Offset, int Length)>
    {
        public bool Equals((object Storage, int Offset, int Length) x, (object Storage, int Offset, int Length) y)
            => ReferenceEquals(x.Storage, y.Storage) && x.Offset == y.Offset && x.Length == y.Length;

        public int GetHashCode((object Storage, int Offset, int Length) key)
            => unchecked((System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(key.Storage) * 397) ^ key.Offset ^ (key.Length << 7));
    }
}
