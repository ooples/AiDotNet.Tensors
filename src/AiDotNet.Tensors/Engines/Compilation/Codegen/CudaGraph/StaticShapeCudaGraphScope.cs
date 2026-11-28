// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;

/// <summary>
/// Categorizes the static-shape execution region managed by <see cref="StaticShapeCudaGraphScope"/>.
/// </summary>
public enum CudaGraphExecutionScopeKind : byte
{
    /// <summary>Deterministic forward-only inference pass over fixed-address input, weight, and output buffers.</summary>
    InferenceForward = 0,

    /// <summary>Full training step (forward, loss, backward, and in-place gradient/parameter work) over static buffers.</summary>
    FullStepForwardBackward = 1,

    /// <summary>Segmented local-credit or activation-checkpointed block step with isolated block-local buffers.</summary>
    SegmentedLocalCreditBlock = 2,

    /// <summary>Multi-tensor GPU optimizer list update over resident parameter, gradient, and moment buffers.</summary>
    MultiTensorOptimizerStep = 3
}

/// <summary>
/// Configuration options for <see cref="StaticShapeCudaGraphScope"/>.
/// </summary>
public sealed class StaticShapeCudaGraphOptions
{
    /// <summary>
    /// Number of eager warmup executions required with stable device pointers before stream capture runs.
    /// Defaults to 2 so lazy kernel compilation, cuBLAS workspaces, and pooled buffers stabilize first.
    /// </summary>
    public int WarmupIterations { get; init; } = 2;

    /// <summary>
    /// When true (default), replays synchronize the backend compute stream via <c>LaunchCapturedGraph</c>.
    /// When false, replays enqueue asynchronously via <c>EnqueueCapturedGraph</c> for multi-segment pipelining.
    /// </summary>
    public bool SynchronizeOnReplay { get; init; } = true;

    /// <summary>
    /// Maximum consecutive capture attempts that may fail for a static signature before staying on eager fallback.
    /// </summary>
    public int MaxCaptureFailuresBeforeFallback { get; init; } = 2;

    /// <summary>
    /// Declares that the step body has no host-side (managed) effects beyond its device work, so running it a second
    /// time after a failed capture is safe. Default <c>false</c>: when a capture fails AFTER the body already ran
    /// (the body threw mid-capture, or ending/instantiating the capture failed), the body's managed effects (for
    /// example an optimizer step counter) have been applied once while its device work was discarded. Re-running it
    /// would apply them twice, so the scope instead disables capture for that signature and throws
    /// <see cref="InvalidOperationException"/>, leaving the caller to decide how to recover. Set this to <c>true</c>
    /// for pure recompute bodies (outputs derived only from device inputs) to fall back to eager execution instead.
    /// </summary>
    public bool StepBodyIsReplayable { get; init; }

    /// <summary>Validates option values; throws <see cref="ArgumentOutOfRangeException"/> on an invalid setting.</summary>
    internal static void Validate(StaticShapeCudaGraphOptions options, string paramName)
    {
        if (options is null) throw new ArgumentNullException(paramName);
        if (options.WarmupIterations < 0)
            throw new ArgumentOutOfRangeException(paramName, "WarmupIterations must be non-negative.");
        if (options.MaxCaptureFailuresBeforeFallback < 1)
            throw new ArgumentOutOfRangeException(paramName, "MaxCaptureFailuresBeforeFallback must be at least 1.");
    }

    /// <summary>Default options singleton.</summary>
    public static StaticShapeCudaGraphOptions Default { get; } = new();
}

/// <summary>
/// Captures and replays static-shape GPU execution scopes (inference forward, full forward+backward step,
/// segmented local-credit block step, and multi-tensor optimizer list step) via CUDA Graphs
/// (<c>cuStreamBeginCapture</c> / <c>cuGraphInstantiate</c> / <c>cuGraphLaunch</c>) with automatic
/// shape, device-pointer, and scalar-hyperparameter invalidation guards.
/// </summary>
public sealed class StaticShapeCudaGraphScope : IDisposable
{
    private readonly struct BufferBindingSnapshot : IEquatable<BufferBindingSnapshot>
    {
        public readonly IntPtr Handle;
        public readonly int Size;

        public BufferBindingSnapshot(IntPtr handle, int size)
        {
            Handle = handle;
            Size = size;
        }

        public bool Equals(BufferBindingSnapshot other) => Handle == other.Handle && Size == other.Size;
        public override bool Equals(object? obj) => obj is BufferBindingSnapshot other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                return (Handle.GetHashCode() * 397) ^ Size;
            }
        }
    }

    private readonly IDirectGpuBackend _backend;
    private readonly IGpuGraphCapture? _captureBackend;
    private readonly StaticShapeCudaGraphOptions _options;
    private readonly object _sync = new();

    private IntPtr _graphExec;
    private BufferBindingSnapshot[] _capturedBindings = Array.Empty<BufferBindingSnapshot>();
    private long _capturedSignature;

    private BufferBindingSnapshot[] _warmupBindings = Array.Empty<BufferBindingSnapshot>();
    private long _warmupSignature;
    private int _warmupCompleted;
    private int _captureFailures;

    private long _warmupExecutionCount;
    private long _captureCount;
    private long _graphReplayCount;
    private long _eagerFallbackCount;
    private long _invalidationCount;
    private string? _lastInvalidationReason;
    private string? _lastCaptureFailure;
    private bool _disposed;

    /// <summary>Creates a guarded static-shape CUDA Graph replay scope.</summary>
    public StaticShapeCudaGraphScope(
        IDirectGpuBackend backend,
        CudaGraphExecutionScopeKind scopeKind,
        StaticShapeCudaGraphOptions? options = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        ScopeKind = scopeKind;
        _options = options ?? StaticShapeCudaGraphOptions.Default;
        StaticShapeCudaGraphOptions.Validate(_options, nameof(options));

        _captureBackend = backend as IGpuGraphCapture;
    }

    /// <summary>The execution region category managed by this scope.</summary>
    public CudaGraphExecutionScopeKind ScopeKind { get; }

    /// <summary>True when the underlying backend supports device graph capture and replay.</summary>
    public bool IsGraphCaptureSupported => _captureBackend is { SupportsGraphCapture: true, ContextIsLive: true };

    /// <summary>True when a valid instantiated CUDA Graph is currently cached for replay.</summary>
    public bool HasCapturedGraph
    {
        get
        {
            lock (_sync) return _graphExec != IntPtr.Zero;
        }
    }

    /// <summary>Total warmup executions performed prior to graph capture.</summary>
    public long WarmupExecutionCount => System.Threading.Interlocked.Read(ref _warmupExecutionCount);

    /// <summary>Total successful <c>cuStreamBeginCapture</c> + <c>cuGraphInstantiate</c> captures.</summary>
    public long CaptureCount => System.Threading.Interlocked.Read(ref _captureCount);

    /// <summary>Total <c>cuGraphLaunch</c> replays executed (including the initial launch following capture).</summary>
    public long GraphReplayCount => System.Threading.Interlocked.Read(ref _graphReplayCount);

    /// <summary>Total eager fallback executions when capture is unsupported, nested, or declined.</summary>
    public long EagerFallbackCount => System.Threading.Interlocked.Read(ref _eagerFallbackCount);

    /// <summary>Total automatic graph invalidations triggered by shape, pointer, or scalar changes.</summary>
    public long InvalidationCount => System.Threading.Interlocked.Read(ref _invalidationCount);

    /// <summary>Human-readable reason for the most recent automatic graph invalidation, or null.</summary>
    public string? LastInvalidationReason
    {
        get { lock (_sync) return _lastInvalidationReason; }
    }

    /// <summary>Diagnostic failure message from the most recent unsuccessful capture attempt, or null.</summary>
    public string? LastCaptureFailure
    {
        get { lock (_sync) return _lastCaptureFailure; }
    }

    /// <summary>
    /// Executes <paramref name="stepBody"/> under automatic CUDA Graph warmup, capture, replay, and invalidation guards
    /// watching <paramref name="watchedBuffers"/>.
    /// </summary>
    public void Execute(Action stepBody, IReadOnlyList<IGpuBuffer> watchedBuffers)
        => Execute(stepBody, watchedBuffers, 0L);

    /// <summary>
    /// Executes <paramref name="stepBody"/> under automatic CUDA Graph warmup, capture, replay, and invalidation guards
    /// watching <paramref name="watchedBuffers"/> and <paramref name="shapeAndScalarSignature"/>.
    /// </summary>
    public void Execute(
        Action stepBody,
        IReadOnlyList<IGpuBuffer> watchedBuffers,
        long shapeAndScalarSignature)
    {
        if (stepBody is null) throw new ArgumentNullException(nameof(stepBody));
        if (watchedBuffers is null) throw new ArgumentNullException(nameof(watchedBuffers));

        lock (_sync)
        {
            ThrowIfDisposed();

            if (!TrySnapshotBuffers(watchedBuffers, out var snapshot, out string? invalidBufferReason))
            {
                if (_graphExec != IntPtr.Zero)
                {
                    InvalidateLocked(invalidBufferReason ?? "A watched buffer is not device-resident.");
                }
                stepBody();
                System.Threading.Interlocked.Increment(ref _eagerFallbackCount);
                return;
            }

            // Eligibility first: a cached graph must not be replayed on a dead context or from inside another
            // capture on this thread (LaunchCapturedGraph / EnqueueCapturedGraph check neither).
            if (_captureBackend is null
                || !_captureBackend.SupportsGraphCapture
                || !_captureBackend.ContextIsLive
                || _captureBackend.CaptureActiveOnThisThread)
            {
                stepBody();
                System.Threading.Interlocked.Increment(ref _eagerFallbackCount);
                return;
            }

            if (_graphExec != IntPtr.Zero)
            {
                if (MatchesCapturedLocked(snapshot, shapeAndScalarSignature, out string? mismatchReason))
                {
                    ReplayLocked();
                    return;
                }

                InvalidateLocked(mismatchReason ?? "Watched buffer binding changed.");
            }

            // Reset failure counter if the caller transitioned to a new signature/pointer set after a failed capture.
            if (!BindingsEqual(_warmupBindings, snapshot) || _warmupSignature != shapeAndScalarSignature)
            {
                _warmupBindings = snapshot;
                _warmupSignature = shapeAndScalarSignature;
                _warmupCompleted = 0;
                _captureFailures = 0;
            }

            if (_captureFailures >= _options.MaxCaptureFailuresBeforeFallback)
            {
                stepBody();
                System.Threading.Interlocked.Increment(ref _eagerFallbackCount);
                return;
            }

            if (_warmupCompleted < _options.WarmupIterations)
            {
                stepBody();
                _warmupCompleted++;
                System.Threading.Interlocked.Increment(ref _warmupExecutionCount);
                return;
            }

            bool bodyInvoked = false;
            IntPtr captured = _captureBackend.CaptureGraph(() =>
            {
                bodyInvoked = true;
                stepBody();
            });
            if (captured == IntPtr.Zero)
            {
                _lastCaptureFailure = _captureBackend.LastCaptureFailure ?? "CaptureGraph returned IntPtr.Zero.";
                _captureFailures++;
                if (bodyInvoked && !_options.StepBodyIsReplayable)
                {
                    // The body already ran (its managed effects are applied) but its device work was discarded with
                    // the failed capture. Re-running it would apply those managed effects a second time, so stop
                    // capturing this signature and surface the failure instead of silently double-applying.
                    _captureFailures = Math.Max(_captureFailures, _options.MaxCaptureFailuresBeforeFallback);
                    throw new InvalidOperationException(
                        "CUDA graph capture failed after the step body had already run, so its device work was " +
                        "discarded while its host-side effects were applied once. The step was not re-run to avoid " +
                        "applying those effects twice; capture is disabled for this signature and later calls run " +
                        "eagerly. Set StaticShapeCudaGraphOptions.StepBodyIsReplayable for bodies without host-side " +
                        "effects. Capture failure: " + _lastCaptureFailure);
                }
                // Either the body never ran (begin-capture failed) or it is declared replayable: execute eagerly so
                // this call still completes its device work exactly once.
                stepBody();
                System.Threading.Interlocked.Increment(ref _eagerFallbackCount);
                return;
            }

            _lastCaptureFailure = null;
            _graphExec = captured;
            _capturedBindings = snapshot;
            _capturedSignature = shapeAndScalarSignature;
            System.Threading.Interlocked.Increment(ref _captureCount);

            // Stream capture records launches without executing them on the compute stream; launch immediately.
            ReplayLocked();
        }
    }

    /// <summary>
    /// Tensor-level overload that automatically extracts resident device buffer pointers, logical sizes, and tensor shapes.
    /// </summary>
    public void Execute<T>(Action stepBody, IReadOnlyList<Tensor<T>> watchedTensors)
        => Execute(stepBody, watchedTensors, 0L);

    /// <summary>
    /// Tensor-level overload that automatically extracts resident device buffer pointers, logical sizes, and tensor shapes.
    /// If any tensor is host-only (no resident GPU buffer), safely executes <paramref name="stepBody"/> eagerly.
    /// </summary>
    public void Execute<T>(
        Action stepBody,
        IReadOnlyList<Tensor<T>> watchedTensors,
        long scalarSignature)
    {
        if (stepBody is null) throw new ArgumentNullException(nameof(stepBody));
        if (watchedTensors is null) throw new ArgumentNullException(nameof(watchedTensors));

        long combinedSignature = ComputeSignature(watchedTensors, scalarSignature);
        var buffers = new IGpuBuffer[watchedTensors.Count];
        for (int i = 0; i < watchedTensors.Count; i++)
        {
            var tensor = watchedTensors[i];
            if (tensor is null)
                throw new ArgumentException($"Watched tensor at index {i} is null.", nameof(watchedTensors));
            IGpuBuffer? buf = tensor._gpuBuffer ?? tensor.TryGetGpuBuffer();
            // A resident buffer whose device copy is stale (a host write bumped the cache version) must not be
            // replayed: the captured graph would read the pre-write device data.
            bool deviceCurrent = buf is not null && tensor._gpuBufferVersion == tensor.GpuCacheVersion;
            if (buf is null || buf.Handle == IntPtr.Zero || !deviceCurrent)
            {
                lock (_sync)
                {
                    ThrowIfDisposed();
                    if (_graphExec != IntPtr.Zero)
                    {
                        InvalidateLocked(buf is null || buf.Handle == IntPtr.Zero
                            ? $"Watched tensor at index {i} is no longer GPU-resident."
                            : $"Watched tensor at index {i} has a stale device copy (host write since upload).");
                    }
                    stepBody();
                    System.Threading.Interlocked.Increment(ref _eagerFallbackCount);
                    return;
                }
            }
            buffers[i] = buf;
        }

        Execute(stepBody, buffers, combinedSignature);
    }

    /// <summary>Explicitly invalidates any cached CUDA Graph without disposing the scope.</summary>
    public void Invalidate(string reason = "Explicit caller invalidation.")
    {
        lock (_sync)
        {
            if (_disposed || _graphExec == IntPtr.Zero) return;
            InvalidateLocked(reason);
        }
    }

    /// <summary>
    /// Computes a deterministic 64-bit structural signature over tensor ranks, dimensions, lengths, and optional scalar seed.
    /// </summary>
    public static long ComputeSignature<T>(IReadOnlyList<Tensor<T>> tensors, long scalarSeed = 0)
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        unchecked
        {
            long hash = (long)14695981039346656037UL ^ scalarSeed;
            hash = (hash * 1099511628211L) ^ tensors.Count;
            for (int i = 0; i < tensors.Count; i++)
            {
                var t = tensors[i];
                if (t is null) continue;
                int[] dims = t._shape;
                hash = (hash * 1099511628211L) ^ dims.Length;
                for (int d = 0; d < dims.Length; d++)
                {
                    hash = (hash * 1099511628211L) ^ dims[d];
                }
            }
            return hash;
        }
    }

    /// <summary>
    /// Packs scalar optimizer/kernel hyperparameters into a deterministic 64-bit signature so changes to
    /// host-passed scalars (learning rate, weight decay, step counter, momentum) automatically invalidate stale graphs.
    /// </summary>
    public static long PackScalarSignature(
        float scalar0,
        float scalar1 = 0.0f,
        int intParam0 = 0,
        float scalar2 = 0.0f,
        float scalar3 = 0.0f)
    {
        unchecked
        {
            long h = 1469598103934665603L;
            h = (h * 1099511628211L) ^ FloatToInt32Bits(scalar0);
            h = (h * 1099511628211L) ^ FloatToInt32Bits(scalar1);
            h = (h * 1099511628211L) ^ intParam0;
            h = (h * 1099511628211L) ^ FloatToInt32Bits(scalar2);
            h = (h * 1099511628211L) ^ FloatToInt32Bits(scalar3);
            return h;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int FloatToInt32Bits(float value) => *(int*)&value;

    private void ReplayLocked()
    {
        if (_captureBackend is null || _graphExec == IntPtr.Zero)
            throw new InvalidOperationException("Cannot replay when no CUDA graph is instantiated.");

        if (_options.SynchronizeOnReplay)
        {
            _captureBackend.LaunchCapturedGraph(_graphExec);
        }
        else
        {
            _captureBackend.EnqueueCapturedGraph(_graphExec);
        }
        System.Threading.Interlocked.Increment(ref _graphReplayCount);
    }

    private void InvalidateLocked(string reason)
    {
        _lastInvalidationReason = reason;
        System.Threading.Interlocked.Increment(ref _invalidationCount);
        DestroyGraphLocked();
        _warmupCompleted = 0;
        _captureFailures = 0;
    }

    private void DestroyGraphLocked()
    {
        IntPtr exec = _graphExec;
        _graphExec = IntPtr.Zero;
        _capturedBindings = Array.Empty<BufferBindingSnapshot>();
        if (exec != IntPtr.Zero && _captureBackend is { ContextIsLive: true })
        {
            _captureBackend.DestroyCapturedGraph(exec);
        }
    }

    private static bool TrySnapshotBuffers(
        IReadOnlyList<IGpuBuffer> buffers,
        out BufferBindingSnapshot[] snapshot,
        out string? failureReason)
    {
        snapshot = new BufferBindingSnapshot[buffers.Count];
        for (int i = 0; i < buffers.Count; i++)
        {
            var buf = buffers[i];
            if (buf is null)
            {
                failureReason = $"Watched buffer at index {i} is null.";
                return false;
            }
            IntPtr ptr = buf.Handle;
            if (ptr == IntPtr.Zero)
            {
                failureReason = $"Watched buffer at index {i} has been released (Handle == 0).";
                return false;
            }
            snapshot[i] = new BufferBindingSnapshot(ptr, buf.Size);
        }
        failureReason = null;
        return true;
    }

    private bool MatchesCapturedLocked(
        BufferBindingSnapshot[] candidate,
        long signature,
        out string? mismatchReason)
    {
        if (_capturedSignature != signature)
        {
            mismatchReason = $"Shape or scalar signature changed ({_capturedSignature} -> {signature}).";
            return false;
        }
        if (_capturedBindings.Length != candidate.Length)
        {
            mismatchReason = $"Watched buffer count changed ({_capturedBindings.Length} -> {candidate.Length}).";
            return false;
        }
        for (int i = 0; i < candidate.Length; i++)
        {
            if (_capturedBindings[i].Handle != candidate[i].Handle)
            {
                mismatchReason = $"Device pointer changed at watched slot {i}.";
                return false;
            }
            if (_capturedBindings[i].Size != candidate[i].Size)
            {
                mismatchReason = $"Buffer logical element size changed at watched slot {i} ({_capturedBindings[i].Size} -> {candidate[i].Size}).";
                return false;
            }
        }
        mismatchReason = null;
        return true;
    }

    private static bool BindingsEqual(BufferBindingSnapshot[] a, BufferBindingSnapshot[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (!a[i].Equals(b[i])) return false;
        }
        return true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(StaticShapeCudaGraphScope));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            DestroyGraphLocked();
        }
        _ = _backend;
    }
}
