// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;

/// <summary>
/// Captures a model's forward pass into a CUDA graph for zero-launch-overhead inference replay —
/// the inference counterpart to <see cref="GraphedTrainingStep"/> (#1630 / #91). On the
/// foundation-model decode path a single forward dispatches hundreds of small kernels; at low
/// batch the per-launch overhead dominates, and CUDA graphs collapse the whole launch sequence
/// into one <c>cuGraphLaunch</c>. This is the standard low-latency-serving lever (TensorRT-LLM /
/// vLLM both capture decode graphs).
/// <para>
/// <b>Static-buffer contract.</b> CUDA-graph replay re-runs the recorded kernels against the
/// EXACT device addresses captured. So the <paramref name="forward"/> closure must read its input
/// from, and write its output to, FIXED device buffers (arena/persistent tensors whose pointers
/// don't move between replays) — never freshly-allocated tensors. To run inference on new data,
/// copy it INTO the captured input buffer (a memcpy on the same stream, outside the recorded
/// graph) and then call <see cref="Replay()"/> / <see cref="Replay(Action)"/>; read the result
/// from the captured output buffer. Shapes are fixed per capture — capture one
/// <see cref="GraphedInferenceStep"/> per inference shape.
/// </para>
/// <para>
/// <b>Determinism.</b> Unlike the training step (which offsets the RNG per replay), inference must
/// be deterministic — the forward closure must contain no stochastic ops (Dropout/GaussianNoise
/// are disabled in inference mode), so replay reproduces the captured computation exactly.
/// </para>
/// </summary>
public sealed class GraphedInferenceStep : IDisposable
{
    private readonly IDirectGpuBackend _backend;
    private readonly IntPtr _stream;
    private readonly Action _forward;
    private readonly GraphedInferenceStepOptions _options;
    private CudaGraphScope? _scope;
    private bool _warmedUp;
    private bool _captured;
    private int _replayCount;
    private bool _disposed;

    /// <summary>Creates a graphed inference-forward wrapper.</summary>
    /// <param name="backend">The GPU backend that owns the captured graph's lifetime.</param>
    /// <param name="stream">A non-default CUDA stream the forward runs on (default-stream capture
    /// is rejected — other threads' launches would be captured too).</param>
    /// <param name="forward">The model's forward closure. Must read from / write to FIXED device
    /// buffers (see the static-buffer contract on the type) so replay sees stable pointers.</param>
    /// <param name="options">Capture options; null uses defaults.</param>
    public GraphedInferenceStep(
        IDirectGpuBackend backend,
        IntPtr stream,
        Action forward,
        GraphedInferenceStepOptions? options = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (stream == IntPtr.Zero) throw new ArgumentException(
            "CUDA graph capture requires a non-default stream. Default-stream capture is rejected "
          + "because other threads may enqueue work on it during capture.", nameof(stream));
        _stream = stream;
        _forward = forward ?? throw new ArgumentNullException(nameof(forward));
        _options = options ?? GraphedInferenceStepOptions.Default;
    }

    /// <summary>True once <see cref="Capture"/> succeeded.</summary>
    public bool HasGraph => _captured;

    /// <summary>How many times <see cref="Replay()"/> has been called.</summary>
    public int ReplayCount => _replayCount;

    /// <summary>
    /// Runs the forward <see cref="GraphedInferenceStepOptions.WarmupIterations"/> times before
    /// capture, so lazy allocations + the arena pool reach the steady-state launch sequence the
    /// graph will record. Must be called before <see cref="Capture"/>.
    /// </summary>
    public void Prepare()
    {
        ThrowIfDisposed();
        for (int i = 0; i < _options.WarmupIterations; i++)
            _forward();
        _warmedUp = true;
    }

    /// <summary>
    /// Captures the forward pass into a CUDA graph (one warm replay records every kernel launch).
    /// After it returns successfully, <see cref="Replay()"/> can be called any number of times.
    /// Honours <see cref="GraphedInferenceStepOptions.ThrowOnUnsupported"/>: throws on a
    /// no-CUDA-graph backend by default, or silently no-ops (leaving <see cref="HasGraph"/> false)
    /// so the caller can fall back to the eager forward.
    /// </summary>
    public void Capture()
    {
        ThrowIfDisposed();
        if (!_warmedUp) throw new InvalidOperationException(
            "GraphedInferenceStep.Prepare must be called before Capture so allocations are warm "
          + "and the kernel-launch sequence has stabilised.");
        if (_captured) return;

        // Tear down any abandoned scope from a prior Capture that threw partway through, so the
        // new BeginCapture runs against a clean slate (no leaked native graph).
        if (_scope is not null)
        {
            _scope.Dispose();
            _scope = null;
        }

        var scope = new CudaGraphScope(_backend, _stream);
        if (!scope.IsSupported)
        {
            scope.Dispose();
            if (_options.ThrowOnUnsupported)
                throw new InvalidOperationException(
                    "CUDA Graph is not supported by this backend or driver version. Catch this and "
                  + "fall back to calling the forward directly (eager inference).");
            return; // HasGraph stays false → Replay throws its "before Capture" contract message.
        }

        // Publish the scope only after a fully successful capture; on any throw, dispose the local
        // and leave _scope null / _captured false so a retry sees a clean slate.
        try
        {
            scope.BeginCapture();
            try { _forward(); }
            finally { scope.EndCapture(); }
            _scope = scope;
            _captured = true;
        }
        catch
        {
            scope.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Replays the captured forward with zero per-kernel launch overhead. The caller must have
    /// already copied the current input into the captured input buffer (see the static-buffer
    /// contract); the result lands in the captured output buffer. Throws if
    /// <see cref="Capture"/> hasn't succeeded — check <see cref="HasGraph"/> when CUDA graph may
    /// be unavailable.
    /// </summary>
    public int Replay()
    {
        ThrowIfDisposed();
        if (!_captured || _scope is null)
            throw new InvalidOperationException(
                "Replay called before Capture. Call Prepare + Capture first.");
        _scope.Replay();
        return _replayCount++;
    }

    /// <summary>
    /// Convenience replay that runs <paramref name="rebindInputs"/> (copy fresh data into the
    /// captured input buffer) immediately before replaying — making the "update-then-replay"
    /// inference contract explicit and atomic at the call site.
    /// </summary>
    public int Replay(Action rebindInputs)
    {
        ThrowIfDisposed();
        if (rebindInputs is null) throw new ArgumentNullException(nameof(rebindInputs));
        if (!_captured || _scope is null)
            throw new InvalidOperationException(
                "Replay called before Capture. Call Prepare + Capture first.");
        rebindInputs();
        _scope.Replay();
        return _replayCount++;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _scope?.Dispose();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GraphedInferenceStep));
    }
}

/// <summary>Configuration for <see cref="GraphedInferenceStep"/>.</summary>
public sealed class GraphedInferenceStepOptions
{
    /// <summary>Warmup forwards run before capture so lazy allocations + the arena pool reach
    /// steady state. Default 2 (one to fill the pool, one to observe steady-state launches).</summary>
    public int WarmupIterations { get; init; } = 2;

    /// <summary>Throw if CUDA graph is unsupported (true, default) vs. silently no-op
    /// <see cref="GraphedInferenceStep.Capture"/> so the caller falls back to eager inference.</summary>
    public bool ThrowOnUnsupported { get; init; } = true;

    /// <summary>Default options.</summary>
    public static GraphedInferenceStepOptions Default { get; } = new();
}
