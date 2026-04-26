// Copyright (c) AiDotNet. All rights reserved.
// CUDA Graph capture of a training step — forward + backward +
// optimizer update recorded into a single cuGraph that replays
// without per-iteration launch overhead.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;

/// <summary>
/// Captures an entire training step — forward, backward, and
/// optimizer update — into a single CUDA graph that the runtime
/// replays without per-iteration kernel-launch overhead. Matches
/// <c>torch.cuda.make_graphed_callables</c>.
/// </summary>
/// <remarks>
/// <para><b>Determinism requirement:</b></para>
/// <para>
/// CUDA graph replay reads the same device pointers every step, so
/// the training loop must produce reproducible allocations.
/// <see cref="Helpers.TensorArena"/> already does this — its
/// Reset/TryAllocate pair returns identical pointer sequences
/// across iterations. If the caller's training step allocates
/// outside the arena (e.g. via raw <c>new Tensor&lt;T&gt;(…)</c>)
/// the graph capture will throw at instantiation time and the
/// fallback non-graphed path is used instead.
/// </para>
/// <para><b>Graph-safe RNG:</b></para>
/// <para>
/// Stochastic ops (Dropout, random init) must be seeded
/// deterministically per replay iteration or the graph will produce
/// identical outputs every step. The
/// <see cref="GraphedTrainingStepOptions.RngSeedOffsetPerReplay"/>
/// option advances the RNG seed by a fixed amount on each
/// <see cref="Replay"/> call — the caller is responsible for
/// threading this into the stochastic ops via
/// <see cref="AiDotNet.Tensors.Engines.Autodiff.GradientTape{T}"/>
/// options.
/// </para>
/// <para><b>Warmup requirement:</b></para>
/// <para>
/// The first captured iteration is a warmup — the arena learns its
/// pool sizes and the kernel launches stabilise into the shape the
/// graph needs. <see cref="Prepare"/> runs the warmup; only after
/// a successful warmup will <see cref="Capture"/> actually record
/// the graph.
/// </para>
/// </remarks>
public sealed class GraphedTrainingStep : IDisposable
{
    private readonly IDirectGpuBackend _backend;
    private readonly IntPtr _stream;
    private readonly Action _trainingStep;
    private readonly GraphedTrainingStepOptions _options;
    private CudaGraphScope? _scope;
    private bool _warmedUp;
    private bool _captured;
    private int _replayCount;
    private bool _disposed;

    /// <summary>
    /// Creates a new graphed training-step wrapper.
    /// </summary>
    /// <param name="backend">The GPU backend that will own the
    /// captured graph's lifetime.</param>
    /// <param name="stream">The CUDA stream the training step runs
    /// on. Must not be the default stream — capture refuses the
    /// default stream because other threads' launches would be
    /// captured too.</param>
    /// <param name="trainingStep">The user's training step closure
    /// — forward + loss + backward + optimizer update. Must only
    /// touch arena-allocated tensors so replay sees stable
    /// pointers.</param>
    /// <param name="options">Capture options (see type for details).
    /// Null uses defaults.</param>
    /// <exception cref="ArgumentNullException">Thrown when any
    /// required argument is null.</exception>
    public GraphedTrainingStep(
        IDirectGpuBackend backend,
        IntPtr stream,
        Action trainingStep,
        GraphedTrainingStepOptions? options = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (stream == IntPtr.Zero) throw new ArgumentException(
            "CUDA graph capture requires a non-default stream. Default-stream capture is rejected "
          + "because other threads may enqueue work on it during capture.", nameof(stream));
        _stream = stream;
        _trainingStep = trainingStep ?? throw new ArgumentNullException(nameof(trainingStep));
        _options = options ?? GraphedTrainingStepOptions.Default;
    }

    /// <summary>
    /// Runs warmup iterations. Must be called before
    /// <see cref="Capture"/>; the warmup fills the
    /// <see cref="Helpers.TensorArena"/> pool and lets any lazy
    /// tape-recording finish so the captured graph sees the steady-
    /// state launch sequence.
    /// </summary>
    public void Prepare()
    {
        ThrowIfDisposed();
        for (int i = 0; i < _options.WarmupIterations; i++)
        {
            _trainingStep();
        }
        _warmedUp = true;
    }

    /// <summary>
    /// Captures the training step into a CUDA graph. The capture
    /// executes the step once, recording every kernel launch into
    /// the graph; after <see cref="Capture"/> returns successfully,
    /// <see cref="Replay"/> can be called any number of times.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if
    /// <see cref="Prepare"/> hasn't been called first, or if CUDA
    /// Graph isn't supported by the backend.</exception>
    public void Capture()
    {
        ThrowIfDisposed();
        if (!_warmedUp) throw new InvalidOperationException(
            "GraphedTrainingStep.Prepare must be called before Capture so the arena pool " +
            "is warm and kernel-launch sequences have stabilised.");
        if (_captured) return;

        _scope = new CudaGraphScope(_backend, _stream);
        if (!_scope.IsSupported)
        {
            _scope.Dispose();
            _scope = null;
            // Honour the option: throw loudly by default, silent
            // no-op when the caller explicitly opted in to fallback
            // semantics. HasGraph stays false, so a subsequent
            // Replay() will throw with its existing
            // "Replay called before Capture" message — the contract
            // documented on GraphedTrainingStepOptions.ThrowOnUnsupported.
            if (_options.ThrowOnUnsupported)
            {
                throw new InvalidOperationException(
                    "CUDA Graph is not supported by this backend or driver version. " +
                    "Falls back to the non-graphed training step — the user should catch this " +
                    "exception and revert to calling their trainingStep directly.");
            }
            return;
        }

        _scope.BeginCapture();
        try
        {
            _trainingStep();
        }
        finally
        {
            _scope.EndCapture();
        }
        _captured = true;
    }

    /// <summary>
    /// Replays the captured graph. Throws
    /// <see cref="InvalidOperationException"/> if <see cref="Capture"/>
    /// has not been called successfully — the caller should check
    /// <see cref="HasGraph"/> first when running on a host where
    /// CUDA Graph may be unavailable (e.g. when
    /// <see cref="GraphedTrainingStepOptions.ThrowOnUnsupported"/>
    /// is <c>false</c> and capture silently no-op'd). Returns the
    /// replay index (starting at 0) so the caller can thread it
    /// into an RNG offset if needed.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if
    /// <see cref="Capture"/> has not been called or did not produce
    /// a graph.</exception>
    /// <exception cref="ObjectDisposedException">Thrown if the
    /// instance has been disposed.</exception>
    public int Replay()
    {
        ThrowIfDisposed();
        if (!_captured || _scope is null)
            throw new InvalidOperationException(
                "Replay called before Capture. Call Prepare + Capture first.");
        _scope.Replay();
        return _replayCount++;
    }

    /// <summary>True once <see cref="Capture"/> succeeded.</summary>
    public bool HasGraph => _captured;

    /// <summary>How many times <see cref="Replay"/> has been called.</summary>
    public int ReplayCount => _replayCount;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _scope?.Dispose();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GraphedTrainingStep));
    }
}

/// <summary>
/// Configuration for <see cref="GraphedTrainingStep"/>.
/// </summary>
public sealed class GraphedTrainingStepOptions
{
    /// <summary>
    /// Number of warmup iterations to run before capture. Warmup
    /// fills the TensorArena pool and lets autograd tape recording
    /// stabilise. Default: 2 (one for arena pool fill, one to
    /// observe steady-state behaviour).
    /// </summary>
    public int WarmupIterations { get; init; } = 2;

    /// <summary>
    /// Amount added to the RNG seed on each <see cref="GraphedTrainingStep.Replay"/>
    /// call. Zero means "same RNG every replay" — only safe when
    /// the training step contains no stochastic ops. Default: 1.
    /// </summary>
    public long RngSeedOffsetPerReplay { get; init; } = 1;

    /// <summary>
    /// Whether to throw <see cref="InvalidOperationException"/> if
    /// CUDA Graph isn't supported (true, default) or to silently
    /// no-op <see cref="GraphedTrainingStep.Capture"/> (false —
    /// caller falls back to the non-graphed step).
    /// </summary>
    public bool ThrowOnUnsupported { get; init; } = true;

    /// <summary>Default options.</summary>
    public static GraphedTrainingStepOptions Default { get; } = new();
}
