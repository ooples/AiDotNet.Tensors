using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>A lexical handoff from one successful training invocation to its consumer.</summary>
internal sealed class RetainedOutputCapture<T> : IDisposable
{
    // Intentional per-thread lexical stacks: Begin pushes; Complete/Dispose restore the parent only in nesting order.
    // These are ambient scope bindings, not process-wide shared capture state.
    [ThreadStatic] private static RetainedOutputCapture<T>? _current;
    [ThreadStatic] private static EagerUpdateScope? _activeUpdate;
    private static readonly ConditionalWeakTable<GradientTape<T>, EagerTapeRegistration> EagerRegistrations = new();

    private sealed class EagerTapeRegistration
    {
        internal long Epoch = -1;
        internal readonly HashSet<string> Keys = new(StringComparer.Ordinal);
    }
    private readonly RetainedOutputCapture<T>? _parent;
    private readonly Dictionary<GradientTape<T>, Dictionary<string, Tensor<T>>> _eager = new();
    private IReadOnlyDictionary<string, Tensor<T>>? _published;
    private CompiledTrainingPlan<T>? _plan;
    private long _planStep;
    private bool _completed;
    private bool _disposed;

    private RetainedOutputCapture()
    {
        _parent = _current;
        _current = this;
    }

    internal static RetainedOutputCapture<T> Begin() => new();
    internal static bool IsActive => _current is not null;

    internal static void RequireReportingUpdater(bool reportsOutcome)
    {
        if (_current is not null && !reportsOutcome)
            throw new NotSupportedException("A retained output handoff requires an updater that explicitly reports acceptance or rejection.");
    }

    internal static EagerUpdateScope? BeginEagerUpdate() => _current is { } capture ? new EagerUpdateScope(capture) : null;
    internal static void MarkUpdateAccepted()
    {
        if (_activeUpdate?.BelongsTo(_current) == true) _activeUpdate.Accept();
    }
    internal static void MarkUpdateRejected()
    {
        if (_activeUpdate?.BelongsTo(_current) == true) _activeUpdate.Reject();
    }

    /// <summary>Acceptance is explicit; returning from a void callback is not proof that it updated.</summary>
    internal sealed class EagerUpdateScope : IDisposable
    {
        private readonly RetainedOutputCapture<T> _owner;
        private readonly EagerUpdateScope? _parent;
        private bool _accepted;
        private bool _rejected;
        private bool _disposed;
        internal bool Accepted => _accepted && !_rejected;
        internal bool BelongsTo(RetainedOutputCapture<T>? owner) => ReferenceEquals(_owner, owner);
        internal EagerUpdateScope(RetainedOutputCapture<T> owner)
        {
            _owner = owner;
            _parent = _activeUpdate;
            _activeUpdate = this;
            owner._published = null;
            owner._plan = null;
        }
        internal void Accept() { if (!_rejected) _accepted = true; }
        internal void Reject() { _rejected = true; }
        public void Dispose()
        {
            if (_disposed) return;
            if (!ReferenceEquals(_activeUpdate, this) || !ReferenceEquals(_current, _owner))
                throw new InvalidOperationException("Retained output update scopes must be disposed in nesting order on their creating thread.");
            _disposed = true;
            if (!Accepted)
            {
                _owner._published = null;
                _owner._plan = null;
            }
            _activeUpdate = _parent;
        }
    }

    internal static void RejectUnsupportedRoute(string route)
    {
        if (_current is not null)
            throw new NotSupportedException($"Retained output handoffs are not supported by {route} training steps.");
    }

    internal static void RegisterEager(GradientTape<T> tape, string key, Tensor<T> output)
    {
        var state = EagerRegistrations.GetValue(tape, static _ => new EagerTapeRegistration());
        if (state.Epoch != tape.RecordingEpoch)
        {
            state.Epoch = tape.RecordingEpoch;
            state.Keys.Clear();
            for (var stale = _current; stale is not null; stale = stale._parent)
                stale._eager.Remove(tape);
        }
        if (!state.Keys.Add(key))
            throw new InvalidOperationException($"The retained output key '{key}' is already registered in this forward.");
        if (_current is not { } capture) return;
        if (!capture._eager.TryGetValue(tape, out var outputs))
            capture._eager.Add(tape, outputs = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal));
        outputs.Add(key, output);
    }

    internal static void DiscardEagerRecording(GradientTape<T> tape)
    {
        if (EagerRegistrations.TryGetValue(tape, out var state))
        {
            state.Epoch = -1;
            state.Keys.Clear();
        }
        for (var capture = _current; capture is not null; capture = capture._parent)
            capture._eager.Remove(tape);
    }

    // Only a completed update marks a tape as the producer. Shape warm-ups do not update and cannot publish.
    internal static void PublishEager(GradientTape<T> tape)
    {
        if (_current is not { } capture) return;
        // Destructive backward advances the epoch before the accepted update publishes its outputs.
        // Older registrations belong to a previous forward, even if this forward retained no keys.
        long epoch = tape.RecordingEpoch;
        bool eligible = EagerRegistrations.TryGetValue(tape, out var registration) && registration.Epoch >= 0
            && (registration.Epoch == epoch || registration.Epoch == epoch - 1);
        if (!eligible) capture._eager.Remove(tape);
        capture._published = eligible && capture._eager.TryGetValue(tape, out var outputs)
            ? new Dictionary<string, Tensor<T>>(outputs, StringComparer.Ordinal)
            : new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
        capture._plan = null;
    }

    internal static void PublishCompiled(CompiledTrainingPlan<T> plan,
        IReadOnlyDictionary<string, Tensor<T>> outputs, long step)
    {
        if (_current is not { } capture) return;
        // Replace the entire map: a missing output in this step cannot reuse an earlier forward's binding.
        capture._published = outputs;
        capture._plan = plan;
        capture._planStep = step;
    }

    /// <summary>Called only after the enclosing training invocation and its validation have succeeded.</summary>
    internal void Complete()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(RetainedOutputCapture<T>));
        if (_completed || !ReferenceEquals(_current, this))
            throw new InvalidOperationException("Retained output captures must complete once, in nesting order.");
        if (_activeUpdate?.BelongsTo(this) == true)
            throw new InvalidOperationException("The producer's update callback has not completed.");
        if (_published is null)
            throw new InvalidOperationException("No successful training step published retained outputs.");
        _completed = true;
        _current = _parent;
        _eager.Clear();
    }

    internal Tensor<T> Get(string key)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(RetainedOutputCapture<T>));
        if (!_completed) throw new InvalidOperationException("The enclosing training invocation has not completed.");
        if (string.IsNullOrEmpty(key)) throw new ArgumentException("A retained output key must be nonempty.", nameof(key));
        if (_published is null || !_published.TryGetValue(key, out var output))
            throw new KeyNotFoundException($"The successful training step did not retain output '{key}'.");
        _plan?.ValidateRetainedOutputStep(_planStep);
        return output;
    }

    public void Dispose()
    {
        if (_disposed) return;
        if (!_completed)
        {
            if (!ReferenceEquals(_current, this))
                throw new InvalidOperationException("Retained output captures must be disposed in nesting order.");
            if (_activeUpdate?.BelongsTo(this) == true)
                throw new InvalidOperationException("The producer's update callback has not completed.");
            _current = _parent;
        }
        _disposed = true;
        _published = null;
        _plan = null;
        _eager.Clear();
    }
}
