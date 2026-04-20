namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Diagnostic switch for the BERT × 100 state-bleed investigation (issue
/// driving this PR). Producers of gather-flavored ops call <see cref="Log"/>
/// only when <see cref="Enabled"/> is true, so the normal fast path pays a
/// single bool read. Messages are guarded by an internal lock, and no
/// external code can mutate the backing list or the counter directly —
/// forcing all mutation through the public methods keeps the state under
/// a single mutex.
/// </summary>
public static class GatherDebug
{
    private static readonly List<string> _messages = new();
    private static readonly object _lock = new();
    private static int _stepCounter;

    /// <summary>
    /// Master switch. Default off so production paths pay nothing; tests
    /// flip this on around the diagnostic region.
    /// </summary>
    public static bool Enabled { get; set; }

    /// <summary>Thread-safe incrementing step counter.</summary>
    public static int StepCounter
    {
        get { lock (_lock) return _stepCounter; }
        set { lock (_lock) _stepCounter = value; }
    }

    /// <summary>Increment the step counter by one and return the new value.</summary>
    public static int NextStep() { lock (_lock) return ++_stepCounter; }

    /// <summary>Append a diagnostic message (no-op when <see cref="Enabled"/> is false).</summary>
    public static void Log(string msg)
    {
        if (!Enabled) return;
        lock (_lock) _messages.Add(msg);
    }

    /// <summary>Clear the message buffer AND reset the step counter.</summary>
    public static void Clear()
    {
        lock (_lock)
        {
            _messages.Clear();
            _stepCounter = 0;
        }
    }

    /// <summary>Take a point-in-time copy of all messages logged so far.</summary>
    public static string[] Snapshot()
    {
        lock (_lock) return _messages.ToArray();
    }
}
