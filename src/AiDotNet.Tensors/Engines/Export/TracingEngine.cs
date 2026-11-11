using System;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>
/// An <see cref="IEngine"/> decorator that forwards every call to an inner engine and reports each outer call to a
/// recorder. Its members are generated from IEngine, so it covers the whole engine surface.
/// </summary>
/// <remarks>
/// Only outer calls are recorded: an engine implementation that calls another public op on itself does not go through
/// the wrapper, and one that calls back through <see cref="AiDotNetEngine.Current"/> is nested, which the recorder
/// suppresses. The recorded graph is therefore the logical program the caller wrote, before any fusion.
/// </remarks>
public sealed partial class TracingEngine
{
    private readonly IEngine _inner;
    private readonly IEngineCallRecorder _recorder;

    internal TracingEngine(IEngine inner, IEngineCallRecorder recorder)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _recorder = recorder ?? throw new ArgumentNullException(nameof(recorder));
    }

    /// <summary>The engine every call is forwarded to.</summary>
    public IEngine Inner => _inner;
}
