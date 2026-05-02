// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Policy that controls how <see cref="DirectGpuTensorEngine"/> handles
/// allocations that exceed the device's per-allocation cap (issue #285).
/// </summary>
public enum GpuChunkingPolicy
{
    /// <summary>
    /// Default. The engine first tries to chunk the operation across
    /// multiple smaller allocations; if chunking fails (op cannot decompose,
    /// or chunk count exceeds <see cref="GpuFallbackOptions.MaxChunkCount"/>),
    /// it falls through to the CPU path.
    /// </summary>
    AutoChunk = 0,

    /// <summary>
    /// Always chunk operations whose buffers exceed the cap. If chunking
    /// cannot decompose the op, the engine falls through to CPU. Equivalent
    /// to <see cref="AutoChunk"/> in current behaviour but reserved for a
    /// future "force chunk even under cap" debugging mode.
    /// </summary>
    AlwaysChunk = 1,

    /// <summary>
    /// Never chunk. If a buffer exceeds the cap, the
    /// <see cref="GpuBufferTooLargeException"/> propagates to the caller.
    /// Use this when you want the user to see the failure and reduce model
    /// size manually (matches pre-#285 behaviour, but with the typed
    /// exception in place of an opaque <c>InvalidOperationException</c>).
    /// </summary>
    NeverChunk_FailFast = 2,

    /// <summary>
    /// Never chunk. If a buffer exceeds the cap, fall through to CPU
    /// silently — same as <see cref="AutoChunk"/> but skip the chunking
    /// attempt. Useful when you don't trust the chunker's correctness for
    /// a specific op and prefer the slower-but-known CPU path.
    /// </summary>
    NeverChunk_FallbackToCpu = 3,
}

/// <summary>
/// User-tunable GPU fallback / chunking knobs for issue #285. Set via the
/// AiDotNet facade's <c>PredictionModelBuilder.ConfigureGpuAcceleration(...)</c>;
/// the facade maps its <c>GpuAccelerationConfig</c> onto these. Defaults
/// (apply when a property is null) are picked to do the right thing for
/// users who don't know about the cap.
/// </summary>
/// <remarks>
/// <para><b>Industry-standard defaults (applied when null):</b></para>
/// <list type="bullet">
/// <item><see cref="MaxBufferBytes"/>: device's <c>MaxBufferAllocBytes</c>
/// (queried at backend init).</item>
/// <item><see cref="ChunkingPolicy"/>: <see cref="GpuChunkingPolicy.AutoChunk"/>.</item>
/// <item><see cref="MaxChunkCount"/>: 64.</item>
/// </list>
/// </remarks>
public sealed class GpuFallbackOptions
{
    /// <summary>
    /// Override the auto-detected per-allocation cap. Useful for testing
    /// (force a low cap to exercise the chunker) or for devices that
    /// misreport. When null, the engine uses
    /// <see cref="IDirectGpuBackend.MaxBufferAllocBytes"/>.
    /// </summary>
    public long? MaxBufferBytes { get; init; }

    /// <summary>
    /// How the engine handles over-cap allocations. When null, defaults to
    /// <see cref="GpuChunkingPolicy.AutoChunk"/>.
    /// </summary>
    public GpuChunkingPolicy? ChunkingPolicy { get; init; }

    /// <summary>
    /// Maximum number of chunks an op can be split into before bailing to
    /// CPU. Prevents pathological cases (e.g. a 100 GiB allocation on a
    /// device with 256 MiB cap would otherwise want 400 chunks). When null,
    /// defaults to 64.
    /// </summary>
    public int? MaxChunkCount { get; init; }

    /// <summary>
    /// Conservative defaults — what the engine uses when no
    /// <see cref="GpuFallbackOptions"/> is explicitly configured.
    /// </summary>
    public static GpuFallbackOptions Default { get; } = new GpuFallbackOptions();

    // ── Internal accessors that apply the industry-standard defaults ──

    /// <summary>
    /// Resolves the effective per-allocation cap. When a user-supplied
    /// override (<see cref="MaxBufferBytes"/>) is set and is lower than
    /// the device's reported cap, the override wins; when it's greater
    /// than the device cap, the device cap wins (the device's actual
    /// rejection is the hard limit and we can't allocate past it).
    /// When no override is set, returns the device cap unchanged.
    /// </summary>
    internal long EffectiveMaxBufferBytes(long deviceCap)
    {
        if (!MaxBufferBytes.HasValue) return deviceCap;
        long userCap = MaxBufferBytes.Value;
        // If device cap is unknown (deviceCap == 0) trust the user value.
        if (deviceCap <= 0) return userCap;
        return Math.Min(userCap, deviceCap);
    }

    internal GpuChunkingPolicy EffectiveChunkingPolicy
        => ChunkingPolicy ?? GpuChunkingPolicy.AutoChunk;

    internal int EffectiveMaxChunkCount
        => MaxChunkCount ?? 64;
}

/// <summary>
/// Process-wide ambient instance of <see cref="GpuFallbackOptions"/>. The
/// AiDotNet facade sets this from the user-configured
/// <c>GpuAccelerationConfig</c>; everything in AiDotNet.Tensors that needs
/// to consult the policy reads <see cref="Current"/>.
/// </summary>
/// <remarks>
/// Process-wide rather than thread-local because GPU buffer caps are a
/// device property — they don't change per-thread, and the same training
/// run typically uses one engine for all worker threads. If a future
/// scenario needs per-thread overrides (e.g. one thread testing chunking
/// while others run normally), this can be promoted to thread-local
/// without breaking callers.
/// </remarks>
public static class GpuFallbackOptionsHolder
{
    private static GpuFallbackOptions _current = GpuFallbackOptions.Default;

    /// <summary>
    /// The currently-active fallback options. Reading and writing both go
    /// through <see cref="System.Threading.Volatile"/> so a writer thread's
    /// update is visible to other threads without explicit
    /// synchronisation. Never returns null — assigning null resets to
    /// <see cref="GpuFallbackOptions.Default"/>.
    /// </summary>
    public static GpuFallbackOptions Current
    {
        get => System.Threading.Volatile.Read(ref _current);
        set => System.Threading.Volatile.Write(ref _current, value ?? GpuFallbackOptions.Default);
    }
}
