using System;
using AiDotNet.Tensors.Engines;  // for FusedActivationType

namespace AiDotNet.Tensors.Engines.BlasManaged;

public readonly ref struct BlasOptions<T> where T : unmanaged
{
    /// <summary>Packing strategy for A and B. <see cref="PackingMode.Auto"/> is safe for all shapes.</summary>
    public PackingMode PackingMode { get; init; }

    /// <summary>Epilogue operations applied after the matrix multiply (bias, activation, skip, dropout, output scale).</summary>
    public Epilogue<T> Epilogue { get; init; }

    /// <summary>
    /// Caller-supplied scratch memory. Empty (default) = allocate internally via the
    /// allocator layers (per-thread pool, ArrayPool, etc.). When supplied, must be
    /// large enough to hold packed A + packed B + any per-thread partials — exact
    /// minimum depends on packing mode and (M, N, K) shape.
    /// </summary>
    public Span<byte> Workspace { get; init; }

    /// <summary>
    /// Pre-packed A buffer from <see cref="BlasManaged.PrePackA"/>. Null = pack on
    /// first call. Reuse across training iterations to amortize pack cost.
    /// </summary>
    public WeightPackHandle? PackedA { get; init; }

    /// <summary>
    /// Pre-packed B buffer from <see cref="BlasManaged.PrePackB"/>. Null = pack on
    /// first call. Reuse across training iterations to amortize pack cost.
    /// </summary>
    public WeightPackHandle? PackedB { get; init; }

    /// <summary>0 = autotune; -1 = single-thread (deterministic); positive = pin to N.</summary>
    public int NumThreads { get; init; }
    /// <summary>0 = derive from shape; nonzero = caller-supplied autotune key.</summary>
    public ulong AutotuneKey { get; init; }
    /// <summary>0 = use process default (64 MB).</summary>
    public long MaxJitCacheBytes { get; init; }
}

public enum PackingMode
{
    /// <summary>
    /// Dispatcher selects a strategy per shape via built-in heuristics
    /// (K-size thresholds, M·N work cutoff). Once Phase H lands, an autotune
    /// cache will further refine the choice based on measured per-shape timings.
    /// </summary>
    Auto,
    /// <summary>Always pack both A and B. Forces the 3-level Goto loop nest.</summary>
    ForcePackBoth,
    /// <summary>Pack A only; B is read in-place from caller memory.</summary>
    ForcePackAOnly,
    /// <summary>No pack. Microkernel reads A and B in native stride. Best for K&lt;32.</summary>
    ForceStreaming,
    /// <summary>Use cached autotune choice if present; never benchmark on first call.</summary>
    DisableAutotune,
}

public readonly ref struct Epilogue<T> where T : unmanaged
{
    /// <summary>Bias vector of length N. Empty = no bias.</summary>
    public ReadOnlySpan<T> BiasN { get; init; }
    /// <summary>
    /// Fused post-multiplication activation. <see cref="FusedActivationType.None"/> = identity (no activation).
    /// </summary>
    public FusedActivationType Activation { get; init; }
    /// <summary>Skip-connection tensor of shape (M, N) in row-major. Empty = no skip.</summary>
    public ReadOnlySpan<T> SkipMxN { get; init; }
    /// <summary>Dropout RNG state. 0 = no dropout (inference).</summary>
    public uint DropoutMask { get; init; }
    /// <summary>
    /// Output scale. When <see cref="HasOutputScale"/> is <c>false</c> (default),
    /// <c>OutputScale = default(T)</c> means "disabled" and a non-default value
    /// implicitly enables scaling. Set <see cref="HasOutputScale"/> to <c>true</c>
    /// to opt in to explicit scaling — including <c>OutputScale = 0</c>, which
    /// otherwise (and as PR #366 / copilot review pointed out) is indistinguishable
    /// from "disabled" and so previously couldn't be used to intentionally zero
    /// the output (useful for masking / debug).
    /// </summary>
    public T OutputScale { get; init; }
    /// <summary>
    /// Explicit opt-in for output scaling. When <c>true</c>, the dispatcher always
    /// applies <see cref="OutputScale"/> (including 0); when <c>false</c>, scaling
    /// is enabled only when <see cref="OutputScale"/> is non-default. Defaults to
    /// <c>false</c> so existing callers that set <see cref="OutputScale"/> to a
    /// non-zero value continue to work without changes.
    /// </summary>
    public bool HasOutputScale { get; init; }
}
