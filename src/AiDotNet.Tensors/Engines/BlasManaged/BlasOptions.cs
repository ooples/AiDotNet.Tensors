using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public readonly ref struct BlasOptions<T> where T : unmanaged
{
    // Fields filled in by Task A3.
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
