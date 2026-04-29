// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.Profiling;

/// <summary>
/// Backends/event-sources the profiler captures from. Mirrors the
/// <c>torch.profiler.ProfilerActivity</c> enum so the mental model carries
/// over for users coming from PyTorch.
/// </summary>
[System.Flags]
public enum ProfilerActivities
{
    /// <summary>Capture nothing. Equivalent to disabling the profiler entirely.</summary>
    None = 0,

    /// <summary>CPU host events: user-annotated ranges, op dispatch, autotune
    /// hits/misses, compile-pass timings.</summary>
    Cpu = 1 << 0,

    /// <summary>GPU device events (kernels, H2D/D2H, cuBLAS/cuDNN). Wired via
    /// the GPU-primitives surface (#219); no-op on CPU-only builds today.</summary>
    Gpu = 1 << 1,

    /// <summary>Memory allocator events (alloc/free, pool churn). Drives the
    /// memory timeline export — see <c>MemoryProfiler</c> (Phase 2).</summary>
    Memory = 1 << 2,

    /// <summary>Convenience: CPU + GPU + Memory.</summary>
    All = Cpu | Gpu | Memory,
}
