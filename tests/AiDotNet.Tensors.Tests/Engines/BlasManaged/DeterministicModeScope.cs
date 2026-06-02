// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Pins the GEMM deterministic mode on the CALLING thread for the lifetime of the scope,
/// then restores the previous thread-local override on dispose.
///
/// <para>
/// Why thread-local rather than <see cref="BlasProvider.SetDeterministicMode"/>: the latter
/// mutates a process-global flag, and BlasManaged.Gemm reads the effective mode on the calling
/// thread (microkernel tile pick, autotune decision, strategy gate). When another test in a
/// parallel collection flips the global flag mid-call, the tile/reduction path changes — which
/// drifts the bit-match assertions and can reject a pre-pack handle into a re-pack from a
/// mutated source array (the long-standing flakiness of PrePackedB_Output_BitMatches_LivePack
/// and the cached-packed-buffer tests). <see cref="BlasProvider.IsDeterministicMode"/> resolves
/// the thread-local override BEFORE the global, so pinning it here makes the whole GEMM on this
/// thread immune to concurrent global flips — without depending on xUnit collection
/// parallelization being perfectly honored (it wasn't, on CI).
/// </para>
/// </summary>
internal readonly struct DeterministicModeScope : IDisposable
{
    private readonly bool? _previousThreadLocal;

    public DeterministicModeScope(bool deterministic)
    {
        _previousThreadLocal = BlasProvider.GetThreadLocalDeterministicMode();
        BlasProvider.SetThreadLocalDeterministicMode(deterministic);
    }

    public void Dispose() => BlasProvider.SetThreadLocalDeterministicMode(_previousThreadLocal);
}
