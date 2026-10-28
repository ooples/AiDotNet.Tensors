// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The identity of the convolution weights the engine is about to hand a backend, for caches of forms derived from
/// them (a Winograd filter transform). A backend sees only a device buffer, which does not say whether the weights in
/// it changed; the engine knows the host tensor and its version. Set only around one backend call, and only when the
/// buffer is host-authoritative (no tape, no resident step, no device-resident weight a device optimizer updates in
/// place), so a hit is exactly "the same host weights at the same version".
/// </summary>
internal static class ConvWeightIdentity
{
    [ThreadStatic] private static object? t_key;
    [ThreadStatic] private static int t_version;

    /// <summary>The weights' identity for the current backend call, or false when derived forms must not be cached.</summary>
    internal static bool TryGet(out object key, out int version)
    {
        key = t_key ?? string.Empty;
        version = t_version;
        return t_key is not null;
    }

    /// <summary>Scopes an identity to one backend call; disposing clears it.</summary>
    internal static Scope Enter(object? key, int version)
    {
        t_key = key;
        t_version = version;
        return default;
    }

    internal readonly struct Scope : IDisposable
    {
        public void Dispose()
        {
            t_key = null;
            t_version = 0;
        }
    }
}
