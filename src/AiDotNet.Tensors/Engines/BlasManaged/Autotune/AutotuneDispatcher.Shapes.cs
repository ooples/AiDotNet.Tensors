using System;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.BlasManaged;

internal static partial class AutotuneDispatcher
{
    // ShapeProfile is immutable and defensively copies its params array. Cache the
    // profile, not the tuning decision: every call still observes cache updates and
    // all measurement/blocking overrides. The persisted shape encoding is unchanged.
    private const int DispatcherShapeCapacity = 1024;
    private static readonly ConcurrentDictionary<(int M, int N, int K, int Mr, int Nr, int Flags), ShapeProfile> DispatcherShapes = new();

    internal static ShapeProfile DispatcherShape<T>(int m, int n, int k, bool transA, bool transB,
        int mr, int nr, bool hasEpilogue, bool isDeterministic) where T : unmanaged
    {
        int precision = typeof(T) == typeof(float) ? 0 : typeof(T) == typeof(double) ? 16
            : throw new NotSupportedException($"Autotune does not support T={typeof(T).Name}.");
        int flags = precision | (transA ? 1 : 0) | (transB ? 2 : 0) | (hasEpilogue ? 4 : 0) | (isDeterministic ? 8 : 0);
        var key = (m, n, k, mr, nr, flags);
        if (DispatcherShapes.TryGetValue(key, out var shape)) return shape;
        shape = BlasManagedAutotune.EncodeShape<T>(m, n, k, transA, transB, mr, nr, hasEpilogue, isDeterministic);
        if (DispatcherShapes.Count >= DispatcherShapeCapacity) DispatcherShapes.Clear();
        DispatcherShapes.TryAdd(key, shape);
        return shape;
    }
}
