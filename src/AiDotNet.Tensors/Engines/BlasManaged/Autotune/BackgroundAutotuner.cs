using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Bounded-LRU shape-sighting tracker (#375 G4). Gates background measurement to a
/// shape's 2nd+ sighting (one-shot shapes never sweep) and de-dups so concurrent
/// callers enqueue a shape once while its measurement is in flight. Shape identity
/// includes the transpose flags — a transB call is a different tuning target than
/// the non-transposed shape, and the shapes reaching strategy selection are typically
/// transposed (Sub-S handles non-transposed aligned GEMM).
/// </summary>
internal sealed class SightingTracker
{
    internal readonly record struct ShapeId(int M, int N, int K, bool Fp64, bool TransA, bool TransB);

    private readonly int _capacity;
    private readonly object _lock = new();
    private readonly LinkedList<ShapeId> _lru = new();
    private readonly Dictionary<ShapeId, (LinkedListNode<ShapeId> node, int count, bool inFlight)> _map = new();

    public SightingTracker(int capacity = 4096) => _capacity = capacity;

    /// <summary>Record a sighting; return true iff this shape should be measured now
    /// (2nd+ sighting and not already in flight).</summary>
    public bool RecordAndShouldMeasure(ShapeId id)
    {
        lock (_lock)
        {
            if (_map.TryGetValue(id, out var e))
            {
                _lru.Remove(e.node);
                _lru.AddFirst(e.node);
                int newCount = e.count + 1;
                bool measure = newCount >= 2 && !e.inFlight;
                _map[id] = (e.node, newCount, e.inFlight || measure);
                return measure;
            }
            if (_map.Count >= _capacity)
            {
                var oldest = _lru.Last!;
                _lru.RemoveLast();
                _map.Remove(oldest.Value);
            }
            var node = new LinkedListNode<ShapeId>(id);
            _lru.AddFirst(node);
            _map[id] = (node, 1, false);
            return false;
        }
    }

    /// <summary>Clear the in-flight flag after a measurement completes.</summary>
    public void MarkDone(ShapeId id)
    {
        lock (_lock)
        {
            if (_map.TryGetValue(id, out var e))
                _map[id] = (e.node, e.count, false);
        }
    }
}
