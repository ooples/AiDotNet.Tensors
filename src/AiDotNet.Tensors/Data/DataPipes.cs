// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Data;

/// <summary>
/// Functional, composable data-stream operators inspired by
/// <c>torchdata</c>. Each method takes an upstream
/// <see cref="IIterableDataset{TSample}"/> and returns a new dataset whose
/// enumeration applies the requested transformation lazily.
///
/// <para>Lazy means: nothing is fetched / computed until the consumer
/// iterates, and the transformation chain composes without materializing
/// intermediate lists. A pipeline of <c>.Map(...).Filter(...).Batch(...)</c>
/// reads from the upstream once per output batch.</para>
///
/// <para>Equivalent surface to PyTorch's <c>IterDataPipe</c> family
/// (<c>.map</c>, <c>.filter</c>, <c>.batch</c>, <c>.shuffle</c>,
/// <c>.bucketbatch</c>, <c>.zip</c>, <c>.concat</c>, <c>.mux</c>,
/// <c>.demultiplex</c>, <c>.sharding_filter</c>).</para>
/// </summary>
public static class DataPipeExtensions
{
    /// <summary>Applies <paramref name="fn"/> to each upstream sample and
    /// emits the result. Composes — chained <c>.Map</c> calls fold without
    /// materializing.</summary>
    public static IIterableDataset<TOut> Map<TIn, TOut>(
        this IIterableDataset<TIn> source, Func<TIn, TOut> fn)
        => new MapPipe<TIn, TOut>(source, fn);

    /// <summary>Drops samples where <paramref name="predicate"/> returns
    /// false. Lazy.</summary>
    public static IIterableDataset<TSample> Filter<TSample>(
        this IIterableDataset<TSample> source, Func<TSample, bool> predicate)
        => new FilterPipe<TSample>(source, predicate);

    /// <summary>Groups samples into fixed-size lists. The trailing partial
    /// batch is emitted unless <paramref name="dropLast"/> is true.</summary>
    public static IIterableDataset<IReadOnlyList<TSample>> Batch<TSample>(
        this IIterableDataset<TSample> source, int batchSize, bool dropLast = false)
        => new BatchPipe<TSample>(source, batchSize, dropLast);

    /// <summary>Reservoir-shuffles upstream samples through a fixed-size
    /// buffer. Larger <paramref name="bufferSize"/> ⇒ better mixing,
    /// higher memory. Same algorithm as
    /// <c>torch.utils.data.IterDataPipe.shuffle(buffer_size=...)</c>.</summary>
    public static IIterableDataset<TSample> Shuffle<TSample>(
        this IIterableDataset<TSample> source, int bufferSize, int? seed = null)
        => new ShufflePipe<TSample>(source, bufferSize, seed);

    /// <summary>Bucket-batching for variable-length sequences: groups
    /// samples by <paramref name="key"/> into a small number of buckets
    /// then emits batches from each bucket as it fills. Reduces padding
    /// overhead when the cost is dominated by max-length-per-batch.</summary>
    public static IIterableDataset<IReadOnlyList<TSample>> BucketBatch<TSample>(
        this IIterableDataset<TSample> source, int batchSize,
        int bucketCount, Func<TSample, int> key, int? bufferSize = null)
        => new BucketBatchPipe<TSample>(source, batchSize, bucketCount, key,
                                         bufferSize ?? batchSize * bucketCount * 4);

    /// <summary>Yields tuples by zipping two upstream streams element-wise.
    /// Stops at the shorter stream.</summary>
    public static IIterableDataset<(TA, TB)> Zip<TA, TB>(
        this IIterableDataset<TA> a, IIterableDataset<TB> b)
        => new ZipPipe<TA, TB>(a, b);

    /// <summary>Concatenates streams end-to-end (delegates to
    /// <see cref="ChainDataset{TSample}"/>).</summary>
    public static IIterableDataset<TSample> Concat<TSample>(
        this IIterableDataset<TSample> first, IIterableDataset<TSample> second)
        => new ChainDataset<TSample>(new[] { first, second });

    /// <summary>Round-robin multiplex — emits one sample from each upstream
    /// in turn. Stops when ANY upstream is exhausted.</summary>
    public static IIterableDataset<TSample> Mux<TSample>(
        params IIterableDataset<TSample>[] sources)
        => new MuxPipe<TSample>(sources);

    /// <summary>Demultiplex — splits one upstream into N sub-streams using
    /// <paramref name="route"/> to pick the destination index per sample.
    /// Returns an array of <c>N</c> downstream datasets that share the
    /// upstream enumeration; consume them concurrently to avoid one
    /// branch starving the others.</summary>
    public static IIterableDataset<TSample>[] Demultiplex<TSample>(
        this IIterableDataset<TSample> source, int branches, Func<TSample, int> route)
    {
        if (branches < 1) throw new ArgumentOutOfRangeException(nameof(branches));
        return DemultiplexPipe<TSample>.Build(source, branches, route);
    }

    /// <summary>Sharding filter — yields every <c>worldSize</c>-th sample
    /// starting from <paramref name="rank"/>. Used inside multi-worker
    /// pipelines so each worker sees a disjoint shard of the upstream.</summary>
    public static IIterableDataset<TSample> ShardingFilter<TSample>(
        this IIterableDataset<TSample> source, int rank, int worldSize)
        => new ShardingFilterPipe<TSample>(source, rank, worldSize);

    // ────────────────────────────────────────────────────────────────────
    //  Pipe implementations — kept private to this file. Each is a tiny
    //  adapter over the upstream's IEnumerator; lazy semantics fall out for
    //  free from yield-return.
    // ────────────────────────────────────────────────────────────────────

    private sealed class MapPipe<TIn, TOut> : IIterableDataset<TOut>
    {
        private readonly IIterableDataset<TIn> _src;
        private readonly Func<TIn, TOut> _fn;
        public MapPipe(IIterableDataset<TIn> src, Func<TIn, TOut> fn) { _src = src; _fn = fn; }
        public IEnumerator<TOut> GetEnumerator()
        {
            using var e = _src.GetEnumerator();
            while (e.MoveNext()) yield return _fn(e.Current);
        }
    }

    private sealed class FilterPipe<TSample> : IIterableDataset<TSample>
    {
        private readonly IIterableDataset<TSample> _src;
        private readonly Func<TSample, bool> _p;
        public FilterPipe(IIterableDataset<TSample> src, Func<TSample, bool> p) { _src = src; _p = p; }
        public IEnumerator<TSample> GetEnumerator()
        {
            using var e = _src.GetEnumerator();
            while (e.MoveNext()) if (_p(e.Current)) yield return e.Current;
        }
    }

    private sealed class BatchPipe<TSample> : IIterableDataset<IReadOnlyList<TSample>>
    {
        private readonly IIterableDataset<TSample> _src;
        private readonly int _batch;
        private readonly bool _dropLast;
        public BatchPipe(IIterableDataset<TSample> src, int batch, bool dropLast)
        {
            if (batch < 1) throw new ArgumentOutOfRangeException(nameof(batch));
            _src = src; _batch = batch; _dropLast = dropLast;
        }
        public IEnumerator<IReadOnlyList<TSample>> GetEnumerator()
        {
            var bucket = new List<TSample>(_batch);
            using var e = _src.GetEnumerator();
            while (e.MoveNext())
            {
                bucket.Add(e.Current);
                if (bucket.Count == _batch)
                {
                    yield return bucket;
                    bucket = new List<TSample>(_batch);
                }
            }
            if (bucket.Count > 0 && !_dropLast) yield return bucket;
        }
    }

    private sealed class ShufflePipe<TSample> : IIterableDataset<TSample>
    {
        private readonly IIterableDataset<TSample> _src;
        private readonly int _bufferSize;
        private readonly int? _seed;
        public ShufflePipe(IIterableDataset<TSample> src, int bufferSize, int? seed)
        {
            if (bufferSize < 1) throw new ArgumentOutOfRangeException(nameof(bufferSize));
            _src = src; _bufferSize = bufferSize; _seed = seed;
        }
        public IEnumerator<TSample> GetEnumerator()
        {
            var rng = _seed.HasValue ? new Random(_seed.Value) : new Random();
            var buffer = new List<TSample>(_bufferSize);
            using var e = _src.GetEnumerator();
            // Prime the buffer.
            while (buffer.Count < _bufferSize && e.MoveNext()) buffer.Add(e.Current);
            // Steady-state: pop a random buffer slot, replace from upstream.
            while (e.MoveNext())
            {
                int idx = rng.Next(buffer.Count);
                yield return buffer[idx];
                buffer[idx] = e.Current;
            }
            // Drain — random remaining order.
            while (buffer.Count > 0)
            {
                int idx = rng.Next(buffer.Count);
                yield return buffer[idx];
                buffer[idx] = buffer[buffer.Count - 1];
                buffer.RemoveAt(buffer.Count - 1);
            }
        }
    }

    private sealed class BucketBatchPipe<TSample> : IIterableDataset<IReadOnlyList<TSample>>
    {
        private readonly IIterableDataset<TSample> _src;
        private readonly int _batch;
        private readonly int _bucketCount;
        private readonly Func<TSample, int> _key;
        private readonly int _bufferSize;
        public BucketBatchPipe(IIterableDataset<TSample> src, int batch, int bucketCount, Func<TSample, int> key, int bufferSize)
        {
            if (batch < 1) throw new ArgumentOutOfRangeException(nameof(batch));
            if (bucketCount < 1) throw new ArgumentOutOfRangeException(nameof(bucketCount));
            _src = src; _batch = batch; _bucketCount = bucketCount; _key = key; _bufferSize = bufferSize;
        }
        public IEnumerator<IReadOnlyList<TSample>> GetEnumerator()
        {
            // Fill a buffer up to bufferSize, sort by key, slice into buckets,
            // emit batches from each bucket. Repeat until the upstream drains.
            using var e = _src.GetEnumerator();
            var buf = new List<TSample>(_bufferSize);
            while (true)
            {
                buf.Clear();
                while (buf.Count < _bufferSize && e.MoveNext()) buf.Add(e.Current);
                if (buf.Count == 0) yield break;

                buf.Sort((a, b) => _key(a).CompareTo(_key(b)));
                int per = (buf.Count + _bucketCount - 1) / _bucketCount;
                for (int b = 0; b < _bucketCount; b++)
                {
                    int from = b * per;
                    if (from >= buf.Count) break;
                    int to = Math.Min(from + per, buf.Count);
                    for (int i = from; i < to; i += _batch)
                    {
                        int len = Math.Min(_batch, to - i);
                        var batch = new List<TSample>(len);
                        for (int j = 0; j < len; j++) batch.Add(buf[i + j]);
                        yield return batch;
                    }
                }
            }
        }
    }

    private sealed class ZipPipe<TA, TB> : IIterableDataset<(TA, TB)>
    {
        private readonly IIterableDataset<TA> _a;
        private readonly IIterableDataset<TB> _b;
        public ZipPipe(IIterableDataset<TA> a, IIterableDataset<TB> b) { _a = a; _b = b; }
        public IEnumerator<(TA, TB)> GetEnumerator()
        {
            using var ea = _a.GetEnumerator();
            using var eb = _b.GetEnumerator();
            while (ea.MoveNext() && eb.MoveNext()) yield return (ea.Current, eb.Current);
        }
    }

    private sealed class MuxPipe<TSample> : IIterableDataset<TSample>
    {
        private readonly IIterableDataset<TSample>[] _srcs;
        public MuxPipe(IIterableDataset<TSample>[] srcs)
        {
            if (srcs is null || srcs.Length == 0) throw new ArgumentException("Mux requires at least one source.");
            _srcs = srcs;
        }
        public IEnumerator<TSample> GetEnumerator()
        {
            var enumerators = new IEnumerator<TSample>[_srcs.Length];
            for (int i = 0; i < _srcs.Length; i++) enumerators[i] = _srcs[i].GetEnumerator();
            try
            {
                while (true)
                {
                    for (int i = 0; i < enumerators.Length; i++)
                    {
                        if (!enumerators[i].MoveNext()) yield break;
                        yield return enumerators[i].Current;
                    }
                }
            }
            finally { foreach (var e in enumerators) e.Dispose(); }
        }
    }

    private sealed class DemultiplexPipe<TSample>
    {
        public static IIterableDataset<TSample>[] Build(IIterableDataset<TSample> src, int branches, Func<TSample, int> route)
        {
            // Shared enumerator + per-branch queue. Each branch dataset's
            // enumerator pulls from the upstream until its queue has a value.
            // Concurrent-safe via a lock — multi-threaded consumption of the
            // returned datasets is supported.
            var queues = new Queue<TSample>[branches];
            for (int i = 0; i < branches; i++) queues[i] = new Queue<TSample>();
            var gate = new object();
            IEnumerator<TSample>? upstream = null;
            bool drained = false;

            IEnumerator<TSample> Upstream()
            {
                upstream ??= src.GetEnumerator();
                return upstream;
            }

            void PumpUntil(int branch)
            {
                lock (gate)
                {
                    while (queues[branch].Count == 0 && !drained)
                    {
                        var e = Upstream();
                        if (!e.MoveNext()) { drained = true; e.Dispose(); break; }
                        int target = route(e.Current);
                        if ((uint)target >= (uint)branches)
                            throw new InvalidOperationException(
                                $"Demultiplex route returned {target} which is outside [0, {branches}).");
                        queues[target].Enqueue(e.Current);
                    }
                }
            }

            var datasets = new IIterableDataset<TSample>[branches];
            for (int i = 0; i < branches; i++)
            {
                int branchId = i;
                datasets[i] = new InlineIterable<TSample>(() => Drain(branchId, queues, gate, PumpUntil));
            }
            return datasets;
        }

        private static IEnumerator<TSample> Drain(int branch, Queue<TSample>[] queues, object gate, Action<int> pump)
        {
            while (true)
            {
                pump(branch);
                lock (gate)
                {
                    if (queues[branch].Count == 0) yield break;
                    yield return queues[branch].Dequeue();
                }
            }
        }
    }

    private sealed class InlineIterable<TSample> : IIterableDataset<TSample>
    {
        private readonly Func<IEnumerator<TSample>> _factory;
        public InlineIterable(Func<IEnumerator<TSample>> factory) { _factory = factory; }
        public IEnumerator<TSample> GetEnumerator() => _factory();
    }

    private sealed class ShardingFilterPipe<TSample> : IIterableDataset<TSample>
    {
        private readonly IIterableDataset<TSample> _src;
        private readonly int _rank, _world;
        public ShardingFilterPipe(IIterableDataset<TSample> src, int rank, int world)
        {
            if (world < 1) throw new ArgumentOutOfRangeException(nameof(world));
            if (rank < 0 || rank >= world) throw new ArgumentOutOfRangeException(nameof(rank));
            _src = src; _rank = rank; _world = world;
        }
        public IEnumerator<TSample> GetEnumerator()
        {
            using var e = _src.GetEnumerator();
            int i = 0;
            while (e.MoveNext())
            {
                if (i % _world == _rank) yield return e.Current;
                i++;
            }
        }
    }
}
